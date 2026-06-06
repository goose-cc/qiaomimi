import numpy as np
from scipy.ndimage import gaussian_filter1d


class Fake97Model:
    """
    Synthetic 97% initializer.

    注意：
    这个不是严格意义上的真实反演模型。
    它输入 fx_true，输出一个接近 fx_true 的 fx_pred。
    适合用来模拟“已有一个 97% 准确的初始谱函数”，然后测试 MCTS refinement。
    """

    def __init__(
        self,
        target_quality=0.97,
        quality_jitter=0.004,
        random_state=None,
    ):
        """
        target_quality:
            近似质量。例如 0.97 表示相对 L2 误差约为 3%。

        quality_jitter:
            每次生成时允许质量有轻微波动，避免所有样本误差完全一样。
        """
        if not 0.0 < target_quality < 1.0:
            raise ValueError("target_quality must be between 0 and 1.")

        self.target_quality = target_quality
        self.quality_jitter = quality_jitter
        self.rng = np.random.default_rng(random_state)
        self.eps = 1e-12

    @staticmethod
    def relative_l2_error(fx_true, fx_pred):
        fx_true = np.asarray(fx_true, dtype=float)
        fx_pred = np.asarray(fx_pred, dtype=float)

        return np.linalg.norm(fx_pred - fx_true) / (
            np.linalg.norm(fx_true) + 1e-12
        )

    @staticmethod
    def quality(fx_true, fx_pred):
        """
        quality = 1 - relative_l2_error
        """
        return 1.0 - Fake97Model.relative_l2_error(fx_true, fx_pred)

    def _make_grid(self, x, n):
        if x is None:
            return np.linspace(0.0, 1.0, n)

        x = np.asarray(x, dtype=float)

        if len(x) != n:
            raise ValueError("x and fx_true must have the same length.")

        span = np.ptp(x)
        if span < self.eps:
            return np.linspace(0.0, 1.0, n)

        return (x - x.min()) / span

    def _gaussian(self, grid, amp, center, width):
        return amp * np.exp(
            -0.5 * ((grid - center) / (width + self.eps)) ** 2
        )

    def _sample_target_relative_error(self):
        q = self.target_quality + self.rng.normal(
            0.0,
            self.quality_jitter,
        )

        q = np.clip(q, 0.75, 0.995)

        return 1.0 - q

    def _raw_perturb(self, fx_true, grid):
        fx = fx_true.copy().astype(float)

        max_fx = max(float(np.max(fx_true)), self.eps)

        # 1. 整体幅度误差
        fx *= self.rng.uniform(0.90, 1.10)

        # 2. 峰位轻微偏移
        if self.rng.random() < 0.70:
            shift = self.rng.uniform(-0.015, 0.015)
            fx = np.interp(
                grid - shift,
                grid,
                fx,
                left=0.0,
                right=0.0,
            )

        # 3. 局部强度扰动
        for _ in range(self.rng.integers(1, 4)):
            center = self.rng.uniform(0.0, 1.0)
            width = self.rng.uniform(0.02, 0.12)
            amp = self.rng.uniform(-0.12, 0.12)

            mask = self._gaussian(
                grid,
                1.0,
                center,
                width,
            )

            fx *= 1.0 + amp * mask

        # 4. 平滑误差
        if self.rng.random() < 0.75:
            fx = gaussian_filter1d(
                fx,
                sigma=self.rng.uniform(0.3, 1.5),
            )

        # 5. 多余小峰
        if self.rng.random() < 0.60:
            for _ in range(self.rng.integers(1, 4)):
                center = self.rng.uniform(0.0, 1.0)
                width = self.rng.uniform(0.004, 0.035)
                amp = self.rng.uniform(0.005, 0.04) * max_fx

                fx += self._gaussian(
                    grid,
                    amp,
                    center,
                    width,
                )

        # 6. 局部漏峰 / 削弱
        if self.rng.random() < 0.60:
            for _ in range(self.rng.integers(1, 3)):
                center = self.rng.uniform(0.0, 1.0)
                width = self.rng.uniform(0.01, 0.06)
                depth = self.rng.uniform(0.04, 0.20)

                mask = self._gaussian(
                    grid,
                    1.0,
                    center,
                    width,
                )

                fx *= 1.0 - depth * mask

        # 7. 低频背景误差
        if self.rng.random() < 0.50:
            freq = self.rng.uniform(0.5, 2.0)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            amp = self.rng.uniform(-0.015, 0.025) * max_fx

            background = amp * np.sin(
                2.0 * np.pi * freq * grid + phase
            )

            fx += background

        # 8. 小噪声
        noise = self.rng.normal(
            0.0,
            0.015 * max_fx,
            size=fx.shape,
        )

        fx += noise

        fx = np.clip(fx, 0.0, None)

        return fx

    def _rescale_to_target_error(self, fx_true, fx_raw, target_rel_error):
        """
        把 raw perturbation 的误差缩放到目标相对误差附近。
        """
        fx_true = np.asarray(fx_true, dtype=float)
        fx_raw = np.asarray(fx_raw, dtype=float)

        base_norm = np.linalg.norm(fx_true) + self.eps
        delta = fx_raw - fx_true
        current_rel_error = np.linalg.norm(delta) / base_norm

        if current_rel_error < self.eps:
            noise = self.rng.normal(0.0, 1.0, size=fx_true.shape)
            noise = gaussian_filter1d(noise, sigma=1.0)
            noise_norm = np.linalg.norm(noise) + self.eps

            delta = noise / noise_norm * target_rel_error * base_norm
            fx_scaled = fx_true + delta
            return np.clip(fx_scaled, 0.0, None)

        scale = target_rel_error / current_rel_error
        fx_scaled = fx_true + scale * delta
        fx_scaled = np.clip(fx_scaled, 0.0, None)

        # clipping 后误差会略变，再校准一次
        delta = fx_scaled - fx_true
        current_rel_error = np.linalg.norm(delta) / base_norm

        if current_rel_error > self.eps:
            scale = target_rel_error / current_rel_error
            fx_scaled = fx_true + scale * delta
            fx_scaled = np.clip(fx_scaled, 0.0, None)

        return fx_scaled

    def predict(self, fx_true, x=None):
        """
        输入:
            fx_true: 真实谱函数
            x: 可选，对应的 x 网格

        输出:
            fx_pred: 模拟 97% 模型输出
        """
        fx_true = np.asarray(fx_true, dtype=float)

        if fx_true.ndim != 1:
            raise ValueError("fx_true must be a 1D array.")

        grid = self._make_grid(x, len(fx_true))

        target_rel_error = self._sample_target_relative_error()

        fx_raw = self._raw_perturb(
            fx_true,
            grid,
        )

        fx_pred = self._rescale_to_target_error(
            fx_true,
            fx_raw,
            target_rel_error,
        )

        return fx_pred

    def predict_batch(self, fx_true_batch, x=None):
        fx_true_batch = np.asarray(fx_true_batch, dtype=float)

        if fx_true_batch.ndim != 2:
            raise ValueError("fx_true_batch must have shape (batch, n).")

        preds = []

        for fx_true in fx_true_batch:
            preds.append(
                self.predict(
                    fx_true,
                    x=x,
                )
            )

        return np.asarray(preds)
