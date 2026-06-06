import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from scipy.ndimage import gaussian_filter1d

from data_generate import compute_gy


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(
        -0.5 * ((x - mu) / (sigma + 1e-12)) ** 2
    )


@dataclass
class Action:
    kind: str
    amp: float
    mu: float
    sigma: float
    smooth_sigma: float = 0.0
    prior: float = 1.0


@dataclass
class Node:
    fx: np.ndarray
    score: float
    parent: Optional["Node"] = None
    action: Optional[Action] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTSRefinement:
    """
    MCTS refinement for inverse problem.

    这个类不会使用 fx_true。
    它只使用:
        1. 当前候选 fx
        2. forward model: compute_gy(x, fx, y)
        3. 目标观测 gy_target
        4. 光滑、曲率等先验
    """

    def __init__(
        self,
        x,
        y,
        gy_target,
        compute_gy_fn: Callable = compute_gy,
        iterations=500,
        rollout_depth=3,
        lambda_tv=0.01,
        lambda_curv=0.002,
        lambda_prior=0.15,
        lambda_mass=0.03,
        exploration=1.5,
        max_children=20,
        progressive_c=2.0,
        progressive_alpha=0.5,
        peak_width_range=(0.003, 0.06),
        peak_amp_range=(0.005, 0.08),
        random_state=None,
        policy_fn=None,
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.gy_target = np.asarray(gy_target, dtype=float)

        self.compute_gy_fn = compute_gy_fn

        self.iterations = iterations
        self.rollout_depth = rollout_depth

        self.lambda_tv = lambda_tv
        self.lambda_curv = lambda_curv
        self.lambda_prior = lambda_prior
        self.lambda_mass = lambda_mass

        self.exploration = exploration
        self.max_children = max_children
        self.progressive_c = progressive_c
        self.progressive_alpha = progressive_alpha

        self.peak_width_range = peak_width_range
        self.peak_amp_range = peak_amp_range

        self.policy_fn = policy_fn

        self.rng = np.random.default_rng(random_state)

        self.eps = 1e-12
        self.gy_norm = np.mean(self.gy_target ** 2) + self.eps
        self.x_span = self.x.max() - self.x.min() + self.eps

        # refinement prior，会在 refine(fx_init) 里设置
        self.fx_prior = None
        self.fx_prior_norm = None
        self.fx_prior_mass = None

    


    def project(self, fx):
        """
        物理约束:
        谱函数非负。
        """
        fx = np.asarray(fx, dtype=float)
        fx = np.clip(fx, 0.0, None)
        return fx

    def score_parts(self, fx):
        fx = self.project(fx)

        gy_pred = self.compute_gy_fn(
            self.x,
            fx,
            self.y,
        )

        mse = np.mean(
            (gy_pred - self.gy_target) ** 2
        ) / self.gy_norm

        fx_level_1 = np.mean(np.abs(fx)) + self.eps
        fx_level_2 = np.mean(fx ** 2) + self.eps

        tv = np.mean(np.abs(np.diff(fx))) / fx_level_1

        if len(fx) >= 3:
            curv = np.mean(np.diff(fx, n=2) ** 2) / fx_level_2
        else:
            curv = 0.0

        # -----------------------------
        # 新增：不要离初始 88% 模型太远
        # -----------------------------
        if self.fx_prior is not None:
            prior = np.mean(
                (fx - self.fx_prior) ** 2
            ) / (self.fx_prior_norm + self.eps)

            mass = (
                (
                    np.trapezoid(fx, self.x)
                    - self.fx_prior_mass
                )
                / (abs(self.fx_prior_mass) + self.eps)
            ) ** 2
        else:
            prior = 0.0
            mass = 0.0

        score = (
            mse
            + self.lambda_tv * tv
            + self.lambda_curv * curv
            + self.lambda_prior * prior
            + self.lambda_mass * mass
        )

        return {
            "score": float(score),
            "mse": float(mse),
            "tv": float(tv),
            "curv": float(curv),
            "prior": float(prior),
            "mass": float(mass),
        }


    def evaluate(self, fx):
        return self.score_parts(fx)["score"]

    def propose_action(self, fx):
        """
        默认随机 action proposal。

        后面你接深度学习时，可以通过 policy_fn 替代这里。
        policy_fn 应该返回一个 Action。
        """

        if self.policy_fn is not None:
            action = self.policy_fn(
                fx=fx,
                x=self.x,
                y=self.y,
                gy_target=self.gy_target,
                compute_gy_fn=self.compute_gy_fn,
                rng=self.rng,
            )

            if action is not None:
                return action

        kind = self.rng.choice(
            ["add", "sub", "scale",  "shift", "sharpen", "smooth"],
            p=[0.25, 0.18, 0.22, 0.22, 0.10, 0.03],
        )

        mu = self.rng.uniform(
            self.x.min(),
            self.x.max(),
        )

        sigma = self.rng.uniform(
            self.peak_width_range[0],
            self.peak_width_range[1],
        ) * self.x_span

        fx_level = max(float(np.max(fx)), 1e-3)

        if kind in ["add", "sub"]:
            amp = self.rng.uniform(
                self.peak_amp_range[0],
                self.peak_amp_range[1],
            ) * fx_level

            return Action(
                kind=kind,
                amp=amp,
                mu=mu,
                sigma=sigma,
                prior=1.0,
            )

        if kind == "scale":
            amp = self.rng.uniform(-0.25, 0.25)

            return Action(
                kind=kind,
                amp=amp,
                mu=mu,
                sigma=sigma,
                prior=1.0,
            )

        if kind == "shift":
            # 局部峰位移动，amp 表示 x 方向位移
            amp = self.rng.uniform(-0.04, 0.04) * self.x_span

            return Action(
                kind=kind,
                amp=amp,
                mu=mu,
                sigma=sigma,
                prior=1.0,
            )

        if kind == "sharpen":
            # 局部锐化，防止主峰被压平
            amp = self.rng.uniform(0.05, 0.35)

            return Action(
                kind=kind,
                amp=amp,
                mu=mu,
                sigma=sigma,
                smooth_sigma=self.rng.uniform(0.8, 2.0),
                prior=1.0,
            )


        smooth_sigma = self.rng.uniform(0.5, 2.0)

        return Action(
            kind="smooth",
            amp=0.0,
            mu=mu,
            sigma=sigma,
            smooth_sigma=smooth_sigma,
            prior=1.0,
        )

    def apply_action(self, fx, action: Action):
        fx_new = fx.copy()

        if action.kind == "add":
            bump = gaussian(
                self.x,
                action.amp,
                action.mu,
                action.sigma,
            )

            fx_new = fx_new + bump

        elif action.kind == "sub":
            bump = gaussian(
                self.x,
                action.amp,
                action.mu,
                action.sigma,
            )

            fx_new = fx_new - bump

        elif action.kind == "scale":
            mask = gaussian(
                self.x,
                1.0,
                action.mu,
                action.sigma,
            )

            fx_new = fx_new * (1.0 + action.amp * mask)

        elif action.kind == "shift":
            mask = gaussian(
                self.x,
                1.0,
                action.mu,
                action.sigma,
            )

            # 正 amp 表示局部往右移动
            displacement = action.amp * mask
            source_x = self.x - displacement

            shifted = np.interp(
                source_x,
                self.x,
                fx,
                left=0.0,
                right=0.0,
            )

            fx_new = (1.0 - mask) * fx + mask * shifted

        elif action.kind == "sharpen":
            mask = gaussian(
                self.x,
                1.0,
                action.mu,
                action.sigma,
            )

            blurred = gaussian_filter1d(
                fx,
                sigma=action.smooth_sigma,
            )

            detail = fx - blurred

            fx_new = fx + action.amp * mask * detail


        elif action.kind == "smooth":
            fx_new = gaussian_filter1d(
                fx_new,
                sigma=action.smooth_sigma,
            )

        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        return self.project(fx_new)

    def allowed_children(self, node: Node):
        """
        Progressive widening.

        连续 action space 不能一次性展开无限多 action，
        所以随着访问次数增加，逐步增加可展开子节点数量。
        """
        allowed = int(
            self.progressive_c
            * ((node.visits + 1) ** self.progressive_alpha)
        )

        allowed = max(1, allowed)
        allowed = min(self.max_children, allowed)

        return allowed

    def can_expand(self, node: Node):
        return len(node.children) < self.allowed_children(node)

    def select_child(self, node: Node):
        """
        PUCT selection.
        """
        parent_visits = max(1, node.visits)

        best_score = -np.inf
        best_child = None

        for child in node.children:
            prior = 1.0

            if child.action is not None:
                prior = child.action.prior

            u = (
                self.exploration
                * prior
                * np.sqrt(parent_visits)
                / (1 + child.visits)
            )

            puct_score = child.q + u

            if puct_score > best_score:
                best_score = puct_score
                best_child = child

        return best_child

    def expand(self, node: Node):
        action = self.propose_action(node.fx)

        fx_child = self.apply_action(
            node.fx,
            action,
        )

        child_score = self.evaluate(fx_child)

        child = Node(
            fx=fx_child,
            score=child_score,
            parent=node,
            action=action,
        )

        node.children.append(child)

        return child

    def rollout(self, node: Node):
        """
        从当前节点继续模拟几步。
        rollout 内部用贪心接受，保持稳定。
        """
        fx = node.fx.copy()
        score = node.score

        best_fx = fx.copy()
        best_score = score

        for _ in range(self.rollout_depth):
            action = self.propose_action(fx)

            candidate = self.apply_action(
                fx,
                action,
            )

            candidate_score = self.evaluate(candidate)

            if candidate_score < score:
                fx = candidate
                score = candidate_score

            if candidate_score < best_score:
                best_score = candidate_score
                best_fx = candidate.copy()

        return best_score, best_fx

    def backpropagate(self, path, reward):
        for node in path:
            node.visits += 1
            node.value_sum += reward

    def refine(
        self,
        fx_init,
        verbose=True,
        return_info=False,
    ):
        fx_init = self.project(fx_init)

            # 用初始模型作为 refinement prior
        self.fx_prior = fx_init.copy()
        self.fx_prior_norm = np.mean(self.fx_prior ** 2) + self.eps
        self.fx_prior_mass = np.trapezoid(self.fx_prior, self.x)


        root_score = self.evaluate(fx_init)

        root = Node(
            fx=fx_init.copy(),
            score=root_score,
        )

        best_fx = fx_init.copy()
        best_score = root_score

        history = {
            "best_score": [],
            "best_mse": [],
            "best_tv": [],
            "best_curv": [],
        }

        if verbose:
            parts = self.score_parts(best_fx)

            print(
                "Initial:",
                f"score={parts['score']:.6e}",
                f"mse={parts['mse']:.6e}",
                f"tv={parts['tv']:.6e}",
                f"curv={parts['curv']:.6e}",
                f"prior={parts['prior']:.6e}",
                f"mass={parts['mass']:.6e}",
            )

        for step in range(self.iterations):
            node = root
            path = [node]

            # 1. Selection
            while node.children and not self.can_expand(node):
                next_node = self.select_child(node)

                if next_node is None:
                    break

                node = next_node
                path.append(node)

            # 2. Expansion
            if self.can_expand(node):
                node = self.expand(node)
                path.append(node)

            # 3. Rollout
            rollout_score, rollout_fx = self.rollout(node)

            # 更新全局最好
            if node.score < best_score:
                best_score = node.score
                best_fx = node.fx.copy()

            if rollout_score < best_score:
                best_score = rollout_score
                best_fx = rollout_fx.copy()

                if verbose:
                    parts = self.score_parts(best_fx)

                    print(
                        f"Step {step}: improved",
                        f"score={parts['score']:.6e}",
                        f"mse={parts['mse']:.6e}",
                        f"tv={parts['tv']:.6e}",
                        f"curv={parts['curv']:.6e}",
                        f"prior={parts['prior']:.6e}",
                        f"mass={parts['mass']:.6e}",
                    )

            # 4. Backpropagation
            reward = (
                root_score - rollout_score
            ) / (abs(root_score) + self.eps)

            self.backpropagate(
                path,
                reward,
            )

            parts = self.score_parts(best_fx)

            history["best_score"].append(parts["score"])
            history["best_mse"].append(parts["mse"])
            history["best_tv"].append(parts["tv"])
            history["best_curv"].append(parts["curv"])

        if return_info:
            info = {
                "root_score": root_score,
                "best_score": best_score,
                "history": history,
                "root": root,
            }

            return best_fx, info

        return best_fx
