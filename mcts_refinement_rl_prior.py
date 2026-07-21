import inspect
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from scipy.ndimage import gaussian_filter1d

from data_generate import compute_gy


ACTION_KINDS = ("add", "sub", "scale", "shift", "sharpen", "smooth")
DEFAULT_KIND_PROBS = {
    "add": 0.25,
    "sub": 0.18,
    "scale": 0.22,
    "shift": 0.22,
    "sharpen": 0.10,
    "smooth": 0.03,
}


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(
        -0.5 * ((x - mu) / (sigma + 1e-12)) ** 2
    )


def _softmax(values, temperature=1.0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    temperature = max(float(temperature), 1e-8)
    z = values / temperature
    z = z - np.max(z)
    exp_z = np.exp(z)
    denom = np.sum(exp_z)

    if not np.isfinite(denom) or denom <= 0.0:
        return np.ones_like(values) / len(values)

    return exp_z / denom


def _normalize_vector(values, fallback_size=None):
    if values is None:
        if fallback_size is None:
            return None
        return np.ones(fallback_size, dtype=float) / fallback_size

    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))

    if values.size == 0:
        if fallback_size is None:
            return values
        return np.ones(fallback_size, dtype=float) / fallback_size

    if not np.isfinite(total) or total <= 0.0:
        return np.ones_like(values) / len(values)

    return values / total


@dataclass
class Action:
    kind: str
    amp: float
    mu: float
    sigma: float
    smooth_sigma: float = 0.0
    # PUCT prior. 旧代码这里一直是 1.0；现在会由 policy / RL model 动态赋值。
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


class ScoreGuidedPriorPolicy:
    """
    一个不需要 fx_true 的 baseline policy。

    它会先随机生成若干候选 action，用当前 MCTS 的 score 做一阶 lookahead，
    然后把“更可能降低 score 的 action”分配更高 prior。

    这不是最终的强化学习模型，而是一个可以直接跑通的 policy-prior baseline。
    后续你训练好 RL policy 后，只需要保持相同返回格式即可替换它。

    返回格式:
        {"candidate_actions": List[Action]}

    每个 Action.prior 会被写成不同值，PUCT selection 会使用这些 prior。
    """

    def __init__(
        self,
        num_candidates=16,
        temperature=0.20,
        min_prior=0.05,
        max_prior=8.0,
    ):
        self.num_candidates = int(num_candidates)
        self.temperature = float(temperature)
        self.min_prior = float(min_prior)
        self.max_prior = float(max_prior)

    def __call__(self, fx, rng, mcts=None, **kwargs):
        if mcts is None:
            return None

        num_candidates = max(1, self.num_candidates)
        base_score = mcts.evaluate(fx)

        actions = []
        advantages = []

        for _ in range(num_candidates):
            action = mcts.sample_random_action(fx)
            candidate_fx = mcts.apply_action(fx, action)
            candidate_score = mcts.evaluate(candidate_fx)

            # score 越低越好；improvement 越大，说明 action 越值得探索。
            advantage = (base_score - candidate_score) / (
                abs(base_score) + mcts.eps
            )

            actions.append(action)
            advantages.append(advantage)

        probs = _softmax(advantages, temperature=self.temperature)

        # 让平均 prior 接近 1，避免把 exploration 整体放大/缩小太多。
        for action, prob in zip(actions, probs):
            scaled_prior = float(prob * len(actions))
            action.prior = float(
                np.clip(scaled_prior, self.min_prior, self.max_prior)
            )

        return {"candidate_actions": actions}


class LearnedPolicyPrior:
    """
    强化学习 / 神经网络策略模型的轻量包装器。

    你的 RL 模型可以是 callable，也可以有 predict(state) 方法。
    模型建议返回 dict，支持下面任意字段:

        {
            "action_priors": {
                "add": 0.35,
                "sub": 0.10,
                "scale": 0.20,
                "shift": 0.25,
                "sharpen": 0.08,
                "smooth": 0.02,
            },
            "position_probs": np.ndarray,   # shape = (len(x),)
            # 或者返回 position_logits，代码会自动 softmax
        }

    action_priors 决定动作类型倾向，position_probs 决定 mu 更常落在哪里。
    MCTSRefinement 会把这些概率转换成 Action.prior，用于 PUCT。
    """

    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = float(temperature)

    def __call__(self, fx, x, y, gy_target, compute_gy_fn, mcts=None, **kwargs):
        gy_pred = compute_gy_fn(x, fx, y)
        residual = gy_target - gy_pred

        state = {
            "fx": np.asarray(fx, dtype=float),
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "gy_target": np.asarray(gy_target, dtype=float),
            "gy_pred": np.asarray(gy_pred, dtype=float),
            "residual": np.asarray(residual, dtype=float),
        }

        if hasattr(self.model, "predict"):
            out = self.model.predict(state)
        else:
            out = self.model(state)

        if not isinstance(out, dict):
            raise TypeError(
                "RL policy model must return a dict, for example "
                "{'action_priors': {...}, 'position_probs': ...}."
            )

        out = dict(out)

        if "action_logits" in out and "action_priors" not in out:
            logits = np.asarray(out["action_logits"], dtype=float)
            probs = _softmax(logits, temperature=self.temperature)
            out["action_priors"] = {
                kind: float(prob)
                for kind, prob in zip(ACTION_KINDS, probs)
            }

        if "position_logits" in out and "position_probs" not in out:
            out["position_probs"] = _softmax(
                out["position_logits"],
                temperature=self.temperature,
            )

        return out


class MCTSRefinement:
    """
    MCTS refinement for inverse problem.

    这个类不会使用 fx_true。
    它只使用:
        1. 当前候选 fx
        2. forward model: compute_gy(x, fx, y)
        3. 目标观测 gy_target
        4. 光滑、曲率等先验
        5. 可选 policy_fn / RL model 输出的 action prior
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
        policy_prior_weight=1.0,
        default_action_priors=None,
        prior_floor=0.05,
        prior_ceiling=8.0,
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

        # policy_fn 可以是:
        #   1. 返回 Action
        #   2. 返回 {"candidate_actions": [Action, ...]}
        #   3. 返回 {"action_priors": ..., "position_probs": ...}
        self.policy_fn = policy_fn
        self.policy_prior_weight = float(np.clip(policy_prior_weight, 0.0, 1.0))
        self.prior_floor = float(prior_floor)
        self.prior_ceiling = float(prior_ceiling)

        self.default_kind_probs = self._normalize_kind_probs(
            default_action_priors or DEFAULT_KIND_PROBS
        )

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

        # 不要离初始模型太远
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

    def _normalize_kind_probs(self, probs):
        if probs is None:
            probs = DEFAULT_KIND_PROBS

        if isinstance(probs, dict):
            raw = np.array(
                [float(probs.get(kind, 0.0)) for kind in ACTION_KINDS],
                dtype=float,
            )
        else:
            raw = np.asarray(probs, dtype=float).reshape(-1)
            if len(raw) != len(ACTION_KINDS):
                raise ValueError(
                    "default_action_priors / action_priors must have "
                    f"length {len(ACTION_KINDS)}."
                )

        raw = _normalize_vector(raw, fallback_size=len(ACTION_KINDS))
        return {
            kind: float(prob)
            for kind, prob in zip(ACTION_KINDS, raw)
        }

    def _blend_kind_probs(self, policy_kind_probs=None):
        base = np.array(
            [self.default_kind_probs[kind] for kind in ACTION_KINDS],
            dtype=float,
        )

        if policy_kind_probs is None:
            mixed = base
        else:
            policy = np.array(
                [policy_kind_probs.get(kind, 0.0) for kind in ACTION_KINDS],
                dtype=float,
            )
            policy = _normalize_vector(policy, fallback_size=len(ACTION_KINDS))
            mixed = (
                (1.0 - self.policy_prior_weight) * base
                + self.policy_prior_weight * policy
            )

        mixed = _normalize_vector(mixed, fallback_size=len(ACTION_KINDS))
        return {
            kind: float(prob)
            for kind, prob in zip(ACTION_KINDS, mixed)
        }

    def _clip_prior(self, prior):
        return float(np.clip(float(prior), self.prior_floor, self.prior_ceiling))

    def _weighted_choice(self, items, probs):
        probs = _normalize_vector(probs, fallback_size=len(items))
        idx = int(self.rng.choice(len(items), p=probs))
        return items[idx], idx, float(probs[idx])

    def _call_policy_fn(self, fx):
        if self.policy_fn is None:
            return None

        kwargs = {
            "fx": fx,
            "x": self.x,
            "y": self.y,
            "gy_target": self.gy_target,
            "compute_gy_fn": self.compute_gy_fn,
            "rng": self.rng,
            "mcts": self,
        }

        try:
            signature = inspect.signature(self.policy_fn)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
            if accepts_var_kw:
                return self.policy_fn(**kwargs)

            filtered = {
                name: value
                for name, value in kwargs.items()
                if name in signature.parameters
            }
            return self.policy_fn(**filtered)
        except (TypeError, ValueError):
            # 兼容你原来写的 policy_fn 签名。
            return self.policy_fn(
                fx=fx,
                x=self.x,
                y=self.y,
                gy_target=self.gy_target,
                compute_gy_fn=self.compute_gy_fn,
                rng=self.rng,
            )

    def _as_action_list(self, raw_policy):
        if raw_policy is None:
            return None

        if isinstance(raw_policy, Action):
            return [raw_policy]

        if isinstance(raw_policy, dict):
            for key in ("candidate_actions", "actions"):
                if key in raw_policy and raw_policy[key] is not None:
                    actions = list(raw_policy[key])
                    if not all(isinstance(a, Action) for a in actions):
                        raise TypeError(f"{key} must be a list of Action objects.")
                    return actions

            if "action" in raw_policy and raw_policy["action"] is not None:
                action = raw_policy["action"]
                if not isinstance(action, Action):
                    raise TypeError("policy output['action'] must be an Action.")
                return [action]

        return None

    def _extract_policy_guidance(self, raw_policy):
        if not isinstance(raw_policy, dict):
            return None, None

        action_priors = None
        for key in ("action_priors", "kind_priors", "kind_probs", "action_probs"):
            if key in raw_policy:
                action_priors = self._normalize_kind_probs(raw_policy[key])
                break

        position_probs = None
        for key in ("position_probs", "mu_probs", "x_probs"):
            if key in raw_policy:
                position_probs = _normalize_vector(raw_policy[key])
                break

        if position_probs is not None and len(position_probs) != len(self.x):
            raise ValueError(
                "position_probs / mu_probs must have the same length as x."
            )

        return action_priors, position_probs

    def sample_random_action(self, fx, kind_probs=None, position_probs=None):
        """
        从给定 action kind 概率和位置概率里采样一个 action。
        同时把采样到的概率转换成 Action.prior。
        """
        kind_probs = self._blend_kind_probs(kind_probs)
        kind_items = list(ACTION_KINDS)
        kind_values = [kind_probs[kind] for kind in kind_items]
        kind, _, kind_prob = self._weighted_choice(kind_items, kind_values)

        if position_probs is None:
            mu = self.rng.uniform(self.x.min(), self.x.max())
            location_multiplier = 1.0
        else:
            _, idx, pos_prob = self._weighted_choice(self.x, position_probs)
            mu = float(self.x[idx])
            # 平均 location_multiplier 约等于 1。
            location_multiplier = float(pos_prob * len(self.x))

        # 平均 kind_multiplier 约等于 1。
        kind_multiplier = float(kind_prob * len(ACTION_KINDS))
        prior = self._clip_prior(kind_multiplier * location_multiplier)

        return self._make_action(
            fx=fx,
            kind=kind,
            mu=mu,
            prior=prior,
        )

    def _make_action(self, fx, kind, mu, prior=1.0):
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
                prior=prior,
            )

        if kind == "scale":
            amp = self.rng.uniform(-0.25, 0.25)

            return Action(
                kind=kind,
                amp=amp,
                mu=mu,
                sigma=sigma,
                prior=prior,
            )

        if kind == "shift":
            # 局部峰位移动，amp 表示 x 方向位移
            amp = self.rng.uniform(-0.04, 0.04) * self.x_span

            return Action(
                kind=kind,
                amp=amp,
                mu=mu,
                sigma=sigma,
                prior=prior,
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
                prior=prior,
            )

        if kind == "smooth":
            smooth_sigma = self.rng.uniform(0.5, 2.0)

            return Action(
                kind="smooth",
                amp=0.0,
                mu=mu,
                sigma=sigma,
                smooth_sigma=smooth_sigma,
                prior=prior,
            )

        raise ValueError(f"Unknown action kind: {kind}")

    def propose_action(self, fx):
        """
        action proposal。

        关键变化:
        1. 如果传入 policy_fn / RL model，就从 policy 输出里读取 action_priors / position_probs；
        2. 生成 Action 时不再固定 prior=1.0，而是按 policy 概率赋值；
        3. 如果 policy 返回一组 candidate_actions，则按它们的 prior 采样。
        """
        raw_policy = self._call_policy_fn(fx)

        policy_actions = self._as_action_list(raw_policy)
        if policy_actions:
            priors = np.array(
                [max(float(action.prior), self.prior_floor) for action in policy_actions],
                dtype=float,
            )
            probs = _normalize_vector(priors, fallback_size=len(policy_actions))
            idx = int(self.rng.choice(len(policy_actions), p=probs))
            action = policy_actions[idx]
            action.prior = self._clip_prior(action.prior)
            return action

        kind_probs, position_probs = self._extract_policy_guidance(raw_policy)

        return self.sample_random_action(
            fx,
            kind_probs=kind_probs,
            position_probs=position_probs,
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
        Progressive widening。

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
        PUCT selection。

        prior 来自 child.action.prior。现在这个值会由 policy / RL model 决定，
        所以高 prior 的动作会被更早、更频繁地探索。
        """
        parent_visits = max(1, node.visits)

        best_score = -np.inf
        best_child = None

        for child in node.children:
            prior = 1.0

            if child.action is not None:
                prior = self._clip_prior(child.action.prior)

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

    def _root_prior_stats(self, root):
        priors = [
            child.action.prior
            for child in root.children
            if child.action is not None
        ]

        if not priors:
            return {
                "min": None,
                "mean": None,
                "max": None,
            }

        priors = np.asarray(priors, dtype=float)
        return {
            "min": float(np.min(priors)),
            "mean": float(np.mean(priors)),
            "max": float(np.max(priors)),
        }

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
            "root_prior_min": [],
            "root_prior_mean": [],
            "root_prior_max": [],
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
            prior_stats = self._root_prior_stats(root)

            history["best_score"].append(parts["score"])
            history["best_mse"].append(parts["mse"])
            history["best_tv"].append(parts["tv"])
            history["best_curv"].append(parts["curv"])
            history["root_prior_min"].append(prior_stats["min"])
            history["root_prior_mean"].append(prior_stats["mean"])
            history["root_prior_max"].append(prior_stats["max"])

        if return_info:
            info = {
                "root_score": root_score,
                "best_score": best_score,
                "history": history,
                "root": root,
                "root_prior_stats": self._root_prior_stats(root),
            }

            return best_fx, info

        return best_fx
