"""
mcts_refinement_rl_prior_v2.py

V2 improvement layer for the policy-prior MCTS code.

Keep your existing files:
    mcts_refinement_rl_prior.py
    data_generate.py
    fake_97_model.py

Run with:
    python mcts_main_rl_prior_v2.py

Main changes:
1. AdaptiveScoreGuidedPriorPolicy:
   - still does one-step score lookahead;
   - adds structured proposals around rough/high-deviation regions;
   - uses a softer prior range to avoid over-dominant priors.

2. MCTSRefinement subclass:
   - adds stagnation restart: when search plateaus, re-root the tree at the current best fx;
   - adds optional polish smoothing candidates, accepted only if the objective improves;
   - records restart steps in info["restart_steps"].
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from mcts_refinement_rl_prior import (
    ACTION_KINDS,
    Action,
    Node,
    MCTSRefinement as _BaseMCTSRefinement,
    ScoreGuidedPriorPolicy,
    LearnedPolicyPrior,
    _normalize_vector,
    _softmax,
)


class AdaptiveScoreGuidedPriorPolicy(ScoreGuidedPriorPolicy):
    """
    Stronger baseline policy for MCTS prior guidance.

    Compared with ScoreGuidedPriorPolicy:
    - half or more candidates are sampled around regions that currently look rough
      or deviate from the initial prior fx;
    - action-type probabilities are weakly adapted from the forward residual sign;
    - prior scaling is softened, so PUCT is guided but not locked by one huge prior.

    It still does not use fx_true.
    """

    def __init__(
        self,
        num_candidates=36,
        structured_fraction=0.60,
        temperature=0.70,
        min_prior=0.20,
        max_prior=4.0,
        roughness_weight=0.65,
        prior_deviation_weight=0.35,
    ):
        super().__init__(
            num_candidates=num_candidates,
            temperature=temperature,
            min_prior=min_prior,
            max_prior=max_prior,
        )
        self.structured_fraction = float(np.clip(structured_fraction, 0.0, 1.0))
        self.roughness_weight = float(max(0.0, roughness_weight))
        self.prior_deviation_weight = float(max(0.0, prior_deviation_weight))

    def _curvature_signal(self, fx):
        fx = np.asarray(fx, dtype=float)
        if fx.size < 3:
            return np.ones_like(fx)

        curv = np.zeros_like(fx)
        curv[1:-1] = np.abs(np.diff(fx, n=2))
        curv[0] = curv[1]
        curv[-1] = curv[-2]

        return curv

    def _position_probs(self, fx, mcts):
        fx = np.asarray(fx, dtype=float)

        rough = self._curvature_signal(fx)
        rough = rough / (np.mean(rough) + mcts.eps)

        if getattr(mcts, "fx_prior", None) is not None:
            dev = np.abs(fx - mcts.fx_prior)
            dev = dev / (np.mean(dev) + mcts.eps)
        else:
            dev = np.zeros_like(fx)

        # A small floor keeps the policy exploratory.
        signal = (
            0.05
            + self.roughness_weight * rough
            + self.prior_deviation_weight * dev
        )

        signal = np.clip(signal, 0.0, None)
        return _normalize_vector(signal, fallback_size=len(fx))

    def _kind_probs_from_residual(self, fx, mcts):
        """
        A weak heuristic: for positive kernels, positive average residual usually
        suggests adding/increasing spectral mass; negative residual suggests
        subtracting/decreasing spectral mass.

        This is only a proposal prior. The one-step score lookahead still decides
        the final prior for each concrete Action.
        """
        try:
            gy_pred = mcts.compute_gy_fn(mcts.x, fx, mcts.y)
            residual = mcts.gy_target - gy_pred
            signed = float(np.mean(residual) / (np.mean(np.abs(mcts.gy_target)) + mcts.eps))
        except Exception:
            signed = 0.0

        probs = {
            "add": 0.22,
            "sub": 0.18,
            "scale": 0.24,
            "shift": 0.22,
            "sharpen": 0.06,
            "smooth": 0.08,
        }

        if signed > 0.002:
            probs["add"] += 0.12
            probs["scale"] += 0.06
            probs["sub"] *= 0.65
        elif signed < -0.002:
            probs["sub"] += 0.12
            probs["scale"] += 0.06
            probs["add"] *= 0.65

        return mcts._normalize_kind_probs(probs)

    def _make_structured_action(self, fx, rng, mcts, position_probs, kind_probs):
        kind_items = list(ACTION_KINDS)
        kind_values = [kind_probs[k] for k in kind_items]
        kind, _, kind_prob = mcts._weighted_choice(kind_items, kind_values)

        _, idx, pos_prob = mcts._weighted_choice(mcts.x, position_probs)
        mu = float(mcts.x[idx])

        kind_multiplier = float(kind_prob * len(ACTION_KINDS))
        location_multiplier = float(pos_prob * len(mcts.x))
        prior = mcts._clip_prior(kind_multiplier * location_multiplier)

        return mcts._make_action(
            fx=fx,
            kind=kind,
            mu=mu,
            prior=prior,
        )

    def __call__(self, fx, rng, mcts=None, **kwargs):
        if mcts is None:
            return None

        num_candidates = max(1, int(self.num_candidates))
        num_structured = int(round(num_candidates * self.structured_fraction))
        num_structured = int(np.clip(num_structured, 0, num_candidates))
        num_random = num_candidates - num_structured

        base_score = mcts.evaluate(fx)
        position_probs = self._position_probs(fx, mcts)
        kind_probs = self._kind_probs_from_residual(fx, mcts)

        actions = []

        for _ in range(num_random):
            actions.append(mcts.sample_random_action(fx))

        for _ in range(num_structured):
            actions.append(
                self._make_structured_action(
                    fx=fx,
                    rng=rng,
                    mcts=mcts,
                    position_probs=position_probs,
                    kind_probs=kind_probs,
                )
            )

        advantages = []
        for action in actions:
            candidate_fx = mcts.apply_action(fx, action)
            candidate_score = mcts.evaluate(candidate_fx)

            # score lower is better.
            improvement = (base_score - candidate_score) / (abs(base_score) + mcts.eps)
            advantages.append(improvement)

        advantages = np.asarray(advantages, dtype=float)

        # Robust standardization prevents one lucky candidate from always saturating max_prior.
        if advantages.size > 1:
            scale = np.std(advantages) + mcts.eps
            ranked = (advantages - np.median(advantages)) / scale
        else:
            ranked = advantages

        probs = _softmax(ranked, temperature=self.temperature)

        for action, prob in zip(actions, probs):
            scaled_prior = float(prob * len(actions))
            action.prior = float(
                np.clip(scaled_prior, self.min_prior, self.max_prior)
            )

        return {"candidate_actions": actions}


class MCTSRefinement(_BaseMCTSRefinement):
    """
    V2 MCTS refiner.

    Adds:
    - restart_patience: re-root at best_fx after plateau;
    - polish_smoothing_sigmas: try small smoothing only if score improves;
    - min_improvement: avoids treating numerical noise as real progress.
    """

    def __init__(
        self,
        *args,
        restart_patience=450,
        min_improvement=1e-12,
        polish_smoothing_sigmas=(0.35, 0.70, 1.00),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.restart_patience = None if restart_patience is None else int(restart_patience)
        self.min_improvement = float(min_improvement)
        self.polish_smoothing_sigmas = tuple(polish_smoothing_sigmas or ())

    def _is_improved(self, candidate_score, best_score):
        return candidate_score < best_score - self.min_improvement * (abs(best_score) + self.eps)

    def _polish_if_better(self, fx, score):
        """
        Try a few very light smoothing candidates. A candidate is accepted only
        when the same score_parts objective gets better.
        """
        best_fx = fx
        best_score = score

        for sigma in self.polish_smoothing_sigmas:
            if sigma is None or sigma <= 0:
                continue

            candidate = gaussian_filter1d(fx, sigma=float(sigma))
            candidate = self.project(candidate)
            candidate_score = self.evaluate(candidate)

            if self._is_improved(candidate_score, best_score):
                best_fx = candidate
                best_score = candidate_score

        return best_score, best_fx

    def refine(
        self,
        fx_init,
        verbose=True,
        return_info=False,
    ):
        fx_init = self.project(fx_init)

        # Use initial model as refinement prior.
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
        steps_since_improvement = 0
        restart_steps = []

        history = {
            "best_score": [],
            "best_mse": [],
            "best_tv": [],
            "best_curv": [],
            "root_prior_min": [],
            "root_prior_mean": [],
            "root_prior_max": [],
            "root_num_children": [],
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

            improved = False

            if self._is_improved(node.score, best_score):
                best_score = node.score
                best_fx = node.fx.copy()
                improved = True

            if self._is_improved(rollout_score, best_score):
                best_score = rollout_score
                best_fx = rollout_fx.copy()
                improved = True

            # 3.5 Optional score-safe polishing
            polished_score, polished_fx = self._polish_if_better(best_fx, best_score)
            if self._is_improved(polished_score, best_score):
                best_score = polished_score
                best_fx = polished_fx.copy()
                improved = True

            if improved:
                steps_since_improvement = 0
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
            else:
                steps_since_improvement += 1

            # 4. Backpropagation
            reward = (
                root_score - rollout_score
            ) / (abs(root_score) + self.eps)

            self.backpropagate(
                path,
                reward,
            )

            # 5. Re-root if search has plateaued.
            if (
                self.restart_patience is not None
                and self.restart_patience > 0
                and steps_since_improvement >= self.restart_patience
                and step < self.iterations - 1
            ):
                restart_steps.append(step)

                if verbose:
                    print(
                        f"Step {step}: restart tree at current best",
                        f"best_score={best_score:.6e}",
                    )

                root_score = best_score
                root = Node(
                    fx=best_fx.copy(),
                    score=best_score,
                )
                steps_since_improvement = 0

            parts = self.score_parts(best_fx)
            prior_stats = self._root_prior_stats(root)

            history["best_score"].append(parts["score"])
            history["best_mse"].append(parts["mse"])
            history["best_tv"].append(parts["tv"])
            history["best_curv"].append(parts["curv"])
            history["root_prior_min"].append(prior_stats["min"])
            history["root_prior_mean"].append(prior_stats["mean"])
            history["root_prior_max"].append(prior_stats["max"])
            history["root_num_children"].append(len(root.children))

        if return_info:
            info = {
                "root_score": root_score,
                "best_score": best_score,
                "history": history,
                "root": root,
                "root_prior_stats": self._root_prior_stats(root),
                "restart_steps": restart_steps,
            }

            return best_fx, info

        return best_fx
