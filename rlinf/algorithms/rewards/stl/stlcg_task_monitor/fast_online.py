"""Exact online-prefix robustness ρ(φ, s[0:t+1]) in numpy O(T²), 3 aggregation methods.

For φ = ∧_k ◇(grasp_k ∧ ◇_[0,τ] goal_k):
  At prefix length t+1, for each step s in [0, t]:
    inner_goal = goal[s : min(s+τ, t)+1]   ← truncated to prefix length
    ψ_k(s) = min(grasp_k[s], inner_goal)   (or inner_goal if no grasp)
  ρ_k(t) = max over s of ψ_k(s)            ← outer ◇
  ρ(t) = min over k of ρ_k(t)              ← ∧

Three aggregation methods (τ_smooth = temperature):
  true:      max → max,  min → min
  logsumexp: max → logsumexp(τ·x)/τ,  min → -logsumexp(-τ·x)/τ
  softmax:   max → (softmax(τ·x)·x).sum(),  min → (softmax(-τ·x)·x).sum()
"""
import numpy as np
from scipy.special import logsumexp


def _smooth_max(x, method, temp):
    if len(x) == 0: return -1e9
    if method == "true":      return float(x.max())
    if method == "logsumexp": return float(logsumexp(temp * x) / temp)
    if method == "softmax":
        mx = x.max(); w = np.exp(temp * (x - mx)); return float((w * x).sum() / w.sum())


def _smooth_min(a, b, method, temp):
    """smooth min of two scalars."""
    if method == "true":      return min(a, b)
    if method == "logsumexp": return -np.log(np.exp(-temp * a) + np.exp(-temp * b)) / temp
    if method == "softmax":
        wa = np.exp(-temp * a); wb = np.exp(-temp * b)
        return float((wa * a + wb * b) / (wa + wb))


def online_prefix_true(grasp_margins, goal_margins, tau: int) -> np.ndarray:
    return online_prefix(grasp_margins, goal_margins, tau, "true", 1.0)["true"]


def online_prefix(grasp_margins, goal_margins, tau: int,
                  methods=("true", "logsumexp", "softmax"), temp=10.0) -> dict:
    """Online prefix ρ(φ, s[0:t+1]) under multiple aggregation methods.

    Returns {method: np.ndarray of shape (T,)}.
    """
    T = len(goal_margins[0]) if goal_margins else 0
    results = {m: np.empty(T) for m in methods}
    if T == 0:
        return results

    for t in range(T):
        for method in methods:
            rho_sub = []
            for grasp, goal in zip(grasp_margins, goal_margins):
                best_psi = -1e9
                for s in range(t + 1):
                    end = min(s + tau + 1, t + 1)
                    inner = _smooth_max(goal[s:end], method, temp)
                    if grasp is not None:
                        psi = _smooth_min(grasp[s], inner, method, temp)
                    else:
                        psi = inner
                    if psi > best_psi:
                        best_psi = psi
                rho_sub.append(best_psi)
            # ∧ across sub-formulas = min
            if method == "true":
                results[method][t] = min(rho_sub)
            else:
                val = rho_sub[0]
                for v in rho_sub[1:]:
                    val = _smooth_min(val, v, method, temp)
                results[method][t] = val
    return results
