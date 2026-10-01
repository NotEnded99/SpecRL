import torch
import functools
from typing import Union, Callable


def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def smooth_mask(T, t_start, t_end, scale, device="cpu"):
    xs = torch.arange(T, device=device).float()
    return torch.sigmoid(scale * (xs - t_start * T)) - torch.sigmoid(
        scale * (xs - t_end * T)
    )


def maxish_softmax(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs,
):

    return (torch.nn.functional.softmax(temperature * signal, dim=dim) * signal).sum(
        dim=dim, keepdim=keepdim
    )


def minish_softmax(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs,
):

    return -maxish_softmax(-signal, temperature, dim, keepdim)


def maxish_logsumexp(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs,
):

    return torch.logsumexp(temperature * signal, dim=dim, keepdim=keepdim) / temperature


def minish_logsumexp(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    temperature: float = 1.0,
    **kwargs,
):

    return -maxish_logsumexp(-signal, temperature, dim, keepdim)


def maxish(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    approx_method: str = "true",
    temperature: float = 1.0,
    **kwargs,
):

    match approx_method:
        case "true":
            return torch.max(signal, dim=dim, keepdim=keepdim)[0]

        case "softmax":
            return maxish_softmax(
                signal, dim=dim, keepdim=keepdim, temperature=temperature
            )

        case "logsumexp":
            return maxish_logsumexp(
                signal, dim=dim, keepdim=keepdim, temperature=temperature
            )


def minish(
    signal: torch.Tensor,
    dim: int = 0,
    keepdim: bool = True,
    approx_method: str = "true",
    temperature: float = 1.0,
    **kwargs,
):

    return -maxish(
        -signal,
        dim=dim,
        keepdim=keepdim,
        approx_method=approx_method,
        temperature=temperature,
    )


def separate_and(formula, signal, **kwargs):
    # base case
    if formula.__class__.__name__ != "And":
        return formula(signal, **kwargs).unsqueeze(-1)
    else:
        if isinstance(signal, tuple):
            return torch.cat(
                [
                    separate_and(formula.subformula1, signal[0], **kwargs),
                    separate_and(formula.subformula2, signal[1], **kwargs),
                ],
                dim=-1,
            )
        else:
            return torch.cat(
                [
                    separate_and(formula.subformula1, signal, **kwargs),
                    separate_and(formula.subformula2, signal, **kwargs),
                ],
                dim=-1,
            )


def separate_or(formula, signal, **kwargs):
    # base case
    if formula.__class__.__name__ != "Or":
        return formula(signal, **kwargs).unsqueeze(-1)
    else:
        if isinstance(signal, tuple):
            return torch.cat(
                [
                    separate_or(formula.subformula1, signal[0], **kwargs),
                    separate_or(formula.subformula2, signal[1], **kwargs),
                ],
                dim=-1,
            )
        else:
            return torch.cat(
                [
                    separate_or(formula.subformula1, signal, **kwargs),
                    separate_or(formula.subformula2, signal, **kwargs),
                ],
                dim=-1,
            )


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)
