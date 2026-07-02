"""AlphaZero loss function: KL divergence policy loss + MSE value loss."""

import torch
import torch.nn.functional as F


def kl_policy_loss(
    policy_logits: torch.Tensor,  # (num_legal,) raw logits
    policy_target: torch.Tensor,  # (num_legal,) probability distribution
) -> torch.Tensor:
    """KL divergence KL(target || network) — paper's Eq. 12.

    KL(p||q) = sum(p * log(p)) - sum(p * log(q))
             = sum(p * log(p/q))

    Uses torch.special.xlogy for the entropy term to handle 0*log(0)=0 safely.

    Args:
        policy_logits: Raw (un-normalised) logits over legal moves, shape (N,).
        policy_target: Target probability distribution over legal moves, shape (N,).
            Must sum to 1 and be non-negative.

    Returns:
        Scalar KL divergence tensor (always >= 0).
    """
    log_q = F.log_softmax(policy_logits, dim=-1)
    cross_entropy = -(policy_target * log_q).sum()
    entropy = torch.special.xlogy(policy_target, policy_target).sum()
    return entropy + cross_entropy  # = KL(target || predicted)


def alphazero_loss(
    policy_logits: torch.Tensor,  # (num_legal,) raw logits
    policy_target: torch.Tensor,  # (num_legal,) probability distribution from MCTS
    value: torch.Tensor,          # scalar predicted value
    value_target: torch.Tensor,   # scalar target (+1/-1 for win/loss, 0 for draw)
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the Gumbel AlphaZero combined loss.

    Policy loss is the KL divergence KL(pi' || pi_network) between the
    improved policy produced by Gumbel MCTS and the network's softmax output,
    per Eq. 12 of Danihelka et al. 2022. The improved policy is
    ``softmax(logits + sigma(completedQ))`` (Eq. 11), not visit counts, and
    is what the Rust self-play binary writes into trajectories as the policy
    target. Uses :func:`kl_policy_loss` internally.

    Value loss is mean-squared error between the scalar value prediction and
    the game outcome.

    Args:
        policy_logits: Raw (un-normalised) logits over legal moves, shape (N,).
        policy_target: Improved policy pi' from Gumbel MCTS over legal moves,
            shape (N,). Must sum to 1 and be non-negative.
        value: Scalar value predicted by the network.
        value_target: Scalar game outcome (+1 win, -1 loss, 0 draw).
        alpha: Weight applied to the policy loss term.
        beta: Weight applied to the value loss term.

    Returns:
        A tuple ``(total_loss, components)`` where *total_loss* is a scalar
        tensor and *components* is a dict with keys ``"policy_loss"`` and
        ``"value_loss"``.
    """
    policy_loss: torch.Tensor = kl_policy_loss(policy_logits, policy_target)
    value_loss: torch.Tensor = F.mse_loss(value, value_target)
    total: torch.Tensor = alpha * policy_loss + beta * value_loss
    return total, {"policy_loss": policy_loss, "value_loss": value_loss}


def path_consistency_loss(values: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """Temporal consistency loss along a game trajectory.

    Penalizes value predictions that are inconsistent with each other
    and with the final outcome. Based on PCZero (Zhao et al., 2022).

    Args:
        values: 1-D tensor of value predictions at each position in a trajectory.
        outcome: scalar ground-truth outcome (+1/-1/0).

    Returns:
        Scalar loss: MSE between consecutive value differences and zero,
        plus MSE between final prediction and outcome.
    """
    if values.numel() <= 1:
        return torch.tensor(0.0, device=values.device)

    # Consecutive consistency: v[t] and v[t+1] should be close (sign-flipped for alternating players)
    diffs = values[1:] + values[:-1]  # sum because players alternate: v_opponent ~ -v_self
    consistency = (diffs ** 2).mean()

    # Terminal consistency: last prediction should match outcome
    terminal = (values[-1] - outcome) ** 2

    return consistency + terminal
