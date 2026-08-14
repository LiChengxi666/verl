import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_prefix_probability_weighted_exact_kl_cumulative_dual_clip
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def test_uses_probability_weighted_bounds_and_raw_prefix_ratio():
    old_log_prob = torch.tensor([[-2.0, -2.0]])
    token_log_ratio = torch.log(torch.tensor([[2.0, 2.0]]))
    log_prob = (old_log_prob + token_log_ratio).requires_grad_(True)
    response_mask = torch.ones_like(old_log_prob)
    advantages = torch.ones_like(old_log_prob)
    config = ActorConfig(
        strategy="fsdp", rollout_n=1, ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(prefix_exact_kl_delta_low=100.0, prefix_exact_kl_delta_high=100.0),
    )

    loss, metrics = compute_policy_loss_prefix_probability_weighted_exact_kl_cumulative_dual_clip(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
        response_mask=response_mask, config=config,
    )

    assert loss.item() == pytest.approx(-3.0)
    prefix = "actor/prefix_probability_weighted_exact_kl_cumulative_dual_clip"
    assert metrics[f"{prefix}/probability_weighted"] == 1.0
    assert metrics[f"{prefix}/geometric_average_surrogate"] == 0.0
    assert metrics[f"{prefix}/cumulative_surrogate"] == 1.0
    assert metrics[f"{prefix}/dual_clip_negative_advantage"] == 1.0
    loss.backward()
    assert torch.allclose(log_prob.grad, torch.tensor([[-1.0, -2.0]]), atol=1e-6)


def test_dual_clip_caps_negative_advantage_at_exact_upper_bound():
    old_log_prob = torch.zeros((2, 1))
    log_prob = torch.full((2, 1), np.log(4.0), requires_grad=True)
    response_mask = torch.ones_like(old_log_prob)
    advantages = torch.tensor([[-1.0], [1.0]])
    config = ActorConfig(
        strategy="fsdp", rollout_n=1, ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(
            prefix_exact_kl_delta_low=100.0,
            prefix_exact_kl_delta_high=1.0 - np.log(2.0),
        ),
    )

    loss, metrics = compute_policy_loss_prefix_probability_weighted_exact_kl_cumulative_dual_clip(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
        response_mask=response_mask, config=config,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    prefix = "actor/prefix_probability_weighted_exact_kl_cumulative_dual_clip"
    assert metrics[f"{prefix}/dual_clipfrac"] == pytest.approx(0.5)
