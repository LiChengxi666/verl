# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_policy_loss_prefix_exact_kl_clip,
    compute_policy_loss_prefix_exact_kl_cumulative_dual_clip,
    compute_policy_loss_prefix_ripo_clip,
    compute_policy_loss_ripo_clip,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    get_policy_loss_fn,
    kl_penalty,
    register_adv_est,
)
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def mock_test_fn():
    pass


def test_ripo_clip_uses_old_action_probability_for_dynamic_bounds():
    """Using a fixed PPO epsilon instead of sqrt(delta / pi_old) must fail."""
    old_prob = torch.tensor([[0.8, 0.01]])
    old_log_prob = torch.log(old_prob)
    raw_ratio = torch.tensor([[1.2, 2.0]])
    log_prob = (old_log_prob + torch.log(raw_ratio)).requires_grad_(True)
    advantages = torch.ones_like(old_log_prob)
    response_mask = torch.ones_like(old_log_prob)

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(
            ripo_delta_low=0.02,
            ripo_delta_high=0.02,
            ripo_ratio_lower=0.5,
            ripo_ratio_upper=10.0,
        ),
    )

    loss, metrics = compute_policy_loss_ripo_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=config,
    )

    # Equation 11: epsilon = sqrt(delta / pi_old). The high-probability
    # action clips at 1.1581139, while the low-probability action's ratio 2.0
    # remains below its 2.4142136 upper bound.
    expected_upper = torch.tensor([[1.1581139, 2.4142137]])
    expected_loss = -torch.tensor([1.1581139, 2.0]).mean()
    assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-6)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(0.5)
    assert metrics["actor/ripo_clip/upper_bound_mean"] == pytest.approx(expected_upper.mean().item())

    loss.backward()
    assert log_prob.grad[0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert log_prob.grad[0, 1].item() == pytest.approx(-1.0, abs=1e-6)


def test_ripo_clip_applies_paper_ratio_bounds_before_dynamic_clip():
    """Removing the paper's [0.5, 10] ratio truncation must fail."""
    old_log_prob = torch.log(torch.tensor([[0.1, 0.1]]))
    raw_ratio = torch.tensor([[0.1, 20.0]])
    log_prob = (old_log_prob + torch.log(raw_ratio)).requires_grad_(True)
    advantages = torch.tensor([[-1.0, 1.0]])
    response_mask = torch.ones_like(old_log_prob)

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(
            ripo_delta_low=100.0,
            ripo_delta_high=100.0,
            ripo_ratio_lower=0.5,
            ripo_ratio_upper=10.0,
        ),
    )

    loss, metrics = compute_policy_loss_ripo_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=config,
    )

    assert loss.item() == pytest.approx((0.5 - 10.0) / 2.0)
    assert metrics["actor/pg_clipfrac"] == pytest.approx(1.0)
    assert metrics["actor/pg_clipfrac_lower"] == pytest.approx(0.5)
    assert metrics["actor/ripo_clip/ratio_lower_clipfrac"] == pytest.approx(0.5)
    assert metrics["actor/ripo_clip/ratio_upper_clipfrac"] == pytest.approx(0.5)

    loss.backward()
    assert torch.allclose(log_prob.grad, torch.zeros_like(log_prob.grad))


def test_ripo_clip_requires_paper_token_mean_aggregation_and_valid_bounds():
    """Silently running RIPO with another aggregation or invalid bounds must fail."""
    tensor = torch.zeros((1, 1))
    mask = torch.ones((1, 1))
    base_kwargs = {
        "old_log_prob": tensor,
        "log_prob": tensor,
        "advantages": mask,
        "response_mask": mask,
    }

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(),
    )
    with pytest.raises(ValueError, match="token-mean"):
        compute_policy_loss_ripo_clip(**base_kwargs, loss_agg_mode="seq-mean-token-mean", config=config)

    bad_config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(ripo_delta_low=0.0),
    )
    with pytest.raises(ValueError, match="positive"):
        compute_policy_loss_ripo_clip(**base_kwargs, loss_agg_mode="token-mean", config=bad_config)

    assert get_policy_loss_fn("ripo_clip") is compute_policy_loss_ripo_clip


def test_prefix_ripo_clip_matches_average_token_kl_bound():
    """RIPO clipping must use the uncapped cumulative prefix log-ratio."""
    seq_len = 100
    delta = 1e-5
    old_log_prob = torch.full((1, seq_len), -1.9)
    log_prob = torch.full((1, seq_len), -0.9, requires_grad=True)
    advantages = torch.ones((1, seq_len))
    response_mask = torch.ones((1, seq_len))

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(prefix_ripo_delta_low=delta, prefix_ripo_delta_high=delta),
    )

    loss, metrics = compute_policy_loss_prefix_ripo_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    prefix_len = torch.arange(1, seq_len + 1, dtype=old_log_prob.dtype)
    old_prefix_log_prob = -1.9 * prefix_len
    radius_log = 0.5 * (torch.log(2.0 * prefix_len * delta) - old_prefix_log_prob)
    upper_log_bound = torch.logaddexp(torch.zeros_like(radius_log), radius_log)
    prefix_log_ratio = prefix_len
    expected_upper_clipped = prefix_log_ratio > upper_log_bound

    assert expected_upper_clipped.any()
    assert prefix_log_ratio[-1] > 80.0
    assert metrics["actor/pg_clipfrac"] == pytest.approx(expected_upper_clipped.float().mean().item())

    loss.backward()
    assert torch.isfinite(loss)
    clipped_grad = log_prob.grad[0, expected_upper_clipped]
    assert torch.allclose(clipped_grad, torch.zeros_like(clipped_grad))


def test_prefix_exact_kl_clip_uses_geometric_average_prefix_ratio_with_token_local_gradient():
    old_log_prob = torch.tensor([[-2.0, -2.0, -2.0]])
    token_log_ratio = torch.tensor([[0.1, 0.2, -0.1]])
    log_prob = (old_log_prob + token_log_ratio).requires_grad_(True)
    response_mask = torch.ones_like(old_log_prob)
    advantages = -torch.ones_like(old_log_prob)

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(prefix_exact_kl_delta_low=100.0, prefix_exact_kl_delta_high=100.0),
    )

    loss, metrics = compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    prefix_len = torch.arange(1, token_log_ratio.shape[-1] + 1)
    geometric_average_prefix_ratio = torch.exp(torch.cumsum(token_log_ratio, dim=-1) / prefix_len)
    assert loss.item() == pytest.approx(geometric_average_prefix_ratio.mean().item())
    assert metrics["actor/prefix_exact_kl_clip/probability_weighted"] == 0.0
    assert metrics["actor/prefix_exact_kl_clip/geometric_average_surrogate"] == 1.0

    loss.backward()
    # Stop-gradient keeps the geometric-average prefix value but routes each loss
    # term's gradient through its current token only.
    assert torch.allclose(log_prob.grad, geometric_average_prefix_ratio / token_log_ratio.shape[-1], atol=1e-6)


def test_prefix_exact_kl_cumulative_dual_clip_uses_cumulative_ratio_with_token_local_gradient():
    """Dividing the cumulative log-ratio by prefix length must fail this test."""
    old_log_prob = torch.zeros((1, 2))
    token_log_ratio = torch.log(torch.tensor([[2.0, 2.0]]))
    log_prob = (old_log_prob + token_log_ratio).requires_grad_(True)
    response_mask = torch.ones_like(old_log_prob)
    advantages = torch.ones_like(old_log_prob)

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(prefix_exact_kl_delta_low=100.0, prefix_exact_kl_delta_high=100.0),
    )

    loss, metrics = compute_policy_loss_prefix_exact_kl_cumulative_dual_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    # Prefix ratios are [2, 4], not geometric-average ratios [2, 2].
    assert loss.item() == pytest.approx(-3.0)
    assert metrics["actor/prefix_exact_kl_cumulative_dual_clip/cumulative_surrogate"] == 1.0

    loss.backward()
    assert torch.allclose(log_prob.grad, torch.tensor([[-1.0, -2.0]]), atol=1e-6)


def test_prefix_exact_kl_cumulative_dual_clip_caps_negative_advantage_at_exact_upper_bound():
    """Letting negative-advantage, high-ratio samples select raw PPO must fail."""
    old_log_prob = torch.zeros((2, 1))
    log_prob = torch.full((2, 1), np.log(4.0), requires_grad=True)
    response_mask = torch.ones_like(old_log_prob)
    advantages = torch.tensor([[-1.0], [1.0]])
    # For t=1, R=2 is exactly the positive root of R - 1 - log(R) = delta.
    delta_high = 1.0 - np.log(2.0)

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(prefix_exact_kl_delta_low=100.0, prefix_exact_kl_delta_high=delta_high),
    )

    loss, metrics = compute_policy_loss_prefix_exact_kl_cumulative_dual_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    # Both signs use R=2 at the upper bound, so the two losses cancel.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["actor/prefix_exact_kl_cumulative_dual_clip/dual_clipfrac"] == pytest.approx(0.5)
    assert metrics["actor/prefix_exact_kl_cumulative_dual_clip/raw_ratio_p95"] == pytest.approx(4.0)
    assert metrics["actor/prefix_exact_kl_cumulative_dual_clip/raw_ratio_max"] == pytest.approx(4.0)


def test_prefix_exact_kl_clip_preserves_cumulative_prefix_clip_decisions():
    old_log_prob = torch.full((1, 4), -2.0)
    token_log_ratio = torch.tensor([[0.01, 0.03, 0.08, 0.12]])
    log_prob = old_log_prob + token_log_ratio
    response_mask = torch.ones_like(old_log_prob)
    advantages = torch.ones_like(old_log_prob)
    delta_high = 1e-3

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=1,
        policy_loss=PolicyLossConfig(
            prefix_exact_kl_delta_low=delta_high,
            prefix_exact_kl_delta_high=delta_high,
        ),
    )

    _, metrics = compute_policy_loss_prefix_exact_kl_clip(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    cumulative_log_ratio = torch.cumsum(token_log_ratio, dim=-1)
    prefix_len = torch.arange(1, token_log_ratio.shape[-1] + 1)
    exact_kl_coordinate = torch.exp(cumulative_log_ratio) - 1.0 - cumulative_log_ratio
    expected_clipped = exact_kl_coordinate > prefix_len * delta_high
    assert metrics["actor/pg_clipfrac"] == pytest.approx(expected_clipped.float().mean().item())


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_grpo_vectorized_matches_original_for_low_variance_rewards():
    token_level_rewards = torch.tensor([[1.0], [1.00001], [2.0], [2.00001]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name,base",
    [
        ("k1+", "k1"),
        ("kl+", "kl"),
        ("abs+", "abs"),
        ("k3+", "k3"),
        ("low_var_kl+", "low_var_kl"),
    ],
)
def test_kl_penalty_straight_through_value_matches_base(name, base):
    """The ``+`` suffix is a straight-through trick that swaps in the k2
    gradient while keeping the base estimator's value. Therefore the forward
    value of e.g. ``k3+`` must match the value of plain ``k3``.

    Regression test for the bug where ``kl_penalty(..., "k3+")`` raised
    ``NotImplementedError`` because the wrapper forwarded the ``+`` suffix to
    ``kl_penalty_forward`` without stripping it.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    plus_value = kl_penalty(logprob, ref_logprob, name)
    base_value = kl_penalty(logprob, ref_logprob, base)
    assert torch.allclose(plus_value, base_value)


def test_kl_penalty_k3_plus_uses_k2_gradient():
    """With ``k3+`` the gradient w.r.t. ``logprob`` should equal the gradient
    obtained from the ``k2`` (``0.5 * log_ratio**2``) estimator, since the
    straight-through trick routes the backward pass through ``k2``.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    out_plus = kl_penalty(logprob, ref_logprob, "k3+").sum()
    (grad_plus,) = torch.autograd.grad(out_plus, logprob)

    logprob_k2 = logprob.detach().clone().requires_grad_(True)
    out_k2 = kl_penalty(logprob_k2, ref_logprob, "k2").sum()
    (grad_k2,) = torch.autograd.grad(out_k2, logprob_k2)

    assert torch.allclose(grad_plus, grad_k2)


if __name__ == "__main__":
    unittest.main()
