# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.utils.reward_score import default_compute_score


@pytest.mark.parametrize(
    "data_source",
    ["amc23", "aime24", "aime25", "math500", "olympiad", "minerva", "gsm8k", "hmmt25"],
)
def test_math_benchmark_uses_boxed_binary_accuracy(data_source):
    correct = default_compute_score(data_source, r"Therefore, \boxed{42}.", "42")
    incorrect = default_compute_score(data_source, r"Therefore, \boxed{41}.", "42")

    assert correct["score"] == 1.0
    assert correct["acc"] is True
    assert correct["pred"] == "42"
    assert incorrect["score"] == 0.0
    assert incorrect["acc"] is False


def test_math_benchmark_does_not_truncate_a_boxed_answer_before_scoring():
    solution = r"The answer is \boxed{42}." + " trailing explanation" * 30

    result = default_compute_score("aime24", solution, "42")

    assert result == {"score": 1.0, "acc": True, "pred": "42"}


def test_processed_math_training_keeps_zero_one_reward_scale():
    assert default_compute_score("lighteval/MATH", r"\boxed{42}", "42") == 1.0
    assert default_compute_score("lighteval/MATH", r"\boxed{41}", "42") == 0.0


def test_legacy_math_dapo_training_keeps_negative_incorrect_reward():
    correct = default_compute_score("math_dapo", "Answer: 42", "42")
    incorrect = default_compute_score("math_dapo", "Answer: 41", "42")

    assert correct["score"] == 1.0
    assert incorrect["score"] == -1.0
