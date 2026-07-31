# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from unittest.mock import Mock

import pytest

from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager


@pytest.mark.parametrize("use_checkpoint_parameters", [False, True])
def test_load_optimizer_always_restores_scheduler_progress(use_checkpoint_parameters):
    manager = MegatronCheckpointManager.__new__(MegatronCheckpointManager)
    manager.optimizer = Mock()
    manager.lr_scheduler = Mock()
    manager.rank = 0
    manager.use_checkpoint_opt_param_scheduler = use_checkpoint_parameters

    optimizer_state = {"state": "optimizer"}
    scheduler_state = {"num_steps": 200}
    manager._load_optimizer_and_scheduler(
        {"optimizer": optimizer_state, "lr_scheduler": scheduler_state},
        "/tmp/global_step_200",
    )

    manager.optimizer.load_state_dict.assert_called_once_with(optimizer_state)
    manager.lr_scheduler.load_state_dict.assert_called_once_with(scheduler_state)


def test_load_optimizer_allows_missing_scheduler_when_scheduler_is_disabled():
    manager = MegatronCheckpointManager.__new__(MegatronCheckpointManager)
    manager.optimizer = Mock()
    manager.lr_scheduler = None
    manager.rank = 0

    manager._load_optimizer_and_scheduler({"optimizer": {}}, "/tmp/global_step_200")

    manager.optimizer.load_state_dict.assert_called_once_with({})
