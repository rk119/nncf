# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
import pytest
import torch
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.backend import BackendType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.torch_fx_backend import FXMinMaxAlgoBackend
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTSoftmaxMetatype
from tests.common.quantization.metatypes import CatTestMetatype
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import SoftmaxTestMetatype
from tests.cross_fw.test_templates.test_ptq_params import TemplateTestPTQParams
from tests.torch.fx.helpers import get_single_conv_nncf_graph
from tests.torch.ptq.helpers import get_single_no_weight_matmul_nncf_graph

def get_hw_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH_FX, device=device)


def get_ignored_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_ignored_pattern_graph(backend=BackendType.TORCH_FX, device=device)

class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return FXMinMaxAlgoBackend()

    def check_quantize_outputs_fq_num(self, quantize_outputs, act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 2
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    def check_unified_scale_layout(self, layout, unified_scale_group):
        assert len(layout.transformations) == len(unified_scale_group)
       
    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    def get_backend_tensor(self, value):
        return torch.tensor(value)

    @property
    def metatypes_mapping(self):
        return {
            Conv2dTestMetatype: PTConv2dMetatype,
            LinearTestMetatype: PTLinearMetatype,
            SoftmaxTestMetatype: PTSoftmaxMetatype,
            CatTestMetatype: PTCatMetatype,
        }

    @property
    def nncf_graph_cls(self):
        return PTNNCFGraph

    @pytest.fixture(scope="session")
    def test_params(self):
        return {
            "test_quantize_outputs": {
                "nncf_graph": get_single_conv_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_model_type_pass": {
                "nncf_graph": get_single_no_weight_matmul_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
        }