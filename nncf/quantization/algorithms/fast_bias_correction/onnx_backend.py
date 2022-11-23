"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Dict, Tuple, List
import onnx
import numpy as np
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry
from nncf.common.graph import NNCFNode

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXIdentityMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.experimental.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.statistics.collectors import ONNXMeanStatisticCollector
from nncf.experimental.onnx.statistics.collectors import ONNXNNCFCollectorTensorProcessor
from nncf.experimental.onnx.tensor import ONNXNNCFTensor
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.fast_bias_correction.backend import FBCAlgoBackend
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph


@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXFBCAlgoBackend(FBCAlgoBackend):

    @property
    def operation_metatypes(self) -> Registry:
        return ONNX_OPERATION_METATYPES

    @property
    def layers_with_bias_metatypes(self) -> Registry:
        return LAYERS_WITH_BIAS_METATYPES

    @property
    def channel_axis_by_types(self) -> Dict[str, int]:
        return {'Conv': 1, 'Gemm': -1, 'ConvTranspose': 1}

    @property
    def tensor_processor(self) -> ONNXNNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    @staticmethod
    def model_transformer(model: onnx.ModelProto) -> ONNXModelTransformer:
        return ONNXModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def bias_correction_command(target_point: ONNXTargetPoint,
                                bias_value: np.ndarray,
                                threshold: float) -> ONNXBiasCorrectionCommand:
        return ONNXBiasCorrectionCommand(target_point, bias_value, threshold)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> ONNXModelExtractionCommand:
        return ONNXModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: int = None,
                                 window_size: int = None) -> ONNXMeanStatisticCollector:
        return ONNXMeanStatisticCollector(reduction_shape, num_samples, window_size)

    @staticmethod
    def nncf_tensor(tensor: np.ndarray) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(tensor)

    @staticmethod
    def get_tensor_names(node: NNCFNode):
        return node.layer_attributes.input_tensor_names, \
               node.layer_attributes.output_tensor_names

    @staticmethod
    def create_blob(shape: Tuple[int], data: List[float]) -> np.ndarray:
        blob = np.zeros(shape)
        for i, value in enumerate(data):
            blob[:, i] = value
        blob = blob.astype(np.float32)
        return blob

    @staticmethod
    def get_bias_value(model: onnx.ModelProto, node: NNCFNode) -> np.ndarray:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(node.node_name)
        # We uses 2nd value from the tensor names
        # because of the bias tensor placement on this position.
        bias_input_name = node.input[2]
        try:
            return onnx_graph.get_initializers_value(bias_input_name)
        except RuntimeError as e:
            node = onnx_graph.get_nodes_by_output(bias_input_name)[0]
            metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node.op_type)
            if metatype == ONNXIdentityMetatype:
                return onnx_graph.get_initializers_value(node.input[0])
            raise RuntimeError('Could not find the bias value of the node') from e

    @staticmethod
    def get_activation_port_ids_for_bias_node(model: onnx.ModelProto, node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def get_bias_port_id(model: onnx.ModelProto, node: NNCFNode) -> int:
        return 2

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(raw_data[output_name])

    @staticmethod
    def is_quantized_weights(node: NNCFNode, model: onnx.ModelProto) -> bool:
        onnx_graph = ONNXGraph(model)
        # We assume that the weight is on the first-index
        weight_input_index = 1
        input_edge_names = onnx_graph.get_node_edge_names(node.node_name)['input']
        nodes_after_weight = onnx_graph.get_nodes_by_output(input_edge_names[weight_input_index])
        if not nodes_after_weight:
            return False
        # We assume that there is only one node after weight
        assert len(nodes_after_weight) == 1
        weight_dequantizer = nodes_after_weight[0]
        metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(weight_dequantizer.op_type)
        return metatype == ONNXDequantizeLinearMetatype
