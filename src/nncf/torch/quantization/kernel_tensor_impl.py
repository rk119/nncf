from nncf import torch

import functools, torch
import gemlite         
from torch.utils._python_dispatch import return_and_correct_aliasing
aten = torch.ops.aten
from typing import Optional
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.weight_lowering import ungroup_weights

def _implements(cls, aten_ops_or_torch_fns):
    if not hasattr(cls, "_IMPL_TABLE"):
        cls._IMPL_TABLE = {}
    if not isinstance(aten_ops_or_torch_fns, (list, tuple)):
        aten_ops_or_torch_fns = [aten_ops_or_torch_fns]

    def decorator(func):
        for op in aten_ops_or_torch_fns:
            @functools.wraps(op)
            def wrapper(f, types, args, kwargs):
                return func(f, types, args, kwargs)
            cls._IMPL_TABLE[op] = wrapper
        return func
    return decorator

def _register_impl(tensor_cls, impl_cls):
    if not hasattr(tensor_cls, "_IMPL_CTOR_TABLE"):
        tensor_cls._IMPL_CTOR_TABLE = {}

    tensor_cls._IMPL_CTOR_TABLE[impl_cls] = impl_cls.pack_weights
    return impl_cls 

    
def _dispatch__torch_function__(cls, func, types, args=(), kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    if hasattr(cls, "_IMPL_TABLE") and func in cls._IMPL_TABLE:
        return cls._IMPL_TABLE[func](func, types, args, kwargs)
    with torch._C.DisableTorchFunctionSubclass():
        return func(*args, **kwargs)


def _dispatch__torch_dispatch__(cls, func, types, args, kwargs):
    if hasattr(cls, "_IMPL_TABLE") and func in cls._IMPL_TABLE:
        return cls._IMPL_TABLE[func](func, types, args, kwargs)
    raise NotImplementedError(f"{cls.__name__} did not implement {func}")

class TorchBaseTensor(torch.Tensor):
    implements          = classmethod(_implements)
    __torch_function__  = classmethod(_dispatch__torch_function__)
    __torch_dispatch__  = classmethod(_dispatch__torch_dispatch__)
    register_impl       = classmethod(_register_impl)

    def get_unpacked_weights(self):
        raise NotImplementedError
    @classmethod
    def pack_weights(cls, *a, **kw):
        raise NotImplementedError

@TorchBaseTensor.register_impl
class GemliteTensorImpl(TorchBaseTensor):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_meta: dict,
        group_size: int,
        bit_width: int,
    ):
        kwargs = dict(device=packed_weight.device,
                      dtype=packed_weight.dtype,
                      requires_grad=False)
        obj = torch.Tensor._make_wrapper_subclass(cls, packed_weight.shape, **kwargs)
        return obj

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_meta: dict,
        group_size: int,
        bit_width: int,
    ):
        self.packed_weight = packed_weight
        self.scale         = scale
        self.zero_point    = zero_point if zero_point is not None else torch.zeros_like(scale)
        self.gemlite_meta  = gemlite_meta
        self.group_size    = group_size
        self.bit_width     = bit_width

    def __tensor_flatten__(self):
        return ["packed_weight", "scale", "zero_point"], [
            dict(group_size=self.group_size,
                 bit_width=self.bit_width,
                 gemlite_meta=self.gemlite_meta)
        ]

    @classmethod
    def __tensor_unflatten__(cls, tdict, attrs, outer_size, outer_stride):
        meta = attrs[0]
        return cls(tdict["packed_weight"], tdict["scale"], tdict["zero_point"],
                   meta["gemlite_meta"],
                   group_size=meta["group_size"],
                   bit_width=meta["bit_width"])

    def get_unpacked_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unpacked_weights = (gemlite.bitpack.unpack_over_rows(
                        self.packed_weight.cuda(),
                        W_nbits=self.bit_width,
                        num_output_rows=self.gemlite_meta["out_features"],
                        dtype=torch.uint8)
                    .t().contiguous()
                   ).to(self.packed_weight.device)
        return unpacked_weights, self.scale.t().contiguous(), self.zero_point.t().contiguous()

    @classmethod
    def pack_weights(
        cls,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        mode: CompressWeightsMode,
        reduction_axis: int,
        group_size: int = 64,
    ):
        if weight.device.type != "cuda":       
            weight = weight.cuda()

        bit_width = 8

        if mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]:
            bit_width = 4
            weight = ungroup_weights(weight, reduction_axis).data
            scale = scale.squeeze(-1)
            if zero_point is None:
                weight = (weight + 8).to(torch.uint8)
                zero_point = torch.zeros_like(scale, dtype=torch.int32)
            else:
                zero_point = zero_point.squeeze(-1)

        scale = scale.to(torch.float16)
        out_f, in_f = weight.shape
        if not weight.is_contiguous():
            weight = weight.contiguous()
        if bit_width == 8 and group_size == in_f and zero_point is None:
            gl = gemlite.helper.A16W8(device=weight.device).from_weights(
                     weight, scales=scale, bias=None)
        else:
            gl = gemlite.helper.A16Wn(device=weight.device).from_weights(
                     weight, scale, zero_point, bit_width, group_size, bias=None)

        packed_w, s, zp = gl.get_tensor_args()
        packed_w = packed_w.to(weight.device).contiguous()
        meta = dict(in_features=in_f, out_features=out_f, meta_args=gl.get_meta_args())

        return cls(packed_w.to(weight.device), s, zp, meta, group_size=group_size, bit_width=bit_width)

    def _gemlite_linear(self, x, bias=None):
        return gemlite.core.forward_functional(
            x=x,
            bias=bias,
            tensor_args=(self.packed_weight, self.scale, self.zero_point),
            meta_args=self.gemlite_meta["meta_args"],
        )

    __torch_function__ = torch._C._disabled_torch_function_impl  # disable fallback

class GemWeight(TorchBaseTensor):
    def __new__(cls, tensor_impl: GemliteTensorImpl):
        return torch.Tensor._make_wrapper_subclass(
            cls, tensor_impl.shape,
            device=tensor_impl.packed_weight.device,
            dtype=tensor_impl.packed_weight.dtype,
            requires_grad=False)

    def __init__(self, tensor_impl):
        self.tensor_impl = tensor_impl

    def _gemlite_linear(self, *a, **kw):
        return self.tensor_impl._gemlite_linear(*a, **kw)
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = kwargs or {}

        if func is aten.detach.default or func is aten.clone.default:
            return return_and_correct_aliasing(func, args, kwargs,
                                               GemWeight(args[0].tensor_impl))

        return TorchBaseTensor.__torch_dispatch__(func, types, args, kwargs)

@TorchBaseTensor.implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    x, w, bias = args[0], args[1], (args[2] if len(args) > 2 else None)
    if not isinstance(w, GemWeight):
        return func(*args, **kwargs)              

    return w._gemlite_linear(x, bias=bias)