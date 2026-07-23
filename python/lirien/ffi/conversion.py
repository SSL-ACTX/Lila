import inspect
import ctypes
from typing import (
    Any,
    Callable,
    get_origin,
    get_args,
    Tuple as typing_Tuple,
)
from ..types.base import TYPE_MAP
from ..types.memory import Buffer, SizedArray, Box, Tensor, List
from ..types.functions import Closure, FnPointer
from ..compiler import (
    _get_type_name,
    is_named_tuple,
    is_typed_dict,
    _get_refinement_parts,
)


class StringHeader(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("len", ctypes.c_uint64)]


def _get_ctypes_type(ann_str: str) -> Any:
    """Map a type name string to a ctypes type."""
    # Prioritize specific Lirien types to avoid 'float' matching 'float32'
    priority_types = [
        "f32x4",
        "i32x4",
        "f64x2",
        "i64x2",
        "i8x16",
        "u8x16",
        "i16x8",
        "u16x8",
        "f64",
        "f32",
        "i64",
        "u64",
        "i32",
        "u32",
        "i16",
        "u16",
        "i8",
        "u8",
        "bool",
    ]
    for name in priority_types:
        if name in ann_str:
            return TYPE_MAP[name]

    # Fallback for other types
    for name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
        if name in ann_str:
            return TYPE_MAP[name]
    return ctypes.c_int64


_optional_ctypes_cache = {}


def get_optional_ctypes(inner_cty):
    if inner_cty in _optional_ctypes_cache:
        return _optional_ctypes_cache[inner_cty]

    class OptionalCtypes(ctypes.Structure):
        _fields_ = [("has_value", ctypes.c_bool), ("value", inner_cty)]

    _optional_ctypes_cache[inner_cty] = OptionalCtypes
    return OptionalCtypes


def _is_value_optional(ann):
    from typing import get_origin, get_args, Union
    import types
    import sys

    actual_ann = getattr(ann, "base_type", ann)
    origin = get_origin(actual_ann) or actual_ann
    if origin is Union or (sys.version_info >= (3, 10) and origin is types.UnionType):
        args = get_args(actual_ann)
        has_none = any(arg is type(None) or arg is None for arg in args)
        if has_none:
            non_none_args = [
                arg for arg in args if arg is not type(None) and arg is not None
            ]
            if non_none_args:
                from ..compiler.signature_helpers import _is_box_type

                return not _is_box_type(non_none_args[0]), non_none_args[0]
    return False, None


def _get_flattened_ctypes_types(
    ty: Any, type_mapping: dict[str, str] = None
) -> list[Any]:
    """Recursively discover all basic ctypes types for a given Lirien type."""
    from ..types import i64
    from typing import Tuple as typing_Tuple, get_origin

    if is_named_tuple(ty):
        res = []
        for f_name in ty._fields:
            f_ann = ty.__annotations__.get(f_name, i64)
            res.extend(_get_flattened_ctypes_types(f_ann, type_mapping))
        return res

    origin = get_origin(ty)
    if origin is tuple or origin is typing_Tuple:
        args = get_args(ty)
        res = []
        for arg in args:
            res.extend(_get_flattened_ctypes_types(arg, type_mapping))
        return res

    ty_str = _get_type_name(ty, type_mapping).lower()
    return [_get_ctypes_type(ty_str)]


def _flatten_values(obj: Any) -> list[Any]:
    """Recursively flatten all values in a Tuple or NamedTuple tree."""
    res = []
    if is_named_tuple(type(obj)) or isinstance(obj, (list, tuple)):
        for val in obj:
            if is_named_tuple(type(val)) or isinstance(val, (list, tuple)):
                res.extend(_flatten_values(val))
            else:
                res.append(val)
    else:
        res.append(obj)
    return res


def _unflatten_values(ty: Any, flattened_values: list[Any]) -> Any:
    """Recursively reconstruct a Tuple or NamedTuple from flattened values."""
    from ..types import i64
    from typing import Tuple as typing_Tuple, get_origin, get_args

    if is_named_tuple(ty):
        fields_vals = []
        idx = 0
        for f_name in ty._fields:
            f_ann = ty.__annotations__.get(f_name, i64)
            count = len(_get_flattened_ctypes_types(f_ann))
            fields_vals.append(
                _unflatten_values(f_ann, flattened_values[idx : idx + count])
            )
            idx += count
        return ty(*fields_vals)

    origin = get_origin(ty)
    if origin is tuple or origin is typing_Tuple:
        args = get_args(ty)
        res = []
        idx = 0
        for arg in args:
            count = len(_get_flattened_ctypes_types(arg))
            res.append(_unflatten_values(arg, flattened_values[idx : idx + count]))
            idx += count
        return tuple(res)

    return flattened_values[0]


def _map_ctypes_arguments(
    sig: inspect.Signature, class_name: str = None, type_mapping: dict[str, str] = None
) -> tuple[list[Any], list[Any]]:
    """Map Python function parameters to ctypes types and tracking info."""
    c_args = [ctypes.c_void_p]
    arg_map = []  # List of (type, c_idx, [metadata])

    for i, param in enumerate(sig.parameters.values()):
        ann = param.annotation

        if param.name == "self" and class_name:
            c_args.append(ctypes.c_void_p)
            arg_map.append(("pointer", len(c_args) - 1))
            continue

        actual_ann = getattr(ann, "base_type", ann)

        # Resolve actual_ann from type_mapping if it was substituted
        if getattr(actual_ann, "__lirien_specialized__", False):
            origin = getattr(actual_ann, "__lirien_origin__", None)
            if origin and type_mapping and origin.__name__ in type_mapping:
                actual_ann = type_mapping[origin.__name__]
        elif (
            isinstance(actual_ann, type)
            and type_mapping
            and actual_ann.__name__ in type_mapping
        ):
            actual_ann = type_mapping[actual_ann.__name__]
        elif type_mapping and getattr(actual_ann, "__name__", None) in type_mapping:
            actual_ann = type_mapping[actual_ann.__name__]
        elif type_mapping and str(actual_ann) in type_mapping:
            actual_ann = type_mapping[str(actual_ann)]

        ann_str = _get_type_name(actual_ann, type_mapping).lower()

        from typing import get_origin, Annotated

        origin = get_origin(actual_ann) or actual_ann

        if is_named_tuple(actual_ann):
            # Unpack NamedTuple into multiple arguments recursively
            flattened_ctypes = _get_flattened_ctypes_types(actual_ann, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("named_tuple", start_idx, len(flattened_ctypes)))
            continue

        from typing import Tuple as typing_Tuple

        if origin is tuple or origin is typing_Tuple:
            # Unpack standard Tuple into multiple arguments recursively
            flattened_ctypes = _get_flattened_ctypes_types(actual_ann, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("tuple", start_idx, len(flattened_ctypes)))
            continue

        is_buffer = (
            isinstance(origin, type) and issubclass(origin, Buffer)
        ) or "buffer" in ann_str

        is_tensor = (
            isinstance(origin, type) and issubclass(origin, Tensor)
        ) or "tensor" in ann_str

        is_ptr_wrapper = False
        if (
            (
                isinstance(origin, type)
                and issubclass(
                    origin,
                    (SizedArray, Closure, FnPointer, Callable, Box, Tensor, List),
                )
            )
            or origin is list
            or origin is str
        ):
            is_ptr_wrapper = True

        # Check for Protocol (duck typing)
        if (
            not is_ptr_wrapper
            and hasattr(actual_ann, "_is_protocol")
            and actual_ann._is_protocol
        ):
            is_ptr_wrapper = True

        if not is_ptr_wrapper and any(
            x in ann_str
            for x in [
                "sizedarray",
                "closure",
                "fnpointer",
                "callable",
                "box",
                "tensor",
                "list",
                "nullable",
                "f32x4",
                "i32x4",
                "f64x2",
                "i64x2",
                "i8x16",
                "u8x16",
                "i16x8",
                "u16x8",
            ]
        ):
            is_ptr_wrapper = True

        if is_buffer:
            c_args.append(ctypes.c_void_p)  # Ptr
            c_args.append(ctypes.c_int64)  # Len
            item_size = 8

            # Check metadata for item type
            item_ty = None
            if origin is Annotated and hasattr(actual_ann, "__metadata__"):
                item_ty = actual_ann.__metadata__[0]

            if item_ty is Ellipsis:
                # Inferred type from ellipsis
                ellipsis_key = f"__ellipsis_{param.name}"
                if type_mapping and ellipsis_key in type_mapping:
                    # mapping[key] is a list [type_name] or [dim1, dim2, ...]
                    m_val = type_mapping[ellipsis_key]
                    if isinstance(m_val, list) and len(m_val) > 0:
                        item_ty_str = str(m_val[0]).lower()
                        priority_types = [
                            "f32x4",
                            "i32x4",
                            "f64x2",
                            "i64x2",
                            "i8x16",
                            "u8x16",
                            "i16x8",
                            "u16x8",
                            "f64",
                            "f32",
                            "i64",
                            "u64",
                            "i32",
                            "u32",
                            "i16",
                            "u16",
                            "i8",
                            "u8",
                            "bool",
                        ]
                        for name in priority_types:
                            if name in item_ty_str:
                                item_size = ctypes.sizeof(TYPE_MAP[name])
                                break
            elif item_ty is not None:
                if isinstance(item_ty, type) and issubclass(
                    item_ty, ctypes._SimpleCData
                ):
                    item_size = ctypes.sizeof(item_ty)
                elif _get_refinement_parts(item_ty) != (None, None):
                    item_base_ty, _ = _get_refinement_parts(item_ty)
                    item_ty_str = _get_type_name(item_base_ty, type_mapping).lower()
                    for name in [
                        "f32x4",
                        "i32x4",
                        "f64x2",
                        "i64x2",
                        "i8x16",
                        "u8x16",
                        "i16x8",
                        "u16x8",
                        "f64",
                        "f32",
                        "i64",
                        "u64",
                        "i32",
                        "u32",
                        "i16",
                        "u16",
                        "i8",
                        "u8",
                        "bool",
                    ]:
                        if name in item_ty_str:
                            item_size = ctypes.sizeof(TYPE_MAP[name])
                            break
                elif getattr(item_ty, "__lirien_struct__", False):
                    item_size = ctypes.sizeof(item_ty.__lirien_ctypes__)
                else:
                    item_ty_str = str(item_ty).lower()
                    for name in [
                        "f32x4",
                        "i32x4",
                        "f64x2",
                        "i64x2",
                        "i8x16",
                        "u8x16",
                        "i16x8",
                        "u16x8",
                        "f64",
                        "f32",
                        "i64",
                        "u64",
                        "i32",
                        "u32",
                        "i16",
                        "u16",
                        "i8",
                        "u8",
                        "bool",
                    ]:
                        if name in item_ty_str:
                            item_size = ctypes.sizeof(TYPE_MAP[name])
                            break
            else:
                # Fallback to ann_str
                for name in [
                    "f32x4",
                    "i32x4",
                    "f64x2",
                    "i64x2",
                    "i8x16",
                    "u8x16",
                    "i16x8",
                    "u16x8",
                    "f64",
                    "f32",
                    "i64",
                    "u64",
                    "i32",
                    "u32",
                    "i16",
                    "u16",
                    "i8",
                    "u8",
                    "bool",
                ]:
                    if name in ann_str:
                        item_size = ctypes.sizeof(TYPE_MAP[name])
                        break
            arg_map.append(("buffer", len(c_args) - 2, item_size))
        elif is_tensor:
            c_args.append(ctypes.c_void_p)
            dim_count = 2  # default to 2D
            if origin is Annotated and hasattr(actual_ann, "__metadata__"):
                metadata = actual_ann.__metadata__[0]
                if isinstance(metadata, tuple) and len(metadata) > 1:
                    shape = metadata[1]
                    # Handle Unpack in shape
                    resolved_dim_count = 0
                    for s in shape:
                        s_origin = get_origin(s)
                        if s_origin is not None and "Unpack" in str(s_origin):
                            s_args = get_args(s)
                            if (
                                s_args
                                and type_mapping
                                and s_args[0].__name__ in type_mapping
                            ):
                                unpack_val = type_mapping[s_args[0].__name__]
                                if isinstance(unpack_val, (list, tuple)):
                                    resolved_dim_count += len(unpack_val)
                                else:
                                    resolved_dim_count += 1
                        else:
                            resolved_dim_count += 1
                    dim_count = resolved_dim_count
            for _ in range(dim_count):
                c_args.append(ctypes.c_int64)
            arg_map.append(("tensor", len(c_args) - 1 - dim_count, dim_count))
        elif (
            is_ptr_wrapper
            or getattr(actual_ann, "__lirien_struct__", False)
            or getattr(actual_ann, "__lirien_enum__", False)
            or is_typed_dict(actual_ann)
        ):
            c_args.append(ctypes.c_void_p)
            arg_map.append(("pointer", len(c_args) - 1, actual_ann))
        elif is_named_tuple(actual_ann):
            # Unpack NamedTuple into multiple arguments recursively
            flattened_ctypes = _get_flattened_ctypes_types(actual_ann, type_mapping)
            start_idx = len(c_args)
            c_args.extend(flattened_ctypes)
            arg_map.append(("named_tuple", start_idx, len(flattened_ctypes)))
        else:
            c_args.append(_get_ctypes_type(ann_str))
            arg_map.append(("value", len(c_args) - 1))
    return c_args, arg_map


def _handle_pointer_return(
    ret_ann: Any,
    c_args: list[Any],
    arg_map: list[Any],
    type_mapping: dict[str, str] = None,
) -> tuple[bool, Any, list[Any], list[Any], list[Any]]:
    ret_ann_str = _get_type_name(ret_ann, type_mapping).lower()
    raw_ann_str = str(ret_ann).lower()
    is_val_opt, inner_type = _is_value_optional(ret_ann)
    is_struct = getattr(ret_ann, "__lirien_struct__", False)

    if is_val_opt or is_struct:
        if is_val_opt:
            if getattr(inner_type, "__lirien_struct__", False):
                inner_cty = inner_type.__lirien_ctypes__
            else:
                inner_cty = _get_ctypes_type(_get_type_name(inner_type, type_mapping))
            ResultStruct = get_optional_ctypes(inner_cty)
        else:
            ResultStruct = ret_ann.__lirien_ctypes__

        new_c_args = [ctypes.c_void_p] + c_args
        new_arg_map = []
        for info in arg_map:
            # Adjust index for SRet
            new_arg_map.append((info[0], info[1] + 1) + info[2:])
        return True, ResultStruct, new_c_args, new_arg_map, []

    # Detect if we need return-by-pointer (SRet style)
    is_tuple = (
        "tuple" in raw_ann_str
        or (ret_ann_str.startswith("(") and ret_ann_str.endswith(")"))
        or is_named_tuple(ret_ann)
    )
    is_simd = any(
        x in ret_ann_str
        for x in [
            "f32x4",
            "i32x4",
            "f64x2",
            "i64x2",
            "i8x16",
            "u8x16",
            "i16x8",
            "u16x8",
        ]
    )

    if not (is_tuple or is_simd):
        return False, None, c_args, arg_map, []

    from ..types import i64

    if is_named_tuple(ret_ann) or get_origin(ret_ann) in [tuple, typing_Tuple]:
        flattened_ctypes = _get_flattened_ctypes_types(ret_ann, type_mapping)

        class TupleResult(ctypes.Structure):
            _fields_ = [(f"f{i}", cty) for i, cty in enumerate(flattened_ctypes)]

        if len(flattened_ctypes) <= 2:
            # Return by registers
            return False, TupleResult, c_args, arg_map, flattened_ctypes
        else:
            # Return by pointer (SRet)
            c_args.insert(0, ctypes.POINTER(TupleResult))
            # Shift existing arg_map indices
            new_arg_map = []
            for item in arg_map:
                if item[0] == "named_tuple" or item[0] == "tuple":
                    new_arg_map.append((item[0], item[1] + 1, item[2]))
                else:
                    new_arg_map.append((item[0], item[1] + 1))
            return True, TupleResult, c_args, new_arg_map, flattened_ctypes

    try:
        ResultStruct = None
        tuple_types = []

        if is_tuple:
            # Try to extract inner types for Tuple
            if hasattr(ret_ann, "__args__"):
                tuple_types = list(ret_ann.__args__)
            else:
                tuple_types = [i64, i64]  # Default

            tuple_fields = []
            for i, t in enumerate(tuple_types):
                f_ty_str = _get_type_name(t, type_mapping).lower()
                tuple_fields.append((f"f{i}", _get_ctypes_type(f_ty_str)))

            class TupleReturn(ctypes.Structure):
                _fields_ = tuple_fields

            ResultStruct = TupleReturn
        elif is_simd:
            # SIMD Return - Match the specific vector type exactly if possible
            # Sort keys by length descending to match 'f32x4' before 'f32'
            for name in sorted(TYPE_MAP.keys(), key=len, reverse=True):
                if name in ret_ann_str:
                    ResultStruct = TYPE_MAP[name]
                    break

        if ResultStruct is None:
            return False, None, c_args, arg_map, []

        new_c_args = [ctypes.c_void_p] + c_args
        new_arg_map = []
        for info in arg_map:
            # Adjust arg_map indices because we inserted a pointer at index 0
            new_arg_map.append((info[0], info[1] + 1) + info[2:])

        return True, ResultStruct, new_c_args, new_arg_map, tuple_types
    except Exception as e:
        print(f"[Lirien Warning] Failed to setup result structure: {e}")
        return False, None, c_args, arg_map, []


def _get_ctypes_return_type(ret_ann: Any, type_mapping: dict[str, str] = None) -> Any:
    """Determine the ctypes return type from the annotation."""
    # Unwrap Refined / Annotated refinement type if necessary
    base_ty, _ = _get_refinement_parts(ret_ann)
    actual_ann = base_ty if base_ty is not None else ret_ann
    ret_ann_str = _get_type_name(actual_ann, type_mapping).lower()

    if (
        actual_ann is None
        or actual_ann is inspect.Signature.empty
        or "none" in ret_ann_str
    ):
        return None

    from typing import get_origin
    from ..types import Tensor, Buffer

    origin = get_origin(actual_ann) or actual_ann
    if (
        (isinstance(origin, type) and issubclass(origin, (Tensor, Buffer)))
        or "tensor" in ret_ann_str
        or "buffer" in ret_ann_str
    ):
        return ctypes.c_int64

    return _get_ctypes_type(ret_ann_str)
