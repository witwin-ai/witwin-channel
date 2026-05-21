// drjit_common.h — DrJit/nanobind infrastructure shared by all binding files.
// This header provides type aliases, type casters, pointer helpers,
// and the WitwinCustomOp template used for Dr.Jit custom AD operations.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <drjit/array.h>
#include <drjit/autodiff.h>
#include <drjit/custom.h>
#include <drjit/jit.h>
#include <drjit/python.h>

namespace drjit {

template <typename T>
struct struct_support {
    using Traversable = traversable_t<T>;

    template <typename T1, typename F>
    static void apply_1(T1 &&value, F &&f) {
        auto fields = Traversable::fields(value);
        traverse_1(fields, std::forward<F>(f));
    }

    template <typename T1, typename T2, typename F>
    static void apply_2(T1 &&value_1, T2 &&value_2, F &&f) {
        auto fields_1 = Traversable::fields(value_1);
        auto fields_2 = Traversable::fields(value_2);
        traverse_2(fields_1, fields_2, std::forward<F>(f));
    }
};

} // namespace drjit

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Core DrJit type aliases
// ---------------------------------------------------------------------------

using Float    = drjit::CUDAArray<float>;
using DiffFloat = drjit::CUDADiffArray<float>;
using Int32    = drjit::CUDAArray<int32_t>;
using DiffInt32 = drjit::CUDADiffArray<int32_t>;
using UInt32   = drjit::CUDAArray<uint32_t>;
using DiffUInt32 = drjit::CUDADiffArray<uint32_t>;

// ---------------------------------------------------------------------------
// WitwinCustomOp — base class for DrJit custom AD operations
// ---------------------------------------------------------------------------

template <typename Output, typename Input>
class WitwinCustomOp : public drjit::detail::CustomOpBase {
public:
    using InputType  = Input;
    using OutputType = Output;

    explicit WitwinCustomOp(const Input &input)
        : m_registered_input(drjit::detail::ad_scan(*this, input, true)) {}

    void register_output(const Output &output) {
        m_registered_output = drjit::detail::ad_scan(*this, output, false);
    }

protected:
    Input  m_registered_input;
    Output m_registered_output;
};

template <typename Op, typename Input>
typename Op::OutputType witwin_custom_op(const Input &input) {
    nanobind::ref<Op> op = new Op(input);
    typename Op::OutputType output = op->eval(drjit::detach(input));
    drjit::detail::new_grad(output);
    op->register_output(output);

    if (!ad_custom_op(op.get()))
        drjit::disable_grad(output);

    return output;
}

// ---------------------------------------------------------------------------
// DrJit ↔ Python type caster infrastructure
// ---------------------------------------------------------------------------

template <typename T>
nb::object drjit_python_type() {
    static nb::object type = []() {
        drjit::ArrayBinding binding;
        return drjit::bind_array<T>(binding);
    }();
    return type;
}

template <typename T>
bool drjit_try_load(nb::handle src, T &value, bool convert) {
    if (!src.is_valid()) {
        return false;
    }

    if (nb::inst_check(src)) {
        try {
            if (nb::type_info(src.type()) == typeid(T)) {
                value = *nb::inst_ptr<T>(src);
                return true;
            }
        } catch (...) {
            PyErr_Clear();
        }
    }

    if (!convert) {
        return false;
    }

    try {
        nb::object converted = nb::steal<nb::object>(
            PyObject_CallOneArg(drjit_python_type<T>().ptr(), src.ptr())
        );
        if (!converted.is_valid()) {
            PyErr_Clear();
            return false;
        }
        value = *nb::inst_ptr<T>(converted);
        return true;
    } catch (...) {
        PyErr_Clear();
        return false;
    }
}

template <typename T>
nb::handle drjit_from_cpp(const T &value) {
    nb::object object = nb::steal<nb::object>(
        PyObject_CallNoArgs(drjit_python_type<T>().ptr())
    );
    if (!object.is_valid()) {
        throw nb::python_error();
    }
    *nb::inst_ptr<T>(object) = value;
    return object.release();
}

template <typename T>
struct drjit_type_caster {
    using Value = T;
    template <typename U> using Cast = nanobind::detail::movable_cast_t<U>;
    template <typename U> static constexpr bool can_cast() { return true; }

    bool from_python(nb::handle src, uint8_t flags, nb::detail::cleanup_list *) noexcept {
        return drjit_try_load<T>(
            src,
            value,
            (flags & (uint8_t) nb::detail::cast_flags::convert) != 0
        );
    }

    static nb::handle from_cpp(const T &src, nb::rv_policy, nb::detail::cleanup_list *) {
        return drjit_from_cpp(src);
    }

    static nb::handle from_cpp(T *src, nb::rv_policy policy, nb::detail::cleanup_list *cleanup) {
        if (!src) {
            return nb::none().release();
        }
        return from_cpp(*src, policy, cleanup);
    }

    explicit operator T *() { return &value; }
    explicit operator T &() { return value; }
    explicit operator T &&() { return std::move(value); }

    Value value;
};

// ---------------------------------------------------------------------------
// Pointer helpers for raw CUDA device pointers
// ---------------------------------------------------------------------------

template <typename T>
const typename T::Value* drjit_data_ptr(const T &arr) {
    return arr.data();
}

template <typename T>
typename T::Value* drjit_data_ptr_mut(T &arr) {
    return arr.data();
}

inline std::uintptr_t drjit_data_ptr_handle(nb::handle value) {
    if (!value.is_valid() || value.is_none()) {
        throw std::runtime_error("Expected a valid Dr.Jit array handle");
    }
    nb::object index_attr;
    try {
        index_attr = nb::steal<nb::object>(PyObject_GetAttrString(value.ptr(), "index"));
    } catch (...) {
        PyErr_Clear();
    }
    if (index_attr.is_valid() && !index_attr.is_none()) {
        try {
            uint32_t index = nb::cast<uint32_t>(index_attr);
            void *ptr = nullptr;
            jit_var_data(index, &ptr);
            if (ptr != nullptr) {
                return reinterpret_cast<std::uintptr_t>(ptr);
            }
        } catch (...) {
            PyErr_Clear();
        }
    }
    auto &supplement = nb::type_supplement<drjit::ArraySupplement>(value.type());
    if (supplement.is_valid && supplement.data != nullptr) {
        auto *array = reinterpret_cast<const drjit::ArrayBase *>(nb::inst_ptr<void>(value));
        return reinterpret_cast<std::uintptr_t>(supplement.data(array));
    }
    throw std::runtime_error("Expected a Dr.Jit array handle");
}

inline std::uintptr_t drjit_data_ptr_mut_handle(nb::handle value) {
    if (!value.is_valid() || value.is_none()) {
        throw std::runtime_error("Expected a valid Dr.Jit array handle");
    }
    auto &supplement = nb::type_supplement<drjit::ArraySupplement>(value.type());
    if (supplement.is_valid && supplement.data != nullptr) {
        auto *array = reinterpret_cast<const drjit::ArrayBase *>(nb::inst_ptr<void>(value));
        return reinterpret_cast<std::uintptr_t>(supplement.data(array));
    }
    return drjit_data_ptr_handle(value);
}

template <typename T>
const T* ptr(std::uintptr_t value) {
    return reinterpret_cast<const T*>(value);
}

template <typename T>
T* ptr_mut(std::uintptr_t value) {
    return reinterpret_cast<T*>(value);
}

inline std::vector<std::uintptr_t> array_pointer_list(nb::list values, const char* name) {
    std::vector<std::uintptr_t> result;
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        try {
            result.push_back(drjit_data_ptr_handle(values[i]));
        } catch (const std::exception &exc) {
            std::string type_name = "<unknown>";
            try {
                type_name = nb::str(values[i].type()).c_str();
            } catch (...) {
            }
            throw std::runtime_error(
                std::string(name)
                + ": item "
                + std::to_string(i)
                + " expected a Dr.Jit array handle, got "
                + type_name
                + " ("
                + exc.what()
                + ")"
            );
        }
    }
    return result;
}

inline std::vector<std::uintptr_t> array_pointer_list_mut(nb::list values, const char* name) {
    std::vector<std::uintptr_t> result;
    result.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        try {
            result.push_back(drjit_data_ptr_mut_handle(values[i]));
        } catch (const std::exception &exc) {
            std::string type_name = "<unknown>";
            try {
                type_name = nb::str(values[i].type()).c_str();
            } catch (...) {
            }
            throw std::runtime_error(
                std::string(name)
                + ": item "
                + std::to_string(i)
                + " expected a writable Dr.Jit array handle, got "
                + type_name
                + " ("
                + exc.what()
                + ")"
            );
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// nanobind type caster specializations
// ---------------------------------------------------------------------------

namespace nanobind::detail {

template <>
struct type_caster<Float> : drjit_type_caster<Float> {
    static constexpr auto Name = const_name("drjit.cuda.Float");
};

template <>
struct type_caster<DiffFloat> : drjit_type_caster<DiffFloat> {
    static constexpr auto Name = const_name("drjit.cuda.ad.Float");
};

template <>
struct type_caster<Int32> : drjit_type_caster<Int32> {
    static constexpr auto Name = const_name("drjit.cuda.Int32");
};

template <>
struct type_caster<DiffInt32> : drjit_type_caster<DiffInt32> {
    static constexpr auto Name = const_name("drjit.cuda.ad.Int32");
};

template <>
struct type_caster<UInt32> : drjit_type_caster<UInt32> {
    static constexpr auto Name = const_name("drjit.cuda.UInt32");
};

template <>
struct type_caster<DiffUInt32> : drjit_type_caster<DiffUInt32> {
    static constexpr auto Name = const_name("drjit.cuda.ad.UInt32");
};

template <typename T>
struct raw_pointer_caster {
    using Value = T;
    template <typename U> using Cast = nanobind::detail::movable_cast_t<U>;
    template <typename U> static constexpr bool can_cast() { return true; }

    bool from_python(nb::handle src, uint8_t, nb::detail::cleanup_list *) noexcept {
        if (!src.is_valid() || src.is_none()) {
            value = nullptr;
            return true;
        }
        try {
            value = reinterpret_cast<T>(nb::cast<std::uintptr_t>(src));
            return true;
        } catch (...) {
            PyErr_Clear();
        }
        try {
            using Pointee = std::remove_pointer_t<T>;
            std::uintptr_t ptr_value = 0;
            if constexpr (std::is_const_v<Pointee>) {
                ptr_value = drjit_data_ptr_handle(src);
            } else {
                ptr_value = drjit_data_ptr_mut_handle(src);
            }
            value = reinterpret_cast<T>(ptr_value);
            return true;
        } catch (...) {
            PyErr_Clear();
            return false;
        }
    }

    static nb::handle from_cpp(T src, nb::rv_policy, nb::detail::cleanup_list *) {
        if (!src) {
            return nb::none().release();
        }
        return nb::int_(reinterpret_cast<std::uintptr_t>(src)).release();
    }

    explicit operator T() { return value; }

    Value value = nullptr;
};

template <>
struct type_caster<const float *> : raw_pointer_caster<const float *> {
    static constexpr auto Name = const_name("int");
};

template <>
struct type_caster<float *> : raw_pointer_caster<float *> {
    static constexpr auto Name = const_name("int");
};

template <>
struct type_caster<const int *> : raw_pointer_caster<const int *> {
    static constexpr auto Name = const_name("int");
};

template <>
struct type_caster<int *> : raw_pointer_caster<int *> {
    static constexpr auto Name = const_name("int");
};

} // namespace nanobind::detail
