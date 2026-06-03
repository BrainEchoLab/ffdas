#include <stdexcept>
#include <iostream>
#include <complex>
#include <cstdio>
#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

#include <vector_types.h>
#include <cuda_fp16.h>

#include "ffdas.h"

namespace nb = nanobind;
using namespace nb::literals;


struct DLManagedTensor {
    nb::dlpack::dltensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(DLManagedTensor *);
};


// prevent GC of buffers backing a DLManagedTensor until the consumer calls
// the deleter or the PyCapsule is destroyed.
struct ManagedCtx {
    nb::object base;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};


struct DType {
    const char *name;
    int code;  // ffdas data type code
    nb::dlpack::dtype dlpack;  // packed DLPack dtype
    nb::dlpack::dtype base;  // expected element dtype in source arrays
    bool is_complex;
    bool operator==(const DType &o) const { return code == o.code; }
};


static const DType dtypes[] = {
    {"half",      FFDAS_R_16F, {(uint8_t)nb::dlpack::dtype_code::Float,   16, 1}, {(uint8_t)nb::dlpack::dtype_code::Float,   16, 1}, false},
    {"float32",   FFDAS_R_32F, {(uint8_t)nb::dlpack::dtype_code::Float,   32, 1}, {(uint8_t)nb::dlpack::dtype_code::Float,   32, 1}, false},
    {"float64",   FFDAS_R_64F, {(uint8_t)nb::dlpack::dtype_code::Float,   64, 1}, {(uint8_t)nb::dlpack::dtype_code::Float,   64, 1}, false},
    {"int16",     FFDAS_R_16I, {(uint8_t)nb::dlpack::dtype_code::Int,     16, 1}, {(uint8_t)nb::dlpack::dtype_code::Int,     16, 1}, false},
    {"int32",     FFDAS_R_32I, {(uint8_t)nb::dlpack::dtype_code::Int,     32, 1}, {(uint8_t)nb::dlpack::dtype_code::Int,     32, 1}, false},
    {"half2",     FFDAS_C_16F, {(uint8_t)nb::dlpack::dtype_code::Float,   16, 2}, {(uint8_t)nb::dlpack::dtype_code::Float,   16, 1}, true},
    {"complex64", FFDAS_C_32F, {(uint8_t)nb::dlpack::dtype_code::Complex, 64, 1}, {(uint8_t)nb::dlpack::dtype_code::Complex, 64, 1}, true},
    {"complex128",FFDAS_C_64F, {(uint8_t)nb::dlpack::dtype_code::Complex, 128, 1}, {(uint8_t)nb::dlpack::dtype_code::Complex, 128, 1}, true},
    {"double2",   FFDAS_C_64F, {(uint8_t)nb::dlpack::dtype_code::Float,   64, 2}, {(uint8_t)nb::dlpack::dtype_code::Float,   64, 1}, true},
    {"float2",    FFDAS_C_32F, {(uint8_t)nb::dlpack::dtype_code::Float,   32, 2}, {(uint8_t)nb::dlpack::dtype_code::Float,   32, 1}, true},
    {"short2",    FFDAS_C_16I, {(uint8_t)nb::dlpack::dtype_code::Int,     16, 2}, {(uint8_t)nb::dlpack::dtype_code::Int,     16, 1}, true},
    {"int2",      FFDAS_C_32I, {(uint8_t)nb::dlpack::dtype_code::Int,     32, 2}, {(uint8_t)nb::dlpack::dtype_code::Int,     32, 1}, true},
};


static ffdas_datatype_t to_ffdas_dtype(nb::dlpack::dtype dt) {
    for (const auto &d : dtypes)
        if (d.dlpack == dt) return static_cast<ffdas_datatype_t>(d.code);
    throw std::invalid_argument("unsupported dtype");
}


namespace nanobind::detail {
    template <> struct dtype_traits<int2> {
        static constexpr dlpack::dtype value {
            (uint8_t) dlpack::dtype_code::Int, 32, 2
        };
        static constexpr auto name = const_name("int2");
    };
    template <> struct dtype_traits<short2> {
        static constexpr dlpack::dtype value {
            (uint8_t) dlpack::dtype_code::Int, 16, 2
        };
        static constexpr auto name = const_name("short2");
    };
    template <> struct dtype_traits<half2> {
        static constexpr dlpack::dtype value {
            (uint8_t) dlpack::dtype_code::Float, 16, 2
        };
        static constexpr auto name = const_name("half2");
    };
}


struct TensorView {
    nb::object base;
    void *data;
    std::vector<int64_t> shape;
    const DType *dtype;
    int32_t device_type;
    int32_t device_id;
};


static void check(ffdas_error_t err) {
    if (err != FFDAS_SUCCESS)
        throw std::runtime_error(ffdas_error_string(err));
}


struct Handle {
    ffdas_handle_t h = nullptr;
    Handle() { 
        check(ffdas_create(&h)); 
    }
    void destroy() {
        if (h) {
            ffdas_destroy(h); 
            h = nullptr;
        }
    }
    ~Handle() { 
        if (h) {
            ffdas_destroy(h); 
            h = nullptr;
        }
    }
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
};


struct ContractionPlan {
    ffdas_contraction_plan_t plan = nullptr;
    Handle *owner = nullptr;

    ~ContractionPlan() {
        if (plan && owner && owner->h)
            ffdas_destroy_contraction(owner->h, plan);
    }
};


struct InterpolationPlan {
    ffdas_interpolation_plan_t plan = nullptr;
    Handle *owner = nullptr;

    ~InterpolationPlan() {
        if (plan && owner && owner->h)
            ffdas_destroy_interpolation_plan(owner->h, plan);
    }
};


struct ScopedTensorDesc {
    ffdas_tensor_desc_t desc = nullptr;

    template <typename... Ts>
    ScopedTensorDesc(const nb::ndarray<Ts...> &a) {
        int64_t ndim = a.ndim();
        std::vector<int64_t> dims(ndim), strides(ndim);
        for (int i = 0; i < ndim; i++) {
            dims[i] = a.shape(i);
            strides[i] = a.stride(i);
        }
        check(ffdas_create_tensor_desc(&desc, ndim, dims.data(), strides.data(), to_ffdas_dtype(a.dtype())));
    }

    ~ScopedTensorDesc() {
        if (desc) ffdas_destroy_tensor_desc(desc);
    }

    ScopedTensorDesc(const ScopedTensorDesc &) = delete;
    ScopedTensorDesc &operator=(const ScopedTensorDesc &) = delete;
};


NB_MODULE(_ffdas, m) {
    nb::class_<DType>(m, "dtype")
        .def_ro("name", &DType::name)
        .def_ro("code", &DType::code)
        .def("__repr__", [](const DType &dt) {
            return std::string("ffdas.") + dt.name;
        })
        .def("__hash__", [](const DType &dt) {
            return nb::hash(nb::int_(static_cast<int>(dt.code)));
        });

    for (auto &dt : dtypes)
        m.attr(dt.name) = nb::cast(&dt, nb::rv_policy::reference);

    nb::class_<TensorView>(m, "TensorView")
        .def_prop_ro("shape", [](TensorView &self) {
            nb::tuple result = nb::steal<nb::tuple>(PyTuple_New(self.shape.size()));
            for (size_t i = 0; i < self.shape.size(); i++)
                PyTuple_SetItem(result.ptr(), i, PyLong_FromLongLong(self.shape[i]));
            return result;
        })
        .def_prop_ro("ndim", [](TensorView &self) {
            return self.shape.size();
        })
        .def_prop_ro("dtype", [](TensorView &self) {
            return self.dtype;
        }, nb::rv_policy::reference)
        .def_prop_ro("base", [](TensorView &self) {
            return self.base;
        }, nb::rv_policy::reference_internal)
        .def("__repr__", [](TensorView &self) {
            std::ostringstream os;
            os << "TensorView(shape=(";
            for (size_t i = 0; i < self.shape.size(); i++) {
                if (i > 0) os << ", ";
                os << self.shape[i];
            }
            if (self.shape.size() == 1) os << ",";
            os << "), dtype=ffdas." << self.dtype->name << ")";
            return os.str();
        })
        .def("__dlpack_device__", [](TensorView &self) {
            return std::make_pair(self.device_type, self.device_id);
        })
        .def("__dlpack__", [](TensorView &self, nb::kwargs) -> nb::object {
            size_t ndim = self.shape.size();

            auto *ctx = new ManagedCtx();
            ctx->base = self.base;
            ctx->shape = self.shape;
            ctx->strides.resize(ndim);
            int64_t stride = 1;
            for (int i = (int)ndim - 1; i >= 0; i--) {
                ctx->strides[i] = stride;
                stride *= ctx->shape[i];
            }

            auto *managed = new DLManagedTensor();
            managed->dl_tensor.data = self.data;
            managed->dl_tensor.device = {self.device_type, self.device_id};
            managed->dl_tensor.ndim = (int32_t)ndim;
            managed->dl_tensor.dtype = self.dtype->dlpack;
            managed->dl_tensor.shape = ctx->shape.data();
            managed->dl_tensor.strides = ctx->strides.data();
            managed->dl_tensor.byte_offset = 0;
            managed->manager_ctx = ctx;
            managed->deleter = [](DLManagedTensor *m) {
                delete static_cast<ManagedCtx *>(m->manager_ctx);
                delete m;
            };
 
            // PyCapsule destructor handles cleanup if the capsule is never
            // consumed (PyCapsule_GetPointer returns NULL once the consumer
            // renames the capsule to "used_dltensor").
            PyObject *capsule = PyCapsule_New(
                managed, "dltensor",
                [](PyObject *cap) {
                    auto *m = static_cast<DLManagedTensor *>(
                        PyCapsule_GetPointer(cap, "dltensor"));
                    if (m && m->deleter)
                        m->deleter(m);
                });
            if (!capsule)
                throw nb::python_error();
            return nb::steal(capsule);
        });

    m.def("view", [](nb::object src_obj, const DType &dtype) -> TensorView {
        auto src = nb::cast<nb::ndarray<>>(src_obj);
 
        if (src.dtype() != dtype.base) {
            std::ostringstream os;
            os << "view: expected base element type compatible with "
               << dtype.name;
            throw nb::value_error(os.str().c_str());
        }

        TensorView view;
        view.base = std::move(src_obj);
        view.data = src.data();
        view.dtype = &dtype;
        view.device_type = src.device_type();
        view.device_id = src.device_id();

        if (dtype.is_complex && dtype.dlpack.lanes == 2) {
            if (src.ndim() == 0 || src.shape(src.ndim() - 1) != 2)
                throw nb::value_error(
                    "view: complex dtype requires trailing dimension of 2");

            for (size_t i = 0; i < src.ndim() - 1; i++)
                view.shape.push_back((int64_t)src.shape(i));
        } else {
            for (size_t i = 0; i < src.ndim(); i++)
                view.shape.push_back((int64_t)src.shape(i));
        }

        return view;
    }, nb::arg("a"), nb::arg("dtype"));

    nb::enum_<ffdas_alg_t>(m, "Algorithm")
        .value("DEFAULT", FFDAS_ALG_DEFAULT)
        .value("ALG1", FFDAS_ALG1)
        .value("ALG2", FFDAS_ALG2)
        .value("ALG3", FFDAS_ALG3)
        .value("ALG4", FFDAS_ALG4);

    nb::enum_<ffdas_interp_mode_t>(m, "InterpMode")
        .value("NEAREST", FFDAS_INTERP_NEAREST)
        .value("LINEAR", FFDAS_INTERP_LINEAR);

    nb::enum_<ffdas_compute_type_t>(m, "ComputeType")
        .value("DEFAULT", FFDAS_COMPUTE_DEFAULT)
        .value("FP16", FFDAS_COMPUTE_16F)
        .value("FP32", FFDAS_COMPUTE_32F)
        .value("FP64", FFDAS_COMPUTE_64F);

    nb::class_<Handle>(m, "Handle")
        .def(nb::init<>())
        .def("destroy", [](Handle &self) {
            self.destroy();
        });

    m.def("event_create", []() {
        uintptr_t event;
        check(ffdas_event_create(&event));
        return event;
    });

    m.def("event_destroy", [](uintptr_t event) {
        check(ffdas_event_destroy(event));
    }, "event"_a);

    m.def("event_record", [](Handle &handle, uintptr_t event) {
        check(ffdas_event_record(handle.h, event));
    }, "handle"_a, "event"_a);

    m.def("event_synchronize", [](uintptr_t event) {
        check(ffdas_event_synchronize(event));
    }, "event"_a);

    m.def("event_elapsed_time", [](uintptr_t start, uintptr_t stop) {
        float ms;
        check(ffdas_event_elapsed_time(start, stop, &ms));
        return ms;
    }, "start"_a, "stop"_a);

    nb::class_<ContractionPlan>(m, "ContractionPlan");

    m.def("create_contraction", [](Handle &handle,
                                    nb::ndarray<nb::ro, nb::device::cuda> x, 
                                    std::vector<int> x_modes,
                                    nb::ndarray<nb::ro, nb::device::cuda> a, 
                                    std::vector<int> a_modes,
                                    nb::ndarray<nb::device::cuda> y, 
                                    std::vector<int> y_modes) {
        ScopedTensorDesc x_desc(x), a_desc(a), y_desc(y);
        auto *cp = new ContractionPlan();
        cp->owner = &handle;
        check(ffdas_create_contraction(
            handle.h, 
            &cp->plan,
            x_desc.desc, 
            x_modes.data(),
            a_desc.desc, 
            a_modes.data(),
            y_desc.desc, 
            y_modes.data()
        ));
        return cp;
    }, "handle"_a, "x"_a, "x_modes"_a, "a"_a, "a_modes"_a, "y"_a, "y_modes"_a,
       nb::rv_policy::take_ownership, nb::keep_alive<0, 1>());

    m.def("contraction", [](Handle &handle, 
                            ContractionPlan &plan,
                             nb::ndarray<nb::ro, nb::device::cuda> x, 
                             nb::ndarray<nb::ro, nb::device::cuda> a, 
                             nb::ndarray<nb::device::cuda> y) {
        check(ffdas_contraction(
            handle.h, 
            plan.plan, 
            x.data(), 
            a.data(), 
            y.data()
        ));
    }, "handle"_a, "plan"_a, "x"_a, "a"_a, "y"_a);

    nb::class_<InterpolationPlan>(m, "InterpolationPlan");

    m.def("create_interpolation_plan", [](
        Handle &handle,
        int64_t nx, 
        int64_t ny, 
        int64_t nz,
        nb::ndarray<const float, nb::device::cuda, nb::c_contig> grid_points,
        ffdas_interp_mode_t mode
    ) {
        auto *ip = new InterpolationPlan();
        ip->owner = &handle;
        check(ffdas_create_interpolation_plan(
            handle.h, 
            &ip->plan,
            nx, 
            ny, 
            nz,
            grid_points.data(), 
            mode
        ));
        return ip;
    }, "handle"_a, "nx"_a, "ny"_a, "nz"_a, "grid_points"_a, "mode"_a,
       nb::rv_policy::take_ownership, nb::keep_alive<0, 1>());

    m.def("interpolation_preprocess", [](
        Handle &handle,
        InterpolationPlan &plan,
        nb::ndarray<const float, nb::device::cuda, nb::c_contig> query_points
    ) {
        int num = 1;
        for (size_t i = 0; i < query_points.ndim() - 1; i++) {
            num *= query_points.shape(i);
        }
        check(ffdas_interpolation_preprocess(
            handle.h, 
            plan.plan,
            num,
            query_points.data()
        ));
    }, "handle"_a, "plan"_a, "query_points"_a);

    m.def("interpolation", [](
        Handle &handle,
        InterpolationPlan &plan,
        nb::ndarray<const float, nb::device::cuda, nb::c_contig> query_points,
        nb::ndarray<nb::ro, nb::device::cuda> values,
        nb::ndarray<nb::device::cuda> output,
        nb::ndarray<nb::ro, nb::device::cpu> fill_value
    ) {
        ScopedTensorDesc values_desc(values);
        int num = 1;
        for (size_t i = 0; i < query_points.ndim() - 1; i++) {
            num *= query_points.shape(i);
        }
        check(ffdas_interpolation(
            handle.h, 
            plan.plan,
            num,
            query_points.data(),
            values_desc.desc, values.data(),
            output.data(), 
            fill_value.data()
        ));
    }, "handle"_a, "plan"_a, "query_points"_a, "values"_a, "output"_a,
       "fill_value"_a);

    m.def("eigfilter", [](Handle &handle,
                           nb::ndarray<nb::ro, nb::device::cuda> x, 
                           int k0, 
                           int k1,
                           nb::ndarray<nb::device::cuda> y
    ) {
        ScopedTensorDesc x_desc(x), y_desc(y);
        check(ffdas_eigfilter(
            handle.h, 
            x_desc.desc, 
            x.data(), 
            k0, 
            k1,
            y_desc.desc, 
            y.data()
        ));
    }, "handle"_a, "x"_a, "k0"_a, "k1"_a, "y"_a);

    m.def("das", [](Handle &handle,
                     nb::ndarray<nb::ro, nb::device::cuda> x,
                     nb::ndarray<const float, nb::device::cuda, nb::c_contig> xpos,
                     nb::ndarray<const float, nb::device::cuda, nb::c_contig> ypos,
                     nb::ndarray<const float, nb::device::cuda, nb::c_contig> offsets,
                     nb::ndarray<const float, nb::device::cuda, nb::c_contig> weights,
                     std::optional<nb::ndarray<const float, nb::device::cuda, nb::c_contig>> xdir,
                     float wavenum,
                     nb::ndarray<nb::ro, nb::device::cpu> beta,
                     nb::ndarray<nb::device::cuda> y,
                     ffdas_alg_t alg,
                     ffdas_compute_type_t compute_type) {
        ScopedTensorDesc x_desc(x), y_desc(y);

        check(ffdas_das(
            handle.h,
            reinterpret_cast<const float3*>(xpos.data()),
            xdir.has_value() ? reinterpret_cast<const float4 *>((*xdir).data()) : nullptr,
            wavenum,
            x_desc.desc, 
            x.data(),
            reinterpret_cast<const float3*>(ypos.data()),
            offsets.data(),
            weights.data(),
            beta.data(),
            y_desc.desc, 
            y.data(),
            compute_type,
            alg
        ));
    }, "handle"_a, "x"_a, "xpos"_a, "ypos"_a, "offsets"_a,
       "weights"_a, nb::arg("xdir").none(), "wavenum"_a, "beta"_a,
       "y"_a, "alg"_a, "compute_type"_a);

    m.def("das_sparse", [](Handle &handle,
                            nb::ndarray<nb::ro, nb::device::cuda> x,
                            nb::ndarray<const float, nb::device::cuda, nb::c_contig> xpos,
                            nb::ndarray<const float, nb::device::cuda, nb::c_contig> ypos,
                            nb::ndarray<const float, nb::device::cuda, nb::c_contig> offsets,
                            nb::ndarray<const float, nb::device::cuda, nb::c_contig> weights,
                            std::optional<nb::ndarray<const float, nb::device::cuda, nb::c_contig>> xdir,
                            float wavenum,
                            nb::ndarray<nb::ro, nb::device::cpu> beta,
                            nb::ndarray<nb::device::cuda> y,
                            nb::ndarray<const int, nb::device::cuda, nb::c_contig> sparse_indices,
                            ffdas_alg_t alg,
                            ffdas_compute_type_t compute_type) {
        ScopedTensorDesc x_desc(x), y_desc(y);
        check(ffdas_das_sparse(
            handle.h,
            reinterpret_cast<const float3 *>(xpos.data()),
            xdir.has_value() ? reinterpret_cast<const float4 *>((*xdir).data()) : nullptr,
            wavenum,
            x_desc.desc, 
            x.data(),
            reinterpret_cast<const float3 *>(ypos.data()),
            offsets.data(),
            weights.data(),
            sparse_indices.shape(0),
            sparse_indices.data(),
            beta.data(),
            y_desc.desc, 
            y.data(),
            compute_type,
            alg
        ));
    }, "handle"_a, "x"_a, "xpos"_a, "ypos"_a, "offsets"_a,
       "weights"_a, nb::arg("xdir").none(), "wavenum"_a, "beta"_a,
       "y"_a, "sparse_indices"_a, "alg"_a, "compute_type"_a);

    m.def("contiguous_copy", [](Handle &handle, nb::ndarray<nb::ro, nb::device::cuda> x, 
        nb::ndarray<nb::device::cuda> y) {
        ScopedTensorDesc x_desc(x), y_desc(y);
        check(ffdas_contiguous_copy(
            handle.h, 
            x_desc.desc, 
            x.data(),
            y_desc.desc, 
            y.data()
        ));
    }, "handle"_a, "x"_a, "y"_a);

    m.def("gather", [](Handle &handle,
                        nb::ndarray<nb::ro, nb::device::cuda> x, 
                        nb::ndarray<nb::device::cuda> y,
                        int mode,
                        nb::ndarray<const int, nb::device::cuda, nb::ndim<1>, nb::c_contig> indices) {
        ScopedTensorDesc x_desc(x), y_desc(y);
        check(ffdas_gather(
            handle.h, 
            x_desc.desc, x.data(),
            y_desc.desc, y.data(),
            mode, 
            indices.shape(0), indices.data()
        ));
    }, "handle"_a, "x"_a, "y"_a, "mode"_a, "indices"_a);

    m.def("scatter", [](Handle &handle,
                         nb::ndarray<nb::ro, nb::device::cuda> x, nb::ndarray<nb::device::cuda> y,
                         int mode,
                         nb::ndarray<const int, nb::device::cuda, nb::ndim<1>, nb::c_contig> indices) {
        ScopedTensorDesc x_desc(x), y_desc(y);
        check(ffdas_scatter(
            handle.h, 
            x_desc.desc, x.data(),
            y_desc.desc, y.data(),
            mode, 
            indices.data()
        ));
    }, "handle"_a, "x"_a, "y"_a, "mode"_a, "indices"_a);

    m.def("greens_sum", [](Handle &handle,
        nb::ndarray<const float, nb::device::cuda, nb::c_contig> xpos,
        nb::ndarray<const float, nb::device::cuda, nb::c_contig> wavenums,
        nb::ndarray<nb::ro, nb::device::cuda> x,
        nb::ndarray<const float, nb::device::cuda, nb::c_contig> ypos,
        nb::ndarray<nb::device::cuda> y
    ) {
        ScopedTensorDesc x_desc(x), y_desc(y);
        check(ffdas_greens_sum(
            handle.h,
            reinterpret_cast<const float3 *>(xpos.data()),
            wavenums.data(),
            x_desc.desc, 
            x.data(),
            reinterpret_cast<const float3 *>(ypos.data()),
            y_desc.desc, 
            y.data()
        ));
    }, "handle"_a, "x"_a, "xpos"_a, "ypos"_a, "wavenums"_a, "y"_a);
}
