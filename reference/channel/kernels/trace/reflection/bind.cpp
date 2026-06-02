#include "drjit_common.h"
#include <trace/reflection/bind.h>

#include <trace/reflection/reflection_types.h>
#include <trace/reflection/reflection_accumulate.h>
#include <trace/reflection/reflection_prefix_filter.h>
#include <trace/reflection/reflection_jvp.h>

// ---------------------------------------------------------------------------
// DRJIT_STRUCT types for reflection accumulation
// ---------------------------------------------------------------------------

struct ReflectionOpArrays {
    DiffFloat image_source_x;
    DiffFloat image_source_y;
    DiffFloat image_source_z;
    DiffFloat slot_plane_point_x;
    DiffFloat slot_plane_point_y;
    DiffFloat slot_plane_point_z;
    DiffFloat slot_plane_normal_x;
    DiffFloat slot_plane_normal_y;
    DiffFloat slot_plane_normal_z;
    Float slot_eta_r;
    Float slot_sigma;
    Float slot_gain;
    DiffFloat rx_x;
    DiffFloat rx_y;
    DiffFloat rx_z;

    DRJIT_STRUCT(
        ReflectionOpArrays,
        image_source_x,
        image_source_y,
        image_source_z,
        slot_plane_point_x,
        slot_plane_point_y,
        slot_plane_point_z,
        slot_plane_normal_x,
        slot_plane_normal_y,
        slot_plane_normal_z,
        slot_eta_r,
        slot_sigma,
        slot_gain,
        rx_x,
        rx_y,
        rx_z
    );
};

struct ReflectionOpInput {
    Int32 path_idx;
    Int32 rx_idx;
    Int32 valid_mask;
    ReflectionOpArrays arrays;
    float tx_pol_x;
    float tx_pol_y;
    float tx_pol_z;
    int n_pairs;
    int n_paths;
    int chain_depth;
    float k;
    float omega;

    DRJIT_STRUCT(
        ReflectionOpInput,
        path_idx,
        rx_idx,
        valid_mask,
        arrays,
        tx_pol_x,
        tx_pol_y,
        tx_pol_z,
        n_pairs,
        n_paths,
        chain_depth,
        k,
        omega
    );
};

struct ReflectionOpOutput {
    DiffFloat vec_x_re;
    DiffFloat vec_x_im;
    DiffFloat vec_y_re;
    DiffFloat vec_y_im;
    DiffFloat vec_z_re;
    DiffFloat vec_z_im;

    DRJIT_STRUCT(
        ReflectionOpOutput,
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im
    );
};

// ---------------------------------------------------------------------------
// Zero / grad helpers
// ---------------------------------------------------------------------------

inline ReflectionOpOutput zero_reflection_output(size_t width) {
    return {
        drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width),
        drjit::zeros<DiffFloat>(width),
    };
}

inline drjit::detached_t<ReflectionOpOutput> zero_reflection_output_grad(size_t width) {
    return {
        drjit::zeros<Float>(width),
        drjit::zeros<Float>(width),
        drjit::zeros<Float>(width),
        drjit::zeros<Float>(width),
        drjit::zeros<Float>(width),
        drjit::zeros<Float>(width),
    };
}

inline void set_reflection_output_grad(
    ReflectionOpOutput &registered_output,
    const drjit::detached_t<ReflectionOpOutput> &grad_output
) {
    drjit::set_grad(registered_output.vec_x_re, grad_output.vec_x_re);
    drjit::set_grad(registered_output.vec_x_im, grad_output.vec_x_im);
    drjit::set_grad(registered_output.vec_y_re, grad_output.vec_y_re);
    drjit::set_grad(registered_output.vec_y_im, grad_output.vec_y_im);
    drjit::set_grad(registered_output.vec_z_re, grad_output.vec_z_re);
    drjit::set_grad(registered_output.vec_z_im, grad_output.vec_z_im);
}

// ---------------------------------------------------------------------------
// CUDA kernel launchers (forward / JVP)
// ---------------------------------------------------------------------------

inline void launch_reflection_forward(
    const drjit::detached_t<ReflectionOpInput> &input,
    ReflectionOpOutput &output
) {
    drjit::eval(
        input.path_idx,
        input.rx_idx,
        input.valid_mask,
        input.arrays.image_source_x,
        input.arrays.image_source_y,
        input.arrays.image_source_z,
        input.arrays.slot_plane_point_x,
        input.arrays.slot_plane_point_y,
        input.arrays.slot_plane_point_z,
        input.arrays.slot_plane_normal_x,
        input.arrays.slot_plane_normal_y,
        input.arrays.slot_plane_normal_z
    );
    drjit::eval(
        input.arrays.slot_eta_r,
        input.arrays.slot_sigma,
        input.arrays.slot_gain,
        input.arrays.rx_x,
        input.arrays.rx_y,
        input.arrays.rx_z,
        output.vec_x_re,
        output.vec_x_im,
        output.vec_y_re,
        output.vec_y_im,
        output.vec_z_re,
        output.vec_z_im
    );

    witwin::channel::native_ext::reflection_accumulate_forward(
        drjit_data_ptr(input.path_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.valid_mask),
        drjit_data_ptr(input.arrays.image_source_x),
        drjit_data_ptr(input.arrays.image_source_y),
        drjit_data_ptr(input.arrays.image_source_z),
        drjit_data_ptr(input.arrays.slot_plane_point_x),
        drjit_data_ptr(input.arrays.slot_plane_point_y),
        drjit_data_ptr(input.arrays.slot_plane_point_z),
        drjit_data_ptr(input.arrays.slot_plane_normal_x),
        drjit_data_ptr(input.arrays.slot_plane_normal_y),
        drjit_data_ptr(input.arrays.slot_plane_normal_z),
        drjit_data_ptr(input.arrays.slot_eta_r),
        drjit_data_ptr(input.arrays.slot_sigma),
        drjit_data_ptr(input.arrays.slot_gain),
        drjit_data_ptr(input.arrays.rx_x),
        drjit_data_ptr(input.arrays.rx_y),
        drjit_data_ptr(input.arrays.rx_z),
        input.tx_pol_x,
        input.tx_pol_y,
        input.tx_pol_z,
        drjit_data_ptr_mut(output.vec_x_re),
        drjit_data_ptr_mut(output.vec_x_im),
        drjit_data_ptr_mut(output.vec_y_re),
        drjit_data_ptr_mut(output.vec_y_im),
        drjit_data_ptr_mut(output.vec_z_re),
        drjit_data_ptr_mut(output.vec_z_im),
        input.n_pairs,
        input.n_paths,
        input.chain_depth,
        input.k,
        input.omega
    );
}

inline void launch_reflection_jvp(
    const drjit::detached_t<ReflectionOpInput> &input,
    const Float &t_image_source_x,
    const Float &t_image_source_y,
    const Float &t_image_source_z,
    const Float &t_slot_plane_point_x,
    const Float &t_slot_plane_point_y,
    const Float &t_slot_plane_point_z,
    const Float &t_slot_plane_normal_x,
    const Float &t_slot_plane_normal_y,
    const Float &t_slot_plane_normal_z,
    const Float &t_rx_x,
    const Float &t_rx_y,
    const Float &t_rx_z,
    drjit::detached_t<ReflectionOpOutput> &output
) {
    drjit::eval(
        input.path_idx,
        input.rx_idx,
        input.valid_mask,
        input.arrays.image_source_x,
        input.arrays.image_source_y,
        input.arrays.image_source_z,
        input.arrays.slot_plane_point_x,
        input.arrays.slot_plane_point_y,
        input.arrays.slot_plane_point_z,
        input.arrays.slot_plane_normal_x,
        input.arrays.slot_plane_normal_y,
        input.arrays.slot_plane_normal_z
    );
    drjit::eval(
        input.arrays.slot_eta_r,
        input.arrays.slot_sigma,
        input.arrays.slot_gain,
        input.arrays.rx_x,
        input.arrays.rx_y,
        input.arrays.rx_z
    );
    drjit::eval(
        t_image_source_x,
        t_image_source_y,
        t_image_source_z,
        t_slot_plane_point_x,
        t_slot_plane_point_y,
        t_slot_plane_point_z,
        t_slot_plane_normal_x,
        t_slot_plane_normal_y,
        t_slot_plane_normal_z,
        t_rx_x,
        t_rx_y,
        t_rx_z
    );
    drjit::eval(
        output.vec_x_re,
        output.vec_x_im,
        output.vec_y_re,
        output.vec_y_im,
        output.vec_z_re,
        output.vec_z_im
    );

    witwin::channel::native_ext::reflection_accumulate_jvp(
        drjit_data_ptr(input.path_idx),
        drjit_data_ptr(input.rx_idx),
        drjit_data_ptr(input.valid_mask),
        drjit_data_ptr(input.arrays.image_source_x),
        drjit_data_ptr(input.arrays.image_source_y),
        drjit_data_ptr(input.arrays.image_source_z),
        drjit_data_ptr(input.arrays.slot_plane_point_x),
        drjit_data_ptr(input.arrays.slot_plane_point_y),
        drjit_data_ptr(input.arrays.slot_plane_point_z),
        drjit_data_ptr(input.arrays.slot_plane_normal_x),
        drjit_data_ptr(input.arrays.slot_plane_normal_y),
        drjit_data_ptr(input.arrays.slot_plane_normal_z),
        drjit_data_ptr(input.arrays.slot_eta_r),
        drjit_data_ptr(input.arrays.slot_sigma),
        drjit_data_ptr(input.arrays.slot_gain),
        drjit_data_ptr(input.arrays.rx_x),
        drjit_data_ptr(input.arrays.rx_y),
        drjit_data_ptr(input.arrays.rx_z),
        input.tx_pol_x,
        input.tx_pol_y,
        input.tx_pol_z,
        drjit_data_ptr(t_image_source_x),
        drjit_data_ptr(t_image_source_y),
        drjit_data_ptr(t_image_source_z),
        drjit_data_ptr(t_slot_plane_point_x),
        drjit_data_ptr(t_slot_plane_point_y),
        drjit_data_ptr(t_slot_plane_point_z),
        drjit_data_ptr(t_slot_plane_normal_x),
        drjit_data_ptr(t_slot_plane_normal_y),
        drjit_data_ptr(t_slot_plane_normal_z),
        drjit_data_ptr(t_rx_x),
        drjit_data_ptr(t_rx_y),
        drjit_data_ptr(t_rx_z),
        drjit_data_ptr_mut(output.vec_x_re),
        drjit_data_ptr_mut(output.vec_x_im),
        drjit_data_ptr_mut(output.vec_y_re),
        drjit_data_ptr_mut(output.vec_y_im),
        drjit_data_ptr_mut(output.vec_z_re),
        drjit_data_ptr_mut(output.vec_z_im),
        input.n_pairs,
        input.n_paths,
        input.chain_depth,
        input.k,
        input.omega
    );
}

// ---------------------------------------------------------------------------
// ReflectionAccumulateOp â€?DrJit custom AD operation
// ---------------------------------------------------------------------------

class ReflectionAccumulateOp
    : public WitwinCustomOp<ReflectionOpOutput, ReflectionOpInput> {
public:
    using Base = WitwinCustomOp<ReflectionOpOutput, ReflectionOpInput>;
    using OutputType = typename Base::OutputType;

    explicit ReflectionAccumulateOp(const ReflectionOpInput &input)
        : Base(input) {}

    OutputType eval(drjit::detached_t<ReflectionOpInput> input) {
        m_input = input;
        OutputType output = zero_reflection_output(input.arrays.rx_x.size());
        if (input.n_pairs > 0) {
            launch_reflection_forward(input, output);
        }
        return output;
    }

    void forward() override {
        auto output = zero_reflection_output_grad(m_input.arrays.rx_x.size());
        if (m_input.n_pairs > 0) {
            auto coerce = [](const Float &value, size_t width) -> Float {
                return value.size() == width ? value : drjit::zeros<Float>(width);
            };
            launch_reflection_jvp(
                m_input,
                coerce(drjit::grad<false>(this->m_registered_input.arrays.image_source_x), m_input.arrays.image_source_x.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.image_source_y), m_input.arrays.image_source_y.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.image_source_z), m_input.arrays.image_source_z.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.slot_plane_point_x), m_input.arrays.slot_plane_point_x.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.slot_plane_point_y), m_input.arrays.slot_plane_point_y.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.slot_plane_point_z), m_input.arrays.slot_plane_point_z.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.slot_plane_normal_x), m_input.arrays.slot_plane_normal_x.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.slot_plane_normal_y), m_input.arrays.slot_plane_normal_y.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.slot_plane_normal_z), m_input.arrays.slot_plane_normal_z.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.rx_x), m_input.arrays.rx_x.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.rx_y), m_input.arrays.rx_y.size()),
                coerce(drjit::grad<false>(this->m_registered_input.arrays.rx_z), m_input.arrays.rx_z.size()),
                output
            );
        }
        set_reflection_output_grad(this->m_registered_output, output);
    }

    const char *name() const override { return "ReflectionAccumulate"; }

private:
    drjit::detached_t<ReflectionOpInput> m_input;
};

// ---------------------------------------------------------------------------
// register_reflection_bindings â€?raw pointer + DrJit AD bindings
// ---------------------------------------------------------------------------

void register_reflection_bindings(nb::module_ &m) {
    m.attr("REFL_MAX_CHAIN_DEPTH") = witwin::channel::native_ext::REFL_MAX_CHAIN_DEPTH;

    m.def(
        "reflection_epc_targets_forward_arrays",
        [](
            Int32 path_idx,
            DiffFloat image_source_x,
            DiffFloat image_source_y,
            DiffFloat image_source_z,
            DiffFloat slot_plane_point_x,
            DiffFloat slot_plane_point_y,
            DiffFloat slot_plane_point_z,
            DiffFloat slot_plane_normal_x,
            DiffFloat slot_plane_normal_y,
            DiffFloat slot_plane_normal_z,
            Float slot_eta_r,
            Float slot_sigma,
            Float slot_gain,
            DiffFloat target_x,
            DiffFloat target_y,
            DiffFloat target_z,
            float tx_pol_x,
            float tx_pol_y,
            float tx_pol_z,
            int n_pairs,
            int n_paths,
            int chain_depth,
            float k,
            float omega
        ) {
            size_t pair_count = static_cast<size_t>(n_pairs);
            size_t hit_count = static_cast<size_t>(n_pairs) * static_cast<size_t>(chain_depth);

            DiffInt32 geom_valid = drjit::zeros<DiffInt32>(pair_count);
            DiffFloat tx_pos_x = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat tx_pos_y = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat tx_pos_z = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat vec_x_re = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat vec_x_im = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat vec_y_re = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat vec_y_im = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat vec_z_re = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat vec_z_im = drjit::zeros<DiffFloat>(pair_count);
            DiffFloat hit_x = drjit::zeros<DiffFloat>(hit_count);
            DiffFloat hit_y = drjit::zeros<DiffFloat>(hit_count);
            DiffFloat hit_z = drjit::zeros<DiffFloat>(hit_count);

            drjit::eval(
                path_idx,
                image_source_x, image_source_y, image_source_z,
                slot_plane_point_x, slot_plane_point_y, slot_plane_point_z,
                slot_plane_normal_x, slot_plane_normal_y, slot_plane_normal_z,
                slot_eta_r, slot_sigma, slot_gain,
                target_x, target_y, target_z
            );
            drjit::eval(
                geom_valid,
                tx_pos_x, tx_pos_y, tx_pos_z,
                vec_x_re, vec_x_im,
                vec_y_re, vec_y_im,
                vec_z_re, vec_z_im,
                hit_x, hit_y, hit_z
            );

            witwin::channel::native_ext::reflection_epc_targets_forward(
                drjit_data_ptr(path_idx),
                drjit_data_ptr(image_source_x),
                drjit_data_ptr(image_source_y),
                drjit_data_ptr(image_source_z),
                drjit_data_ptr(slot_plane_point_x),
                drjit_data_ptr(slot_plane_point_y),
                drjit_data_ptr(slot_plane_point_z),
                drjit_data_ptr(slot_plane_normal_x),
                drjit_data_ptr(slot_plane_normal_y),
                drjit_data_ptr(slot_plane_normal_z),
                drjit_data_ptr(slot_eta_r),
                drjit_data_ptr(slot_sigma),
                drjit_data_ptr(slot_gain),
                drjit_data_ptr(target_x),
                drjit_data_ptr(target_y),
                drjit_data_ptr(target_z),
                tx_pol_x,
                tx_pol_y,
                tx_pol_z,
                drjit_data_ptr_mut(geom_valid),
                drjit_data_ptr_mut(tx_pos_x),
                drjit_data_ptr_mut(tx_pos_y),
                drjit_data_ptr_mut(tx_pos_z),
                drjit_data_ptr_mut(vec_x_re),
                drjit_data_ptr_mut(vec_x_im),
                drjit_data_ptr_mut(vec_y_re),
                drjit_data_ptr_mut(vec_y_im),
                drjit_data_ptr_mut(vec_z_re),
                drjit_data_ptr_mut(vec_z_im),
                drjit_data_ptr_mut(hit_x),
                drjit_data_ptr_mut(hit_y),
                drjit_data_ptr_mut(hit_z),
                n_pairs,
                n_paths,
                chain_depth,
                k,
                omega
            );

            return nb::make_tuple(
                geom_valid,
                tx_pos_x,
                tx_pos_y,
                tx_pos_z,
                vec_x_re,
                vec_x_im,
                vec_y_re,
                vec_y_im,
                vec_z_re,
                vec_z_im,
                hit_x,
                hit_y,
                hit_z
            );
        },
        "Run exact path calculation (EPC) for reflection chains to arbitrary targets and return batch geometry plus Jones-chain vectors."
    );

    m.def(
        "reflection_prefix_filter_arrays",
        [](
            Int32 has_reflected_support,
            Float source_x,
            Float source_y,
            Float source_z,
            Float edge_pos_x,
            Float edge_pos_y,
            Float edge_pos_z,
            Float edge_dir_x,
            Float edge_dir_y,
            Float edge_dir_z,
            Float n0_x,
            Float n0_y,
            Float n0_z,
            Float nn_x,
            Float nn_y,
            Float nn_z,
            DiffFloat vec_x_re,
            DiffFloat vec_x_im,
            DiffFloat vec_y_re,
            DiffFloat vec_y_im,
            DiffFloat vec_z_re,
            DiffFloat vec_z_im,
            float wavelength,
            float field_power_threshold
        ) {
            size_t pair_count = source_x.size();
            Int32 support_mask = drjit::zeros<Int32>(pair_count);
            Int32 keep_mask = drjit::zeros<Int32>(pair_count);

            drjit::eval(
                has_reflected_support,
                source_x,
                source_y,
                source_z,
                edge_pos_x,
                edge_pos_y,
                edge_pos_z,
                edge_dir_x,
                edge_dir_y,
                edge_dir_z,
                n0_x,
                n0_y,
                n0_z,
                nn_x,
                nn_y,
                nn_z,
                vec_x_re,
                vec_x_im,
                vec_y_re,
                vec_y_im,
                vec_z_re,
                vec_z_im,
                support_mask,
                keep_mask
            );

            witwin::channel::native_ext::reflection_prefix_filter(
                drjit_data_ptr(has_reflected_support),
                drjit_data_ptr(source_x),
                drjit_data_ptr(source_y),
                drjit_data_ptr(source_z),
                drjit_data_ptr(edge_pos_x),
                drjit_data_ptr(edge_pos_y),
                drjit_data_ptr(edge_pos_z),
                drjit_data_ptr(edge_dir_x),
                drjit_data_ptr(edge_dir_y),
                drjit_data_ptr(edge_dir_z),
                drjit_data_ptr(n0_x),
                drjit_data_ptr(n0_y),
                drjit_data_ptr(n0_z),
                drjit_data_ptr(nn_x),
                drjit_data_ptr(nn_y),
                drjit_data_ptr(nn_z),
                drjit_data_ptr(vec_x_re),
                drjit_data_ptr(vec_x_im),
                drjit_data_ptr(vec_y_re),
                drjit_data_ptr(vec_y_im),
                drjit_data_ptr(vec_z_re),
                drjit_data_ptr(vec_z_im),
                wavelength,
                field_power_threshold,
                static_cast<int>(pair_count),
                drjit_data_ptr_mut(support_mask),
                drjit_data_ptr_mut(keep_mask)
            );

            return nb::make_tuple(support_mask, keep_mask);
        },
        "Fused support and field-power filtering for reflection-prefix first-order diffraction pairs."
    );

    // Reflection accumulation with C++-side Dr.Jit custom AD support
    m.def(
        "reflection_accumulate",
        [](
            Int32 path_idx,
            Int32 rx_idx,
            Int32 valid_mask,
            DiffFloat image_source_x,
            DiffFloat image_source_y,
            DiffFloat image_source_z,
            DiffFloat slot_plane_point_x,
            DiffFloat slot_plane_point_y,
            DiffFloat slot_plane_point_z,
            DiffFloat slot_plane_normal_x,
            DiffFloat slot_plane_normal_y,
            DiffFloat slot_plane_normal_z,
            Float slot_eta_r,
            Float slot_sigma,
            Float slot_gain,
            DiffFloat rx_x,
            DiffFloat rx_y,
            DiffFloat rx_z,
            float tx_pol_x,
            float tx_pol_y,
            float tx_pol_z,
            int n_pairs,
            int n_paths,
            int chain_depth,
            float k,
            float omega
        ) {
            ReflectionOpInput input{
                path_idx,
                rx_idx,
                valid_mask,
                {
                    image_source_x,
                    image_source_y,
                    image_source_z,
                    slot_plane_point_x,
                    slot_plane_point_y,
                    slot_plane_point_z,
                    slot_plane_normal_x,
                    slot_plane_normal_y,
                    slot_plane_normal_z,
                    slot_eta_r,
                    slot_sigma,
                    slot_gain,
                    rx_x,
                    rx_y,
                    rx_z,
                },
                tx_pol_x,
                tx_pol_y,
                tx_pol_z,
                n_pairs,
                n_paths,
                chain_depth,
                k,
                omega,
            };
            ReflectionOpOutput output = witwin_custom_op<ReflectionAccumulateOp>(input);
            return nb::make_tuple(
                output.vec_x_re,
                output.vec_x_im,
                output.vec_y_re,
                output.vec_y_im,
                output.vec_z_re,
                output.vec_z_im
            );
        },
        "Reflection accumulation with C++-side Dr.Jit custom AD support."
    );
}
