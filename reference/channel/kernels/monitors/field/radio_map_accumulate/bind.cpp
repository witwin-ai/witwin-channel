#include "drjit_common.h"
#include <monitors/field/radio_map_accumulate/bind.h>

#include <monitors/field/radio_map_accumulate/radio_map_accumulate.h>

namespace {

template <typename Array>
Array zeros_array(int width) {
    Array value = drjit::zeros<Array>(static_cast<size_t>(width));
    drjit::eval(value);
    return value;
}

} // namespace

void register_radio_map_accumulate_bindings(nb::module_ &m) {
    m.def(
        "radiomap_accumulate_vector_power_pairs_into",
        [](
            nb::handle rx_idx_value,
            nb::tuple pair_vector_arrays,
            nb::tuple arrival_arrays,
            nb::tuple output_arrays,
            float rx_pol_x,
            float rx_pol_y,
            float rx_pol_z,
            int n_pairs
        ) {
            if (nb::len(pair_vector_arrays) != 6) {
                throw std::runtime_error(
                    "radiomap_accumulate_vector_power_pairs_into expected 6 vector arrays"
                );
            }
            if (nb::len(arrival_arrays) != 3) {
                throw std::runtime_error(
                    "radiomap_accumulate_vector_power_pairs_into expected 3 arrival arrays"
                );
            }
            if (nb::len(output_arrays) != 10) {
                throw std::runtime_error(
                    "radiomap_accumulate_vector_power_pairs_into expected 10 output arrays"
                );
            }

            witwin::channel::native_ext::radiomap_accumulate_vector_power_forward(
                ptr<int>(drjit_data_ptr_handle(rx_idx_value)),
                ptr<float>(drjit_data_ptr_handle(pair_vector_arrays[0])),
                ptr<float>(drjit_data_ptr_handle(pair_vector_arrays[1])),
                ptr<float>(drjit_data_ptr_handle(pair_vector_arrays[2])),
                ptr<float>(drjit_data_ptr_handle(pair_vector_arrays[3])),
                ptr<float>(drjit_data_ptr_handle(pair_vector_arrays[4])),
                ptr<float>(drjit_data_ptr_handle(pair_vector_arrays[5])),
                ptr<float>(drjit_data_ptr_handle(arrival_arrays[0])),
                ptr<float>(drjit_data_ptr_handle(arrival_arrays[1])),
                ptr<float>(drjit_data_ptr_handle(arrival_arrays[2])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[0])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[1])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[2])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[3])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[4])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[5])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[6])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[7])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[8])),
                ptr_mut<float>(drjit_data_ptr_handle(output_arrays[9])),
                n_pairs,
                rx_pol_x,
                rx_pol_y,
                rx_pol_z
            );
        },
        "Accumulate radiomap diffraction pair vectors into dense receiver outputs."
    );
    m.def(
        "radiomap_accumulate_vector_power_pairs",
        [](
            const Int32 &rx_idx_value,
            const Float &pair_vec_x_re,
            const Float &pair_vec_x_im,
            const Float &pair_vec_y_re,
            const Float &pair_vec_y_im,
            const Float &pair_vec_z_re,
            const Float &pair_vec_z_im,
            const Float &arrival_x,
            const Float &arrival_y,
            const Float &arrival_z,
            int n_output_rx,
            int n_pairs,
            float rx_pol_x,
            float rx_pol_y,
            float rx_pol_z
        ) {
            Float coherent_re = zeros_array<Float>(n_output_rx);
            Float coherent_im = zeros_array<Float>(n_output_rx);
            Float power = zeros_array<Float>(n_output_rx);
            Float vector_x_re = zeros_array<Float>(n_output_rx);
            Float vector_x_im = zeros_array<Float>(n_output_rx);
            Float vector_y_re = zeros_array<Float>(n_output_rx);
            Float vector_y_im = zeros_array<Float>(n_output_rx);
            Float vector_z_re = zeros_array<Float>(n_output_rx);
            Float vector_z_im = zeros_array<Float>(n_output_rx);
            Float valid_pair_count = zeros_array<Float>(1);

            witwin::channel::native_ext::radiomap_accumulate_vector_power_forward(
                drjit_data_ptr(rx_idx_value),
                drjit_data_ptr(pair_vec_x_re),
                drjit_data_ptr(pair_vec_x_im),
                drjit_data_ptr(pair_vec_y_re),
                drjit_data_ptr(pair_vec_y_im),
                drjit_data_ptr(pair_vec_z_re),
                drjit_data_ptr(pair_vec_z_im),
                drjit_data_ptr(arrival_x),
                drjit_data_ptr(arrival_y),
                drjit_data_ptr(arrival_z),
                drjit_data_ptr_mut(coherent_re),
                drjit_data_ptr_mut(coherent_im),
                drjit_data_ptr_mut(power),
                drjit_data_ptr_mut(vector_x_re),
                drjit_data_ptr_mut(vector_x_im),
                drjit_data_ptr_mut(vector_y_re),
                drjit_data_ptr_mut(vector_y_im),
                drjit_data_ptr_mut(vector_z_re),
                drjit_data_ptr_mut(vector_z_im),
                drjit_data_ptr_mut(valid_pair_count),
                n_pairs,
                rx_pol_x,
                rx_pol_y,
                rx_pol_z
            );

            return nb::make_tuple(
                coherent_re,
                coherent_im,
                power,
                vector_x_re,
                vector_x_im,
                vector_y_re,
                vector_y_im,
                vector_z_re,
                vector_z_im,
                valid_pair_count
            );
        },
        "Accumulate radiomap diffraction pair vectors and return dense receiver outputs."
    );
    m.def(
        "radiomap_vector_power_forward_raw",
        [](
            const DiffFloat &vec_x_re,
            const DiffFloat &vec_x_im,
            const DiffFloat &vec_y_re,
            const DiffFloat &vec_y_im,
            const DiffFloat &vec_z_re,
            const DiffFloat &vec_z_im,
            int n_rx
        ) {
            DiffFloat power = zeros_array<DiffFloat>(n_rx);
            witwin::channel::native_ext::radiomap_vector_power_forward(
                drjit_data_ptr(vec_x_re),
                drjit_data_ptr(vec_x_im),
                drjit_data_ptr(vec_y_re),
                drjit_data_ptr(vec_y_im),
                drjit_data_ptr(vec_z_re),
                drjit_data_ptr(vec_z_im),
                drjit_data_ptr_mut(power),
                n_rx
            );
            return power;
        },
        "Compute dense matched-isotropic vector power."
    );
    m.def(
        "radiomap_vector_power_jvp_raw",
        [](
            const DiffFloat &vec_x_re,
            const DiffFloat &vec_x_im,
            const DiffFloat &vec_y_re,
            const DiffFloat &vec_y_im,
            const DiffFloat &vec_z_re,
            const DiffFloat &vec_z_im,
            const DiffFloat &t_vec_x_re,
            const DiffFloat &t_vec_x_im,
            const DiffFloat &t_vec_y_re,
            const DiffFloat &t_vec_y_im,
            const DiffFloat &t_vec_z_re,
            const DiffFloat &t_vec_z_im,
            int n_rx
        ) {
            DiffFloat t_power = zeros_array<DiffFloat>(n_rx);
            witwin::channel::native_ext::radiomap_vector_power_jvp(
                drjit_data_ptr(vec_x_re),
                drjit_data_ptr(vec_x_im),
                drjit_data_ptr(vec_y_re),
                drjit_data_ptr(vec_y_im),
                drjit_data_ptr(vec_z_re),
                drjit_data_ptr(vec_z_im),
                drjit_data_ptr(t_vec_x_re),
                drjit_data_ptr(t_vec_x_im),
                drjit_data_ptr(t_vec_y_re),
                drjit_data_ptr(t_vec_y_im),
                drjit_data_ptr(t_vec_z_re),
                drjit_data_ptr(t_vec_z_im),
                drjit_data_ptr_mut(t_power),
                n_rx
            );
            return t_power;
        },
        "Compute the JVP of dense matched-isotropic vector power."
    );
    m.def(
        "radiomap_vector_power_backward_raw",
        [](
            const DiffFloat &vec_x_re,
            const DiffFloat &vec_x_im,
            const DiffFloat &vec_y_re,
            const DiffFloat &vec_y_im,
            const DiffFloat &vec_z_re,
            const DiffFloat &vec_z_im,
            const DiffFloat &grad_power,
            int n_rx
        ) {
            DiffFloat grad_vec_x_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_vec_x_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_vec_y_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_vec_y_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_vec_z_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_vec_z_im = zeros_array<DiffFloat>(n_rx);
            witwin::channel::native_ext::radiomap_vector_power_backward(
                drjit_data_ptr(vec_x_re),
                drjit_data_ptr(vec_x_im),
                drjit_data_ptr(vec_y_re),
                drjit_data_ptr(vec_y_im),
                drjit_data_ptr(vec_z_re),
                drjit_data_ptr(vec_z_im),
                drjit_data_ptr(grad_power),
                drjit_data_ptr_mut(grad_vec_x_re),
                drjit_data_ptr_mut(grad_vec_x_im),
                drjit_data_ptr_mut(grad_vec_y_re),
                drjit_data_ptr_mut(grad_vec_y_im),
                drjit_data_ptr_mut(grad_vec_z_re),
                drjit_data_ptr_mut(grad_vec_z_im),
                n_rx
            );
            return nb::make_tuple(
                grad_vec_x_re,
                grad_vec_x_im,
                grad_vec_y_re,
                grad_vec_y_im,
                grad_vec_z_re,
                grad_vec_z_im
            );
        },
        "Compute the VJP of dense matched-isotropic vector power."
    );
    m.def(
        "radiomap_matched_isb_completion_forward_raw",
        [](
            const DiffFloat &continued_direct_re,
            const DiffFloat &continued_direct_im,
            const DiffFloat &tx_basis_x,
            const DiffFloat &tx_basis_y,
            const DiffFloat &tx_basis_z,
            const DiffFloat &rx_basis_x,
            const DiffFloat &rx_basis_y,
            const DiffFloat &rx_basis_z,
            const DiffFloat &hard_visibility,
            const Int32 &interior_mask,
            const DiffFloat &incident_weight,
            const DiffFloat &incident_response_re,
            const DiffFloat &incident_response_im,
            const DiffFloat &raw_vec_x_re,
            const DiffFloat &raw_vec_x_im,
            const DiffFloat &raw_vec_y_re,
            const DiffFloat &raw_vec_y_im,
            const DiffFloat &raw_vec_z_re,
            const DiffFloat &raw_vec_z_im,
            int n_rx
        ) {
            DiffFloat coherent_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat coherent_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat power = zeros_array<DiffFloat>(n_rx);
            DiffFloat vector_x_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat vector_x_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat vector_y_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat vector_y_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat vector_z_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat vector_z_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat continued_direct_power = zeros_array<DiffFloat>(n_rx);
            DiffFloat transition_magnitude = zeros_array<DiffFloat>(n_rx);
            DiffFloat transition_phase = zeros_array<DiffFloat>(n_rx);
            witwin::channel::native_ext::radiomap_matched_isb_completion_forward(
                drjit_data_ptr(continued_direct_re),
                drjit_data_ptr(continued_direct_im),
                drjit_data_ptr(tx_basis_x),
                drjit_data_ptr(tx_basis_y),
                drjit_data_ptr(tx_basis_z),
                drjit_data_ptr(rx_basis_x),
                drjit_data_ptr(rx_basis_y),
                drjit_data_ptr(rx_basis_z),
                drjit_data_ptr(hard_visibility),
                drjit_data_ptr(interior_mask),
                drjit_data_ptr(incident_weight),
                drjit_data_ptr(incident_response_re),
                drjit_data_ptr(incident_response_im),
                drjit_data_ptr(raw_vec_x_re),
                drjit_data_ptr(raw_vec_x_im),
                drjit_data_ptr(raw_vec_y_re),
                drjit_data_ptr(raw_vec_y_im),
                drjit_data_ptr(raw_vec_z_re),
                drjit_data_ptr(raw_vec_z_im),
                drjit_data_ptr_mut(coherent_re),
                drjit_data_ptr_mut(coherent_im),
                drjit_data_ptr_mut(power),
                drjit_data_ptr_mut(vector_x_re),
                drjit_data_ptr_mut(vector_x_im),
                drjit_data_ptr_mut(vector_y_re),
                drjit_data_ptr_mut(vector_y_im),
                drjit_data_ptr_mut(vector_z_re),
                drjit_data_ptr_mut(vector_z_im),
                drjit_data_ptr_mut(continued_direct_power),
                drjit_data_ptr_mut(transition_magnitude),
                drjit_data_ptr_mut(transition_phase),
                n_rx
            );
            return nb::make_tuple(
                coherent_re,
                coherent_im,
                power,
                vector_x_re,
                vector_x_im,
                vector_y_re,
                vector_y_im,
                vector_z_re,
                vector_z_im,
                continued_direct_power,
                transition_magnitude,
                transition_phase
            );
        },
        "Compute matched ISB completion outputs."
    );
    m.def(
        "radiomap_matched_isb_completion_jvp_raw",
        [](
            const DiffFloat &continued_direct_re,
            const DiffFloat &continued_direct_im,
            const DiffFloat &tx_basis_x,
            const DiffFloat &tx_basis_y,
            const DiffFloat &tx_basis_z,
            const DiffFloat &rx_basis_x,
            const DiffFloat &rx_basis_y,
            const DiffFloat &rx_basis_z,
            const DiffFloat &hard_visibility,
            const Int32 &interior_mask,
            const DiffFloat &incident_weight,
            const DiffFloat &incident_response_re,
            const DiffFloat &incident_response_im,
            const DiffFloat &raw_vec_x_re,
            const DiffFloat &raw_vec_x_im,
            const DiffFloat &raw_vec_y_re,
            const DiffFloat &raw_vec_y_im,
            const DiffFloat &raw_vec_z_re,
            const DiffFloat &raw_vec_z_im,
            const DiffFloat &t_continued_direct_re,
            const DiffFloat &t_continued_direct_im,
            const DiffFloat &t_tx_basis_x,
            const DiffFloat &t_tx_basis_y,
            const DiffFloat &t_tx_basis_z,
            const DiffFloat &t_rx_basis_x,
            const DiffFloat &t_rx_basis_y,
            const DiffFloat &t_rx_basis_z,
            const DiffFloat &t_incident_weight,
            const DiffFloat &t_incident_response_re,
            const DiffFloat &t_incident_response_im,
            const DiffFloat &t_raw_vec_x_re,
            const DiffFloat &t_raw_vec_x_im,
            const DiffFloat &t_raw_vec_y_re,
            const DiffFloat &t_raw_vec_y_im,
            const DiffFloat &t_raw_vec_z_re,
            const DiffFloat &t_raw_vec_z_im,
            int n_rx
        ) {
            DiffFloat t_coherent_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_coherent_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_power = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_vector_x_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_vector_x_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_vector_y_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_vector_y_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_vector_z_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_vector_z_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_continued_direct_power = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_transition_magnitude = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_transition_phase = zeros_array<DiffFloat>(n_rx);
            witwin::channel::native_ext::radiomap_matched_isb_completion_jvp(
                drjit_data_ptr(continued_direct_re),
                drjit_data_ptr(continued_direct_im),
                drjit_data_ptr(tx_basis_x),
                drjit_data_ptr(tx_basis_y),
                drjit_data_ptr(tx_basis_z),
                drjit_data_ptr(rx_basis_x),
                drjit_data_ptr(rx_basis_y),
                drjit_data_ptr(rx_basis_z),
                drjit_data_ptr(hard_visibility),
                drjit_data_ptr(interior_mask),
                drjit_data_ptr(incident_weight),
                drjit_data_ptr(incident_response_re),
                drjit_data_ptr(incident_response_im),
                drjit_data_ptr(raw_vec_x_re),
                drjit_data_ptr(raw_vec_x_im),
                drjit_data_ptr(raw_vec_y_re),
                drjit_data_ptr(raw_vec_y_im),
                drjit_data_ptr(raw_vec_z_re),
                drjit_data_ptr(raw_vec_z_im),
                drjit_data_ptr(t_continued_direct_re),
                drjit_data_ptr(t_continued_direct_im),
                drjit_data_ptr(t_tx_basis_x),
                drjit_data_ptr(t_tx_basis_y),
                drjit_data_ptr(t_tx_basis_z),
                drjit_data_ptr(t_rx_basis_x),
                drjit_data_ptr(t_rx_basis_y),
                drjit_data_ptr(t_rx_basis_z),
                drjit_data_ptr(t_incident_weight),
                drjit_data_ptr(t_incident_response_re),
                drjit_data_ptr(t_incident_response_im),
                drjit_data_ptr(t_raw_vec_x_re),
                drjit_data_ptr(t_raw_vec_x_im),
                drjit_data_ptr(t_raw_vec_y_re),
                drjit_data_ptr(t_raw_vec_y_im),
                drjit_data_ptr(t_raw_vec_z_re),
                drjit_data_ptr(t_raw_vec_z_im),
                drjit_data_ptr_mut(t_coherent_re),
                drjit_data_ptr_mut(t_coherent_im),
                drjit_data_ptr_mut(t_power),
                drjit_data_ptr_mut(t_vector_x_re),
                drjit_data_ptr_mut(t_vector_x_im),
                drjit_data_ptr_mut(t_vector_y_re),
                drjit_data_ptr_mut(t_vector_y_im),
                drjit_data_ptr_mut(t_vector_z_re),
                drjit_data_ptr_mut(t_vector_z_im),
                drjit_data_ptr_mut(t_continued_direct_power),
                drjit_data_ptr_mut(t_transition_magnitude),
                drjit_data_ptr_mut(t_transition_phase),
                n_rx
            );
            return nb::make_tuple(
                t_coherent_re,
                t_coherent_im,
                t_power,
                t_vector_x_re,
                t_vector_x_im,
                t_vector_y_re,
                t_vector_y_im,
                t_vector_z_re,
                t_vector_z_im,
                t_continued_direct_power,
                t_transition_magnitude,
                t_transition_phase
            );
        },
        "Compute the JVP of matched ISB completion outputs."
    );
    m.def(
        "radiomap_matched_isb_completion_backward_raw",
        [](
            const DiffFloat &continued_direct_re,
            const DiffFloat &continued_direct_im,
            const DiffFloat &tx_basis_x,
            const DiffFloat &tx_basis_y,
            const DiffFloat &tx_basis_z,
            const DiffFloat &rx_basis_x,
            const DiffFloat &rx_basis_y,
            const DiffFloat &rx_basis_z,
            const DiffFloat &hard_visibility,
            const Int32 &interior_mask,
            const DiffFloat &incident_weight,
            const DiffFloat &incident_response_re,
            const DiffFloat &incident_response_im,
            const DiffFloat &raw_vec_x_re,
            const DiffFloat &raw_vec_x_im,
            const DiffFloat &raw_vec_y_re,
            const DiffFloat &raw_vec_y_im,
            const DiffFloat &raw_vec_z_re,
            const DiffFloat &raw_vec_z_im,
            const DiffFloat &grad_coherent_re,
            const DiffFloat &grad_coherent_im,
            const DiffFloat &grad_power,
            const DiffFloat &grad_vector_x_re,
            const DiffFloat &grad_vector_x_im,
            const DiffFloat &grad_vector_y_re,
            const DiffFloat &grad_vector_y_im,
            const DiffFloat &grad_vector_z_re,
            const DiffFloat &grad_vector_z_im,
            const DiffFloat &grad_continued_direct_power,
            const DiffFloat &grad_transition_magnitude,
            const DiffFloat &grad_transition_phase,
            int n_rx
        ) {
            DiffFloat grad_continued_direct_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_continued_direct_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_tx_basis_x = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_tx_basis_y = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_tx_basis_z = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_rx_basis_x = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_rx_basis_y = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_rx_basis_z = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_incident_weight = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_incident_response_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_incident_response_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_raw_vec_x_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_raw_vec_x_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_raw_vec_y_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_raw_vec_y_im = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_raw_vec_z_re = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_raw_vec_z_im = zeros_array<DiffFloat>(n_rx);
            witwin::channel::native_ext::radiomap_matched_isb_completion_backward(
                drjit_data_ptr(continued_direct_re),
                drjit_data_ptr(continued_direct_im),
                drjit_data_ptr(tx_basis_x),
                drjit_data_ptr(tx_basis_y),
                drjit_data_ptr(tx_basis_z),
                drjit_data_ptr(rx_basis_x),
                drjit_data_ptr(rx_basis_y),
                drjit_data_ptr(rx_basis_z),
                drjit_data_ptr(hard_visibility),
                drjit_data_ptr(interior_mask),
                drjit_data_ptr(incident_weight),
                drjit_data_ptr(incident_response_re),
                drjit_data_ptr(incident_response_im),
                drjit_data_ptr(raw_vec_x_re),
                drjit_data_ptr(raw_vec_x_im),
                drjit_data_ptr(raw_vec_y_re),
                drjit_data_ptr(raw_vec_y_im),
                drjit_data_ptr(raw_vec_z_re),
                drjit_data_ptr(raw_vec_z_im),
                drjit_data_ptr(grad_coherent_re),
                drjit_data_ptr(grad_coherent_im),
                drjit_data_ptr(grad_power),
                drjit_data_ptr(grad_vector_x_re),
                drjit_data_ptr(grad_vector_x_im),
                drjit_data_ptr(grad_vector_y_re),
                drjit_data_ptr(grad_vector_y_im),
                drjit_data_ptr(grad_vector_z_re),
                drjit_data_ptr(grad_vector_z_im),
                drjit_data_ptr(grad_continued_direct_power),
                drjit_data_ptr(grad_transition_magnitude),
                drjit_data_ptr(grad_transition_phase),
                drjit_data_ptr_mut(grad_continued_direct_re),
                drjit_data_ptr_mut(grad_continued_direct_im),
                drjit_data_ptr_mut(grad_tx_basis_x),
                drjit_data_ptr_mut(grad_tx_basis_y),
                drjit_data_ptr_mut(grad_tx_basis_z),
                drjit_data_ptr_mut(grad_rx_basis_x),
                drjit_data_ptr_mut(grad_rx_basis_y),
                drjit_data_ptr_mut(grad_rx_basis_z),
                drjit_data_ptr_mut(grad_incident_weight),
                drjit_data_ptr_mut(grad_incident_response_re),
                drjit_data_ptr_mut(grad_incident_response_im),
                drjit_data_ptr_mut(grad_raw_vec_x_re),
                drjit_data_ptr_mut(grad_raw_vec_x_im),
                drjit_data_ptr_mut(grad_raw_vec_y_re),
                drjit_data_ptr_mut(grad_raw_vec_y_im),
                drjit_data_ptr_mut(grad_raw_vec_z_re),
                drjit_data_ptr_mut(grad_raw_vec_z_im),
                n_rx
            );
            return nb::make_tuple(
                grad_continued_direct_re,
                grad_continued_direct_im,
                grad_tx_basis_x,
                grad_tx_basis_y,
                grad_tx_basis_z,
                grad_rx_basis_x,
                grad_rx_basis_y,
                grad_rx_basis_z,
                grad_incident_weight,
                grad_incident_response_re,
                grad_incident_response_im,
                grad_raw_vec_x_re,
                grad_raw_vec_x_im,
                grad_raw_vec_y_re,
                grad_raw_vec_y_im,
                grad_raw_vec_z_re,
                grad_raw_vec_z_im
            );
        },
        "Compute the VJP of matched ISB completion outputs."
    );
    m.def(
        "radiomap_shadow_boundary_incident_stats_forward_raw",
        [](
            const DiffFloat &tx_x,
            const DiffFloat &tx_y,
            const DiffFloat &tx_z,
            const DiffFloat &rx_x,
            const DiffFloat &rx_y,
            const DiffFloat &rx_z,
            const DiffFloat &edge_pos_x,
            const DiffFloat &edge_pos_y,
            const DiffFloat &edge_pos_z,
            const Float &edge_dir_x,
            const Float &edge_dir_y,
            const Float &edge_dir_z,
            const Float &n0_x,
            const Float &n0_y,
            const Float &n0_z,
            const Float &nn_x,
            const Float &nn_y,
            const Float &nn_z,
            const Float &wedge_n,
            const Float &edge_line_min,
            const Float &edge_line_max,
            const Int32 &source_visible,
            int n_rx,
            int n_edges,
            float k
        ) {
            DiffFloat sum_incident_weight = zeros_array<DiffFloat>(n_rx);
            DiffFloat max_incident_weight = zeros_array<DiffFloat>(n_rx);
            DiffFloat weighted_incident_response_real = zeros_array<DiffFloat>(n_rx);
            DiffFloat weighted_incident_response_imag = zeros_array<DiffFloat>(n_rx);
            Int32 argmax_edge_idx = zeros_array<Int32>(n_rx);
            DiffFloat second_max_incident_weight = zeros_array<DiffFloat>(n_rx);
            Int32 support_edge_count = zeros_array<Int32>(n_rx);
            witwin::channel::native_ext::radiomap_shadow_boundary_incident_statistics_forward(
                drjit_data_ptr(tx_x),
                drjit_data_ptr(tx_y),
                drjit_data_ptr(tx_z),
                drjit_data_ptr(rx_x),
                drjit_data_ptr(rx_y),
                drjit_data_ptr(rx_z),
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
                drjit_data_ptr(wedge_n),
                drjit_data_ptr(edge_line_min),
                drjit_data_ptr(edge_line_max),
                drjit_data_ptr(source_visible),
                drjit_data_ptr_mut(sum_incident_weight),
                drjit_data_ptr_mut(max_incident_weight),
                drjit_data_ptr_mut(weighted_incident_response_real),
                drjit_data_ptr_mut(weighted_incident_response_imag),
                drjit_data_ptr_mut(argmax_edge_idx),
                drjit_data_ptr_mut(second_max_incident_weight),
                drjit_data_ptr_mut(support_edge_count),
                n_rx,
                n_edges,
                k
            );
            return nb::make_tuple(
                sum_incident_weight,
                max_incident_weight,
                weighted_incident_response_real,
                weighted_incident_response_imag,
                argmax_edge_idx,
                second_max_incident_weight,
                support_edge_count
            );
        },
        "Compute radiomap shadow-boundary incident statistics."
    );
    m.def(
        "radiomap_shadow_boundary_incident_stats_jvp_raw",
        [](
            const DiffFloat &tx_x,
            const DiffFloat &tx_y,
            const DiffFloat &tx_z,
            const DiffFloat &rx_x,
            const DiffFloat &rx_y,
            const DiffFloat &rx_z,
            const DiffFloat &edge_pos_x,
            const DiffFloat &edge_pos_y,
            const DiffFloat &edge_pos_z,
            const Float &edge_dir_x,
            const Float &edge_dir_y,
            const Float &edge_dir_z,
            const Float &n0_x,
            const Float &n0_y,
            const Float &n0_z,
            const Float &nn_x,
            const Float &nn_y,
            const Float &nn_z,
            const Float &wedge_n,
            const Float &edge_line_min,
            const Float &edge_line_max,
            const Int32 &source_visible,
            const Int32 &argmax_edge_idx,
            const DiffFloat &t_tx_x,
            const DiffFloat &t_tx_y,
            const DiffFloat &t_tx_z,
            const DiffFloat &t_rx_x,
            const DiffFloat &t_rx_y,
            const DiffFloat &t_rx_z,
            const DiffFloat &t_edge_pos_x,
            const DiffFloat &t_edge_pos_y,
            const DiffFloat &t_edge_pos_z,
            int n_rx,
            int n_edges,
            float k
        ) {
            DiffFloat t_sum_incident_weight = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_max_incident_weight = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_weighted_incident_response_real = zeros_array<DiffFloat>(n_rx);
            DiffFloat t_weighted_incident_response_imag = zeros_array<DiffFloat>(n_rx);
            witwin::channel::native_ext::radiomap_shadow_boundary_incident_statistics_jvp(
                drjit_data_ptr(tx_x),
                drjit_data_ptr(tx_y),
                drjit_data_ptr(tx_z),
                drjit_data_ptr(rx_x),
                drjit_data_ptr(rx_y),
                drjit_data_ptr(rx_z),
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
                drjit_data_ptr(wedge_n),
                drjit_data_ptr(edge_line_min),
                drjit_data_ptr(edge_line_max),
                drjit_data_ptr(source_visible),
                drjit_data_ptr(argmax_edge_idx),
                drjit_data_ptr(t_tx_x),
                drjit_data_ptr(t_tx_y),
                drjit_data_ptr(t_tx_z),
                drjit_data_ptr(t_rx_x),
                drjit_data_ptr(t_rx_y),
                drjit_data_ptr(t_rx_z),
                drjit_data_ptr(t_edge_pos_x),
                drjit_data_ptr(t_edge_pos_y),
                drjit_data_ptr(t_edge_pos_z),
                drjit_data_ptr_mut(t_sum_incident_weight),
                drjit_data_ptr_mut(t_max_incident_weight),
                drjit_data_ptr_mut(t_weighted_incident_response_real),
                drjit_data_ptr_mut(t_weighted_incident_response_imag),
                n_rx,
                n_edges,
                k
            );
            return nb::make_tuple(
                t_sum_incident_weight,
                t_max_incident_weight,
                t_weighted_incident_response_real,
                t_weighted_incident_response_imag
            );
        },
        "Compute the JVP of radiomap shadow-boundary incident statistics."
    );
    m.def(
        "radiomap_shadow_boundary_incident_stats_backward_raw",
        [](
            const DiffFloat &tx_x,
            const DiffFloat &tx_y,
            const DiffFloat &tx_z,
            const DiffFloat &rx_x,
            const DiffFloat &rx_y,
            const DiffFloat &rx_z,
            const DiffFloat &edge_pos_x,
            const DiffFloat &edge_pos_y,
            const DiffFloat &edge_pos_z,
            const Float &edge_dir_x,
            const Float &edge_dir_y,
            const Float &edge_dir_z,
            const Float &n0_x,
            const Float &n0_y,
            const Float &n0_z,
            const Float &nn_x,
            const Float &nn_y,
            const Float &nn_z,
            const Float &wedge_n,
            const Float &edge_line_min,
            const Float &edge_line_max,
            const Int32 &source_visible,
            const Int32 &argmax_edge_idx,
            const DiffFloat &grad_sum_incident_weight,
            const DiffFloat &grad_max_incident_weight,
            const DiffFloat &grad_weighted_incident_response_real,
            const DiffFloat &grad_weighted_incident_response_imag,
            int n_rx,
            int n_edges,
            float k
        ) {
            DiffFloat grad_tx_x = zeros_array<DiffFloat>(1);
            DiffFloat grad_tx_y = zeros_array<DiffFloat>(1);
            DiffFloat grad_tx_z = zeros_array<DiffFloat>(1);
            DiffFloat grad_rx_x = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_rx_y = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_rx_z = zeros_array<DiffFloat>(n_rx);
            DiffFloat grad_edge_pos_x = zeros_array<DiffFloat>(n_edges);
            DiffFloat grad_edge_pos_y = zeros_array<DiffFloat>(n_edges);
            DiffFloat grad_edge_pos_z = zeros_array<DiffFloat>(n_edges);
            witwin::channel::native_ext::radiomap_shadow_boundary_incident_statistics_backward(
                drjit_data_ptr(tx_x),
                drjit_data_ptr(tx_y),
                drjit_data_ptr(tx_z),
                drjit_data_ptr(rx_x),
                drjit_data_ptr(rx_y),
                drjit_data_ptr(rx_z),
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
                drjit_data_ptr(wedge_n),
                drjit_data_ptr(edge_line_min),
                drjit_data_ptr(edge_line_max),
                drjit_data_ptr(source_visible),
                drjit_data_ptr(argmax_edge_idx),
                drjit_data_ptr(grad_sum_incident_weight),
                drjit_data_ptr(grad_max_incident_weight),
                drjit_data_ptr(grad_weighted_incident_response_real),
                drjit_data_ptr(grad_weighted_incident_response_imag),
                drjit_data_ptr_mut(grad_tx_x),
                drjit_data_ptr_mut(grad_tx_y),
                drjit_data_ptr_mut(grad_tx_z),
                drjit_data_ptr_mut(grad_rx_x),
                drjit_data_ptr_mut(grad_rx_y),
                drjit_data_ptr_mut(grad_rx_z),
                drjit_data_ptr_mut(grad_edge_pos_x),
                drjit_data_ptr_mut(grad_edge_pos_y),
                drjit_data_ptr_mut(grad_edge_pos_z),
                n_rx,
                n_edges,
                k
            );
            return nb::make_tuple(
                grad_tx_x,
                grad_tx_y,
                grad_tx_z,
                grad_rx_x,
                grad_rx_y,
                grad_rx_z,
                grad_edge_pos_x,
                grad_edge_pos_y,
                grad_edge_pos_z
            );
        },
        "Compute the VJP of radiomap shadow-boundary incident statistics."
    );
}
