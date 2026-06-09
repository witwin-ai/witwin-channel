import unittest
from pathlib import Path

import raydn as rt


ROOT = Path(__file__).resolve().parents[2]


class PublicApiContractTests(unittest.TestCase):
    def test_scene_does_not_expose_benchmark_specific_fast_paths(self):
        self.assertFalse(hasattr(rt.Scene, "trace_reflections_minimal"))
        self.assertFalse(hasattr(rt.Scene, "intersect_t_sum"))
        self.assertFalse(hasattr(rt.Scene, "intersect_t_sum_vjp"))

    def test_acceptance_benchmarks_use_public_raydn_api(self):
        stress = (ROOT / "tests" / "benchmark_raydn_rayd_mitsuba_stress.py").read_text()
        multipath = (ROOT / "tests" / "benchmark_raydn_rayd_mitsuba_multipath.py").read_text()

        forbidden = (
            "trace_reflections_minimal",
            "intersect_t_sum",
            "intersect_t_sum_vjp",
            "trace_reflections_forward_minimal",
        )
        for token in forbidden:
            self.assertNotIn(token, stress)
            self.assertNotIn(token, multipath)

    def test_full_intersection_backward_does_not_contiguous_upstream_grads(self):
        source = (ROOT / "src" / "torch_ext" / "scene" / "ops_intersect.cpp").read_text()
        forbidden = (
            "optional_contiguous_tensor",
            "grad_t.contiguous()",
            "grad_p.contiguous()",
            "grad_n.contiguous()",
            "grad_geo_n.contiguous()",
            "grad_uv.contiguous()",
            "grad_barycentric.contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_full_intersection_ad_does_not_materialize_unused_output_grads(self):
        source = (ROOT / "src" / "torch_ext" / "scene" / "ops_intersect.cpp").read_text()
        start = source.index("class IntersectAdFunction")
        end = source.index("class IntersectTAdFunction")
        intersect_ad_source = source[start:end]
        self.assertIn("ctx->set_materialize_grads(false);", intersect_ad_source)
        t_start = source.index("class IntersectTAdFunction")
        t_end = source.index("} // namespace")
        intersect_t_source = source[t_start:t_end]
        self.assertIn("ctx->set_materialize_grads(false);", intersect_t_source)

    def test_intersection_is_valid_uses_native_kernel(self):
        source = (ROOT / "raydn" / "types.py").read_text()
        start = source.index("class Intersection")
        end = source.index("@dataclass(frozen=True)\nclass NearestPointEdge")
        intersection_source = source[start:end]
        self.assertIn("torch.ops.raydn.intersection_valid", intersection_source)
        self.assertLess(
            intersection_source.index("torch.ops.raydn.intersection_valid"),
            intersection_source.index("torch.isfinite"),
        )

    def test_camera_public_path_does_not_stage_contiguous_copies(self):
        source = (ROOT / "raydn" / "camera.py").read_text()
        forbidden = (
            "sample.contiguous()",
            "point.contiguous()",
            "grad_world.contiguous()",
            "grad_sample.contiguous()",
            "grad_direction.contiguous()",
            "sample.new_empty((0, 3))",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_diffraction_active_masks_are_not_expanded_or_contiguous_staged(self):
        python_source = (ROOT / "raydn" / "autograd.py").read_text()
        cpp_source = (ROOT / "src" / "torch_ext" / "diffraction" / "ops.cpp").read_text()
        forbidden = (
            "active.contiguous()",
            "recursive_active.contiguous()",
            "active.expand({state_count}).contiguous()",
            ".expand({state_count}).contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, python_source)
            self.assertNotIn(token, cpp_source)

    def test_trace_dfr_paths_public_wrapper_does_not_stage_contiguous_inputs(self):
        source = (ROOT / "raydn" / "autograd.py").read_text()
        start = source.index("def trace_dfr_paths_order1_native")
        end = source.index("class _DfrDirectAccumFunction")
        path_source = source[start:end]
        forbidden = (
            "_contig_states(",
            "_contig_material(",
            ".contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, path_source)

    def test_coherent_diffraction_public_wrapper_does_not_stage_contiguous_inputs(self):
        source = (ROOT / "raydn" / "autograd.py").read_text()
        start = source.index("def accum_dfr_coherent_direct_native")
        end = source.index("class NativeOpUnavailable")
        coherent_source = source[start:end]
        forbidden = (
            "_contig_states(",
            "_contig_material(",
            ".contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, coherent_source)

    def test_chain_diffraction_public_ad_path_does_not_stage_states_or_material(self):
        source = (ROOT / "raydn" / "autograd.py").read_text()
        start = source.index("def accum_dfr_chain_native")
        end = source.index("def accum_dfr_coherent_direct_native")
        chain_source = source[start:end]
        forbidden = (
            "_contig_states(",
            "_contig_material(",
            ".contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, chain_source)
        self.assertIn("initial_states.state_count", chain_source)
        self.assertIn("recursive_states.state_count", chain_source)

    def test_diffraction_accumulation_forward_does_not_split_vec3_with_aten_copies(self):
        source = (ROOT / "src" / "torch_ext" / "diffraction" / "ops.cpp").read_text()
        start = source.index("py::tuple diffraction_accumulation_forward_op")
        end = source.index("py::tuple diffraction_accumulation_direct_backward_op")
        forward_source = source[start:end]
        coherent_start = source.index("py::tuple diffraction_coherent_accumulation_forward_op")
        coherent_end = source.index("} // namespace raydn", coherent_start)
        coherent_source = source[coherent_start:coherent_end]
        forbidden = (
            "split_vec3(",
            "split_optional_vec3(",
            ".contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, forward_source)
            self.assertNotIn(token, coherent_source)

    def test_direct_diffraction_ad_does_not_stage_vec3_or_upstream_grads_with_aten_copies(self):
        source = (ROOT / "src" / "torch_ext" / "diffraction" / "ops.cpp").read_text()
        backward_start = source.index("py::tuple diffraction_accumulation_direct_backward_op")
        backward_end = source.index("py::tuple diffraction_accumulation_direct_jvp_op")
        backward_source = source[backward_start:backward_end]
        jvp_start = backward_end
        jvp_end = source.index("py::tuple diffraction_accumulation_chain_backward_op")
        jvp_source = source[jvp_start:jvp_end]
        forbidden = (
            "split_vec3(",
            "split_optional_vec3(",
            "flatten_optional_f32(",
            "stack_vec3(",
            ".contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, backward_source)
            self.assertNotIn(token, jvp_source)

    def test_chain_diffraction_ad_does_not_stage_vec3_or_upstream_grads_with_aten_copies(self):
        source = (ROOT / "src" / "torch_ext" / "diffraction" / "ops.cpp").read_text()
        backward_start = source.index("py::tuple diffraction_accumulation_chain_backward_op")
        backward_end = source.index("py::tuple diffraction_accumulation_chain_jvp_op")
        backward_source = source[backward_start:backward_end]
        jvp_start = backward_end
        jvp_end = source.index("py::tuple diffraction_coherent_accumulation_forward_op")
        jvp_source = source[jvp_start:jvp_end]
        forbidden = (
            "split_vec3(",
            "split_optional_vec3(",
            "flatten_optional_f32(",
            "stack_vec3(",
            ".contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, backward_source)
            self.assertNotIn(token, jvp_source)

    def test_direct_diffraction_public_ad_path_does_not_stage_states_or_material(self):
        source = (ROOT / "raydn" / "autograd.py").read_text()
        start = source.index("def accum_dfr_direct_native")
        end = source.index("class _DfrChainAccumFunction")
        direct_source = source[start:end]
        forbidden = (
            "_contig_states(",
            "_contig_material(",
            ".contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, direct_source)

    def test_python_autograd_functions_do_not_materialize_unused_output_grads(self):
        source = (ROOT / "raydn" / "autograd.py").read_text()
        self.assertEqual(source.count("def setup_context"), source.count("ctx.set_materialize_grads(False)"))

    def test_nearest_edge_ad_does_not_contiguous_upstream_grads_or_tangents(self):
        source = (ROOT / "src" / "torch_ext" / "edge" / "ops_edge.cpp").read_text()
        forbidden = (
            "optional_contiguous_tensor",
            "grad_distance.contiguous()",
            "grad_edge_point.contiguous()",
            "grad_edge_t.contiguous()",
            "tangent_vertices.contiguous()",
            "tangent_point.contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_reflection_epc_ad_does_not_contiguous_upstream_grads_or_tangents(self):
        source = (ROOT / "src" / "torch_ext" / "reflection" / "ops.cpp").read_text()
        start = source.index("py::tuple trace_refl_epc_field_backward_op")
        end = source.index("py::tuple reflection_dedup_forward_op")
        epc_source = source[start:end]
        forbidden = (
            "optional_contiguous_tensor",
            "grad_field_real.contiguous()",
            "grad_field_imag.contiguous()",
            "grad_path_length.contiguous()",
            "tangent_vertices.contiguous()",
            "tangent_source.contiguous()",
            "tangent_receiver.contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, epc_source)

    def test_reflection_chain_ad_does_not_stage_upstream_grads_or_tangents(self):
        source = (ROOT / "src" / "torch_ext" / "reflection" / "ops.cpp").read_text()
        start = source.index("py::tuple trace_reflections_backward_op")
        end = source.index("py::tuple trace_refl_epc_field_forward_op")
        chain_source = source[start:end]
        forbidden = (
            "optional_contiguous_or_empty",
            "optional_contiguous_or_zeros_like",
            "grad_t.contiguous()",
            "grad_image_sources.contiguous()",
            "tangent_vertices.contiguous()",
            "tangent_ray_o.contiguous()",
            "tangent_ray_d.contiguous()",
        )
        for token in forbidden:
            self.assertNotIn(token, chain_source)

    def test_reflection_chain_ad_uses_fused_kernels_not_aten_bounce_loop(self):
        source = (ROOT / "src" / "torch_ext" / "reflection" / "backward.cu").read_text()
        backward_start = source.index("ReflectionBackwardOutputs reflection_chain_backward_cuda")
        backward_end = source.index("ReflectionJvpOutputs reflection_jvp_cuda")
        backward_source = source[backward_start:backward_end]
        jvp_start = source.index("ReflectionJvpOutputs reflection_chain_jvp_cuda")
        jvp_end = source.index("ReflEpcBackwardOutputs refl_epc_backward_cuda")
        jvp_source = source[jvp_start:jvp_end]
        forbidden = (
            "select_bounce(",
            "mask_vec(",
            "normal_sign(",
            "intersect_backward_cuda(",
            ".contiguous()",
            "copy_(",
            "at::sum",
            "at::where",
        )
        for token in forbidden:
            self.assertNotIn(token, backward_source)
            self.assertNotIn(token, jvp_source)


if __name__ == "__main__":
    unittest.main()
