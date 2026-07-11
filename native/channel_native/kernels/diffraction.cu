#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <rayd/shared/utd/utd_math.h>

#include "../tensor_checks.h"
#include <vector>

namespace {

constexpr int kDiffractionBlockSize = 256;
namespace utd = witwin::channel::native_ext;

__device__ __forceinline__ unsigned int dfr_hash(unsigned int x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x;
}
__device__ __forceinline__ float dfr_uniform(unsigned int lane,unsigned int stream,unsigned int seed) {
    const unsigned int h=dfr_hash(lane^(stream*0x9e3779b9u)^seed);
    return static_cast<float>(h&0x00ffffffu)*(1.0f/16777216.0f);
}

__device__ __forceinline__ void slab_reflection(
    float ct,float er,float sg,float thickness,float wavelength,utd::Complex &rte,utd::Complex &rtm) {
    ct=fminf(fmaxf(fabsf(ct),1.0e-6f),1.0f);
    const float omega=2.0f*utd::UTD_PI*299792458.0f/wavelength;
    const utd::Complex eta=utd::cplx(er,-sg/(omega*utd::UTD_EPSILON_0));
    const utd::Complex root=utd::cplx_sqrt(utd::cplx_sub(eta,utd::cplx(1.0f-ct*ct,0)));
    const utd::Complex rp_te=utd::cplx_div(utd::cplx_sub(utd::cplx(ct,0),root),utd::cplx_add(utd::cplx(ct,0),root));
    const utd::Complex ect=utd::cplx_mul_real(eta,ct);
    const utd::Complex rp_tm=utd::cplx_div(utd::cplx_sub(ect,root),utd::cplx_add(ect,root));
    const utd::Complex q=utd::cplx_mul_real(root,2.0f*utd::UTD_PI*fmaxf(thickness,0.0f)/wavelength);
    float sn,cs; sincosf(2.0f*q.re,&sn,&cs);
    const float amplitude=expf(fminf(2.0f*q.im,80.0f));
    const utd::Complex phase=utd::cplx(amplitude*cs,-amplitude*sn);
    const utd::Complex one=utd::cplx(1,0), term=utd::cplx_sub(one,phase);
    rte=utd::cplx_div(utd::cplx_mul(rp_te,term),utd::cplx_sub(one,utd::cplx_mul(utd::cplx_mul(rp_te,rp_te),phase)));
    rtm=utd::cplx_div(utd::cplx_mul(rp_tm,term),utd::cplx_sub(one,utd::cplx_mul(utd::cplx_mul(rp_tm,rp_tm),phase)));
}

__device__ __forceinline__ utd::JonesOperator slab_face_operator(
    float ct,float er,float sg,float gain,float thickness,float wavelength,
    utd::float3a normal,utd::float3a in_hat,utd::float3a out_hat,
    utd::Basis3 in_edge,utd::Basis3 out_edge) {
    utd::Complex rte,rtm; slab_reflection(ct,er,sg,thickness,wavelength,rte,rtm);
    utd::JonesOperator diag={utd::cplx_mul_real(rte,gain),utd::cplx_zero(),utd::cplx_zero(),utd::cplx_mul_real(rtm,gain)};
    const utd::float3a face_in=utd::f3_cross(normal,in_hat);
    const utd::float3a raw_out=utd::f3_cross(normal,out_hat);
    const utd::float3a reference_axis=utd::stable_perp_basis(out_hat,face_in);
    const utd::float3a face_out=utd::f3_dot(raw_out,reference_axis)<0?utd::f3_neg(raw_out):raw_out;
    const utd::Basis3 fin=utd::basis_from_first_vector(in_hat,face_in,utd::stable_perp_basis(in_hat,utd::make_f3(0,0,1)));
    const utd::Basis3 fout=utd::basis_from_first_vector(out_hat,face_out,reference_axis);
    return utd::jop_in_basis(diag,fin,fout,in_edge,out_edge);
}

__device__ __forceinline__ utd::float3a load_utd3(const float *p, int i) {
    return utd::make_f3(p[3*i], p[3*i+1], p[3*i+2]);
}

__device__ __forceinline__ float component_utd(utd::float3a v, int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

__global__ void sionna_diffraction_tape_accumulate_kernel(
    const bool *tape_active, const int *tape_state, const int *tape_cell, const float *tape_u,
    const float *edge_pos, const float *edge_dir, const float *t_min, const float *t_max,
    const float *n0, const float *nn, const int *prim0, const int *prim1,
    const float *exterior_angle, const float *source, const float *source_power,
    const float *eta_r, const float *sigma, const float *mu_r, const float *gain, const float *thickness,
    const bool *material_valid, float *output, int64_t sample_count, int state_count,
    int axis, float plane, float c0min, float c0max, float c1min, float c1max,
    int r0, int r1, float wavelength, float cell_area, int seed, float total_edge_length) {
    const int64_t stride = static_cast<int64_t>(blockDim.x)*gridDim.x;
    for (int64_t lane=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         lane<sample_count; lane+=stride) {
        if (!tape_active[lane]) continue;
        const int sidx=tape_state[lane], cell=tape_cell[lane];
        if (sidx<0 || sidx>=state_count || cell<0 || cell>=r0*r1) continue;
        const utd::float3a ep=load_utd3(edge_pos,sidx);
        const utd::float3a eh=utd::safe_normalize(load_utd3(edge_dir,sidx),utd::make_f3(0,0,1));
        const float length=fmaxf(t_max[sidx]-t_min[sidx],0.0f);
        const float ell=t_min[sidx]+tape_u[lane]*length;
        const utd::float3a edge_point=utd::f3_add(ep,utd::f3_mul(eh,ell));
        const utd::float3a src=load_utd3(source,sidx);
        const utd::float3a incident=utd::safe_normalize(utd::f3_sub(edge_point,src),utd::make_f3(0,0,1));
        const float axial=fminf(fmaxf(utd::f3_dot(incident,eh),-1.0f),1.0f);
        const float radial=sqrtf(fmaxf(1.0f-axial*axial,0.0f));
        const utd::float3a basis0=utd::stable_perp_basis(eh,incident);
        const utd::float3a basis1=utd::safe_normalize(utd::f3_cross(eh,basis0),utd::make_f3(0,1,0));
        float sa,ca; sincosf(2.0f*utd::UTD_PI*dfr_uniform(static_cast<unsigned int>(lane),1u,static_cast<unsigned int>(seed)),&sa,&ca);
        const utd::float3a ko_exact=utd::safe_normalize(
            utd::f3_add(utd::f3_mul(eh,axial),utd::f3_mul(utd::f3_add(utd::f3_mul(basis0,ca),utd::f3_mul(basis1,sa)),radial)),basis0);
        const float denom=component_utd(ko_exact,axis);
        if(fabsf(denom)<1.0e-8f) continue;
        const float distance=(plane-component_utd(edge_point,axis))/denom;
        if(!(distance>0.0f)) continue;
        const utd::float3a target=utd::f3_add(edge_point,utd::f3_mul(ko_exact,distance));

        utd::PairInputs p={};
        p.edgePos=edge_point; p.edgeDir=eh; p.n0=load_utd3(n0,sidx); p.nn=load_utd3(nn,sidx);
        p.wedgeN=exterior_angle[sidx]/utd::UTD_PI; p.edgeLineMin=-1.0e5f; p.edgeLineMax=1.0e5f;
        p.sourcePos=src; p.selectStationaryPoint=0.0f;
        const float k=2.0f*utd::UTD_PI/wavelength;
        utd::MaterialParams mat={}; mat.useFresnel=1; mat.etaR=1; mat.muR=1; mat.sigma=0; mat.gain=1;
        mat.omega=2.0f*utd::UTD_PI*299792458.0f/wavelength; mat.txPolZ=1.0f;
        const utd::float3a incident_dir=incident;
        const utd::float3a pol=utd::stable_perp_basis(incident_dir,utd::make_f3(0,0,1));
        p.incidentBasis=utd::basis_from_first_vector(incident_dir,pol,utd::make_f3(1,0,0));
        p.incidentJones=utd::jones_from_vector(utd::direct_source_vector(src,edge_point,k,mat),p.incidentBasis);
        auto load_mat=[&](int prim)->utd::FaceMaterialParams {
            utd::FaceMaterialParams m={1,1,0,1,1,0};
            if(prim>=0 && material_valid[prim]) { m={eta_r[prim],mu_r[prim],sigma[prim],gain[prim],1,1}; }
            return m;
        };
        p.face0Material=load_mat(prim0[sidx]); p.face1Material=load_mat(prim1[sidx]);
        float gphi,gphi_p,gs,gs_p,gsb;
        utd::compute_edge_geometry_3d(src,edge_point,eh,p.n0,target,gphi,gphi_p,gs,gs_p,gsb);
        const utd::Basis3 in_edge=utd::diffraction_edge_basis(utd::f3_sub(edge_point,src),eh,false);
        const utd::Basis3 out_edge=utd::diffraction_edge_basis(utd::f3_sub(target,edge_point),eh,true);
        const int f0=prim0[sidx], f1=prim1[sidx];
        if(f0>=0 && material_valid[f0]) p.face0Operator=slab_face_operator(
            fabsf(sinf(gphi_p)),eta_r[f0],sigma[f0],gain[f0],thickness[f0],wavelength,
            p.n0,in_edge.k,out_edge.k,in_edge,out_edge);
        if(f1>=0 && material_valid[f1]) p.face1Operator=slab_face_operator(
            fabsf(sinf(p.wedgeN*utd::UTD_PI-gphi)),eta_r[f1],sigma[f1],gain[f1],thickness[f1],wavelength,
            p.nn,in_edge.k,out_edge.k,in_edge,out_edge);
        mat.omega=0.0f;
        const utd::PairOutputs field=utd::compute_pair_contribution(p,target,k,mat);
        const float field_power=utd::cplx_abs_sqr(field.vectorField.x)+utd::cplx_abs_sqr(field.vectorField.y)+utd::cplx_abs_sqr(field.vectorField.z);
        if (!(field_power>0) || !isfinite(field_power)) continue;

        const utd::float3a t0v=utd::safe_normalize(utd::f3_cross(p.n0,eh),utd::make_f3(1,0,0));
        const utd::float3a ko=ko_exact;
        float phi=atan2f(utd::f3_dot(ko,p.n0),utd::f3_dot(ko,t0v));
        if (phi < 0.0f) phi += 2.0f*utd::UTD_PI;
        // RayD proposes the complete Keller cone. Sionna's lit-region
        // estimator only accepts the exterior angular interval [0, 2pi-i].
        // Rejection keeps the full-cone 1/(2pi) proposal density, hence the
        // accepted sample weight below remains 2pi rather than the interval
        // width.
        if (phi > exterior_angle[sidx]) continue;
        const utd::float3a dko=utd::f3_mul(utd::f3_add(utd::f3_mul(basis0,-sa),utd::f3_mul(basis1,ca)),radial);
        const utd::float3a je=utd::f3_sub(eh,utd::f3_mul(ko,component_utd(eh,axis)/denom));
        const utd::float3a jp=utd::f3_mul(utd::f3_sub(dko,utd::f3_mul(ko,component_utd(dko,axis)/denom)),distance);
        const float jacobian=utd::safe_length(utd::f3_cross(jp,je));
        const float edge_weight=total_edge_length/fmaxf(static_cast<float>(sample_count),1.0f);
        const float value=field_power*source_power[sidx]*jacobian*(2.0f*utd::UTD_PI)*edge_weight/fmaxf(cell_area,1.0e-8f);
        if(value>0 && isfinite(value)) atomicAdd(output+cell,value);
    }
}

using channel_native::check_tensor;

__global__ void diffraction_state_wi_kernel(
    const float *__restrict__ state_edge_pos,
    const float *__restrict__ state_src,
    float *__restrict__ state_wi,
    int64_t state_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        const float *edge_pos = state_edge_pos + state * 3;
        const float *src = state_src + state * 3;
        float dx = edge_pos[0] - src[0];
        float dy = edge_pos[1] - src[1];
        float dz = edge_pos[2] - src[2];
        const float norm = sqrtf(dx * dx + dy * dy + dz * dz);
        const float scale = norm > 1.0e-6f ? 1.0f / norm : 1.0e6f;

        float *out = state_wi + state * 3;
        out[0] = dx * scale;
        out[1] = dy * scale;
        out[2] = dz * scale;
    }
}

__global__ void diffraction_state_pack_kernel(
    const int *__restrict__ edge_indices,
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ line_min,
    const float *__restrict__ line_max,
    const float *__restrict__ n0,
    const float *__restrict__ n1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const float *__restrict__ exterior_angle,
    const float *__restrict__ tx,
    const float *__restrict__ tx_power,
    int *__restrict__ state_edge_index,
    float *__restrict__ state_edge_pos,
    float *__restrict__ state_edge_dir,
    float *__restrict__ state_line_min,
    float *__restrict__ state_line_max,
    float *__restrict__ state_n0,
    float *__restrict__ state_n1,
    int *__restrict__ state_face0,
    int *__restrict__ state_face1,
    float *__restrict__ state_exterior_angle,
    float *__restrict__ state_src,
    float *__restrict__ state_src_power,
    int64_t state_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float tx_x = tx[0];
    const float tx_y = tx[1];
    const float tx_z = tx[2];
    const float power = tx_power[0];
    for (int64_t state = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         state < state_count;
         state += stride) {
        const int edge = edge_indices[state];
        state_edge_index[state] = edge;
        state_line_min[state] = line_min[edge];
        state_line_max[state] = line_max[edge];
        state_face0[state] = face0[edge];
        state_face1[state] = face1[edge];
        state_exterior_angle[state] = exterior_angle[edge];
        state_src_power[state] = power;

        const int64_t edge_base = static_cast<int64_t>(edge) * 3;
        const int64_t state_base = state * 3;
        state_edge_pos[state_base + 0] = edge_pos[edge_base + 0];
        state_edge_pos[state_base + 1] = edge_pos[edge_base + 1];
        state_edge_pos[state_base + 2] = edge_pos[edge_base + 2];
        state_edge_dir[state_base + 0] = edge_dir[edge_base + 0];
        state_edge_dir[state_base + 1] = edge_dir[edge_base + 1];
        state_edge_dir[state_base + 2] = edge_dir[edge_base + 2];
        state_n0[state_base + 0] = n0[edge_base + 0];
        state_n0[state_base + 1] = n0[edge_base + 1];
        state_n0[state_base + 2] = n0[edge_base + 2];
        state_n1[state_base + 0] = n1[edge_base + 0];
        state_n1[state_base + 1] = n1[edge_base + 1];
        state_n1[state_base + 2] = n1[edge_base + 2];
        state_src[state_base + 0] = tx_x;
        state_src[state_base + 1] = tx_y;
        state_src[state_base + 2] = tx_z;
    }
}

__global__ void diffraction_state_pack_selected_kernel(
    const bool *__restrict__ selected,
    const float *__restrict__ edge_pos,
    const float *__restrict__ edge_dir,
    const float *__restrict__ line_min,
    const float *__restrict__ line_max,
    const float *__restrict__ n0,
    const float *__restrict__ n1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const float *__restrict__ exterior_angle,
    const float *__restrict__ tx,
    const float *__restrict__ tx_power,
    int *__restrict__ state_edge_index,
    float *__restrict__ state_edge_pos,
    float *__restrict__ state_edge_dir,
    float *__restrict__ state_line_min,
    float *__restrict__ state_line_max,
    float *__restrict__ state_n0,
    float *__restrict__ state_n1,
    int *__restrict__ state_face0,
    int *__restrict__ state_face1,
    float *__restrict__ state_exterior_angle,
    float *__restrict__ state_src,
    float *__restrict__ state_src_power,
    int64_t edge_count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float tx_x = tx[0];
    const float tx_y = tx[1];
    const float tx_z = tx[2];
    const float power = tx_power[0];
    for (int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < edge_count;
         edge += stride) {
        state_edge_index[edge] = static_cast<int>(edge);
        state_line_min[edge] = line_min[edge];
        state_line_max[edge] = line_max[edge];
        state_face0[edge] = face0[edge];
        state_face1[edge] = face1[edge];
        state_exterior_angle[edge] = exterior_angle[edge];
        state_src_power[edge] = selected[edge] ? power : 0.0f;

        const int64_t base = edge * 3;
        state_edge_pos[base + 0] = edge_pos[base + 0];
        state_edge_pos[base + 1] = edge_pos[base + 1];
        state_edge_pos[base + 2] = edge_pos[base + 2];
        state_edge_dir[base + 0] = edge_dir[base + 0];
        state_edge_dir[base + 1] = edge_dir[base + 1];
        state_edge_dir[base + 2] = edge_dir[base + 2];
        state_n0[base + 0] = n0[base + 0];
        state_n0[base + 1] = n0[base + 1];
        state_n0[base + 2] = n0[base + 2];
        state_n1[base + 0] = n1[base + 0];
        state_n1[base + 1] = n1[base + 1];
        state_n1[base + 2] = n1[base + 2];
        state_src[base + 0] = tx_x;
        state_src[base + 1] = tx_y;
        state_src[base + 2] = tx_z;
    }
}

__device__ __forceinline__ float3 load_vec3(const float *data, int64_t index) {
    const float *ptr = data + index * 3;
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__device__ __forceinline__ void store_vec3(float *data, int64_t index, float3 value) {
    float *ptr = data + index * 3;
    ptr[0] = value.x;
    ptr[1] = value.y;
    ptr[2] = value.z;
}

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    const float xz = __fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.z, b.z));
    return __fadd_rn(xz, __fmul_rn(a.y, b.y));
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float norm3(float3 value) {
    return sqrtf(dot3(value, value));
}

__device__ __forceinline__ float3 normalize3(float3 value, float eps) {
    const float norm = norm3(value);
    const float denom = fmaxf(norm, eps);
    return make_float3(
        __fdiv_rn(value.x, denom),
        __fdiv_rn(value.y, denom),
        __fdiv_rn(value.z, denom));
}

__device__ __forceinline__ float signf_like_torch(float value) {
    return (value > 0.0f) ? 1.0f : ((value < 0.0f) ? -1.0f : 0.0f);
}

__device__ __forceinline__ float unsigned_angle(float3 a, float3 b, float3 axis) {
    const float3 cross = cross3(a, b);
    const float signed_norm = signf_like_torch(dot3(cross, axis)) * norm3(cross);
    float angle = atan2f(signed_norm, dot3(a, b));
    return angle < 0.0f ? angle + 6.28318530717958647692f : angle;
}

__device__ __forceinline__ int opposite_vertex(const int *faces, int face, int shared0, int shared1) {
    const int *tri = faces + static_cast<int64_t>(face) * 3;
    const int v0 = tri[0];
    const int v1 = tri[1];
    const int v2 = tri[2];
    if (v0 != shared0 && v0 != shared1) {
        return v0;
    }
    if (v1 != shared0 && v1 != shared1) {
        return v1;
    }
    return v2;
}

__global__ void diffraction_edge_geometry_kernel(
    const float *__restrict__ vertices,
    const int *__restrict__ faces,
    const float *__restrict__ face_normals,
    const int *__restrict__ edge_v0,
    const int *__restrict__ edge_v1,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    bool *__restrict__ selected,
    float *__restrict__ edge_pos,
    float *__restrict__ edge_dir,
    float *__restrict__ lengths,
    float *__restrict__ line_min,
    float *__restrict__ line_max,
    float *__restrict__ n0,
    float *__restrict__ n1,
    float *__restrict__ exterior_angle,
    int64_t edge_count,
    float plane_tol) {
    constexpr float edge_epsilon = 1.0e-6f;
    constexpr float normal_cos_tol = 1.0f - 1.0e-5f;
    constexpr float two_pi = 6.28318530717958647692f;

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t edge = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         edge < edge_count;
         edge += stride) {
        const int v0 = edge_v0[edge];
        const int v1 = edge_v1[edge];
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const bool boundary = valid0 && !valid1;
        const bool interior = valid0 && valid1;
        const int safe0 = f0 >= 0 ? f0 : 0;
        const int safe1 = f1 >= 0 ? f1 : 0;

        const float3 start = load_vec3(vertices, v0);
        const float3 end = load_vec3(vertices, v1);
        const float3 vector = sub3(end, start);
        const float length = fmaxf(norm3(vector), 1.0e-12f);
        const float3 dir = make_float3(
            __fdiv_rn(vector.x, length),
            __fdiv_rn(vector.y, length),
            __fdiv_rn(vector.z, length));
        const float half_length = 0.5f * length;

        const float3 n0_cand = normalize3(load_vec3(face_normals, safe0), edge_epsilon);
        const float3 n1_cand = normalize3(load_vec3(face_normals, safe1), edge_epsilon);
        const float3 to1 = normalize3(cross3(n0_cand, dir), edge_epsilon);
        const float3 tn1 = normalize3(cross3(n1_cand, dir), edge_epsilon);
        const float3 to2 = normalize3(cross3(n1_cand, dir), edge_epsilon);
        const float3 tn2 = normalize3(cross3(n0_cand, dir), edge_epsilon);
        const bool choose_first = unsigned_angle(to1, tn1, dir) < unsigned_angle(to2, tn2, dir);
        const float3 ordered_n0 = choose_first ? n0_cand : n1_cand;
        const float3 ordered_n1 = choose_first ? n1_cand : n0_cand;
        float3 out_n0 = interior ? ordered_n0 : n0_cand;
        float3 out_n1 = interior ? ordered_n1 : n1_cand;
        if (f1 < 0) {
            out_n1 = mul3(n0_cand, -1.0f);
        }
        const float output_normal_dot = dot3(out_n0, out_n1);
        const float output_clamped_neg_dot = fminf(fmaxf(-output_normal_dot, -1.0f), 1.0f);
        const float output_interior_angle = acosf(output_clamped_neg_dot);
        const float out_exterior_angle = interior ? (two_pi - output_interior_angle) : two_pi;

        bool coplanar = false;
        if (interior) {
            const float selected_normal_dot = dot3(n0_cand, n1_cand);
            const bool aligned = fabsf(selected_normal_dot) >= normal_cos_tol;
            const int opp0 = opposite_vertex(faces, safe0, v0, v1);
            const int opp1 = opposite_vertex(faces, safe1, v0, v1);
            const float3 point_a = load_vec3(vertices, opp0);
            const float3 point_b = load_vec3(vertices, opp1);
            const float plane_dist_a = fabsf(dot3(sub3(point_a, start), n0_cand));
            const float plane_dist_b = fabsf(dot3(sub3(point_b, start), n0_cand));
            coplanar = aligned && plane_dist_a <= plane_tol && plane_dist_b <= plane_tol;
        }
        const float selected_normal_dot = dot3(n0_cand, n1_cand);
        const bool selected_wedge_angle = boundary || (interior && selected_normal_dot < 1.0f);
        selected[edge] =
            (interior || boundary) && !coplanar && length > edge_epsilon && selected_wedge_angle;

        store_vec3(edge_pos, edge, mul3(add3(start, end), 0.5f));
        store_vec3(edge_dir, edge, dir);
        lengths[edge] = length;
        line_min[edge] = -half_length;
        line_max[edge] = half_length;
        store_vec3(n0, edge, out_n0);
        store_vec3(n1, edge, out_n1);
        exterior_angle[edge] = out_exterior_angle;
    }
}

__device__ __forceinline__ int find_root_const(const int *__restrict__ parent, int x) {
    int p = parent[x];
    while (p != parent[p]) {
        p = parent[p];
    }
    return p;
}

__global__ void init_parent_kernel(int *__restrict__ parent, int count) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {
        parent[idx] = idx;
    }
}

__global__ void count_surface_group_edges_kernel(
    const bool *__restrict__ selected,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const int *__restrict__ parent,
    int *__restrict__ root_count,
    int *__restrict__ max_count,
    int64_t edge_count,
    int face_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int face = 0; face < face_count; ++face) {
        root_count[face] = 0;
    }
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (!selected[edge]) {
            continue;
        }
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const int root0 = valid0 ? find_root_const(parent, f0) : -1;
        if (valid0) {
            root_count[root0] += 1;
        }
        if (valid1) {
            const int root1 = find_root_const(parent, f1);
            if (!valid0 || root1 != root0) {
                root_count[root1] += 1;
            }
        }
    }
    int local_max = 0;
    for (int face = 0; face < face_count; ++face) {
        local_max = root_count[face] > local_max ? root_count[face] : local_max;
    }
    max_count[0] = local_max;
}

__global__ void fill_int_kernel(int *__restrict__ data, int value, int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += stride) {
        data[idx] = value;
    }
}

__global__ void fill_surface_group_root_edges_kernel(
    const bool *__restrict__ selected,
    const int *__restrict__ face0,
    const int *__restrict__ face1,
    const int *__restrict__ parent,
    int *__restrict__ root_cursor,
    int *__restrict__ root_indices,
    int64_t edge_count,
    int face_count,
    int max_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int face = 0; face < face_count; ++face) {
        root_cursor[face] = 0;
    }
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (!selected[edge]) {
            continue;
        }
        const int edge_i = static_cast<int>(edge);
        const int f0 = face0[edge];
        const int f1 = face1[edge];
        const bool valid0 = f0 >= 0;
        const bool valid1 = f1 >= 0;
        const int root0 = valid0 ? find_root_const(parent, f0) : -1;
        if (valid0) {
            const int slot = root_cursor[root0]++;
            if (slot < max_count) {
                root_indices[static_cast<int64_t>(root0) * max_count + slot] = edge_i;
            }
        }
        if (valid1) {
            const int root1 = find_root_const(parent, f1);
            if (!valid0 || root1 != root0) {
                const int slot = root_cursor[root1]++;
                if (slot < max_count) {
                    root_indices[static_cast<int64_t>(root1) * max_count + slot] = edge_i;
                }
            }
        }
    }
}

__global__ void emit_surface_group_face_rows_kernel(
    const int *__restrict__ parent,
    const int *__restrict__ root_count,
    const int *__restrict__ root_indices,
    int *__restrict__ counts,
    int *__restrict__ indices,
    int face_count,
    int max_count) {
    const int stride = blockDim.x * gridDim.x;
    for (int face = blockIdx.x * blockDim.x + threadIdx.x; face < face_count; face += stride) {
        const int root = find_root_const(parent, face);
        const int count = root_count[root];
        counts[face] = count;
        for (int slot = 0; slot < count && slot < max_count; ++slot) {
            indices[static_cast<int64_t>(face) * max_count + slot] =
                root_indices[static_cast<int64_t>(root) * max_count + slot];
        }
    }
}

__global__ void count_selected_edge_indices_kernel(
    const bool *__restrict__ selected,
    int *__restrict__ count,
    int64_t edge_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int local_count = 0;
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (selected[edge]) {
            ++local_count;
        }
    }
    count[0] = local_count;
}

__global__ void fill_selected_edge_indices_kernel(
    const bool *__restrict__ selected,
    int *__restrict__ indices,
    int64_t edge_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int cursor = 0;
    for (int64_t edge = 0; edge < edge_count; ++edge) {
        if (selected[edge]) {
            indices[cursor++] = static_cast<int>(edge);
        }
    }
}

}  // namespace

at::Tensor diffraction_state_wi_cuda_impl(at::Tensor state_edge_pos, at::Tensor state_src) {
    check_tensor(state_edge_pos, "state_edge_pos", at::kFloat, 2);
    check_tensor(state_src, "state_src", at::kFloat, 2);
    TORCH_CHECK(state_edge_pos.size(1) == 3, "state_edge_pos must have shape (N, 3)");
    TORCH_CHECK(state_src.size(1) == 3, "state_src must have shape (N, 3)");
    TORCH_CHECK(state_src.size(0) == state_edge_pos.size(0), "state_src must match state_edge_pos");

    const int64_t state_count = state_edge_pos.size(0);
    auto state_wi = at::empty({state_count, 3}, state_edge_pos.options());
    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(state_edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_wi_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            state_edge_pos.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_wi.data_ptr<float>(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return state_wi;
}

at::Tensor selected_edge_indices_cuda_impl(at::Tensor selected) {
    check_tensor(selected, "selected", at::kBool, 1);
    const int64_t edge_count = selected.size(0);
    auto int_options = selected.options().dtype(at::kInt);
    auto count_tensor = at::empty({1}, int_options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(selected.get_device()).stream();
    count_selected_edge_indices_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        count_tensor.data_ptr<int>(),
        edge_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int host_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_count,
        count_tensor.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host_count >= 0, "selected edge count must be non-negative");

    auto indices = at::empty({host_count}, int_options);
    if (host_count > 0) {
        fill_selected_edge_indices_kernel<<<1, 1, 0, stream>>>(
            selected.data_ptr<bool>(),
            indices.data_ptr<int>(),
            edge_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return indices;
}

std::vector<at::Tensor> diffraction_state_pack_cuda_impl(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    check_tensor(edge_indices, "edge_indices", at::kInt, 1);
    check_tensor(edge_pos, "edge_pos", at::kFloat, 2);
    check_tensor(edge_dir, "edge_dir", at::kFloat, 2);
    check_tensor(line_min, "line_min", at::kFloat, 1);
    check_tensor(line_max, "line_max", at::kFloat, 1);
    check_tensor(n0, "n0", at::kFloat, 2);
    check_tensor(n1, "n1", at::kFloat, 2);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    check_tensor(exterior_angle, "exterior_angle", at::kFloat, 1);
    check_tensor(tx, "tx", at::kFloat, 1);
    TORCH_CHECK(tx_power.is_cuda(), "tx_power must be a CUDA tensor");
    TORCH_CHECK(tx_power.scalar_type() == at::kFloat, "tx_power has the wrong dtype");
    TORCH_CHECK(tx_power.is_contiguous(), "tx_power must be contiguous");
    TORCH_CHECK(tx_power.dim() == 0 || tx_power.dim() == 1, "tx_power must be scalar or 1-D");
    if (tx_power.dim() == 0) {
        TORCH_CHECK(tx_power_index == 0, "tx_power_index must be zero for scalar tx_power");
    } else {
        TORCH_CHECK(tx_power_index >= 0 && tx_power_index < tx_power.size(0), "tx_power_index is out of range");
    }
    TORCH_CHECK(edge_pos.size(1) == 3, "edge_pos must have shape (N, 3)");
    TORCH_CHECK(edge_dir.sizes() == edge_pos.sizes(), "edge_dir must match edge_pos");
    TORCH_CHECK(n0.sizes() == edge_pos.sizes(), "n0 must match edge_pos");
    TORCH_CHECK(n1.sizes() == edge_pos.sizes(), "n1 must match edge_pos");
    TORCH_CHECK(line_min.size(0) == edge_pos.size(0), "line_min must match edge count");
    TORCH_CHECK(line_max.size(0) == edge_pos.size(0), "line_max must match edge count");
    TORCH_CHECK(face0.size(0) == edge_pos.size(0), "face0 must match edge count");
    TORCH_CHECK(face1.size(0) == edge_pos.size(0), "face1 must match edge count");
    TORCH_CHECK(exterior_angle.size(0) == edge_pos.size(0), "exterior_angle must match edge count");
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    TORCH_CHECK(edge_pos.get_device() == edge_indices.get_device(), "edge tensors must be on the same device");

    const int64_t state_count = edge_indices.size(0);
    auto int_options = edge_indices.options();
    auto float_options = edge_pos.options();
    auto state_edge_index = at::empty({state_count}, int_options);
    auto state_edge_pos = at::empty({state_count, 3}, float_options);
    auto state_edge_dir = at::empty({state_count, 3}, float_options);
    auto state_line_min = at::empty({state_count}, float_options);
    auto state_line_max = at::empty({state_count}, float_options);
    auto state_n0 = at::empty({state_count, 3}, float_options);
    auto state_n1 = at::empty({state_count, 3}, float_options);
    auto state_face0 = at::empty({state_count}, int_options);
    auto state_face1 = at::empty({state_count}, int_options);
    auto state_exterior_angle = at::empty({state_count}, float_options);
    auto state_src = at::empty({state_count, 3}, float_options);
    auto state_src_power = at::empty({state_count}, float_options);

    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_pack_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            edge_indices.data_ptr<int>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            line_min.data_ptr<float>(),
            line_max.data_ptr<float>(),
            n0.data_ptr<float>(),
            n1.data_ptr<float>(),
            face0.data_ptr<int>(),
            face1.data_ptr<int>(),
            exterior_angle.data_ptr<float>(),
            tx.data_ptr<float>(),
            tx_power.data_ptr<float>() + tx_power_index,
            state_edge_index.data_ptr<int>(),
            state_edge_pos.data_ptr<float>(),
            state_edge_dir.data_ptr<float>(),
            state_line_min.data_ptr<float>(),
            state_line_max.data_ptr<float>(),
            state_n0.data_ptr<float>(),
            state_n1.data_ptr<float>(),
            state_face0.data_ptr<int>(),
            state_face1.data_ptr<int>(),
            state_exterior_angle.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_src_power.data_ptr<float>(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        state_edge_index,
        state_edge_pos,
        state_edge_dir,
        state_line_min,
        state_line_max,
        state_n0,
        state_n1,
        state_face0,
        state_face1,
        state_exterior_angle,
        state_src,
        state_src_power,
    };
}

std::vector<at::Tensor> diffraction_state_pack_selected_cuda_impl(
    at::Tensor selected,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    check_tensor(selected, "selected", at::kBool, 1);
    check_tensor(edge_pos, "edge_pos", at::kFloat, 2);
    check_tensor(edge_dir, "edge_dir", at::kFloat, 2);
    check_tensor(line_min, "line_min", at::kFloat, 1);
    check_tensor(line_max, "line_max", at::kFloat, 1);
    check_tensor(n0, "n0", at::kFloat, 2);
    check_tensor(n1, "n1", at::kFloat, 2);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    check_tensor(exterior_angle, "exterior_angle", at::kFloat, 1);
    check_tensor(tx, "tx", at::kFloat, 1);
    TORCH_CHECK(tx_power.is_cuda(), "tx_power must be a CUDA tensor");
    TORCH_CHECK(tx_power.scalar_type() == at::kFloat, "tx_power has the wrong dtype");
    TORCH_CHECK(tx_power.is_contiguous(), "tx_power must be contiguous");
    TORCH_CHECK(tx_power.dim() == 0 || tx_power.dim() == 1, "tx_power must be scalar or 1-D");
    if (tx_power.dim() == 0) {
        TORCH_CHECK(tx_power_index == 0, "tx_power_index must be zero for scalar tx_power");
    } else {
        TORCH_CHECK(tx_power_index >= 0 && tx_power_index < tx_power.size(0), "tx_power_index is out of range");
    }
    TORCH_CHECK(edge_pos.size(1) == 3, "edge_pos must have shape (N, 3)");
    TORCH_CHECK(edge_dir.sizes() == edge_pos.sizes(), "edge_dir must match edge_pos");
    TORCH_CHECK(n0.sizes() == edge_pos.sizes(), "n0 must match edge_pos");
    TORCH_CHECK(n1.sizes() == edge_pos.sizes(), "n1 must match edge_pos");
    TORCH_CHECK(line_min.size(0) == edge_pos.size(0), "line_min must match edge count");
    TORCH_CHECK(line_max.size(0) == edge_pos.size(0), "line_max must match edge count");
    TORCH_CHECK(face0.size(0) == edge_pos.size(0), "face0 must match edge count");
    TORCH_CHECK(face1.size(0) == edge_pos.size(0), "face1 must match edge count");
    TORCH_CHECK(exterior_angle.size(0) == edge_pos.size(0), "exterior_angle must match edge count");
    TORCH_CHECK(selected.size(0) == edge_pos.size(0), "selected must match edge count");
    TORCH_CHECK(tx.size(0) == 3, "tx must have shape (3,)");
    TORCH_CHECK(edge_pos.get_device() == selected.get_device(), "edge tensors must be on the same device");

    const int64_t state_count = edge_pos.size(0);
    auto int_options = face0.options();
    auto float_options = edge_pos.options();
    auto state_edge_index = at::empty({state_count}, int_options);
    auto state_edge_pos = at::empty({state_count, 3}, float_options);
    auto state_edge_dir = at::empty({state_count, 3}, float_options);
    auto state_line_min = at::empty({state_count}, float_options);
    auto state_line_max = at::empty({state_count}, float_options);
    auto state_n0 = at::empty({state_count, 3}, float_options);
    auto state_n1 = at::empty({state_count, 3}, float_options);
    auto state_face0 = at::empty({state_count}, int_options);
    auto state_face1 = at::empty({state_count}, int_options);
    auto state_exterior_angle = at::empty({state_count}, float_options);
    auto state_src = at::empty({state_count, 3}, float_options);
    auto state_src_power = at::empty({state_count}, float_options);

    if (state_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_pos.get_device()).stream();
        const int block_count = static_cast<int>((state_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_state_pack_selected_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            selected.data_ptr<bool>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            line_min.data_ptr<float>(),
            line_max.data_ptr<float>(),
            n0.data_ptr<float>(),
            n1.data_ptr<float>(),
            face0.data_ptr<int>(),
            face1.data_ptr<int>(),
            exterior_angle.data_ptr<float>(),
            tx.data_ptr<float>(),
            tx_power.data_ptr<float>() + tx_power_index,
            state_edge_index.data_ptr<int>(),
            state_edge_pos.data_ptr<float>(),
            state_edge_dir.data_ptr<float>(),
            state_line_min.data_ptr<float>(),
            state_line_max.data_ptr<float>(),
            state_n0.data_ptr<float>(),
            state_n1.data_ptr<float>(),
            state_face0.data_ptr<int>(),
            state_face1.data_ptr<int>(),
            state_exterior_angle.data_ptr<float>(),
            state_src.data_ptr<float>(),
            state_src_power.data_ptr<float>(),
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        state_edge_index,
        state_edge_pos,
        state_edge_dir,
        state_line_min,
        state_line_max,
        state_n0,
        state_n1,
        state_face0,
        state_face1,
        state_exterior_angle,
        state_src,
        state_src_power,
    };
}

std::vector<at::Tensor> diffraction_edge_geometry_cuda_impl(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol) {
    check_tensor(vertices, "vertices", at::kFloat, 2);
    check_tensor(faces, "faces", at::kInt, 2);
    check_tensor(face_normals, "face_normals", at::kFloat, 2);
    check_tensor(edge_v0, "edge_v0", at::kInt, 1);
    check_tensor(edge_v1, "edge_v1", at::kInt, 1);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    TORCH_CHECK(vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.size(1) == 3, "face_normals must have shape (F, 3)");
    TORCH_CHECK(edge_v1.size(0) == edge_v0.size(0), "edge_v1 must match edge_v0");
    TORCH_CHECK(face0.size(0) == edge_v0.size(0), "face0 must match edge_v0");
    TORCH_CHECK(face1.size(0) == edge_v0.size(0), "face1 must match edge_v0");

    const int64_t edge_count = edge_v0.size(0);
    auto bool_options = edge_v0.options().dtype(at::kBool);
    auto float_options = vertices.options();
    auto selected = at::empty({edge_count}, bool_options);
    auto edge_pos = at::empty({edge_count, 3}, float_options);
    auto edge_dir = at::empty({edge_count, 3}, float_options);
    auto lengths = at::empty({edge_count}, float_options);
    auto line_min = at::empty({edge_count}, float_options);
    auto line_max = at::empty({edge_count}, float_options);
    auto n0 = at::empty({edge_count, 3}, float_options);
    auto n1 = at::empty({edge_count, 3}, float_options);
    auto exterior_angle = at::empty({edge_count}, float_options);

    if (edge_count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
        const int block_count = static_cast<int>((edge_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        diffraction_edge_geometry_kernel<<<block_count, kDiffractionBlockSize, 0, stream>>>(
            vertices.data_ptr<float>(),
            faces.data_ptr<int>(),
            face_normals.data_ptr<float>(),
            edge_v0.data_ptr<int>(),
            edge_v1.data_ptr<int>(),
            face0.data_ptr<int>(),
            face1.data_ptr<int>(),
            selected.data_ptr<bool>(),
            edge_pos.data_ptr<float>(),
            edge_dir.data_ptr<float>(),
            lengths.data_ptr<float>(),
            line_min.data_ptr<float>(),
            line_max.data_ptr<float>(),
            n0.data_ptr<float>(),
            n1.data_ptr<float>(),
            exterior_angle.data_ptr<float>(),
            edge_count,
            static_cast<float>(plane_tol));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {
        selected,
        edge_pos,
        edge_dir,
        lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    };
}

std::vector<at::Tensor> surface_group_edge_candidates_cuda_impl(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol) {
    check_tensor(vertices, "vertices", at::kFloat, 2);
    check_tensor(faces, "faces", at::kInt, 2);
    check_tensor(face_normals, "face_normals", at::kFloat, 2);
    check_tensor(edge_v0, "edge_v0", at::kInt, 1);
    check_tensor(edge_v1, "edge_v1", at::kInt, 1);
    check_tensor(face0, "face0", at::kInt, 1);
    check_tensor(face1, "face1", at::kInt, 1);
    check_tensor(selected, "selected", at::kBool, 1);
    TORCH_CHECK(vertices.size(1) == 3, "vertices must have shape (N, 3)");
    TORCH_CHECK(faces.size(1) == 3, "faces must have shape (F, 3)");
    TORCH_CHECK(face_normals.size(1) == 3, "face_normals must have shape (F, 3)");
    TORCH_CHECK(edge_v1.size(0) == edge_v0.size(0), "edge_v1 must match edge_v0");
    TORCH_CHECK(face0.size(0) == edge_v0.size(0), "face0 must match edge_v0");
    TORCH_CHECK(face1.size(0) == edge_v0.size(0), "face1 must match edge_v0");
    TORCH_CHECK(selected.size(0) == edge_v0.size(0), "selected must match edge_v0");

    const int face_count = static_cast<int>(faces.size(0));
    const int64_t edge_count = edge_v0.size(0);
    auto int_options = faces.options();
    auto parent = at::empty({face_count}, int_options);
    auto root_count = at::empty({face_count}, int_options);
    auto root_cursor = at::empty({face_count}, int_options);
    auto max_count_tensor = at::empty({1}, int_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(vertices.get_device()).stream();
    const int face_blocks = static_cast<int>((face_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
    if (face_count > 0) {
        init_parent_kernel<<<face_blocks, kDiffractionBlockSize, 0, stream>>>(
            parent.data_ptr<int>(),
            face_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Mitsuba's primitive_silhouette_projection() samples the perimeter of the
    // intersected primitive. Keep every triangle as its own root: merging
    // coplanar neighbours changes the sampling domain and both drops and adds
    // wedges relative to Sionna RT.
    (void)vertices;
    (void)face_normals;
    (void)edge_v0;
    (void)edge_v1;
    (void)plane_tol;

    count_surface_group_edges_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        face0.data_ptr<int>(),
        face1.data_ptr<int>(),
        parent.data_ptr<int>(),
        root_count.data_ptr<int>(),
        max_count_tensor.data_ptr<int>(),
        edge_count,
        face_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int host_max_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_max_count,
        max_count_tensor.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host_max_count >= 0, "surface group candidate max count must be non-negative");

    auto counts = at::empty({face_count}, int_options);
    auto indices = at::empty({face_count, host_max_count}, int_options);
    auto root_indices = at::empty({face_count, host_max_count}, int_options);
    const int64_t table_count = static_cast<int64_t>(face_count) * host_max_count;
    if (table_count > 0) {
        const int table_blocks = static_cast<int>((table_count + kDiffractionBlockSize - 1) / kDiffractionBlockSize);
        fill_int_kernel<<<table_blocks, kDiffractionBlockSize, 0, stream>>>(
            indices.data_ptr<int>(),
            -1,
            table_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        fill_int_kernel<<<table_blocks, kDiffractionBlockSize, 0, stream>>>(
            root_indices.data_ptr<int>(),
            -1,
            table_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    fill_surface_group_root_edges_kernel<<<1, 1, 0, stream>>>(
        selected.data_ptr<bool>(),
        face0.data_ptr<int>(),
        face1.data_ptr<int>(),
        parent.data_ptr<int>(),
        root_cursor.data_ptr<int>(),
        root_indices.data_ptr<int>(),
        edge_count,
        face_count,
        host_max_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (face_count > 0) {
        emit_surface_group_face_rows_kernel<<<face_blocks, kDiffractionBlockSize, 0, stream>>>(
            parent.data_ptr<int>(),
            root_count.data_ptr<int>(),
            root_indices.data_ptr<int>(),
            counts.data_ptr<int>(),
            indices.data_ptr<int>(),
            face_count,
            host_max_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {counts, indices};
}

at::Tensor cn_mc_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src) {
    return diffraction_state_wi_cuda_impl(state_edge_pos, state_src);
}

at::Tensor cn_bdpt_diffraction_state_wi_cuda(at::Tensor state_edge_pos, at::Tensor state_src) {
    return diffraction_state_wi_cuda_impl(state_edge_pos, state_src);
}

at::Tensor cn_mc_selected_edge_indices_cuda(at::Tensor selected) {
    return selected_edge_indices_cuda_impl(selected);
}

at::Tensor cn_bdpt_selected_edge_indices_cuda(at::Tensor selected) {
    return selected_edge_indices_cuda_impl(selected);
}

std::vector<at::Tensor> cn_mc_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power) {
    return diffraction_state_pack_cuda_impl(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        0);
}

std::vector<at::Tensor> cn_bdpt_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power) {
    return diffraction_state_pack_cuda_impl(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        0);
}

std::vector<at::Tensor> cn_deterministic_diffraction_state_pack_cuda(
    at::Tensor edge_indices,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    return diffraction_state_pack_cuda_impl(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index);
}

std::vector<at::Tensor> cn_deterministic_diffraction_state_pack_selected_cuda(
    at::Tensor selected,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index) {
    return diffraction_state_pack_selected_cuda_impl(
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        tx_power_index);
}

std::vector<at::Tensor> cn_mc_diffraction_edge_geometry_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol) {
    return diffraction_edge_geometry_cuda_impl(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        plane_tol);
}

std::vector<at::Tensor> cn_bdpt_diffraction_edge_geometry_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    double plane_tol) {
    return diffraction_edge_geometry_cuda_impl(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        plane_tol);
}

std::vector<at::Tensor> cn_mc_surface_group_edge_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol) {
    return surface_group_edge_candidates_cuda_impl(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        selected,
        plane_tol);
}

std::vector<at::Tensor> cn_bdpt_surface_group_edge_candidates_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor face_normals,
    at::Tensor edge_v0,
    at::Tensor edge_v1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor selected,
    double plane_tol) {
    return surface_group_edge_candidates_cuda_impl(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        selected,
        plane_tol);
}

at::Tensor cn_mc_sionna_diffraction_tape_accumulate_cuda(
    at::Tensor tape_active, at::Tensor tape_state, at::Tensor tape_cell, at::Tensor tape_u,
    at::Tensor edge_pos, at::Tensor edge_dir, at::Tensor t_min, at::Tensor t_max,
    at::Tensor n0, at::Tensor nn, at::Tensor prim0, at::Tensor prim1,
    at::Tensor exterior_angle, at::Tensor source, at::Tensor source_power,
    at::Tensor eta_r, at::Tensor sigma, at::Tensor mu_r, at::Tensor gain,
    at::Tensor material_valid, at::Tensor thickness, int64_t axis, double plane,
    double c0min, double c0max, double c1min, double c1max,
    int64_t r0, int64_t r1, double wavelength, double cell_area, int64_t seed, double total_edge_length) {
    auto output=at::empty({r1,r0},source.options());
    const auto stream=at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.data_ptr<float>(),0,static_cast<size_t>(output.numel())*sizeof(float),stream));
    const int64_t samples=tape_active.numel();
    if(samples==0) return output;
    const int blocks=static_cast<int>(std::min<int64_t>((samples+kDiffractionBlockSize-1)/kDiffractionBlockSize,65535));
    sionna_diffraction_tape_accumulate_kernel<<<blocks,kDiffractionBlockSize,0,stream>>>(
        tape_active.data_ptr<bool>(),tape_state.data_ptr<int>(),tape_cell.data_ptr<int>(),tape_u.data_ptr<float>(),
        edge_pos.data_ptr<float>(),edge_dir.data_ptr<float>(),t_min.data_ptr<float>(),t_max.data_ptr<float>(),
        n0.data_ptr<float>(),nn.data_ptr<float>(),prim0.data_ptr<int>(),prim1.data_ptr<int>(),
        exterior_angle.data_ptr<float>(),source.data_ptr<float>(),source_power.data_ptr<float>(),
        eta_r.data_ptr<float>(),sigma.data_ptr<float>(),mu_r.data_ptr<float>(),gain.data_ptr<float>(),thickness.data_ptr<float>(),
        material_valid.data_ptr<bool>(),output.data_ptr<float>(),samples,static_cast<int>(edge_pos.size(0)),
        static_cast<int>(axis),static_cast<float>(plane),static_cast<float>(c0min),static_cast<float>(c0max),
        static_cast<float>(c1min),static_cast<float>(c1max),static_cast<int>(r0),static_cast<int>(r1),
        static_cast<float>(wavelength),static_cast<float>(cell_area),static_cast<int>(seed),static_cast<float>(total_edge_length));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
