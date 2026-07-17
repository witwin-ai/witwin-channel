#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../tensor_checks.h"
#include "scattering_table.cuh"

namespace {

constexpr int kBlockSize = 256;
constexpr float kTwoPi = channel_native::scattering_tables::kTwoPi;
constexpr float kC0 = 299792458.0f;

using channel_native::scattering_tables::interp4;
using channel_native::scattering_tables::linear_axis;
using channel_native::scattering_tables::nearest_axis;
using channel_native::scattering_tables::positive_phi;

__device__ __forceinline__ float table_pdf(
    const float* __restrict__ density, int nti, int npi, int nto, int npo,
    const float* wi, const float* wo) {
    if (wi[2] <= 0.0f || wo[2] <= 0.0f) return 0.0f;
    const float phi_i = positive_phi(wi[1], wi[0]);
    float phi_o = positive_phi(wo[1], wo[0]);
    if (npi == 1) { phi_o -= phi_i; if (phi_o < 0.0f) phi_o += kTwoPi; }
    const int ti = nearest_axis(wi[2], nti, 1.0f, false);
    const int pi = npi == 1 ? 0 : nearest_axis(phi_i, npi, kTwoPi, true);
    const int to = nearest_axis(wo[2], nto, 1.0f, false);
    const int po = nearest_axis(phi_o, npo, kTwoPi, true);
    return density[((static_cast<int64_t>(ti) * npi + pi) * nto + to) * npo + po];
}

__global__ void scattering_eval_kernel(
    int64_t count, const float* __restrict__ wi, const float* __restrict__ wo,
    const float* __restrict__ fte, const float* __restrict__ ftm,
    int nti, int npi, int nto, int npo, float* out_te, float* out_tm) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        channel_native::scattering_tables::eval_te_tm(
            fte, ftm, nti, npi, nto, npo, wi + 3 * row, wo + 3 * row,
            out_te[row], out_tm[row]);
    }
}

__global__ void scattering_pdf_kernel(
    int64_t count, const float* wi, const float* wo, const float* density,
    int nti, int npi, int nto, int npo, bool reverse, float* out) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float* a = wi + 3 * row; const float* b = wo + 3 * row;
        out[row] = reverse ? table_pdf(density,nti,npi,nto,npo,b,a)
                           : table_pdf(density,nti,npi,nto,npo,a,b);
    }
}

__global__ void scattering_sample_kernel(
    int64_t count, const float* wi, const float* uniforms,
    const float* marginal, const float* conditional, const float* density,
    int nti, int npi, int nto, int npo,
    float* wo, float* pdf_fwd, float* pdf_rev) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float* a = wi + 3 * row;
        const float phi_i = positive_phi(a[1], a[0]);
        const int ti = nearest_axis(a[2], nti, 1.0f, false);
        const int pi = npi == 1 ? 0 : nearest_axis(phi_i, npi, kTwoPi, true);
        const float u1 = fminf(fmaxf(uniforms[2*row],0.0f),1.0f-1e-7f);
        const float u2 = fminf(fmaxf(uniforms[2*row+1],0.0f),1.0f-1e-7f);
        const int64_t mbase = (static_cast<int64_t>(ti)*npi+pi)*nto;
        int lo=0, hi=nto;
        while (lo<hi) { const int mid=(lo+hi)>>1; if (marginal[mbase+mid]<=u1) lo=mid+1; else hi=mid; }
        const int to=min(lo,nto-1);
        const float mlo = to ? marginal[mbase+to-1] : 0.0f;
        const float mhi = marginal[mbase+to];
        const float mf = mhi > mlo ? (u1-mlo)/(mhi-mlo) : 0.5f;
        const int64_t cbase = ((static_cast<int64_t>(ti)*npi+pi)*nto+to)*npo;
        lo=0; hi=npo;
        while (lo<hi) { const int mid=(lo+hi)>>1; if (conditional[cbase+mid]<=u2) lo=mid+1; else hi=mid; }
        const int po=min(lo,npo-1);
        const float clo = po ? conditional[cbase+po-1] : 0.0f;
        const float chi = conditional[cbase+po];
        const float pf = chi > clo ? (u2-clo)/(chi-clo) : 0.5f;
        const float cos_o = fminf(fmaxf((static_cast<float>(to)+mf)/nto,1e-6f),1.0f);
        float phi_o = (static_cast<float>(po)+pf)*(kTwoPi/npo);
        if (npi == 1) phi_o += phi_i;
        const float sin_o = sqrtf(fmaxf(0.0f,1.0f-cos_o*cos_o));
        float* b=wo+3*row; b[0]=sin_o*cosf(phi_o); b[1]=sin_o*sinf(phi_o); b[2]=cos_o;
        pdf_fwd[row]=density[cbase+po];
        pdf_rev[row]=table_pdf(density,nti,npi,nto,npo,b,a);
    }
}

__global__ void scattering_event_kernel(
    int64_t count, const float* cos_theta, const int* material_id,
    const float* cap_r_te, const float* cap_r_tm,
    const float* cap_t_te, const float* cap_t_tm,
    const float* rough_sigma, const int* scatter_model, int64_t material_count,
    float frequency, float probability_floor,
    float* p_scatter, float* p_transmit, float* coherent, bool* rough) {
    const float k0 = kTwoPi * frequency / kC0;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int mat=material_id[row];
        if (mat<0 || mat>=material_count || scatter_model[mat]!=1) {
            p_scatter[row]=p_transmit[row]=0.0f; coherent[row]=1.0f; rough[row]=false; continue;
        }
        const float c=expf(-2.0f*(k0*fabsf(cos_theta[row])*rough_sigma[mat])*(k0*fabsf(cos_theta[row])*rough_sigma[mat]));
        const float rbar=0.5f*(cap_r_te[row]+cap_r_tm[row]);
        const float tbar=0.5f*(cap_t_te[row]+cap_t_tm[row]);
        const float rcoh=rbar*c*c, rdiff=fmaxf(0.0f,rbar-rcoh);
        const float total=fmaxf(rcoh+rdiff+tbar,1e-12f);
        float pr=rcoh/total, ps=rdiff/total, pt=tbar/total;
        if (rcoh>0.0f) pr=fmaxf(pr,probability_floor);
        if (rdiff>0.0f) ps=fmaxf(ps,probability_floor);
        if (tbar>0.0f) pt=fmaxf(pt,probability_floor);
        const float norm=fmaxf(pr+ps+pt,1e-12f);
        p_scatter[row]=ps/norm; p_transmit[row]=pt/norm; coherent[row]=c; rough[row]=true;
    }
}

int blocks(int64_t n) { return static_cast<int>((n+kBlockSize-1)/kBlockSize); }

void check_table(const at::Tensor& t, const char* name) {
    channel_native::check_tensor(t,name,at::kFloat,4);
}

} // namespace

pybind11::dict cn_scattering_table_eval(at::Tensor wi, at::Tensor wo, at::Tensor fte, at::Tensor ftm) {
    channel_native::check_vec3_table(wi,"wi"); channel_native::check_vec3_table(wo,"wo");
    check_table(fte,"f_te"); check_table(ftm,"f_tm");
    TORCH_CHECK(wi.sizes()==wo.sizes(),"wi and wo shapes must match");
    TORCH_CHECK(fte.sizes()==ftm.sizes(),"f_te and f_tm shapes must match");
    TORCH_CHECK(wi.get_device()==fte.get_device() && wo.get_device()==wi.get_device() && ftm.get_device()==wi.get_device(),"scattering tensors must share device");
    auto te=at::empty({wi.size(0)},wi.options()), tm=at::empty_like(te);
    if (wi.size(0)>0) { auto s=at::cuda::getCurrentCUDAStream(wi.get_device()).stream(); scattering_eval_kernel<<<blocks(wi.size(0)),kBlockSize,0,s>>>(wi.size(0),wi.data_ptr<float>(),wo.data_ptr<float>(),fte.data_ptr<float>(),ftm.data_ptr<float>(),fte.size(0),fte.size(1),fte.size(2),fte.size(3),te.data_ptr<float>(),tm.data_ptr<float>()); C10_CUDA_KERNEL_LAUNCH_CHECK(); }
    pybind11::dict out; out["f_te"]=te; out["f_tm"]=tm; return out;
}

at::Tensor cn_scattering_table_pdf(at::Tensor wi, at::Tensor wo, at::Tensor density, bool reverse) {
    channel_native::check_vec3_table(wi,"wi"); channel_native::check_vec3_table(wo,"wo"); check_table(density,"sample_density");
    TORCH_CHECK(wi.sizes()==wo.sizes(),"wi and wo shapes must match"); TORCH_CHECK(wi.get_device()==density.get_device(),"scattering tensors must share device");
    auto out=at::empty({wi.size(0)},wi.options()); if(wi.size(0)>0){auto s=at::cuda::getCurrentCUDAStream(wi.get_device()).stream();scattering_pdf_kernel<<<blocks(wi.size(0)),kBlockSize,0,s>>>(wi.size(0),wi.data_ptr<float>(),wo.data_ptr<float>(),density.data_ptr<float>(),density.size(0),density.size(1),density.size(2),density.size(3),reverse,out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();} return out;
}

pybind11::dict cn_scattering_table_sample(at::Tensor wi, at::Tensor uniforms, at::Tensor marginal, at::Tensor conditional, at::Tensor density) {
    channel_native::check_vec3_table(wi,"wi"); channel_native::check_tensor(uniforms,"uniforms",at::kFloat,2); channel_native::check_tensor(marginal,"marginal_cdf",at::kFloat,3); check_table(conditional,"conditional_cdf"); check_table(density,"sample_density");
    TORCH_CHECK(uniforms.size(0)==wi.size(0)&&uniforms.size(1)==2,"uniforms must have shape (N,2)"); TORCH_CHECK(marginal.get_device()==wi.get_device()&&conditional.get_device()==wi.get_device()&&density.get_device()==wi.get_device()&&uniforms.get_device()==wi.get_device(),"scattering tensors must share device");
    auto wo=at::empty_like(wi), pf=at::empty({wi.size(0)},wi.options()), pr=at::empty_like(pf); if(wi.size(0)>0){auto s=at::cuda::getCurrentCUDAStream(wi.get_device()).stream();scattering_sample_kernel<<<blocks(wi.size(0)),kBlockSize,0,s>>>(wi.size(0),wi.data_ptr<float>(),uniforms.data_ptr<float>(),marginal.data_ptr<float>(),conditional.data_ptr<float>(),density.data_ptr<float>(),density.size(0),density.size(1),density.size(2),density.size(3),wo.data_ptr<float>(),pf.data_ptr<float>(),pr.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();} pybind11::dict out;out["wo"]=wo;out["pdf_forward"]=pf;out["pdf_reverse"]=pr;return out;
}

pybind11::dict cn_scattering_event_probabilities(at::Tensor cos_theta,at::Tensor material_id,at::Tensor cap_r_te,at::Tensor cap_r_tm,at::Tensor cap_t_te,at::Tensor cap_t_tm,at::Tensor rough_sigma,at::Tensor scatter_model,double frequency,double probability_floor){
    channel_native::check_flat_tensor(cos_theta,"cos_theta",at::kFloat);channel_native::check_flat_tensor(material_id,"material_id",at::kInt);channel_native::check_flat_tensor(cap_r_te,"cap_R_te",at::kFloat);channel_native::check_flat_tensor(cap_r_tm,"cap_R_tm",at::kFloat);channel_native::check_flat_tensor(cap_t_te,"cap_T_te",at::kFloat);channel_native::check_flat_tensor(cap_t_tm,"cap_T_tm",at::kFloat);channel_native::check_flat_tensor(rough_sigma,"rough_sigma_h_m",at::kFloat);channel_native::check_flat_tensor(scatter_model,"scatter_model_id",at::kInt);
    const int64_t count=cos_theta.size(0); TORCH_CHECK(material_id.size(0)==count&&cap_r_te.size(0)==count&&cap_r_tm.size(0)==count&&cap_t_te.size(0)==count&&cap_t_tm.size(0)==count,"per-ray scattering arrays must match cos_theta"); TORCH_CHECK(scatter_model.size(0)==rough_sigma.size(0),"material arrays must match");
    for(const auto&t:{material_id,cap_r_te,cap_r_tm,cap_t_te,cap_t_tm,rough_sigma,scatter_model})TORCH_CHECK(t.get_device()==cos_theta.get_device(),"scattering tensors must share device");
    auto ps=at::empty_like(cos_theta),pt=at::empty_like(cos_theta),cr=at::empty_like(cos_theta);auto rb=at::empty({count},cos_theta.options().dtype(at::kBool));if(count>0){auto s=at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();scattering_event_kernel<<<blocks(count),kBlockSize,0,s>>>(count,cos_theta.data_ptr<float>(),material_id.data_ptr<int>(),cap_r_te.data_ptr<float>(),cap_r_tm.data_ptr<float>(),cap_t_te.data_ptr<float>(),cap_t_tm.data_ptr<float>(),rough_sigma.data_ptr<float>(),scatter_model.data_ptr<int>(),rough_sigma.size(0),static_cast<float>(frequency),static_cast<float>(probability_floor),ps.data_ptr<float>(),pt.data_ptr<float>(),cr.data_ptr<float>(),rb.data_ptr<bool>());C10_CUDA_KERNEL_LAUNCH_CHECK();}pybind11::dict out;out["p_scatter"]=ps;out["p_transmit"]=pt;out["r_coh_amplitude"]=cr;out["rough"]=rb;return out;
}
