#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../tensor_checks.h"

namespace {

constexpr int kBlockSize = 256;
constexpr float kTwoPi = 6.2831853071795864769f;
constexpr float kC0 = 299792458.0f;

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

} // namespace

pybind11::dict channel_scattering_event_probabilities(at::Tensor cos_theta,at::Tensor material_id,at::Tensor cap_r_te,at::Tensor cap_r_tm,at::Tensor cap_t_te,at::Tensor cap_t_tm,at::Tensor rough_sigma,at::Tensor scatter_model,double frequency,double probability_floor){
    channel::check_flat_tensor(cos_theta,"cos_theta",at::kFloat);channel::check_flat_tensor(material_id,"material_id",at::kInt);channel::check_flat_tensor(cap_r_te,"cap_R_te",at::kFloat);channel::check_flat_tensor(cap_r_tm,"cap_R_tm",at::kFloat);channel::check_flat_tensor(cap_t_te,"cap_T_te",at::kFloat);channel::check_flat_tensor(cap_t_tm,"cap_T_tm",at::kFloat);channel::check_flat_tensor(rough_sigma,"rough_sigma_h_m",at::kFloat);channel::check_flat_tensor(scatter_model,"scatter_model_id",at::kInt);
    const int64_t count=cos_theta.size(0); TORCH_CHECK(material_id.size(0)==count&&cap_r_te.size(0)==count&&cap_r_tm.size(0)==count&&cap_t_te.size(0)==count&&cap_t_tm.size(0)==count,"per-ray scattering arrays must match cos_theta"); TORCH_CHECK(scatter_model.size(0)==rough_sigma.size(0),"material arrays must match");
    for(const auto&t:{material_id,cap_r_te,cap_r_tm,cap_t_te,cap_t_tm,rough_sigma,scatter_model})TORCH_CHECK(t.get_device()==cos_theta.get_device(),"scattering tensors must share device");
    auto ps=at::empty_like(cos_theta),pt=at::empty_like(cos_theta),cr=at::empty_like(cos_theta);auto rb=at::empty({count},cos_theta.options().dtype(at::kBool));if(count>0){auto s=at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();scattering_event_kernel<<<blocks(count),kBlockSize,0,s>>>(count,cos_theta.data_ptr<float>(),material_id.data_ptr<int>(),cap_r_te.data_ptr<float>(),cap_r_tm.data_ptr<float>(),cap_t_te.data_ptr<float>(),cap_t_tm.data_ptr<float>(),rough_sigma.data_ptr<float>(),scatter_model.data_ptr<int>(),rough_sigma.size(0),static_cast<float>(frequency),static_cast<float>(probability_floor),ps.data_ptr<float>(),pt.data_ptr<float>(),cr.data_ptr<float>(),rb.data_ptr<bool>());C10_CUDA_KERNEL_LAUNCH_CHECK();}pybind11::dict out;out["p_scatter"]=ps;out["p_transmit"]=pt;out["r_coh_amplitude"]=cr;out["rough"]=rb;return out;
}
