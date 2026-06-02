#include <cuda_runtime.h>

#include <common/cuda_check.h>
#include <trace/utd/utd_types.h>
#include <trace/utd/utd_math.h>
#include <trace/utd/utd_jvp.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

UTD_DINLINE PairInputs load_pair_inputs_jvp(
    int sIdx,
    const float* epx, const float* epy, const float* epz,
    const float* edx, const float* edy, const float* edz,
    const float* n0x, const float* n0y, const float* n0z,
    const float* nnx, const float* nny, const float* nnz,
    const float* wn,
    const float* spx, const float* spy, const float* spz,
    const float* ifr, const float* ifi,
    const float* inr, const float* ini,
    const float* r0r, const float* r0i,
    const float* rnr, const float* rni,
    const float* vxr, const float* vxi,
    const float* vyr, const float* vyi,
    const float* vzr, const float* vzi,
    const float* dxr, const float* dxi,
    const float* dyr, const float* dyi,
    const float* dzr, const float* dzi,
    const float* jur, const float* jui,
    const float* jvr, const float* jvi,
    const float* djur, const float* djui,
    const float* djvr, const float* djvi,
    const float* bux, const float* buy, const float* buz,
    const float* bvx, const float* bvy, const float* bvz,
    const float* bkx, const float* bky, const float* bkz,
    const float* f0m00r, const float* f0m00i,
    const float* f0m01r, const float* f0m01i,
    const float* f0m10r, const float* f0m10i,
    const float* f0m11r, const float* f0m11i,
    const float* f1m00r, const float* f1m00i,
    const float* f1m01r, const float* f1m01i,
    const float* f1m10r, const float* f1m10i,
    const float* f1m11r, const float* f1m11i,
    const float* f0er, const float* f0sg, const float* f0g, const float* f0uf, const float* f0pr,
    const float* f1er, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr)
{
    PairInputs p;
    p.edgePos   = make_f3(epx[sIdx], epy[sIdx], epz[sIdx]);
    p.edgeDir   = make_f3(edx[sIdx], edy[sIdx], edz[sIdx]);
    p.n0        = make_f3(n0x[sIdx], n0y[sIdx], n0z[sIdx]);
    p.nn        = make_f3(nnx[sIdx], nny[sIdx], nnz[sIdx]);
    p.wedgeN    = wn[sIdx];
    p.sourcePos = make_f3(spx[sIdx], spy[sIdx], spz[sIdx]);
    p.incidentField = cplx(ifr[sIdx], ifi[sIdx]);
    p.incidentNormalDerivative = cplx(inr[sIdx], ini[sIdx]);
    p.r0 = cplx(r0r[sIdx], r0i[sIdx]);
    p.rn = cplx(rnr[sIdx], rni[sIdx]);
    p.incidentVector = {
        cplx(vxr[sIdx], vxi[sIdx]),
        cplx(vyr[sIdx], vyi[sIdx]),
        cplx(vzr[sIdx], vzi[sIdx]),
    };
    p.incidentDerivativeVector = {
        cplx(dxr[sIdx], dxi[sIdx]),
        cplx(dyr[sIdx], dyi[sIdx]),
        cplx(dzr[sIdx], dzi[sIdx]),
    };
    p.incidentJones = {cplx(jur[sIdx], jui[sIdx]), cplx(jvr[sIdx], jvi[sIdx])};
    p.incidentDerivativeJones = {cplx(djur[sIdx], djui[sIdx]), cplx(djvr[sIdx], djvi[sIdx])};
    p.incidentBasis = {
        make_f3(bux[sIdx], buy[sIdx], buz[sIdx]),
        make_f3(bvx[sIdx], bvy[sIdx], bvz[sIdx]),
        make_f3(bkx[sIdx], bky[sIdx], bkz[sIdx]),
    };
    p.face0Operator = {
        cplx(f0m00r[sIdx], f0m00i[sIdx]),
        cplx(f0m01r[sIdx], f0m01i[sIdx]),
        cplx(f0m10r[sIdx], f0m10i[sIdx]),
        cplx(f0m11r[sIdx], f0m11i[sIdx]),
    };
    p.face1Operator = {
        cplx(f1m00r[sIdx], f1m00i[sIdx]),
        cplx(f1m01r[sIdx], f1m01i[sIdx]),
        cplx(f1m10r[sIdx], f1m10i[sIdx]),
        cplx(f1m11r[sIdx], f1m11i[sIdx]),
    };
    p.face0Material = {f0er[sIdx], f0sg[sIdx], f0g[sIdx], f0uf[sIdx], f0pr[sIdx]};
    p.face1Material = {f1er[sIdx], f1sg[sIdx], f1g[sIdx], f1uf[sIdx], f1pr[sIdx]};
    return p;
}

UTD_DINLINE PairInputsGrad load_pair_tangent_jvp(
    int sIdx,
    const float* t_epx, const float* t_epy, const float* t_epz,
    const float* t_edx, const float* t_edy, const float* t_edz,
    const float* t_n0x, const float* t_n0y, const float* t_n0z,
    const float* t_nnx, const float* t_nny, const float* t_nnz,
    const float* t_wn,
    const float* t_spx, const float* t_spy, const float* t_spz,
    const float* t_r0r, const float* t_r0i,
    const float* t_rnr, const float* t_rni,
    const float* t_vxr, const float* t_vxi,
    const float* t_vyr, const float* t_vyi,
    const float* t_vzr, const float* t_vzi,
    const float* t_dxr, const float* t_dxi,
    const float* t_dyr, const float* t_dyi,
    const float* t_dzr, const float* t_dzi,
    const float* t_f0m00r, const float* t_f0m00i,
    const float* t_f0m01r, const float* t_f0m01i,
    const float* t_f0m10r, const float* t_f0m10i,
    const float* t_f0m11r, const float* t_f0m11i,
    const float* t_f1m00r, const float* t_f1m00i,
    const float* t_f1m01r, const float* t_f1m01i,
    const float* t_f1m10r, const float* t_f1m10i,
    const float* t_f1m11r, const float* t_f1m11i,
    const float* t_f0etaR, const float* t_f0sigma, const float* t_f0gain,
    const float* t_f1etaR, const float* t_f1sigma, const float* t_f1gain)
{
    PairInputsGrad g = pig_zero();
    g.edgePos = make_f3(t_epx[sIdx], t_epy[sIdx], t_epz[sIdx]);
    g.edgeDir = make_f3(t_edx[sIdx], t_edy[sIdx], t_edz[sIdx]);
    g.n0 = make_f3(t_n0x[sIdx], t_n0y[sIdx], t_n0z[sIdx]);
    g.nn = make_f3(t_nnx[sIdx], t_nny[sIdx], t_nnz[sIdx]);
    g.wedgeN = t_wn[sIdx];
    g.sourcePos = make_f3(t_spx[sIdx], t_spy[sIdx], t_spz[sIdx]);
    g.r0 = cplx(t_r0r[sIdx], t_r0i[sIdx]);
    g.rn = cplx(t_rnr[sIdx], t_rni[sIdx]);
    g.incidentVector = {
        cplx(t_vxr[sIdx], t_vxi[sIdx]),
        cplx(t_vyr[sIdx], t_vyi[sIdx]),
        cplx(t_vzr[sIdx], t_vzi[sIdx]),
    };
    g.incidentDerivativeVector = {
        cplx(t_dxr[sIdx], t_dxi[sIdx]),
        cplx(t_dyr[sIdx], t_dyi[sIdx]),
        cplx(t_dzr[sIdx], t_dzi[sIdx]),
    };
    g.face0Operator = {
        cplx(t_f0m00r[sIdx], t_f0m00i[sIdx]),
        cplx(t_f0m01r[sIdx], t_f0m01i[sIdx]),
        cplx(t_f0m10r[sIdx], t_f0m10i[sIdx]),
        cplx(t_f0m11r[sIdx], t_f0m11i[sIdx]),
    };
    g.face1Operator = {
        cplx(t_f1m00r[sIdx], t_f1m00i[sIdx]),
        cplx(t_f1m01r[sIdx], t_f1m01i[sIdx]),
        cplx(t_f1m10r[sIdx], t_f1m10i[sIdx]),
        cplx(t_f1m11r[sIdx], t_f1m11i[sIdx]),
    };
    g.face0Material = {t_f0etaR[sIdx], t_f0sigma[sIdx], t_f0gain[sIdx], 0.f, 0.f};
    g.face1Material = {t_f1etaR[sIdx], t_f1sigma[sIdx], t_f1gain[sIdx], 0.f, 0.f};
    return g;
}

UTD_DINLINE float pair_input_tangent_dot(
    const PairInputsGrad& grad_state,
    const PairInputsGrad& tangent_state,
    float3a grad_rx,
    float3a tangent_rx)
{
    float result = 0.f;
    result += f3_dot(grad_state.edgePos, tangent_state.edgePos);
    result += f3_dot(grad_state.edgeDir, tangent_state.edgeDir);
    result += f3_dot(grad_state.n0, tangent_state.n0);
    result += f3_dot(grad_state.nn, tangent_state.nn);
    result += grad_state.wedgeN * tangent_state.wedgeN;
    result += f3_dot(grad_state.sourcePos, tangent_state.sourcePos);
    result += cplx_adj_dot(grad_state.r0, tangent_state.r0);
    result += cplx_adj_dot(grad_state.rn, tangent_state.rn);
    result += cplx_adj_dot(grad_state.incidentVector.x, tangent_state.incidentVector.x);
    result += cplx_adj_dot(grad_state.incidentVector.y, tangent_state.incidentVector.y);
    result += cplx_adj_dot(grad_state.incidentVector.z, tangent_state.incidentVector.z);
    result += cplx_adj_dot(grad_state.incidentDerivativeVector.x, tangent_state.incidentDerivativeVector.x);
    result += cplx_adj_dot(grad_state.incidentDerivativeVector.y, tangent_state.incidentDerivativeVector.y);
    result += cplx_adj_dot(grad_state.incidentDerivativeVector.z, tangent_state.incidentDerivativeVector.z);
    result += cplx_adj_dot(grad_state.face0Operator.m00, tangent_state.face0Operator.m00);
    result += cplx_adj_dot(grad_state.face0Operator.m01, tangent_state.face0Operator.m01);
    result += cplx_adj_dot(grad_state.face0Operator.m10, tangent_state.face0Operator.m10);
    result += cplx_adj_dot(grad_state.face0Operator.m11, tangent_state.face0Operator.m11);
    result += cplx_adj_dot(grad_state.face1Operator.m00, tangent_state.face1Operator.m00);
    result += cplx_adj_dot(grad_state.face1Operator.m01, tangent_state.face1Operator.m01);
    result += cplx_adj_dot(grad_state.face1Operator.m10, tangent_state.face1Operator.m10);
    result += cplx_adj_dot(grad_state.face1Operator.m11, tangent_state.face1Operator.m11);
    result += grad_state.face0Material.etaR * tangent_state.face0Material.etaR;
    result += grad_state.face0Material.sigma * tangent_state.face0Material.sigma;
    result += grad_state.face0Material.gain * tangent_state.face0Material.gain;
    result += grad_state.face1Material.etaR * tangent_state.face1Material.etaR;
    result += grad_state.face1Material.sigma * tangent_state.face1Material.sigma;
    result += grad_state.face1Material.gain * tangent_state.face1Material.gain;
    result += f3_dot(grad_rx, tangent_rx);
    return result;
}

UTD_DINLINE Complex3 basis_output_seed(int component) {
    Complex3 seed = c3_zero();
    if (component == 0) {
        seed.x.re = 1.f;
    } else if (component == 1) {
        seed.x.im = 1.f;
    } else if (component == 2) {
        seed.y.re = 1.f;
    } else if (component == 3) {
        seed.y.im = 1.f;
    } else if (component == 4) {
        seed.z.re = 1.f;
    } else {
        seed.z.im = 1.f;
    }
    return seed;
}

__global__ void utd_accumulate_jvp_kernel(
    const int* __restrict__ stateIdx,
    const int* __restrict__ rxIdx,
    const int* __restrict__ ownerCode,
    const float* __restrict__ epx, const float* __restrict__ epy, const float* __restrict__ epz,
    const float* __restrict__ edx, const float* __restrict__ edy, const float* __restrict__ edz,
    const float* __restrict__ n0x, const float* __restrict__ n0y, const float* __restrict__ n0z,
    const float* __restrict__ nnx, const float* __restrict__ nny, const float* __restrict__ nnz,
    const float* __restrict__ wn,
    const float* __restrict__ spx, const float* __restrict__ spy, const float* __restrict__ spz,
    const float* __restrict__ ifr, const float* __restrict__ ifi,
    const float* __restrict__ inr, const float* __restrict__ ini,
    const float* __restrict__ r0r, const float* __restrict__ r0i,
    const float* __restrict__ rnr, const float* __restrict__ rni,
    const float* __restrict__ vxr, const float* __restrict__ vxi,
    const float* __restrict__ vyr, const float* __restrict__ vyi,
    const float* __restrict__ vzr, const float* __restrict__ vzi,
    const float* __restrict__ dxr, const float* __restrict__ dxi,
    const float* __restrict__ dyr, const float* __restrict__ dyi,
    const float* __restrict__ dzr, const float* __restrict__ dzi,
    const float* __restrict__ jur, const float* __restrict__ jui,
    const float* __restrict__ jvr, const float* __restrict__ jvi,
    const float* __restrict__ djur, const float* __restrict__ djui,
    const float* __restrict__ djvr, const float* __restrict__ djvi,
    const float* __restrict__ bux, const float* __restrict__ buy, const float* __restrict__ buz,
    const float* __restrict__ bvx, const float* __restrict__ bvy, const float* __restrict__ bvz,
    const float* __restrict__ bkx, const float* __restrict__ bky, const float* __restrict__ bkz,
    const float* __restrict__ f0m00r, const float* __restrict__ f0m00i,
    const float* __restrict__ f0m01r, const float* __restrict__ f0m01i,
    const float* __restrict__ f0m10r, const float* __restrict__ f0m10i,
    const float* __restrict__ f0m11r, const float* __restrict__ f0m11i,
    const float* __restrict__ f1m00r, const float* __restrict__ f1m00i,
    const float* __restrict__ f1m01r, const float* __restrict__ f1m01i,
    const float* __restrict__ f1m10r, const float* __restrict__ f1m10i,
    const float* __restrict__ f1m11r, const float* __restrict__ f1m11i,
    const float* __restrict__ f0er, const float* __restrict__ f0sg,
    const float* __restrict__ f0g,  const float* __restrict__ f0uf,
    const float* __restrict__ f0pr,
    const float* __restrict__ f1er, const float* __restrict__ f1sg,
    const float* __restrict__ f1g,  const float* __restrict__ f1uf,
    const float* __restrict__ f1pr,
    const float* __restrict__ rxx, const float* __restrict__ rxy, const float* __restrict__ rxz,
    const float* __restrict__ t_epx, const float* __restrict__ t_epy, const float* __restrict__ t_epz,
    const float* __restrict__ t_edx, const float* __restrict__ t_edy, const float* __restrict__ t_edz,
    const float* __restrict__ t_n0x, const float* __restrict__ t_n0y, const float* __restrict__ t_n0z,
    const float* __restrict__ t_nnx, const float* __restrict__ t_nny, const float* __restrict__ t_nnz,
    const float* __restrict__ t_wn,
    const float* __restrict__ t_spx, const float* __restrict__ t_spy, const float* __restrict__ t_spz,
    const float* __restrict__ t_ifr, const float* __restrict__ t_ifi,
    const float* __restrict__ t_inr, const float* __restrict__ t_ini,
    const float* __restrict__ t_r0r, const float* __restrict__ t_r0i,
    const float* __restrict__ t_rnr, const float* __restrict__ t_rni,
    const float* __restrict__ t_vxr, const float* __restrict__ t_vxi,
    const float* __restrict__ t_vyr, const float* __restrict__ t_vyi,
    const float* __restrict__ t_vzr, const float* __restrict__ t_vzi,
    const float* __restrict__ t_dxr, const float* __restrict__ t_dxi,
    const float* __restrict__ t_dyr, const float* __restrict__ t_dyi,
    const float* __restrict__ t_dzr, const float* __restrict__ t_dzi,
    const float* __restrict__ t_f0m00r, const float* __restrict__ t_f0m00i,
    const float* __restrict__ t_f0m01r, const float* __restrict__ t_f0m01i,
    const float* __restrict__ t_f0m10r, const float* __restrict__ t_f0m10i,
    const float* __restrict__ t_f0m11r, const float* __restrict__ t_f0m11i,
    const float* __restrict__ t_f1m00r, const float* __restrict__ t_f1m00i,
    const float* __restrict__ t_f1m01r, const float* __restrict__ t_f1m01i,
    const float* __restrict__ t_f1m10r, const float* __restrict__ t_f1m10i,
    const float* __restrict__ t_f1m11r, const float* __restrict__ t_f1m11i,
    const float* __restrict__ t_f0etaR, const float* __restrict__ t_f0sigma, const float* __restrict__ t_f0gain,
    const float* __restrict__ t_f1etaR, const float* __restrict__ t_f1sigma, const float* __restrict__ t_f1gain,
    const float* __restrict__ t_rxx, const float* __restrict__ t_rxy, const float* __restrict__ t_rxz,
    float* __restrict__ to_dvxr, float* __restrict__ to_dvxi,
    float* __restrict__ to_dvyr, float* __restrict__ to_dvyi,
    float* __restrict__ to_dvzr, float* __restrict__ to_dvzi,
    float* __restrict__ to_mvxr, float* __restrict__ to_mvxi,
    float* __restrict__ to_mvyr, float* __restrict__ to_mvyi,
    float* __restrict__ to_mvzr, float* __restrict__ to_mvzi,
    int nPairs, float k, MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs)
        return;

    int sI = stateIdx[tid];
    int rI = rxIdx[tid];
    int own = ownerCode[sI];
    if (own != OWNERSHIP_DIRECT && own != OWNERSHIP_MIXED)
        return;

    (void) t_ifr;
    (void) t_ifi;
    (void) t_inr;
    (void) t_ini;

    PairInputs pi = load_pair_inputs_jvp(
        sI,
        epx, epy, epz,
        edx, edy, edz,
        n0x, n0y, n0z,
        nnx, nny, nnz,
        wn,
        spx, spy, spz,
        ifr, ifi,
        inr, ini,
        r0r, r0i,
        rnr, rni,
        vxr, vxi, vyr, vyi, vzr, vzi,
        dxr, dxi, dyr, dyi, dzr, dzi,
        jur, jui, jvr, jvi, djur, djui, djvr, djvi,
        bux, buy, buz, bvx, bvy, bvz, bkx, bky, bkz,
        f0m00r, f0m00i, f0m01r, f0m01i, f0m10r, f0m10i, f0m11r, f0m11i,
        f1m00r, f1m00i, f1m01r, f1m01i, f1m10r, f1m10i, f1m11r, f1m11i,
        f0er, f0sg, f0g, f0uf, f0pr,
        f1er, f1sg, f1g, f1uf, f1pr
    );
    PairInputsGrad tangent_state = load_pair_tangent_jvp(
        sI,
        t_epx, t_epy, t_epz,
        t_edx, t_edy, t_edz,
        t_n0x, t_n0y, t_n0z,
        t_nnx, t_nny, t_nnz,
        t_wn,
        t_spx, t_spy, t_spz,
        t_r0r, t_r0i,
        t_rnr, t_rni,
        t_vxr, t_vxi, t_vyr, t_vyi, t_vzr, t_vzi,
        t_dxr, t_dxi, t_dyr, t_dyi, t_dzr, t_dzi
        , t_f0m00r, t_f0m00i, t_f0m01r, t_f0m01i, t_f0m10r, t_f0m10i, t_f0m11r, t_f0m11i
        , t_f1m00r, t_f1m00i, t_f1m01r, t_f1m01i, t_f1m10r, t_f1m10i, t_f1m11r, t_f1m11i
        , t_f0etaR, t_f0sigma, t_f0gain, t_f1etaR, t_f1sigma, t_f1gain
    );
    float3a tgtPos = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    float3a tangent_rx = make_f3(t_rxx[rI], t_rxy[rI], t_rxz[rI]);

    float* out_re[3];
    float* out_im[3];
    if (own == OWNERSHIP_DIRECT) {
        out_re[0] = to_dvxr; out_im[0] = to_dvxi;
        out_re[1] = to_dvyr; out_im[1] = to_dvyi;
        out_re[2] = to_dvzr; out_im[2] = to_dvzi;
    } else {
        out_re[0] = to_mvxr; out_im[0] = to_mvxi;
        out_re[1] = to_mvyr; out_im[1] = to_mvyi;
        out_re[2] = to_mvzr; out_im[2] = to_mvzi;
    }

    for (int component = 0; component < 6; ++component) {
        PairInputsGrad grad_state = pig_zero();
        float3a grad_rx = f3_zero();
        pair_vector_output_vjp(pi, tgtPos, k, mat, basis_output_seed(component), grad_state, grad_rx);
        float tangent_value = pair_input_tangent_dot(grad_state, tangent_state, grad_rx, tangent_rx);
        int axis = component / 2;
        if ((component & 1) == 0)
            atomicAdd(&out_re[axis][rI], tangent_value);
        else
            atomicAdd(&out_im[axis][rI], tangent_value);
    }
}

} // anonymous namespace

void utd_accumulate_jvp(
    const int* state_index, const int* rx_index, const int* ownership_code,
    const float* epx, const float* epy, const float* epz,
    const float* edx, const float* edy, const float* edz,
    const float* n0x, const float* n0y, const float* n0z,
    const float* nnx, const float* nny, const float* nnz,
    const float* wn,
    const float* spx, const float* spy, const float* spz,
    const float* ifr, const float* ifi,
    const float* inr, const float* ini,
    const float* r0r, const float* r0i,
    const float* rnr, const float* rni,
    const float* vxr, const float* vxi, const float* vyr, const float* vyi,
    const float* vzr, const float* vzi,
    const float* dxr, const float* dxi, const float* dyr, const float* dyi,
    const float* dzr, const float* dzi,
    const float* jur, const float* jui, const float* jvr, const float* jvi,
    const float* djur, const float* djui, const float* djvr, const float* djvi,
    const float* bux, const float* buy, const float* buz,
    const float* bvx, const float* bvy, const float* bvz,
    const float* bkx, const float* bky, const float* bkz,
    const float* f0m00r, const float* f0m00i, const float* f0m01r, const float* f0m01i,
    const float* f0m10r, const float* f0m10i, const float* f0m11r, const float* f0m11i,
    const float* f1m00r, const float* f1m00i, const float* f1m01r, const float* f1m01i,
    const float* f1m10r, const float* f1m10i, const float* f1m11r, const float* f1m11i,
    const float* f0er, const float* f0sg, const float* f0g, const float* f0uf, const float* f0pr,
    const float* f1er, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
    const float* rxx, const float* rxy, const float* rxz,
    const float* t_edge_pos_x, const float* t_edge_pos_y, const float* t_edge_pos_z,
    const float* t_edge_dir_x, const float* t_edge_dir_y, const float* t_edge_dir_z,
    const float* t_n0_x, const float* t_n0_y, const float* t_n0_z,
    const float* t_nn_x, const float* t_nn_y, const float* t_nn_z,
    const float* t_wedge_n,
    const float* t_source_pos_x, const float* t_source_pos_y, const float* t_source_pos_z,
    const float* t_ifr, const float* t_ifi,
    const float* t_inr, const float* t_ini,
    const float* t_r0r, const float* t_r0i,
    const float* t_rnr, const float* t_rni,
    const float* t_vxr, const float* t_vxi, const float* t_vyr, const float* t_vyi,
    const float* t_vzr, const float* t_vzi,
    const float* t_dxr, const float* t_dxi, const float* t_dyr, const float* t_dyi,
    const float* t_dzr, const float* t_dzi,
    const float* t_f0m00r, const float* t_f0m00i, const float* t_f0m01r, const float* t_f0m01i,
    const float* t_f0m10r, const float* t_f0m10i, const float* t_f0m11r, const float* t_f0m11i,
    const float* t_f1m00r, const float* t_f1m00i, const float* t_f1m01r, const float* t_f1m01i,
    const float* t_f1m10r, const float* t_f1m10i, const float* t_f1m11r, const float* t_f1m11i,
    const float* t_f0etaR, const float* t_f0sigma, const float* t_f0gain,
    const float* t_f1etaR, const float* t_f1sigma, const float* t_f1gain,
    const float* t_rxx, const float* t_rxy, const float* t_rxz,
    float* to_dvxr, float* to_dvxi, float* to_dvyr, float* to_dvyi,
    float* to_dvzr, float* to_dvzi,
    float* to_mvxr, float* to_mvxi, float* to_mvyr, float* to_mvyi,
    float* to_mvzr, float* to_mvzi,
    int n_pairs, float k, MaterialParams material)
{
    if (n_pairs <= 0)
        return;

    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;

    utd_accumulate_jvp_kernel<<<grid, BLOCK>>>(
        state_index, rx_index, ownership_code,
        epx, epy, epz, edx, edy, edz, n0x, n0y, n0z, nnx, nny, nnz, wn,
        spx, spy, spz, ifr, ifi, inr, ini, r0r, r0i, rnr, rni,
        vxr, vxi, vyr, vyi, vzr, vzi, dxr, dxi, dyr, dyi, dzr, dzi,
        jur, jui, jvr, jvi, djur, djui, djvr, djvi,
        bux, buy, buz, bvx, bvy, bvz, bkx, bky, bkz,
        f0m00r, f0m00i, f0m01r, f0m01i, f0m10r, f0m10i, f0m11r, f0m11i,
        f1m00r, f1m00i, f1m01r, f1m01i, f1m10r, f1m10i, f1m11r, f1m11i,
        f0er, f0sg, f0g, f0uf, f0pr, f1er, f1sg, f1g, f1uf, f1pr,
        rxx, rxy, rxz,
        t_edge_pos_x, t_edge_pos_y, t_edge_pos_z,
        t_edge_dir_x, t_edge_dir_y, t_edge_dir_z,
        t_n0_x, t_n0_y, t_n0_z,
        t_nn_x, t_nn_y, t_nn_z,
        t_wedge_n,
        t_source_pos_x, t_source_pos_y, t_source_pos_z,
        t_ifr, t_ifi, t_inr, t_ini,
        t_r0r, t_r0i, t_rnr, t_rni,
        t_vxr, t_vxi, t_vyr, t_vyi, t_vzr, t_vzi,
        t_dxr, t_dxi, t_dyr, t_dyi, t_dzr, t_dzi,
        t_f0m00r, t_f0m00i, t_f0m01r, t_f0m01i, t_f0m10r, t_f0m10i, t_f0m11r, t_f0m11i,
        t_f1m00r, t_f1m00i, t_f1m01r, t_f1m01i, t_f1m10r, t_f1m10i, t_f1m11r, t_f1m11i,
        t_f0etaR, t_f0sigma, t_f0gain, t_f1etaR, t_f1sigma, t_f1gain,
        t_rxx, t_rxy, t_rxz,
        to_dvxr, to_dvxi, to_dvyr, to_dvyi, to_dvzr, to_dvzi,
        to_mvxr, to_mvxi, to_mvyr, to_mvyi, to_mvzr, to_mvzi,
        n_pairs, k, material
    );

    throw_cuda(cudaGetLastError(), "utd_accumulate_jvp_kernel launch");
}

} // namespace witwin::channel::native_ext
