#include <cuda_runtime.h>
#include <stdexcept>

#include <common/cuda_check.h>
#include <trace/utd/utd_types.h>
#include <trace/utd/utd_math.h>
#include <trace/utd/utd_accumulate.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

// -------------------------------------------------------------------------
// SoA â†?PairInputs loader  (one thread, one state index)
// -------------------------------------------------------------------------
__device__ __forceinline__ PairInputs load_pair_inputs(
    int sIdx,
    const float* epx, const float* epy, const float* epz,
    const float* edx, const float* edy, const float* edz,
    const float* n0x, const float* n0y, const float* n0z,
    const float* nnx, const float* nny, const float* nnz,
    const float* wn,
    const float* elm, const float* elx,
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
    p.edgeLineMin = elm != nullptr ? elm[sIdx] : nanf("");
    p.edgeLineMax = elx != nullptr ? elx[sIdx] : nanf("");
    p.sourcePos = make_f3(spx[sIdx], spy[sIdx], spz[sIdx]);
    p.incidentField = cplx(ifr[sIdx], ifi[sIdx]);
    p.incidentNormalDerivative = cplx(inr[sIdx], ini[sIdx]);
    p.r0 = cplx(r0r[sIdx], r0i[sIdx]);
    p.rn = cplx(rnr[sIdx], rni[sIdx]);
    p.incidentVector = {cplx(vxr[sIdx],vxi[sIdx]),
                        cplx(vyr[sIdx],vyi[sIdx]),
                        cplx(vzr[sIdx],vzi[sIdx])};
    p.incidentDerivativeVector = {cplx(dxr[sIdx],dxi[sIdx]),
                                  cplx(dyr[sIdx],dyi[sIdx]),
                                  cplx(dzr[sIdx],dzi[sIdx])};
    p.incidentJones = {cplx(jur[sIdx],jui[sIdx]), cplx(jvr[sIdx],jvi[sIdx])};
    p.incidentDerivativeJones = {cplx(djur[sIdx],djui[sIdx]), cplx(djvr[sIdx],djvi[sIdx])};
    p.incidentBasis = {make_f3(bux[sIdx],buy[sIdx],buz[sIdx]),
                       make_f3(bvx[sIdx],bvy[sIdx],bvz[sIdx]),
                       make_f3(bkx[sIdx],bky[sIdx],bkz[sIdx])};
    p.face0Operator = {cplx(f0m00r[sIdx],f0m00i[sIdx]), cplx(f0m01r[sIdx],f0m01i[sIdx]),
                       cplx(f0m10r[sIdx],f0m10i[sIdx]), cplx(f0m11r[sIdx],f0m11i[sIdx])};
    p.face1Operator = {cplx(f1m00r[sIdx],f1m00i[sIdx]), cplx(f1m01r[sIdx],f1m01i[sIdx]),
                       cplx(f1m10r[sIdx],f1m10i[sIdx]), cplx(f1m11r[sIdx],f1m11i[sIdx])};
    p.face0Material = {f0er[sIdx], f0sg[sIdx], f0g[sIdx], f0uf[sIdx], f0pr[sIdx]};
    p.face1Material = {f1er[sIdx], f1sg[sIdx], f1g[sIdx], f1uf[sIdx], f1pr[sIdx]};
    return p;
}

__device__ __forceinline__ PairInputs load_pair_inputs_from_slots(
    int sIdx,
    const float* const* slots)
{
    return load_pair_inputs(
        sIdx,
        slots[0], slots[1], slots[2],
        slots[3], slots[4], slots[5],
        slots[6], slots[7], slots[8],
        slots[9], slots[10], slots[11],
        slots[12],
        slots[13], slots[14],
        slots[15], slots[16], slots[17],
        slots[18], slots[19],
        slots[20], slots[21],
        slots[22], slots[23],
        slots[24], slots[25],
        slots[26], slots[27],
        slots[28], slots[29],
        slots[30], slots[31],
        slots[32], slots[33],
        slots[34], slots[35],
        slots[36], slots[37],
        slots[38], slots[39],
        slots[40], slots[41],
        slots[42], slots[43],
        slots[44], slots[45],
        slots[46], slots[47], slots[48],
        slots[49], slots[50], slots[51],
        slots[52], slots[53], slots[54],
        slots[55], slots[56],
        slots[57], slots[58],
        slots[59], slots[60],
        slots[61], slots[62],
        slots[63], slots[64],
        slots[65], slots[66],
        slots[67], slots[68],
        slots[69], slots[70],
        slots[71], slots[72], slots[73], slots[74], slots[75],
        slots[76], slots[77], slots[78], slots[79], slots[80]
    );
}

__device__ __forceinline__ PairContributionDebug debug_pair_contribution_device(
    PairInputs state,
    float3a target,
    float k,
    MaterialParams material)
{
    PairContributionDebug debug{};

    bool src_ext = wedge_exterior_mask(
        f3_sub(state.sourcePos, state.edgePos),
        state.edgeDir,
        state.n0,
        state.nn
    );
    bool tgt_ext = wedge_exterior_mask(
        f3_sub(target, state.edgePos),
        state.edgeDir,
        state.n0,
        state.nn
    );

    float phi = 0.f;
    float phi_prime = 0.f;
    float s = 0.f;
    float s_prime = 0.f;
    float sin_beta0 = 0.f;
    compute_edge_geometry_3d(
        state.sourcePos,
        state.edgePos,
        state.edgeDir,
        state.n0,
        target,
        phi,
        phi_prime,
        s,
        s_prime,
        sin_beta0
    );

    bool pole_safe = cot_pole_safe_mask(phi, phi_prime, state.wedgeN, 1.0e-6f);
    float safe_phi = pole_safe ? phi : 0.5f * state.wedgeN * UTD_PI;
    float safe_phi_prime = pole_safe ? phi_prime : 0.5f * state.wedgeN * UTD_PI;
    bool slope_safe = slope_safe_mask(safe_phi, safe_phi_prime, state.wedgeN, UTD_SLOPE_STEP);
    bool geom_valid = src_ext && tgt_ext && (s_prime > UTD_MIN_DISTANCE) && (s > UTD_MIN_DISTANCE);

    Complex field = cplx_zero();
    Complex direct_gain = cplx_zero();
    Complex derivative_gain = cplx_zero();
    bool field_geom_valid = false;
    compute_pair_field_terms(
        state,
        target,
        k,
        material,
        field_geom_valid,
        field,
        direct_gain,
        derivative_gain
    );

    debug.srcExt = src_ext ? 1 : 0;
    debug.tgtExt = tgt_ext ? 1 : 0;
    debug.poleSafe = pole_safe ? 1 : 0;
    debug.slopeSafe = slope_safe ? 1 : 0;
    debug.geomValid = (geom_valid && field_geom_valid) ? 1 : 0;
    debug.phi = phi;
    debug.phiPrime = phi_prime;
    debug.s = s;
    debug.sPrime = s_prime;
    debug.sinBeta0 = sin_beta0;
    debug.finiteFactor = finite_wedge_truncation_factor(state, target, k);
    debug.field = field;
    debug.directGain = direct_gain;
    debug.derivativeGain = derivative_gain;
    debug.vectorField = compute_pair_vector_contribution(state, target, k, material);
    return debug;
}

__global__ void utd_debug_pair_device_kernel(
    PairInputs state,
    float3a target,
    float k,
    MaterialParams material,
    PairContributionDebug* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    *output = debug_pair_contribution_device(state, target, k, material);
}

__global__ void utd_debug_pair_from_state_slots_kernel(
    const float* const* state_slots,
    int state_index,
    float3a target,
    float k,
    MaterialParams material,
    PairContributionDebug* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    PairInputs state = load_pair_inputs_from_slots(state_index, state_slots);
    *output = debug_pair_contribution_device(state, target, k, material);
}

__global__ void utd_debug_pair_outputs_device_kernel(
    PairInputs state,
    float3a target,
    float k,
    MaterialParams material,
    PairOutputs* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    *output = compute_pair_contribution(state, target, k, material);
}

// -------------------------------------------------------------------------
// Atomic scatter helpers
// -------------------------------------------------------------------------
__device__ __forceinline__ void atomic_add_pair_output(
    int ownership, int rIdx, PairOutputs out,
    float* dR, float* dI, float* mR, float* mI,
    float* dvxR, float* dvxI, float* dvyR, float* dvyI, float* dvzR, float* dvzI,
    float* mvxR, float* mvxI, float* mvyR, float* mvyI, float* mvzR, float* mvzI)
{
    if (ownership == OWNERSHIP_DIRECT) {
        atomicAdd(&dR[rIdx],   out.field.re);
        atomicAdd(&dI[rIdx],   out.field.im);
        atomicAdd(&dvxR[rIdx], out.vectorField.x.re);
        atomicAdd(&dvxI[rIdx], out.vectorField.x.im);
        atomicAdd(&dvyR[rIdx], out.vectorField.y.re);
        atomicAdd(&dvyI[rIdx], out.vectorField.y.im);
        atomicAdd(&dvzR[rIdx], out.vectorField.z.re);
        atomicAdd(&dvzI[rIdx], out.vectorField.z.im);
    } else if (ownership == OWNERSHIP_MIXED) {
        atomicAdd(&mR[rIdx],   out.field.re);
        atomicAdd(&mI[rIdx],   out.field.im);
        atomicAdd(&mvxR[rIdx], out.vectorField.x.re);
        atomicAdd(&mvxI[rIdx], out.vectorField.x.im);
        atomicAdd(&mvyR[rIdx], out.vectorField.y.re);
        atomicAdd(&mvyI[rIdx], out.vectorField.y.im);
        atomicAdd(&mvzR[rIdx], out.vectorField.z.re);
        atomicAdd(&mvzI[rIdx], out.vectorField.z.im);
    }
}

__device__ __forceinline__ Complex scalarize_pair_vector(
    Complex3 vectorField,
    float3a arrivalDir,
    float3a rxPol)
{
    float3a basis = stable_perp_basis(arrivalDir, rxPol);
    return cplx_dot_real(vectorField, basis);
}

__device__ __forceinline__ void atomic_add_scalar_power_output(
    int rIdx,
    Complex coeff,
    float* coherentRe,
    float* coherentIm,
    float* power,
    float* validPairCount)
{
    atomicAdd(&coherentRe[rIdx], coeff.re);
    atomicAdd(&coherentIm[rIdx], coeff.im);
    atomicAdd(&power[rIdx], cplx_abs_sqr(coeff));
    atomicAdd(validPairCount, 1.f);
}

// -------------------------------------------------------------------------
// Load upstream gradient for a (ownership, rIdx) pair
// -------------------------------------------------------------------------
__device__ __forceinline__ PairOutputs load_output_grad(
    int ownership, int rIdx,
    const float* gdR, const float* gdI, const float* gmR, const float* gmI,
    const float* gdvxR, const float* gdvxI, const float* gdvyR, const float* gdvyI,
    const float* gdvzR, const float* gdvzI,
    const float* gmvxR, const float* gmvxI, const float* gmvyR, const float* gmvyI,
    const float* gmvzR, const float* gmvzI)
{
    PairOutputs g;
    if (ownership == OWNERSHIP_DIRECT) {
        g.field = cplx(gdR[rIdx], gdI[rIdx]);
        g.vectorField = {cplx(gdvxR[rIdx],gdvxI[rIdx]),
                         cplx(gdvyR[rIdx],gdvyI[rIdx]),
                         cplx(gdvzR[rIdx],gdvzI[rIdx])};
    } else if (ownership == OWNERSHIP_MIXED) {
        g.field = cplx(gmR[rIdx], gmI[rIdx]);
        g.vectorField = {cplx(gmvxR[rIdx],gmvxI[rIdx]),
                         cplx(gmvyR[rIdx],gmvyI[rIdx]),
                         cplx(gmvzR[rIdx],gmvzI[rIdx])};
    } else {
        g.field = cplx_zero();
        g.vectorField = c3_zero();
    }
    return g;
}

// Atomically scatter a PairInputsGrad back to state SoA gradient arrays
__device__ __forceinline__ void atomic_add_state_grad(
    int sIdx, PairInputsGrad g,
    float* gEpx, float* gEpy, float* gEpz,
    float* gEdx, float* gEdy, float* gEdz,
    float* gN0x, float* gN0y, float* gN0z,
    float* gNnx, float* gNny, float* gNnz,
    float* gWn,
    float* gSpx, float* gSpy, float* gSpz,
    float* gIfr, float* gIfi,
    float* gInr, float* gIni,
    float* gR0r, float* gR0i,
    float* gRnr, float* gRni,
    float* gVxr, float* gVxi, float* gVyr, float* gVyi, float* gVzr, float* gVzi,
    float* gDxr, float* gDxi, float* gDyr, float* gDyi, float* gDzr, float* gDzi,
    float* gF0m00r, float* gF0m00i, float* gF0m01r, float* gF0m01i,
    float* gF0m10r, float* gF0m10i, float* gF0m11r, float* gF0m11i,
    float* gF1m00r, float* gF1m00i, float* gF1m01r, float* gF1m01i,
    float* gF1m10r, float* gF1m10i, float* gF1m11r, float* gF1m11i,
    float* gF0etaR, float* gF0sigma, float* gF0gain,
    float* gF1etaR, float* gF1sigma, float* gF1gain)
{
    atomicAdd(&gEpx[sIdx], g.edgePos.x); atomicAdd(&gEpy[sIdx], g.edgePos.y); atomicAdd(&gEpz[sIdx], g.edgePos.z);
    atomicAdd(&gEdx[sIdx], g.edgeDir.x); atomicAdd(&gEdy[sIdx], g.edgeDir.y); atomicAdd(&gEdz[sIdx], g.edgeDir.z);
    atomicAdd(&gN0x[sIdx], g.n0.x); atomicAdd(&gN0y[sIdx], g.n0.y); atomicAdd(&gN0z[sIdx], g.n0.z);
    atomicAdd(&gNnx[sIdx], g.nn.x); atomicAdd(&gNny[sIdx], g.nn.y); atomicAdd(&gNnz[sIdx], g.nn.z);
    atomicAdd(&gWn[sIdx],  g.wedgeN);
    atomicAdd(&gSpx[sIdx], g.sourcePos.x); atomicAdd(&gSpy[sIdx], g.sourcePos.y); atomicAdd(&gSpz[sIdx], g.sourcePos.z);
    atomicAdd(&gIfr[sIdx], g.incidentField.re); atomicAdd(&gIfi[sIdx], g.incidentField.im);
    atomicAdd(&gInr[sIdx], g.incidentNormalDerivative.re); atomicAdd(&gIni[sIdx], g.incidentNormalDerivative.im);
    atomicAdd(&gR0r[sIdx], g.r0.re); atomicAdd(&gR0i[sIdx], g.r0.im);
    atomicAdd(&gRnr[sIdx], g.rn.re); atomicAdd(&gRni[sIdx], g.rn.im);
    atomicAdd(&gVxr[sIdx], g.incidentVector.x.re); atomicAdd(&gVxi[sIdx], g.incidentVector.x.im);
    atomicAdd(&gVyr[sIdx], g.incidentVector.y.re); atomicAdd(&gVyi[sIdx], g.incidentVector.y.im);
    atomicAdd(&gVzr[sIdx], g.incidentVector.z.re); atomicAdd(&gVzi[sIdx], g.incidentVector.z.im);
    atomicAdd(&gDxr[sIdx], g.incidentDerivativeVector.x.re); atomicAdd(&gDxi[sIdx], g.incidentDerivativeVector.x.im);
    atomicAdd(&gDyr[sIdx], g.incidentDerivativeVector.y.re); atomicAdd(&gDyi[sIdx], g.incidentDerivativeVector.y.im);
    atomicAdd(&gDzr[sIdx], g.incidentDerivativeVector.z.re); atomicAdd(&gDzi[sIdx], g.incidentDerivativeVector.z.im);
    atomicAdd(&gF0m00r[sIdx], g.face0Operator.m00.re); atomicAdd(&gF0m00i[sIdx], g.face0Operator.m00.im);
    atomicAdd(&gF0m01r[sIdx], g.face0Operator.m01.re); atomicAdd(&gF0m01i[sIdx], g.face0Operator.m01.im);
    atomicAdd(&gF0m10r[sIdx], g.face0Operator.m10.re); atomicAdd(&gF0m10i[sIdx], g.face0Operator.m10.im);
    atomicAdd(&gF0m11r[sIdx], g.face0Operator.m11.re); atomicAdd(&gF0m11i[sIdx], g.face0Operator.m11.im);
    atomicAdd(&gF1m00r[sIdx], g.face1Operator.m00.re); atomicAdd(&gF1m00i[sIdx], g.face1Operator.m00.im);
    atomicAdd(&gF1m01r[sIdx], g.face1Operator.m01.re); atomicAdd(&gF1m01i[sIdx], g.face1Operator.m01.im);
    atomicAdd(&gF1m10r[sIdx], g.face1Operator.m10.re); atomicAdd(&gF1m10i[sIdx], g.face1Operator.m10.im);
    atomicAdd(&gF1m11r[sIdx], g.face1Operator.m11.re); atomicAdd(&gF1m11i[sIdx], g.face1Operator.m11.im);
    atomicAdd(&gF0etaR[sIdx], g.face0Material.etaR);
    atomicAdd(&gF0sigma[sIdx], g.face0Material.sigma);
    atomicAdd(&gF0gain[sIdx], g.face0Material.gain);
    atomicAdd(&gF1etaR[sIdx], g.face1Material.etaR);
    atomicAdd(&gF1sigma[sIdx], g.face1Material.sigma);
    atomicAdd(&gF1gain[sIdx], g.face1Material.gain);
}

// =========================================================================
// FORWARD MEGA-KERNEL
// =========================================================================
__global__ void utd_accumulate_forward_kernel(
    const int* __restrict__ stateIdx,
    const int* __restrict__ rxIdx,
    const int* __restrict__ ownerCode,
    // state SoA (pointers passed through to load_pair_inputs)
    const float* __restrict__ epx, const float* __restrict__ epy, const float* __restrict__ epz,
    const float* __restrict__ edx, const float* __restrict__ edy, const float* __restrict__ edz,
    const float* __restrict__ n0x, const float* __restrict__ n0y, const float* __restrict__ n0z,
    const float* __restrict__ nnx, const float* __restrict__ nny, const float* __restrict__ nnz,
    const float* __restrict__ wn,
    const float* __restrict__ elm, const float* __restrict__ elx,
    const float* __restrict__ spx, const float* __restrict__ spy, const float* __restrict__ spz,
    const float* __restrict__ ifr, const float* __restrict__ ifi,
    const float* __restrict__ inr, const float* __restrict__ ini,
    const float* __restrict__ r0r, const float* __restrict__ r0i,
    const float* __restrict__ rnr, const float* __restrict__ rni,
    const float* __restrict__ vxr, const float* __restrict__ vxi,
    const float* __restrict__ vyr, const float* __restrict__ vyi,
    const float* __restrict__ vzr, const float* __restrict__ vzi,
    const float* __restrict__ dxr_, const float* __restrict__ dxi_,
    const float* __restrict__ dyr_, const float* __restrict__ dyi_,
    const float* __restrict__ dzr_, const float* __restrict__ dzi_,
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
    // rx
    const float* __restrict__ rxx, const float* __restrict__ rxy, const float* __restrict__ rxz,
    // output
    float* __restrict__ odr, float* __restrict__ odi,
    float* __restrict__ omr, float* __restrict__ omi,
    float* __restrict__ odvxr, float* __restrict__ odvxi,
    float* __restrict__ odvyr, float* __restrict__ odvyi,
    float* __restrict__ odvzr, float* __restrict__ odvzi,
    float* __restrict__ omvxr, float* __restrict__ omvxi,
    float* __restrict__ omvyr, float* __restrict__ omvyi,
    float* __restrict__ omvzr, float* __restrict__ omvzi,
    int nPairs, float k, MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;

    int sI = stateIdx[tid];
    int rI = rxIdx[tid];
    int own = ownerCode[sI];

    PairInputs pi = load_pair_inputs(sI,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr_,dxi_, dyr_,dyi_, dzr_,dzi_,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr);

    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    PairOutputs po = compute_pair_contribution(pi, tgt, k, mat);

    atomic_add_pair_output(own, rI, po,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi);
}

__global__ void utd_accumulate_tiled_forward_kernel(
    const int* __restrict__ stateLocalIdx,
    const int* __restrict__ rxLocalIdx,
    const int* __restrict__ validMask,
    const int* __restrict__ ownerCode,
    const float* __restrict__ epx, const float* __restrict__ epy, const float* __restrict__ epz,
    const float* __restrict__ edx, const float* __restrict__ edy, const float* __restrict__ edz,
    const float* __restrict__ n0x, const float* __restrict__ n0y, const float* __restrict__ n0z,
    const float* __restrict__ nnx, const float* __restrict__ nny, const float* __restrict__ nnz,
    const float* __restrict__ wn,
    const float* __restrict__ elm, const float* __restrict__ elx,
    const float* __restrict__ spx, const float* __restrict__ spy, const float* __restrict__ spz,
    const float* __restrict__ ifr, const float* __restrict__ ifi,
    const float* __restrict__ inr, const float* __restrict__ ini,
    const float* __restrict__ r0r, const float* __restrict__ r0i,
    const float* __restrict__ rnr, const float* __restrict__ rni,
    const float* __restrict__ vxr, const float* __restrict__ vxi,
    const float* __restrict__ vyr, const float* __restrict__ vyi,
    const float* __restrict__ vzr, const float* __restrict__ vzi,
    const float* __restrict__ dxr_, const float* __restrict__ dxi_,
    const float* __restrict__ dyr_, const float* __restrict__ dyi_,
    const float* __restrict__ dzr_, const float* __restrict__ dzi_,
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
    float* __restrict__ odr, float* __restrict__ odi,
    float* __restrict__ omr, float* __restrict__ omi,
    float* __restrict__ odvxr, float* __restrict__ odvxi,
    float* __restrict__ odvyr, float* __restrict__ odvyi,
    float* __restrict__ odvzr, float* __restrict__ odvzi,
    float* __restrict__ omvxr, float* __restrict__ omvxi,
    float* __restrict__ omvyr, float* __restrict__ omvyi,
    float* __restrict__ omvzr, float* __restrict__ omvzi,
    int nLocalStates, int nLocalRx, float k, MaterialParams mat)
{
    int nPairs = nLocalStates * nLocalRx;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;
    if (validMask != nullptr && validMask[tid] == 0) return;

    int localState = tid / nLocalRx;
    int localRx = tid - localState * nLocalRx;
    int sI = stateLocalIdx[localState];
    int rI = rxLocalIdx[localRx];
    int own = ownerCode[sI];

    PairInputs pi = load_pair_inputs(sI,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr_,dxi_, dyr_,dyi_, dzr_,dzi_,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr);

    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    PairOutputs po = compute_pair_contribution(pi, tgt, k, mat);

    atomic_add_pair_output(own, rI, po,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi);
}

__global__ void utd_accumulate_tiled_vector_power_forward_kernel(
    const int* __restrict__ stateLocalIdx,
    const int* __restrict__ rxLocalIdx,
    const int* __restrict__ validMask,
    const int* __restrict__ ownerCode,
    const float* __restrict__ epx, const float* __restrict__ epy, const float* __restrict__ epz,
    const float* __restrict__ edx, const float* __restrict__ edy, const float* __restrict__ edz,
    const float* __restrict__ n0x, const float* __restrict__ n0y, const float* __restrict__ n0z,
    const float* __restrict__ nnx, const float* __restrict__ nny, const float* __restrict__ nnz,
    const float* __restrict__ wn,
    const float* __restrict__ elm, const float* __restrict__ elx,
    const float* __restrict__ spx, const float* __restrict__ spy, const float* __restrict__ spz,
    const float* __restrict__ ifr, const float* __restrict__ ifi,
    const float* __restrict__ inr, const float* __restrict__ ini,
    const float* __restrict__ r0r, const float* __restrict__ r0i,
    const float* __restrict__ rnr, const float* __restrict__ rni,
    const float* __restrict__ vxr, const float* __restrict__ vxi,
    const float* __restrict__ vyr, const float* __restrict__ vyi,
    const float* __restrict__ vzr, const float* __restrict__ vzi,
    const float* __restrict__ dxr_, const float* __restrict__ dxi_,
    const float* __restrict__ dyr_, const float* __restrict__ dyi_,
    const float* __restrict__ dzr_, const float* __restrict__ dzi_,
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
    float* __restrict__ odr, float* __restrict__ odi,
    float* __restrict__ omr, float* __restrict__ omi,
    float* __restrict__ odvxr, float* __restrict__ odvxi,
    float* __restrict__ odvyr, float* __restrict__ odvyi,
    float* __restrict__ odvzr, float* __restrict__ odvzi,
    float* __restrict__ omvxr, float* __restrict__ omvxi,
    float* __restrict__ omvyr, float* __restrict__ omvyi,
    float* __restrict__ omvzr, float* __restrict__ omvzi,
    float* __restrict__ matchedPower,
    float* __restrict__ validPairCount,
    int nLocalStates, int nLocalRx, float k, float3a rxPol, MaterialParams mat)
{
    int nPairs = nLocalStates * nLocalRx;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;
    if (validMask != nullptr && validMask[tid] == 0) return;

    int localState = tid / nLocalRx;
    int localRx = tid - localState * nLocalRx;
    int sI = stateLocalIdx[localState];
    int rI = rxLocalIdx[localRx];
    int own = ownerCode[sI];

    PairInputs pi = load_pair_inputs(sI,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr_,dxi_, dyr_,dyi_, dzr_,dzi_,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr);

    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    PairOutputs po = compute_pair_contribution(pi, tgt, k, mat);
    float fieldNorm = cplx_abs_sqr(po.field);
    float vectorNorm =
        cplx_abs_sqr(po.vectorField.x) +
        cplx_abs_sqr(po.vectorField.y) +
        cplx_abs_sqr(po.vectorField.z);
    if (fieldNorm <= 0.0f && vectorNorm <= 0.0f) return;
    float3a arrivalDir = f3_sub(tgt, pi.edgePos);
    po.field = scalarize_pair_vector(po.vectorField, arrivalDir, rxPol);

    atomic_add_pair_output(own, rI, po,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi);
    atomicAdd(&matchedPower[rI], vectorNorm);
    atomicAdd(validPairCount, 1.0f);
}

__global__ void utd_accumulate_scalar_power_forward_kernel(
    const float* __restrict__ epx, const float* __restrict__ epy, const float* __restrict__ epz,
    const float* __restrict__ edx, const float* __restrict__ edy, const float* __restrict__ edz,
    const float* __restrict__ n0x, const float* __restrict__ n0y, const float* __restrict__ n0z,
    const float* __restrict__ nnx, const float* __restrict__ nny, const float* __restrict__ nnz,
    const float* __restrict__ wn,
    const float* __restrict__ elm, const float* __restrict__ elx,
    const float* __restrict__ spx, const float* __restrict__ spy, const float* __restrict__ spz,
    const float* __restrict__ ifr, const float* __restrict__ ifi,
    const float* __restrict__ inr, const float* __restrict__ ini,
    const float* __restrict__ r0r, const float* __restrict__ r0i,
    const float* __restrict__ rnr, const float* __restrict__ rni,
    const float* __restrict__ vxr, const float* __restrict__ vxi,
    const float* __restrict__ vyr, const float* __restrict__ vyi,
    const float* __restrict__ vzr, const float* __restrict__ vzi,
    const float* __restrict__ dxr_, const float* __restrict__ dxi_,
    const float* __restrict__ dyr_, const float* __restrict__ dyi_,
    const float* __restrict__ dzr_, const float* __restrict__ dzi_,
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
    const int* __restrict__ outputRxIdx,
    const float* __restrict__ pairRxx, const float* __restrict__ pairRxy, const float* __restrict__ pairRxz,
    float* __restrict__ coherentRe,
    float* __restrict__ coherentIm,
    float* __restrict__ power,
    float* __restrict__ validPairCount,
    int nPairs,
    float k,
    float3a rxPol,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;

    int rI = outputRxIdx[tid];
    PairInputs pi = load_pair_inputs(
        tid,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr_,dxi_, dyr_,dyi_, dzr_,dzi_,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr);

    float3a tgt = make_f3(pairRxx[tid], pairRxy[tid], pairRxz[tid]);
    PairOutputs pairOutput = compute_pair_contribution(pi, tgt, k, mat);
    float fieldNorm = cplx_abs_sqr(pairOutput.field);
    float vectorNorm =
        cplx_abs_sqr(pairOutput.vectorField.x) +
        cplx_abs_sqr(pairOutput.vectorField.y) +
        cplx_abs_sqr(pairOutput.vectorField.z);
    if (fieldNorm <= 0.0f && vectorNorm <= 0.0f) {
        return;
    }
    float3a arrivalDir = f3_sub(tgt, pi.edgePos);
    Complex scalarCoeff = scalarize_pair_vector(pairOutput.vectorField, arrivalDir, rxPol);
    atomic_add_scalar_power_output(
        rI,
        scalarCoeff,
        coherentRe,
        coherentIm,
        power,
        validPairCount);
}

__global__ void utd_accumulate_tiled_forward_slots_kernel(
    const int* __restrict__ stateLocalIdx,
    const int* __restrict__ rxLocalIdx,
    const int* __restrict__ validMask,
    const int* __restrict__ ownerCode,
    const float* const* __restrict__ stateSlots,
    const float* __restrict__ rxx,
    const float* __restrict__ rxy,
    const float* __restrict__ rxz,
    float* __restrict__ odr,
    float* __restrict__ odi,
    float* __restrict__ omr,
    float* __restrict__ omi,
    float* __restrict__ odvxr,
    float* __restrict__ odvxi,
    float* __restrict__ odvyr,
    float* __restrict__ odvyi,
    float* __restrict__ odvzr,
    float* __restrict__ odvzi,
    float* __restrict__ omvxr,
    float* __restrict__ omvxi,
    float* __restrict__ omvyr,
    float* __restrict__ omvyi,
    float* __restrict__ omvzr,
    float* __restrict__ omvzi,
    int nLocalStates,
    int nLocalReceivers,
    float k,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nPairs = nLocalStates * nLocalReceivers;
    if (tid >= nPairs) return;

    int localState = tid / nLocalReceivers;
    int localRx = tid - localState * nLocalReceivers;
    if (validMask != nullptr && validMask[tid] == 0) {
        return;
    }

    int sI = stateLocalIdx[localState];
    int rI = rxLocalIdx[localRx];
    int own = ownerCode[sI];
    PairInputs pi = load_pair_inputs_from_slots(sI, stateSlots);
    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    PairOutputs po = compute_pair_contribution(pi, tgt, k, mat);

    atomic_add_pair_output(own, rI, po,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi);
}

__global__ void utd_accumulate_tiled_vector_power_forward_slots_kernel(
    const int* __restrict__ stateLocalIdx,
    const int* __restrict__ rxLocalIdx,
    const int* __restrict__ validMask,
    const int* __restrict__ ownerCode,
    const float* const* __restrict__ stateSlots,
    const float* __restrict__ rxx,
    const float* __restrict__ rxy,
    const float* __restrict__ rxz,
    float* __restrict__ odr,
    float* __restrict__ odi,
    float* __restrict__ omr,
    float* __restrict__ omi,
    float* __restrict__ odvxr,
    float* __restrict__ odvxi,
    float* __restrict__ odvyr,
    float* __restrict__ odvyi,
    float* __restrict__ odvzr,
    float* __restrict__ odvzi,
    float* __restrict__ omvxr,
    float* __restrict__ omvxi,
    float* __restrict__ omvyr,
    float* __restrict__ omvyi,
    float* __restrict__ omvzr,
    float* __restrict__ omvzi,
    float* __restrict__ matchedPower,
    float* __restrict__ validPairCount,
    int nLocalStates,
    int nLocalReceivers,
    float k,
    float3a rxPol,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nPairs = nLocalStates * nLocalReceivers;
    if (tid >= nPairs) return;

    int localState = tid / nLocalReceivers;
    int localRx = tid - localState * nLocalReceivers;
    if (validMask != nullptr && validMask[tid] == 0) {
        return;
    }

    int sI = stateLocalIdx[localState];
    int rI = rxLocalIdx[localRx];
    int own = ownerCode[sI];
    PairInputs pi = load_pair_inputs_from_slots(sI, stateSlots);
    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    PairOutputs po = compute_pair_contribution(pi, tgt, k, mat);
    float fieldNorm = cplx_abs_sqr(po.field);
    float vectorNorm =
        cplx_abs_sqr(po.vectorField.x) +
        cplx_abs_sqr(po.vectorField.y) +
        cplx_abs_sqr(po.vectorField.z);
    if (fieldNorm <= 0.0f && vectorNorm <= 0.0f) {
        return;
    }

    float3a arrivalDir = f3_sub(tgt, pi.edgePos);
    po.field = scalarize_pair_vector(po.vectorField, arrivalDir, rxPol);

    atomic_add_pair_output(own, rI, po,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi);
    atomicAdd(&matchedPower[rI], vectorNorm);
    atomicAdd(validPairCount, 1.0f);
}

__global__ void utd_accumulate_scalar_power_forward_slots_kernel(
    const float* const* __restrict__ stateSlots,
    const int* __restrict__ outputRxIdx,
    const float* __restrict__ pairRxx,
    const float* __restrict__ pairRxy,
    const float* __restrict__ pairRxz,
    float* __restrict__ coherentRe,
    float* __restrict__ coherentIm,
    float* __restrict__ power,
    float* __restrict__ validPairCount,
    int nPairs,
    float k,
    float3a rxPol,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;

    int rI = outputRxIdx[tid];
    PairInputs pi = load_pair_inputs_from_slots(tid, stateSlots);
    float3a tgt = make_f3(pairRxx[tid], pairRxy[tid], pairRxz[tid]);
    PairOutputs pairOutput = compute_pair_contribution(pi, tgt, k, mat);
    float fieldNorm = cplx_abs_sqr(pairOutput.field);
    float vectorNorm =
        cplx_abs_sqr(pairOutput.vectorField.x) +
        cplx_abs_sqr(pairOutput.vectorField.y) +
        cplx_abs_sqr(pairOutput.vectorField.z);
    if (fieldNorm <= 0.0f && vectorNorm <= 0.0f) {
        return;
    }
    float3a arrivalDir = f3_sub(tgt, pi.edgePos);
    Complex scalarCoeff = scalarize_pair_vector(pairOutput.vectorField, arrivalDir, rxPol);
    atomic_add_scalar_power_output(
        rI,
        scalarCoeff,
        coherentRe,
        coherentIm,
        power,
        validPairCount);
}

inline const float** copy_state_slots_to_device(const float* const* stateSlots, size_t slotCount, const char* label) {
    const float** deviceSlots = nullptr;
    throw_cuda(cudaMalloc(&deviceSlots, slotCount * sizeof(const float*)), label);
    try {
        throw_cuda(
            cudaMemcpy(
                deviceSlots,
                stateSlots,
                slotCount * sizeof(const float*),
                cudaMemcpyHostToDevice
            ),
            label
        );
    } catch (...) {
        cudaFree(deviceSlots);
        throw;
    }
    return deviceSlots;
}

// =========================================================================
// BACKWARD MEGA-KERNEL
//
// Recomputes forward intermediates (recompute strategy, saves memory).
// Uses finite-difference based numerical VJP for the full vector field
// contribution, or a hand-written scalar adjoint for the scalar field
// component. For the first compilable version we use a pragmatic approach:
// accumulate scalar-field gradients analytically and vector-field gradients
// via the vector transport adjoint.
// =========================================================================
__global__ void utd_accumulate_backward_kernel(
    const int* __restrict__ stateIdx,
    const int* __restrict__ rxIdx,
    const int* __restrict__ ownerCode,
    // state SoA
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
    const float* __restrict__ dxr_, const float* __restrict__ dxi_,
    const float* __restrict__ dyr_, const float* __restrict__ dyi_,
    const float* __restrict__ dzr_, const float* __restrict__ dzi_,
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
    // rx
    const float* __restrict__ rxx, const float* __restrict__ rxy, const float* __restrict__ rxz,
    // upstream grads
    const float* __restrict__ gdR, const float* __restrict__ gdI,
    const float* __restrict__ gmR, const float* __restrict__ gmI,
    const float* __restrict__ gdvxR, const float* __restrict__ gdvxI,
    const float* __restrict__ gdvyR, const float* __restrict__ gdvyI,
    const float* __restrict__ gdvzR, const float* __restrict__ gdvzI,
    const float* __restrict__ gmvxR, const float* __restrict__ gmvxI,
    const float* __restrict__ gmvyR, const float* __restrict__ gmvyI,
    const float* __restrict__ gmvzR, const float* __restrict__ gmvzI,
    // state grad accumulators
    float* gEpx, float* gEpy, float* gEpz,
    float* gEdx, float* gEdy, float* gEdz,
    float* gN0x_, float* gN0y_, float* gN0z_,
    float* gNnx_, float* gNny_, float* gNnz_,
    float* gWn_,
    float* gSpx, float* gSpy, float* gSpz,
    float* gIfr_, float* gIfi_,
    float* gInr_, float* gIni_,
    float* gR0r_, float* gR0i_,
    float* gRnr_, float* gRni_,
    float* gVxr_, float* gVxi_,
    float* gVyr_, float* gVyi_,
    float* gVzr_, float* gVzi_,
    float* gDxr__, float* gDxi__,
    float* gDyr__, float* gDyi__,
    float* gDzr__, float* gDzi__,
    float* gF0m00r_, float* gF0m00i_,
    float* gF0m01r_, float* gF0m01i_,
    float* gF0m10r_, float* gF0m10i_,
    float* gF0m11r_, float* gF0m11i_,
    float* gF1m00r_, float* gF1m00i_,
    float* gF1m01r_, float* gF1m01i_,
    float* gF1m10r_, float* gF1m10i_,
    float* gF1m11r_, float* gF1m11i_,
    float* gF0etaR_, float* gF0sigma_, float* gF0gain_,
    float* gF1etaR_, float* gF1sigma_, float* gF1gain_,
    // rx grad accumulators
    float* gRxx, float* gRxy, float* gRxz,
    int nPairs, float k, MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;

    int sI = stateIdx[tid];
    int rI = rxIdx[tid];
    int own = ownerCode[sI];

    PairInputs pi = load_pair_inputs(sI,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, nullptr, nullptr,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr_,dxi_, dyr_,dyi_, dzr_,dzi_,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr);

    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);

    // Load upstream gradient
    PairOutputs gOut = load_output_grad(own, rI,
        gdR,gdI, gmR,gmI,
        gdvxR,gdvxI, gdvyR,gdvyI, gdvzR,gdvzI,
        gmvxR,gmvxI, gmvyR,gmvyI, gmvzR,gmvzI);

    // --- Scalar field backward (incidentField, incidentNormalDerivative) ---
    // Recompute forward scalars
    bool gv; Complex field_, dg_, dvg_;
    compute_pair_field_terms(pi, tgt, k, mat, gv, field_, dg_, dvg_);
    if (!gv) return;

    // grad(field) = gOut.field
    // field = incidentField*directGain + incidentNormalDerivative*derivativeGain
    // => grad(incidentField) += gOut.field * conj(directGain)?  No â€?adjoint of complex mul
    PairInputsGrad sg = pig_zero();
    float3a gTgt = f3_zero();

    // Scalar field adjoint: dL/d(incidentField) etc.
    // field = cplx_mul(incidentField, directGain) + cplx_mul(incidentNormDeriv, derivativeGain)
    Complex gIF = cplx_zero(), gDG = cplx_zero(), gIND = cplx_zero(), gDVG = cplx_zero();
    adj_cplx_mul(pi.incidentField, dg_, gOut.field, gIF, gDG);
    adj_cplx_mul(pi.incidentNormalDerivative, dvg_, gOut.field, gIND, gDVG);
    sg.incidentField = gIF;
    sg.incidentNormalDerivative = gIND;

    // --- Vector field backward ---
    pair_vector_output_vjp(pi, tgt, k, mat, gOut.vectorField, sg, gTgt);

    // Scatter gradients
    atomic_add_state_grad(sI, sg,
        gEpx,gEpy,gEpz, gEdx,gEdy,gEdz,
        gN0x_,gN0y_,gN0z_, gNnx_,gNny_,gNnz_, gWn_,
        gSpx,gSpy,gSpz, gIfr_,gIfi_, gInr_,gIni_,
        gR0r_,gR0i_, gRnr_,gRni_,
        gVxr_,gVxi_, gVyr_,gVyi_, gVzr_,gVzi_,
        gDxr__,gDxi__, gDyr__,gDyi__, gDzr__,gDzi__,
        gF0m00r_,gF0m00i_, gF0m01r_,gF0m01i_, gF0m10r_,gF0m10i_, gF0m11r_,gF0m11i_,
        gF1m00r_,gF1m00i_, gF1m01r_,gF1m01i_, gF1m10r_,gF1m10i_, gF1m11r_,gF1m11i_,
        gF0etaR_,gF0sigma_,gF0gain_, gF1etaR_,gF1sigma_,gF1gain_);
    atomicAdd(&gRxx[rI], gTgt.x);
    atomicAdd(&gRxy[rI], gTgt.y);
    atomicAdd(&gRxz[rI], gTgt.z);
}

} // anonymous namespace

// =========================================================================
// Host launcher: forward
// =========================================================================
void utd_accumulate_forward(
    const int* state_index, const int* rx_index, const int* ownership_code,
    const float* epx, const float* epy, const float* epz,
    const float* edx, const float* edy, const float* edz,
    const float* n0x, const float* n0y, const float* n0z,
    const float* nnx, const float* nny, const float* nnz,
    const float* wn,
    const float* elm, const float* elx,
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
    const float* f1er, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
    const float* rxx, const float* rxy, const float* rxz,
    float* odr, float* odi, float* omr, float* omi,
    float* odvxr, float* odvxi, float* odvyr, float* odvyi, float* odvzr, float* odvzi,
    float* omvxr, float* omvxi, float* omvyr, float* omvyi, float* omvzr, float* omvzi,
    int n_pairs, float k, MaterialParams material)
{
    if (n_pairs <= 0) return;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    utd_accumulate_forward_kernel<<<grid, BLOCK>>>(
        state_index, rx_index, ownership_code,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr,dxi, dyr,dyi, dzr,dzi,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr,
        rxx,rxy,rxz,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi,
        n_pairs, k, material);
    throw_cuda(cudaGetLastError(), "utd_accumulate_forward_kernel launch");
}

void utd_accumulate_tiled_forward(
    const int* state_index, const int* rx_index, const int* valid_mask, const int* ownership_code,
    const float* epx, const float* epy, const float* epz,
    const float* edx, const float* edy, const float* edz,
    const float* n0x, const float* n0y, const float* n0z,
    const float* nnx, const float* nny, const float* nnz,
    const float* wn,
    const float* elm, const float* elx,
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
    const float* f1er, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
    const float* rxx, const float* rxy, const float* rxz,
    float* odr, float* odi, float* omr, float* omi,
    float* odvxr, float* odvxi, float* odvyr, float* odvyi, float* odvzr, float* odvzi,
    float* omvxr, float* omvxi, float* omvyr, float* omvyi, float* omvzr, float* omvzi,
    int n_local_states, int n_local_receivers, float k, MaterialParams material)
{
    int n_pairs = n_local_states * n_local_receivers;
    if (n_pairs <= 0) return;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    utd_accumulate_tiled_forward_kernel<<<grid, BLOCK>>>(
        state_index, rx_index, valid_mask, ownership_code,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr,dxi, dyr,dyi, dzr,dzi,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr,
        rxx,rxy,rxz,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi,
        n_local_states, n_local_receivers, k, material);
    throw_cuda(cudaGetLastError(), "utd_accumulate_tiled_forward_kernel launch");
}

void utd_accumulate_tiled_vector_power_forward(
    const int* state_index, const int* rx_index, const int* valid_mask, const int* ownership_code,
    const float* epx, const float* epy, const float* epz,
    const float* edx, const float* edy, const float* edz,
    const float* n0x, const float* n0y, const float* n0z,
    const float* nnx, const float* nny, const float* nnz,
    const float* wn,
    const float* elm, const float* elx,
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
    const float* f1er, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
    const float* rxx, const float* rxy, const float* rxz,
    float* odr, float* odi, float* omr, float* omi,
    float* odvxr, float* odvxi, float* odvyr, float* odvyi, float* odvzr, float* odvzi,
    float* omvxr, float* omvxi, float* omvyr, float* omvyi, float* omvzr, float* omvzi,
    float* matched_power, float* valid_pair_count,
    int n_local_states, int n_local_receivers,
    float k, float rx_pol_x, float rx_pol_y, float rx_pol_z, MaterialParams material)
{
    int n_pairs = n_local_states * n_local_receivers;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    float3a rx_pol = {rx_pol_x, rx_pol_y, rx_pol_z};
    utd_accumulate_tiled_vector_power_forward_kernel<<<grid, BLOCK>>>(
        state_index, rx_index, valid_mask, ownership_code,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr,dxi, dyr,dyi, dzr,dzi,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr,
        rxx,rxy,rxz,
        odr,odi, omr,omi,
        odvxr,odvxi, odvyr,odvyi, odvzr,odvzi,
        omvxr,omvxi, omvyr,omvyi, omvzr,omvzi,
        matched_power, valid_pair_count,
        n_local_states, n_local_receivers, k,
        rx_pol,
        material);
    throw_cuda(cudaGetLastError(), "utd_accumulate_tiled_vector_power_forward_kernel launch");
}

void utd_accumulate_scalar_power_forward(
    const float* epx, const float* epy, const float* epz,
    const float* edx, const float* edy, const float* edz,
    const float* n0x, const float* n0y, const float* n0z,
    const float* nnx, const float* nny, const float* nnz,
    const float* wn,
    const float* elm, const float* elx,
    const float* spx, const float* spy, const float* spz,
    const float* ifr, const float* ifi,
    const float* inr, const float* ini,
    const float* r0r, const float* r0i,
    const float* rnr, const float* rni,
    const float* vxr, const float* vxi,
    const float* vyr, const float* vyi,
    const float* vzr, const float* vzi,
    const float* dxr_, const float* dxi_,
    const float* dyr_, const float* dyi_,
    const float* dzr_, const float* dzi_,
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
    const float* f0er, const float* f0sg,
    const float* f0g, const float* f0uf,
    const float* f0pr,
    const float* f1er, const float* f1sg,
    const float* f1g, const float* f1uf,
    const float* f1pr,
    const int* output_rx_index,
    const float* pair_rxx, const float* pair_rxy, const float* pair_rxz,
    float* coherentRe,
    float* coherentIm,
    float* power,
    float* validPairCount,
    int n_pairs,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material)
{
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    float3a rxPol = {rx_pol_x, rx_pol_y, rx_pol_z};
    utd_accumulate_scalar_power_forward_kernel<<<grid, BLOCK>>>(
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn, elm, elx,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr_,dxi_, dyr_,dyi_, dzr_,dzi_,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr,
        output_rx_index,
        pair_rxx,pair_rxy,pair_rxz,
        coherentRe, coherentIm, power, validPairCount,
        n_pairs, k, rxPol, material);
    throw_cuda(cudaGetLastError(), "utd_accumulate_scalar_power_forward_kernel launch");
}

void utd_accumulate_tiled_forward_slots(
    const int* state_index,
    const int* rx_index,
    const int* valid_mask,
    const int* ownership_code,
    const float* const* state_slots,
    const float* rxx,
    const float* rxy,
    const float* rxz,
    float* odr,
    float* odi,
    float* omr,
    float* omi,
    float* odvxr,
    float* odvxi,
    float* odvyr,
    float* odvyi,
    float* odvzr,
    float* odvzi,
    float* omvxr,
    float* omvxi,
    float* omvyr,
    float* omvyi,
    float* omvzr,
    float* omvzi,
    int n_local_states,
    int n_local_receivers,
    float k,
    MaterialParams material)
{
    int n_pairs = n_local_states * n_local_receivers;
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 81;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    const float** device_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_accumulate_tiled_forward_slots(state_slots)"
    );

    try {
        utd_accumulate_tiled_forward_slots_kernel<<<grid, BLOCK>>>(
            state_index,
            rx_index,
            valid_mask,
            ownership_code,
            device_slots,
            rxx,
            rxy,
            rxz,
            odr,
            odi,
            omr,
            omi,
            odvxr,
            odvxi,
            odvyr,
            odvyi,
            odvzr,
            odvzi,
            omvxr,
            omvxi,
            omvyr,
            omvyi,
            omvzr,
            omvzi,
            n_local_states,
            n_local_receivers,
            k,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_accumulate_tiled_forward_slots_kernel launch");
    } catch (...) {
        cudaFree(device_slots);
        throw;
    }

    cudaFree(device_slots);
}

void utd_accumulate_tiled_vector_power_forward_slots(
    const int* state_index,
    const int* rx_index,
    const int* valid_mask,
    const int* ownership_code,
    const float* const* state_slots,
    const float* rxx,
    const float* rxy,
    const float* rxz,
    float* odr,
    float* odi,
    float* omr,
    float* omi,
    float* odvxr,
    float* odvxi,
    float* odvyr,
    float* odvyi,
    float* odvzr,
    float* odvzi,
    float* omvxr,
    float* omvxi,
    float* omvyr,
    float* omvyi,
    float* omvzr,
    float* omvzi,
    float* matched_power,
    float* valid_pair_count,
    int n_local_states,
    int n_local_receivers,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material)
{
    int n_pairs = n_local_states * n_local_receivers;
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 81;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    float3a rx_pol = {rx_pol_x, rx_pol_y, rx_pol_z};
    const float** device_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_accumulate_tiled_vector_power_forward_slots(state_slots)"
    );

    try {
        utd_accumulate_tiled_vector_power_forward_slots_kernel<<<grid, BLOCK>>>(
            state_index,
            rx_index,
            valid_mask,
            ownership_code,
            device_slots,
            rxx,
            rxy,
            rxz,
            odr,
            odi,
            omr,
            omi,
            odvxr,
            odvxi,
            odvyr,
            odvyi,
            odvzr,
            odvzi,
            omvxr,
            omvxi,
            omvyr,
            omvyi,
            omvzr,
            omvzi,
            matched_power,
            valid_pair_count,
            n_local_states,
            n_local_receivers,
            k,
            rx_pol,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_accumulate_tiled_vector_power_forward_slots_kernel launch");
    } catch (...) {
        cudaFree(device_slots);
        throw;
    }

    cudaFree(device_slots);
}

void utd_accumulate_scalar_power_forward_slots(
    const float* const* state_slots,
    const int* output_rx_index,
    const float* pair_rxx,
    const float* pair_rxy,
    const float* pair_rxz,
    float* coherentRe,
    float* coherentIm,
    float* power,
    float* validPairCount,
    int n_pairs,
    float k,
    float rx_pol_x,
    float rx_pol_y,
    float rx_pol_z,
    MaterialParams material)
{
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 81;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    float3a rxPol = {rx_pol_x, rx_pol_y, rx_pol_z};
    const float** device_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_accumulate_scalar_power_forward_slots(state_slots)"
    );

    try {
        utd_accumulate_scalar_power_forward_slots_kernel<<<grid, BLOCK>>>(
            device_slots,
            output_rx_index,
            pair_rxx,
            pair_rxy,
            pair_rxz,
            coherentRe,
            coherentIm,
            power,
            validPairCount,
            n_pairs,
            k,
            rxPol,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_accumulate_scalar_power_forward_slots_kernel launch");
    } catch (...) {
        cudaFree(device_slots);
        throw;
    }

    cudaFree(device_slots);
}

PairContributionDebug utd_debug_pair_device(
    PairInputs state,
    float3a target,
    float k,
    MaterialParams material)
{
    PairContributionDebug host_output{};
    PairContributionDebug* device_output = nullptr;

    throw_cuda(cudaMalloc(&device_output, sizeof(PairContributionDebug)), "cudaMalloc utd_debug_pair_device");

    try {
        utd_debug_pair_device_kernel<<<1, 1>>>(state, target, k, material, device_output);
        throw_cuda(cudaGetLastError(), "utd_debug_pair_device_kernel launch");
        throw_cuda(
            cudaMemcpy(
                &host_output,
                device_output,
                sizeof(PairContributionDebug),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy utd_debug_pair_device"
        );
    } catch (...) {
        cudaFree(device_output);
        throw;
    }

    cudaFree(device_output);
    return host_output;
}

PairOutputs utd_debug_pair_outputs_device(
    PairInputs state,
    float3a target,
    float k,
    MaterialParams material)
{
    PairOutputs host_output{};
    PairOutputs* device_output = nullptr;

    throw_cuda(cudaMalloc(&device_output, sizeof(PairOutputs)), "cudaMalloc utd_debug_pair_outputs_device");

    try {
        utd_debug_pair_outputs_device_kernel<<<1, 1>>>(state, target, k, material, device_output);
        throw_cuda(cudaGetLastError(), "utd_debug_pair_outputs_device_kernel launch");
        throw_cuda(
            cudaMemcpy(
                &host_output,
                device_output,
                sizeof(PairOutputs),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy utd_debug_pair_outputs_device"
        );
    } catch (...) {
        cudaFree(device_output);
        throw;
    }

    cudaFree(device_output);
    return host_output;
}

PairContributionDebug utd_debug_pair_from_state_slots(
    const float* const* state_slots,
    int state_index,
    float3a target,
    float k,
    MaterialParams material)
{
    constexpr size_t SLOT_COUNT = 81;
    PairContributionDebug host_output{};
    const float** device_slots = nullptr;
    PairContributionDebug* device_output = nullptr;

    throw_cuda(
        cudaMalloc(&device_slots, SLOT_COUNT * sizeof(const float*)),
        "cudaMalloc utd_debug_pair_from_state_slots(state_slots)"
    );

    try {
        throw_cuda(
            cudaMemcpy(
                device_slots,
                state_slots,
                SLOT_COUNT * sizeof(const float*),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy utd_debug_pair_from_state_slots(state_slots)"
        );

        throw_cuda(
            cudaMalloc(&device_output, sizeof(PairContributionDebug)),
            "cudaMalloc utd_debug_pair_from_state_slots(output)"
        );

        utd_debug_pair_from_state_slots_kernel<<<1, 1>>>(
            device_slots,
            state_index,
            target,
            k,
            material,
            device_output
        );
        throw_cuda(cudaGetLastError(), "utd_debug_pair_from_state_slots_kernel launch");
        throw_cuda(
            cudaMemcpy(
                &host_output,
                device_output,
                sizeof(PairContributionDebug),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy utd_debug_pair_from_state_slots(output)"
        );
    } catch (...) {
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        if (device_slots != nullptr) {
            cudaFree(device_slots);
        }
        throw;
    }

    cudaFree(device_output);
    cudaFree(device_slots);
    return host_output;
}

void utd_accumulate_forward(
    const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*, const float*,
    float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float*,
    int, float, MaterialParams)
{
    throw std::runtime_error(
        "Native finite-wedge UTD requires edge_line_min and edge_line_max. "
        "The legacy forward entrypoint without finite-edge bounds is unsupported."
    );
}

void utd_accumulate_tiled_forward(
    const int*, const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*, const float*,
    float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float*,
    int, int, float, MaterialParams)
{
    throw std::runtime_error(
        "Native finite-wedge UTD requires edge_line_min and edge_line_max. "
        "The legacy tiled forward entrypoint without finite-edge bounds is unsupported."
    );
}

void utd_accumulate_scalar_power_forward(
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*,
    const float*, const float*,
    const float*,
    const int*,
    const float*, const float*, const float*,
    float*, float*, float*, float*,
    int, float, float, float, float, MaterialParams)
{
    throw std::runtime_error(
        "Native finite-wedge UTD requires edge_line_min and edge_line_max. "
        "The legacy scalar-power entrypoint without finite-edge bounds is unsupported."
    );
}

// =========================================================================
// Host launcher: backward
// =========================================================================
void utd_accumulate_backward(
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
    const float* f1er, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
    const float* rxx, const float* rxy, const float* rxz,
    const float* gdR, const float* gdI, const float* gmR, const float* gmI,
    const float* gdvxR, const float* gdvxI, const float* gdvyR, const float* gdvyI,
    const float* gdvzR, const float* gdvzI,
    const float* gmvxR, const float* gmvxI, const float* gmvyR, const float* gmvyI,
    const float* gmvzR, const float* gmvzI,
    float* gEpx, float* gEpy, float* gEpz,
    float* gEdx, float* gEdy, float* gEdz,
    float* gN0x_, float* gN0y_, float* gN0z_,
    float* gNnx_, float* gNny_, float* gNnz_,
    float* gWn_,
    float* gSpx, float* gSpy, float* gSpz,
    float* gIfr_, float* gIfi_,
    float* gInr_, float* gIni_,
    float* gR0r_, float* gR0i_,
    float* gRnr_, float* gRni_,
    float* gVxr_, float* gVxi_,
    float* gVyr_, float* gVyi_,
    float* gVzr_, float* gVzi_,
    float* gDxr_, float* gDxi_,
    float* gDyr_, float* gDyi_,
    float* gDzr_, float* gDzi_,
    float* gF0m00r_, float* gF0m00i_,
    float* gF0m01r_, float* gF0m01i_,
    float* gF0m10r_, float* gF0m10i_,
    float* gF0m11r_, float* gF0m11i_,
    float* gF1m00r_, float* gF1m00i_,
    float* gF1m01r_, float* gF1m01i_,
    float* gF1m10r_, float* gF1m10i_,
    float* gF1m11r_, float* gF1m11i_,
    float* gF0etaR_, float* gF0sigma_, float* gF0gain_,
    float* gF1etaR_, float* gF1sigma_, float* gF1gain_,
    float* gRxx, float* gRxy, float* gRxz,
    int n_pairs, float k, MaterialParams material)
{
    throw std::runtime_error(
        "Native finite-wedge explicit backward is unsupported on the legacy entrypoint without "
        "edge_line_min and edge_line_max. Use the Dr.Jit finite-wedge replay path instead."
    );
    if (n_pairs <= 0) return;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    utd_accumulate_backward_kernel<<<grid, BLOCK>>>(
        state_index, rx_index, ownership_code,
        epx,epy,epz, edx,edy,edz, n0x,n0y,n0z, nnx,nny,nnz, wn,
        spx,spy,spz, ifr,ifi, inr,ini, r0r,r0i, rnr,rni,
        vxr,vxi, vyr,vyi, vzr,vzi, dxr,dxi, dyr,dyi, dzr,dzi,
        jur,jui, jvr,jvi, djur,djui, djvr,djvi,
        bux,buy,buz, bvx,bvy,bvz, bkx,bky,bkz,
        f0m00r,f0m00i, f0m01r,f0m01i, f0m10r,f0m10i, f0m11r,f0m11i,
        f1m00r,f1m00i, f1m01r,f1m01i, f1m10r,f1m10i, f1m11r,f1m11i,
        f0er,f0sg,f0g,f0uf,f0pr, f1er,f1sg,f1g,f1uf,f1pr,
        rxx,rxy,rxz,
        gdR,gdI, gmR,gmI,
        gdvxR,gdvxI, gdvyR,gdvyI, gdvzR,gdvzI,
        gmvxR,gmvxI, gmvyR,gmvyI, gmvzR,gmvzI,
        gEpx,gEpy,gEpz, gEdx,gEdy,gEdz,
        gN0x_,gN0y_,gN0z_, gNnx_,gNny_,gNnz_, gWn_,
        gSpx,gSpy,gSpz, gIfr_,gIfi_, gInr_,gIni_,
        gR0r_,gR0i_, gRnr_,gRni_,
        gVxr_,gVxi_, gVyr_,gVyi_, gVzr_,gVzi_,
        gDxr_,gDxi_, gDyr_,gDyi_, gDzr_,gDzi_,
        gF0m00r_,gF0m00i_, gF0m01r_,gF0m01i_, gF0m10r_,gF0m10i_, gF0m11r_,gF0m11i_,
        gF1m00r_,gF1m00i_, gF1m01r_,gF1m01i_, gF1m10r_,gF1m10i_, gF1m11r_,gF1m11i_,
        gF0etaR_,gF0sigma_,gF0gain_, gF1etaR_,gF1sigma_,gF1gain_,
        gRxx,gRxy,gRxz,
        n_pairs, k, material);
    throw_cuda(cudaGetLastError(), "utd_accumulate_backward_kernel launch");
}

} // namespace witwin::channel::native_ext
