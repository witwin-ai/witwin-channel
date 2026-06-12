#include <cuda_runtime.h>
#include <stdexcept>

#include <common/cuda_check.h>
#include <utd/utd_types.h>
#include <utd/utd_math.h>
#include <utd/utd_accumulate.h>

namespace witwin::channel::native_ext {
namespace {

using common::throw_cuda;

#define WITWIN_DEVICE_NOINLINE __device__ __noinline__

__device__ __forceinline__ bool pair_mask_valid(const int* validMask, int pairIdx)
{
    if (validMask == nullptr) {
        return true;
    }
    return (validMask[pairIdx] & UTD_PAIR_VALID_FLAG) != 0;
}

// -------------------------------------------------------------------------
// SoA to PairInputs loader (one thread, one state index)
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
    const float* f0er, const float* f0mu, const float* f0sg, const float* f0g, const float* f0uf, const float* f0pr,
    const float* f1er, const float* f1mu, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr)
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
    p.face0Material = {f0er[sIdx], f0mu[sIdx], f0sg[sIdx], f0g[sIdx], f0uf[sIdx], f0pr[sIdx]};
    p.face1Material = {f1er[sIdx], f1mu[sIdx], f1sg[sIdx], f1g[sIdx], f1uf[sIdx], f1pr[sIdx]};
    p.selectStationaryPoint = 0.f;
    p.directFirstOrder = 0.f;
    p.pathLengthPrefix = 0.f;
    return p;
}

__device__ __forceinline__ PairInputs load_pair_inputs_from_slots(
    int sIdx,
    const float* const* slots)
{
    PairInputs p = load_pair_inputs(
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
        slots[71], slots[72], slots[73], slots[74], slots[75], slots[76],
        slots[77], slots[78], slots[79], slots[80], slots[81], slots[82]
    );
    p.selectStationaryPoint = slots[83][sIdx];
    p.directFirstOrder = slots[84][sIdx];
    p.pathLengthPrefix = slots[85][sIdx];
    return p;
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
    const float* __restrict__ f0er, const float* __restrict__ f0mu, const float* __restrict__ f0sg,
    const float* __restrict__ f0g,  const float* __restrict__ f0uf,
    const float* __restrict__ f0pr,
    const float* __restrict__ f1er, const float* __restrict__ f1mu, const float* __restrict__ f1sg,
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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr);

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
    const float* __restrict__ f0er, const float* __restrict__ f0mu, const float* __restrict__ f0sg,
    const float* __restrict__ f0g,  const float* __restrict__ f0uf,
    const float* __restrict__ f0pr,
    const float* __restrict__ f1er, const float* __restrict__ f1mu, const float* __restrict__ f1sg,
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
    if (!pair_mask_valid(validMask, tid)) return;

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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr);

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
    const float* __restrict__ f0er, const float* __restrict__ f0mu, const float* __restrict__ f0sg,
    const float* __restrict__ f0g,  const float* __restrict__ f0uf,
    const float* __restrict__ f0pr,
    const float* __restrict__ f1er, const float* __restrict__ f1mu, const float* __restrict__ f1sg,
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
    if (!pair_mask_valid(validMask, tid)) return;

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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr);

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
    const float* __restrict__ f0er, const float* __restrict__ f0mu, const float* __restrict__ f0sg,
    const float* __restrict__ f0g,  const float* __restrict__ f0uf,
    const float* __restrict__ f0pr,
    const float* __restrict__ f1er, const float* __restrict__ f1mu, const float* __restrict__ f1sg,
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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr);

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
    if (!pair_mask_valid(validMask, tid)) {
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
    if (!pair_mask_valid(validMask, tid)) {
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

inline float** copy_mutable_state_slots_to_device(float* const* stateSlots, size_t slotCount, const char* label) {
    float** deviceSlots = nullptr;
    throw_cuda(cudaMalloc(&deviceSlots, slotCount * sizeof(float*)), label);
    try {
        throw_cuda(
            cudaMemcpy(
                deviceSlots,
                stateSlots,
                slotCount * sizeof(float*),
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

__device__ __forceinline__ PairInputsGrad load_pair_tangent_from_slots(
    int sIdx,
    const float* const* slots)
{
    PairInputsGrad g = pig_zero();
    g.edgePos = make_f3(slots[0][sIdx], slots[1][sIdx], slots[2][sIdx]);
    g.edgeDir = make_f3(slots[3][sIdx], slots[4][sIdx], slots[5][sIdx]);
    g.n0 = make_f3(slots[6][sIdx], slots[7][sIdx], slots[8][sIdx]);
    g.nn = make_f3(slots[9][sIdx], slots[10][sIdx], slots[11][sIdx]);
    g.wedgeN = slots[12][sIdx];
    g.sourcePos = make_f3(slots[15][sIdx], slots[16][sIdx], slots[17][sIdx]);
    g.incidentField = cplx(slots[18][sIdx], slots[19][sIdx]);
    g.incidentNormalDerivative = cplx(slots[20][sIdx], slots[21][sIdx]);
    g.r0 = cplx(slots[22][sIdx], slots[23][sIdx]);
    g.rn = cplx(slots[24][sIdx], slots[25][sIdx]);
    g.incidentVector = {
        cplx(slots[26][sIdx], slots[27][sIdx]),
        cplx(slots[28][sIdx], slots[29][sIdx]),
        cplx(slots[30][sIdx], slots[31][sIdx]),
    };
    g.incidentDerivativeVector = {
        cplx(slots[32][sIdx], slots[33][sIdx]),
        cplx(slots[34][sIdx], slots[35][sIdx]),
        cplx(slots[36][sIdx], slots[37][sIdx]),
    };
    g.incidentJones = {
        cplx(slots[38][sIdx], slots[39][sIdx]),
        cplx(slots[40][sIdx], slots[41][sIdx]),
    };
    g.incidentDerivativeJones = {
        cplx(slots[42][sIdx], slots[43][sIdx]),
        cplx(slots[44][sIdx], slots[45][sIdx]),
    };
    g.incidentBasis = {
        make_f3(slots[46][sIdx], slots[47][sIdx], slots[48][sIdx]),
        make_f3(slots[49][sIdx], slots[50][sIdx], slots[51][sIdx]),
        make_f3(slots[52][sIdx], slots[53][sIdx], slots[54][sIdx]),
    };
    g.face0Operator = {
        cplx(slots[55][sIdx], slots[56][sIdx]),
        cplx(slots[57][sIdx], slots[58][sIdx]),
        cplx(slots[59][sIdx], slots[60][sIdx]),
        cplx(slots[61][sIdx], slots[62][sIdx]),
    };
    g.face1Operator = {
        cplx(slots[63][sIdx], slots[64][sIdx]),
        cplx(slots[65][sIdx], slots[66][sIdx]),
        cplx(slots[67][sIdx], slots[68][sIdx]),
        cplx(slots[69][sIdx], slots[70][sIdx]),
    };
    g.face0Material = {slots[71][sIdx], slots[72][sIdx], slots[73][sIdx], slots[74][sIdx], 0.f, 0.f};
    g.face1Material = {slots[77][sIdx], slots[78][sIdx], slots[79][sIdx], slots[80][sIdx], 0.f, 0.f};
    return g;
}

__device__ __forceinline__ float basis_tangent_dot(Basis3 grad, Basis3 tangent)
{
    return f3_dot(grad.u, tangent.u)
        + f3_dot(grad.v, tangent.v)
        + f3_dot(grad.k, tangent.k);
}

__device__ __forceinline__ float pair_input_tangent_dot(
    const PairInputsGrad& gradState,
    const PairInputsGrad& tangentState,
    float3a gradRx,
    float3a tangentRx)
{
    float result = 0.f;
    result += f3_dot(gradState.edgePos, tangentState.edgePos);
    result += f3_dot(gradState.edgeDir, tangentState.edgeDir);
    result += f3_dot(gradState.n0, tangentState.n0);
    result += f3_dot(gradState.nn, tangentState.nn);
    result += gradState.wedgeN * tangentState.wedgeN;
    result += f3_dot(gradState.sourcePos, tangentState.sourcePos);
    result += cplx_adj_dot(gradState.incidentField, tangentState.incidentField);
    result += cplx_adj_dot(
        gradState.incidentNormalDerivative,
        tangentState.incidentNormalDerivative);
    result += cplx_adj_dot(gradState.r0, tangentState.r0);
    result += cplx_adj_dot(gradState.rn, tangentState.rn);
    result += cplx_adj_dot(gradState.incidentVector.x, tangentState.incidentVector.x);
    result += cplx_adj_dot(gradState.incidentVector.y, tangentState.incidentVector.y);
    result += cplx_adj_dot(gradState.incidentVector.z, tangentState.incidentVector.z);
    result += cplx_adj_dot(
        gradState.incidentDerivativeVector.x,
        tangentState.incidentDerivativeVector.x);
    result += cplx_adj_dot(
        gradState.incidentDerivativeVector.y,
        tangentState.incidentDerivativeVector.y);
    result += cplx_adj_dot(
        gradState.incidentDerivativeVector.z,
        tangentState.incidentDerivativeVector.z);
    result += cplx_adj_dot(gradState.incidentJones.u, tangentState.incidentJones.u);
    result += cplx_adj_dot(gradState.incidentJones.v, tangentState.incidentJones.v);
    result += cplx_adj_dot(
        gradState.incidentDerivativeJones.u,
        tangentState.incidentDerivativeJones.u);
    result += cplx_adj_dot(
        gradState.incidentDerivativeJones.v,
        tangentState.incidentDerivativeJones.v);
    result += basis_tangent_dot(gradState.incidentBasis, tangentState.incidentBasis);
    result += cplx_adj_dot(gradState.face0Operator.m00, tangentState.face0Operator.m00);
    result += cplx_adj_dot(gradState.face0Operator.m01, tangentState.face0Operator.m01);
    result += cplx_adj_dot(gradState.face0Operator.m10, tangentState.face0Operator.m10);
    result += cplx_adj_dot(gradState.face0Operator.m11, tangentState.face0Operator.m11);
    result += cplx_adj_dot(gradState.face1Operator.m00, tangentState.face1Operator.m00);
    result += cplx_adj_dot(gradState.face1Operator.m01, tangentState.face1Operator.m01);
    result += cplx_adj_dot(gradState.face1Operator.m10, tangentState.face1Operator.m10);
    result += cplx_adj_dot(gradState.face1Operator.m11, tangentState.face1Operator.m11);
    result += gradState.face0Material.etaR * tangentState.face0Material.etaR;
    result += gradState.face0Material.muR * tangentState.face0Material.muR;
    result += gradState.face0Material.sigma * tangentState.face0Material.sigma;
    result += gradState.face0Material.gain * tangentState.face0Material.gain;
    result += gradState.face1Material.etaR * tangentState.face1Material.etaR;
    result += gradState.face1Material.muR * tangentState.face1Material.muR;
    result += gradState.face1Material.sigma * tangentState.face1Material.sigma;
    result += gradState.face1Material.gain * tangentState.face1Material.gain;
    result += f3_dot(gradRx, tangentRx);
    return result;
}

__device__ __forceinline__ bool slot_has_vjp(int slot)
{
    return slot != 75 && slot != 76 && slot != 81 && slot != 82;
}

__device__ __forceinline__ PairInputs perturb_pair_input_slot(PairInputs value, int slot, float delta)
{
    switch (slot) {
    case 0: value.edgePos.x += delta; break;
    case 1: value.edgePos.y += delta; break;
    case 2: value.edgePos.z += delta; break;
    case 3: value.edgeDir.x += delta; break;
    case 4: value.edgeDir.y += delta; break;
    case 5: value.edgeDir.z += delta; break;
    case 6: value.n0.x += delta; break;
    case 7: value.n0.y += delta; break;
    case 8: value.n0.z += delta; break;
    case 9: value.nn.x += delta; break;
    case 10: value.nn.y += delta; break;
    case 11: value.nn.z += delta; break;
    case 12: value.wedgeN += delta; break;
    case 13: value.edgeLineMin += delta; break;
    case 14: value.edgeLineMax += delta; break;
    case 15: value.sourcePos.x += delta; break;
    case 16: value.sourcePos.y += delta; break;
    case 17: value.sourcePos.z += delta; break;
    case 18: value.incidentField.re += delta; break;
    case 19: value.incidentField.im += delta; break;
    case 20: value.incidentNormalDerivative.re += delta; break;
    case 21: value.incidentNormalDerivative.im += delta; break;
    case 22: value.r0.re += delta; break;
    case 23: value.r0.im += delta; break;
    case 24: value.rn.re += delta; break;
    case 25: value.rn.im += delta; break;
    case 26: value.incidentVector.x.re += delta; break;
    case 27: value.incidentVector.x.im += delta; break;
    case 28: value.incidentVector.y.re += delta; break;
    case 29: value.incidentVector.y.im += delta; break;
    case 30: value.incidentVector.z.re += delta; break;
    case 31: value.incidentVector.z.im += delta; break;
    case 32: value.incidentDerivativeVector.x.re += delta; break;
    case 33: value.incidentDerivativeVector.x.im += delta; break;
    case 34: value.incidentDerivativeVector.y.re += delta; break;
    case 35: value.incidentDerivativeVector.y.im += delta; break;
    case 36: value.incidentDerivativeVector.z.re += delta; break;
    case 37: value.incidentDerivativeVector.z.im += delta; break;
    case 38: value.incidentJones.u.re += delta; break;
    case 39: value.incidentJones.u.im += delta; break;
    case 40: value.incidentJones.v.re += delta; break;
    case 41: value.incidentJones.v.im += delta; break;
    case 42: value.incidentDerivativeJones.u.re += delta; break;
    case 43: value.incidentDerivativeJones.u.im += delta; break;
    case 44: value.incidentDerivativeJones.v.re += delta; break;
    case 45: value.incidentDerivativeJones.v.im += delta; break;
    case 46: value.incidentBasis.u.x += delta; break;
    case 47: value.incidentBasis.u.y += delta; break;
    case 48: value.incidentBasis.u.z += delta; break;
    case 49: value.incidentBasis.v.x += delta; break;
    case 50: value.incidentBasis.v.y += delta; break;
    case 51: value.incidentBasis.v.z += delta; break;
    case 52: value.incidentBasis.k.x += delta; break;
    case 53: value.incidentBasis.k.y += delta; break;
    case 54: value.incidentBasis.k.z += delta; break;
    case 55: value.face0Operator.m00.re += delta; break;
    case 56: value.face0Operator.m00.im += delta; break;
    case 57: value.face0Operator.m01.re += delta; break;
    case 58: value.face0Operator.m01.im += delta; break;
    case 59: value.face0Operator.m10.re += delta; break;
    case 60: value.face0Operator.m10.im += delta; break;
    case 61: value.face0Operator.m11.re += delta; break;
    case 62: value.face0Operator.m11.im += delta; break;
    case 63: value.face1Operator.m00.re += delta; break;
    case 64: value.face1Operator.m00.im += delta; break;
    case 65: value.face1Operator.m01.re += delta; break;
    case 66: value.face1Operator.m01.im += delta; break;
    case 67: value.face1Operator.m10.re += delta; break;
    case 68: value.face1Operator.m10.im += delta; break;
    case 69: value.face1Operator.m11.re += delta; break;
    case 70: value.face1Operator.m11.im += delta; break;
    case 71: value.face0Material.etaR += delta; break;
    case 72: value.face0Material.muR += delta; break;
    case 73: value.face0Material.sigma += delta; break;
    case 74: value.face0Material.gain += delta; break;
    case 77: value.face1Material.etaR += delta; break;
    case 78: value.face1Material.muR += delta; break;
    case 79: value.face1Material.sigma += delta; break;
    case 80: value.face1Material.gain += delta; break;
    default: break;
    }
    return value;
}

WITWIN_DEVICE_NOINLINE PairOutputs compute_pair_outputs_for_vjp(
    PairInputs pi,
    float3a tgt,
    float k,
    MaterialParams mat,
    bool vector_completion)
{
    PairOutputs out;
    out.field = cplx_zero();
    out.vectorField = c3_zero();
    bool geom_valid;
    Complex direct_gain;
    Complex derivative_gain;
    compute_pair_field_terms(pi, tgt, k, mat, geom_valid, out.field, direct_gain, derivative_gain);
    out.vectorField = vector_completion
        ? compute_pair_vector_contribution(pi, tgt, k, mat)
        : compute_pair_vector_contribution_no_completion(pi, tgt, k, mat);
    return out;
}

__device__ __forceinline__ PairOutputs pair_outputs_scaled_difference(
    PairOutputs plus,
    PairOutputs minus,
    float scale)
{
    PairOutputs out;
    out.field = cplx_mul_real(cplx_sub(plus.field, minus.field), scale);
    out.vectorField = c3_scale_real(complex3_sub(plus.vectorField, minus.vectorField), scale);
    return out;
}

WITWIN_DEVICE_NOINLINE PairOutputs pair_outputs_jvp_finite_difference(
    PairInputs pi,
    PairInputsGrad tangentState,
    float3a tgt,
    float3a tangentTgt,
    float k,
    MaterialParams mat,
    bool vector_completion)
{
    constexpr float eps = 1.0e-3f;
    PairOutputs plus = compute_pair_outputs_for_vjp(
        pair_inputs_add_scaled(pi, tangentState, eps),
        f3_add(tgt, f3_mul(tangentTgt, eps)),
        k,
        mat,
        vector_completion);
    PairOutputs minus = compute_pair_outputs_for_vjp(
        pair_inputs_add_scaled(pi, tangentState, -eps),
        f3_add(tgt, f3_mul(tangentTgt, -eps)),
        k,
        mat,
        vector_completion);
    return pair_outputs_scaled_difference(plus, minus, 0.5f / eps);
}

__device__ __forceinline__ float pair_outputs_adjoint_dot(PairOutputs value, PairOutputs grad)
{
    float result = cplx_adj_dot(grad.field, value.field);
    result += cplx_adj_dot(grad.vectorField.x, value.vectorField.x);
    result += cplx_adj_dot(grad.vectorField.y, value.vectorField.y);
    result += cplx_adj_dot(grad.vectorField.z, value.vectorField.z);
    return result;
}

__device__ __forceinline__ bool pair_outputs_grad_nonzero(PairOutputs grad)
{
    return cplx_any_nonzero(grad.field) || c3_grad_any_nonzero(grad.vectorField);
}

WITWIN_DEVICE_NOINLINE float state_slot_vjp_finite_difference(
    PairInputs pi,
    float3a tgt,
    PairOutputs grad,
    int slot,
    float k,
    MaterialParams mat,
    bool vector_completion)
{
    constexpr float eps = 1.0e-3f;
    PairOutputs plus = compute_pair_outputs_for_vjp(
        perturb_pair_input_slot(pi, slot, eps),
        tgt,
        k,
        mat,
        vector_completion);
    PairOutputs minus = compute_pair_outputs_for_vjp(
        perturb_pair_input_slot(pi, slot, -eps),
        tgt,
        k,
        mat,
        vector_completion);
    return (pair_outputs_adjoint_dot(plus, grad) - pair_outputs_adjoint_dot(minus, grad))
        * (0.5f / eps);
}

WITWIN_DEVICE_NOINLINE float receiver_slot_vjp_finite_difference(
    PairInputs pi,
    float3a tgt,
    PairOutputs grad,
    int axis,
    float k,
    MaterialParams mat,
    bool vector_completion)
{
    constexpr float eps = 1.0e-3f;
    float3a plus_tgt = tgt;
    float3a minus_tgt = tgt;
    if (axis == 0) {
        plus_tgt.x += eps;
        minus_tgt.x -= eps;
    } else if (axis == 1) {
        plus_tgt.y += eps;
        minus_tgt.y -= eps;
    } else {
        plus_tgt.z += eps;
        minus_tgt.z -= eps;
    }
    PairOutputs plus = compute_pair_outputs_for_vjp(pi, plus_tgt, k, mat, vector_completion);
    PairOutputs minus = compute_pair_outputs_for_vjp(pi, minus_tgt, k, mat, vector_completion);
    return (pair_outputs_adjoint_dot(plus, grad) - pair_outputs_adjoint_dot(minus, grad))
        * (0.5f / eps);
}

__device__ __forceinline__ void atomic_add_if_finite(float* target, int index, float value)
{
    if (value != 0.0f && isfinite(value)) {
        atomicAdd(&target[index], value);
    }
}

WITWIN_DEVICE_NOINLINE void accumulate_pair_vjp_to_slots(
    int sI,
    int rI,
    PairInputs pi,
    float3a tgt,
    PairOutputs grad,
    float* const* gradSlots,
    float* gradRxX,
    float* gradRxY,
    float* gradRxZ,
    float k,
    MaterialParams mat,
    bool vector_completion)
{
    if (!pair_outputs_grad_nonzero(grad)) {
        return;
    }
    for (int slot = 0; slot < 83; ++slot) {
        if (!slot_has_vjp(slot)) {
            continue;
        }
        atomic_add_if_finite(
            gradSlots[slot],
            sI,
            state_slot_vjp_finite_difference(pi, tgt, grad, slot, k, mat, vector_completion));
    }
    atomic_add_if_finite(
        gradRxX,
        rI,
        receiver_slot_vjp_finite_difference(pi, tgt, grad, 0, k, mat, vector_completion));
    atomic_add_if_finite(
        gradRxY,
        rI,
        receiver_slot_vjp_finite_difference(pi, tgt, grad, 1, k, mat, vector_completion));
    atomic_add_if_finite(
        gradRxZ,
        rI,
        receiver_slot_vjp_finite_difference(pi, tgt, grad, 2, k, mat, vector_completion));
}

__device__ __forceinline__ Complex3 vector_component_seed(int component)
{
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

__global__ void utd_accumulate_tiled_jvp_slots_kernel(
    const int* __restrict__ stateLocalIdx,
    const int* __restrict__ rxLocalIdx,
    const int* __restrict__ validMask,
    const int* __restrict__ ownerCode,
    const float* const* __restrict__ stateSlots,
    const float* __restrict__ rxx,
    const float* __restrict__ rxy,
    const float* __restrict__ rxz,
    const float* const* __restrict__ tangentSlots,
    const float* __restrict__ tRxx,
    const float* __restrict__ tRxy,
    const float* __restrict__ tRxz,
    float* __restrict__ tdvxr,
    float* __restrict__ tdvxi,
    float* __restrict__ tdvyr,
    float* __restrict__ tdvyi,
    float* __restrict__ tdvzr,
    float* __restrict__ tdvzi,
    float* __restrict__ tmvxr,
    float* __restrict__ tmvxi,
    float* __restrict__ tmvyr,
    float* __restrict__ tmvyi,
    float* __restrict__ tmvzr,
    float* __restrict__ tmvzi,
    int nLocalStates,
    int nLocalReceivers,
    float k,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nPairs = nLocalStates * nLocalReceivers;
    if (tid >= nPairs) return;
    if (!pair_mask_valid(validMask, tid)) return;

    int localState = tid / nLocalReceivers;
    int localReceiver = tid - localState * nLocalReceivers;
    int sI = stateLocalIdx[localState];
    int rI = rxLocalIdx[localReceiver];
    int ownership = ownerCode[sI];
    if (ownership != OWNERSHIP_DIRECT && ownership != OWNERSHIP_MIXED) return;

    PairInputs pi = load_pair_inputs_from_slots(sI, stateSlots);
    PairInputsGrad tangentState = load_pair_tangent_from_slots(sI, tangentSlots);
    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    float3a tangentRx = make_f3(tRxx[rI], tRxy[rI], tRxz[rI]);

    float* outRe[3];
    float* outIm[3];
    if (ownership == OWNERSHIP_DIRECT) {
        outRe[0] = tdvxr; outIm[0] = tdvxi;
        outRe[1] = tdvyr; outIm[1] = tdvyi;
        outRe[2] = tdvzr; outIm[2] = tdvzi;
    } else {
        outRe[0] = tmvxr; outIm[0] = tmvxi;
        outRe[1] = tmvyr; outIm[1] = tmvyi;
        outRe[2] = tmvzr; outIm[2] = tmvzi;
    }

    Complex3 tangent = pair_vector_output_jvp_completion(
        pi,
        tangentState,
        tgt,
        tangentRx,
        k,
        mat);
    atomicAdd(&outRe[0][rI], tangent.x.re);
    atomicAdd(&outIm[0][rI], tangent.x.im);
    atomicAdd(&outRe[1][rI], tangent.y.re);
    atomicAdd(&outIm[1][rI], tangent.y.im);
    atomicAdd(&outRe[2][rI], tangent.z.re);
    atomicAdd(&outIm[2][rI], tangent.z.im);
}

__global__ void utd_pair_forward_slots_kernel(
    const int* __restrict__ stateIdx,
    const int* __restrict__ rxIdx,
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
    int nPairs,
    float k,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;

    int sI = stateIdx[tid];
    int rI = rxIdx[tid];
    int ownership = ownerCode[sI];
    if (ownership != OWNERSHIP_DIRECT && ownership != OWNERSHIP_MIXED) return;

    PairInputs pi = load_pair_inputs_from_slots(sI, stateSlots);
    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    PairOutputs po;
    po.field = cplx_zero();
    po.vectorField = c3_zero();
    bool geomValid;
    Complex directGain;
    Complex derivativeGain;
    compute_pair_field_terms(pi, tgt, k, mat, geomValid, po.field, directGain, derivativeGain);
    po.vectorField = compute_pair_vector_contribution_no_completion(pi, tgt, k, mat);
    atomic_add_pair_output(ownership, rI, po,
        odr, odi, omr, omi,
        odvxr, odvxi, odvyr, odvyi, odvzr, odvzi,
        omvxr, omvxi, omvyr, omvyi, omvzr, omvzi);
}

__global__ void utd_pair_jvp_slots_kernel(
    const int* __restrict__ stateIdx,
    const int* __restrict__ rxIdx,
    const int* __restrict__ ownerCode,
    const float* const* __restrict__ stateSlots,
    const float* __restrict__ rxx,
    const float* __restrict__ rxy,
    const float* __restrict__ rxz,
    const float* const* __restrict__ tangentSlots,
    const float* __restrict__ tRxx,
    const float* __restrict__ tRxy,
    const float* __restrict__ tRxz,
    float* __restrict__ tdR,
    float* __restrict__ tdI,
    float* __restrict__ tmR,
    float* __restrict__ tmI,
    float* __restrict__ tdvxr,
    float* __restrict__ tdvxi,
    float* __restrict__ tdvyr,
    float* __restrict__ tdvyi,
    float* __restrict__ tdvzr,
    float* __restrict__ tdvzi,
    float* __restrict__ tmvxr,
    float* __restrict__ tmvxi,
    float* __restrict__ tmvyr,
    float* __restrict__ tmvyi,
    float* __restrict__ tmvzr,
    float* __restrict__ tmvzi,
    int nPairs,
    float k,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;

    int sI = stateIdx[tid];
    int rI = rxIdx[tid];
    int ownership = ownerCode[sI];
    if (ownership != OWNERSHIP_DIRECT && ownership != OWNERSHIP_MIXED) return;

    PairInputs pi = load_pair_inputs_from_slots(sI, stateSlots);
    PairInputsGrad tangentState = load_pair_tangent_from_slots(sI, tangentSlots);
    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    float3a tangentRx = make_f3(tRxx[rI], tRxy[rI], tRxz[rI]);

    PairOutputs tangent = pair_outputs_jvp_finite_difference(
        pi,
        tangentState,
        tgt,
        tangentRx,
        k,
        mat,
        false);

    float* scalarRe;
    float* scalarIm;
    float* outRe[3];
    float* outIm[3];
    if (ownership == OWNERSHIP_DIRECT) {
        scalarRe = tdR; scalarIm = tdI;
        outRe[0] = tdvxr; outIm[0] = tdvxi;
        outRe[1] = tdvyr; outIm[1] = tdvyi;
        outRe[2] = tdvzr; outIm[2] = tdvzi;
    } else {
        scalarRe = tmR; scalarIm = tmI;
        outRe[0] = tmvxr; outIm[0] = tmvxi;
        outRe[1] = tmvyr; outIm[1] = tmvyi;
        outRe[2] = tmvzr; outIm[2] = tmvzi;
    }

    atomicAdd(&scalarRe[rI], tangent.field.re);
    atomicAdd(&scalarIm[rI], tangent.field.im);
    atomicAdd(&outRe[0][rI], tangent.vectorField.x.re);
    atomicAdd(&outIm[0][rI], tangent.vectorField.x.im);
    atomicAdd(&outRe[1][rI], tangent.vectorField.y.re);
    atomicAdd(&outIm[1][rI], tangent.vectorField.y.im);
    atomicAdd(&outRe[2][rI], tangent.vectorField.z.re);
    atomicAdd(&outIm[2][rI], tangent.vectorField.z.im);
}

__global__ void utd_accumulate_tiled_vjp_slots_kernel(
    const int* __restrict__ stateLocalIdx,
    const int* __restrict__ rxLocalIdx,
    const int* __restrict__ validMask,
    const int* __restrict__ ownerCode,
    const float* const* __restrict__ stateSlots,
    const float* __restrict__ rxx,
    const float* __restrict__ rxy,
    const float* __restrict__ rxz,
    const float* __restrict__ gdvxr,
    const float* __restrict__ gdvxi,
    const float* __restrict__ gdvyr,
    const float* __restrict__ gdvyi,
    const float* __restrict__ gdvzr,
    const float* __restrict__ gdvzi,
    const float* __restrict__ gmvxr,
    const float* __restrict__ gmvxi,
    const float* __restrict__ gmvyr,
    const float* __restrict__ gmvyi,
    const float* __restrict__ gmvzr,
    const float* __restrict__ gmvzi,
    float* const* __restrict__ gradSlots,
    float* __restrict__ gRxx,
    float* __restrict__ gRxy,
    float* __restrict__ gRxz,
    int nLocalStates,
    int nLocalReceivers,
    float k,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nPairs = nLocalStates * nLocalReceivers;
    if (tid >= nPairs) return;
    if (!pair_mask_valid(validMask, tid)) return;

    int localState = tid / nLocalReceivers;
    int localReceiver = tid - localState * nLocalReceivers;
    int sI = stateLocalIdx[localState];
    int rI = rxLocalIdx[localReceiver];
    int ownership = ownerCode[sI];
    if (ownership != OWNERSHIP_DIRECT && ownership != OWNERSHIP_MIXED) return;

    PairOutputs grad;
    grad.field = cplx_zero();
    if (ownership == OWNERSHIP_DIRECT) {
        grad.vectorField = {
            cplx(gdvxr[rI], gdvxi[rI]),
            cplx(gdvyr[rI], gdvyi[rI]),
            cplx(gdvzr[rI], gdvzi[rI]),
        };
    } else {
        grad.vectorField = {
            cplx(gmvxr[rI], gmvxi[rI]),
            cplx(gmvyr[rI], gmvyi[rI]),
            cplx(gmvzr[rI], gmvzi[rI]),
        };
    }

    PairInputs pi = load_pair_inputs_from_slots(sI, stateSlots);
    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    accumulate_pair_vjp_to_slots(
        sI, rI, pi, tgt, grad, gradSlots, gRxx, gRxy, gRxz, k, mat, true);
}

__global__ void utd_pair_vjp_slots_kernel(
    const int* __restrict__ stateIdx,
    const int* __restrict__ rxIdx,
    const int* __restrict__ ownerCode,
    const float* const* __restrict__ stateSlots,
    const float* __restrict__ rxx,
    const float* __restrict__ rxy,
    const float* __restrict__ rxz,
    const float* __restrict__ gdR,
    const float* __restrict__ gdI,
    const float* __restrict__ gmR,
    const float* __restrict__ gmI,
    const float* __restrict__ gdvxr,
    const float* __restrict__ gdvxi,
    const float* __restrict__ gdvyr,
    const float* __restrict__ gdvyi,
    const float* __restrict__ gdvzr,
    const float* __restrict__ gdvzi,
    const float* __restrict__ gmvxr,
    const float* __restrict__ gmvxi,
    const float* __restrict__ gmvyr,
    const float* __restrict__ gmvyi,
    const float* __restrict__ gmvzr,
    const float* __restrict__ gmvzi,
    float* const* __restrict__ gradSlots,
    float* __restrict__ gRxx,
    float* __restrict__ gRxy,
    float* __restrict__ gRxz,
    int nPairs,
    float k,
    MaterialParams mat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nPairs) return;

    int sI = stateIdx[tid];
    int rI = rxIdx[tid];
    int ownership = ownerCode[sI];
    if (ownership != OWNERSHIP_DIRECT && ownership != OWNERSHIP_MIXED) return;

    PairOutputs grad;
    if (ownership == OWNERSHIP_DIRECT) {
        grad.field = cplx(gdR[rI], gdI[rI]);
        grad.vectorField = {
            cplx(gdvxr[rI], gdvxi[rI]),
            cplx(gdvyr[rI], gdvyi[rI]),
            cplx(gdvzr[rI], gdvzi[rI]),
        };
    } else {
        grad.field = cplx(gmR[rI], gmI[rI]);
        grad.vectorField = {
            cplx(gmvxr[rI], gmvxi[rI]),
            cplx(gmvyr[rI], gmvyi[rI]),
            cplx(gmvzr[rI], gmvzi[rI]),
        };
    }

    PairInputs pi = load_pair_inputs_from_slots(sI, stateSlots);
    float3a tgt = make_f3(rxx[rI], rxy[rI], rxz[rI]);
    accumulate_pair_vjp_to_slots(
        sI, rI, pi, tgt, grad, gradSlots, gRxx, gRxy, gRxz, k, mat, false);
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
    const float* f0er, const float* f0mu, const float* f0sg, const float* f0g, const float* f0uf, const float* f0pr,
    const float* f1er, const float* f1mu, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr,
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
    const float* f0er, const float* f0mu, const float* f0sg, const float* f0g, const float* f0uf, const float* f0pr,
    const float* f1er, const float* f1mu, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr,
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
    const float* f0er, const float* f0mu, const float* f0sg, const float* f0g, const float* f0uf, const float* f0pr,
    const float* f1er, const float* f1mu, const float* f1sg, const float* f1g, const float* f1uf, const float* f1pr,
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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr,
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
    const float* f0er, const float* f0mu, const float* f0sg,
    const float* f0g, const float* f0uf,
    const float* f0pr,
    const float* f1er, const float* f1mu, const float* f1sg,
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
        f0er,f0mu,f0sg,f0g,f0uf,f0pr, f1er,f1mu,f1sg,f1g,f1uf,f1pr,
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

    constexpr size_t SLOT_COUNT = 86;
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
        throw_cuda(cudaDeviceSynchronize(), "utd_accumulate_tiled_forward_slots_kernel sync");
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

    constexpr size_t SLOT_COUNT = 86;
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
        throw_cuda(cudaDeviceSynchronize(), "utd_accumulate_tiled_vector_power_forward_slots_kernel sync");
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

    constexpr size_t SLOT_COUNT = 86;
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
        throw_cuda(cudaDeviceSynchronize(), "utd_accumulate_scalar_power_forward_slots_kernel sync");
    } catch (...) {
        cudaFree(device_slots);
        throw;
    }

    cudaFree(device_slots);
}

void utd_accumulate_tiled_jvp_slots(
    const int* state_index,
    const int* rx_index,
    const int* valid_mask,
    const int* ownership_code,
    const float* const* state_slots,
    const float* rxx,
    const float* rxy,
    const float* rxz,
    const float* const* tangent_slots,
    const float* tangent_rxx,
    const float* tangent_rxy,
    const float* tangent_rxz,
    float* tangent_direct_vec_x_re,
    float* tangent_direct_vec_x_im,
    float* tangent_direct_vec_y_re,
    float* tangent_direct_vec_y_im,
    float* tangent_direct_vec_z_re,
    float* tangent_direct_vec_z_im,
    float* tangent_multi_vec_x_re,
    float* tangent_multi_vec_x_im,
    float* tangent_multi_vec_y_re,
    float* tangent_multi_vec_y_im,
    float* tangent_multi_vec_z_re,
    float* tangent_multi_vec_z_im,
    int n_local_states,
    int n_local_receivers,
    float k,
    MaterialParams material)
{
    int n_pairs = n_local_states * n_local_receivers;
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 86;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    const float** device_state_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_accumulate_tiled_jvp_slots(state_slots)"
    );
    const float** device_tangent_slots = nullptr;
    try {
        device_tangent_slots = copy_state_slots_to_device(
            tangent_slots,
            SLOT_COUNT,
            "utd_accumulate_tiled_jvp_slots(tangent_slots)"
        );
        utd_accumulate_tiled_jvp_slots_kernel<<<grid, BLOCK>>>(
            state_index,
            rx_index,
            valid_mask,
            ownership_code,
            device_state_slots,
            rxx,
            rxy,
            rxz,
            device_tangent_slots,
            tangent_rxx,
            tangent_rxy,
            tangent_rxz,
            tangent_direct_vec_x_re,
            tangent_direct_vec_x_im,
            tangent_direct_vec_y_re,
            tangent_direct_vec_y_im,
            tangent_direct_vec_z_re,
            tangent_direct_vec_z_im,
            tangent_multi_vec_x_re,
            tangent_multi_vec_x_im,
            tangent_multi_vec_y_re,
            tangent_multi_vec_y_im,
            tangent_multi_vec_z_re,
            tangent_multi_vec_z_im,
            n_local_states,
            n_local_receivers,
            k,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_accumulate_tiled_jvp_slots_kernel launch");
        throw_cuda(cudaDeviceSynchronize(), "utd_accumulate_tiled_jvp_slots_kernel sync");
    } catch (...) {
        cudaFree(device_tangent_slots);
        cudaFree(device_state_slots);
        throw;
    }

    cudaFree(device_tangent_slots);
    cudaFree(device_state_slots);
}

void utd_pair_forward_slots(
    const int* state_index,
    const int* rx_index,
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
    int n_pairs,
    float k,
    MaterialParams material)
{
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 86;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    const float** device_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_pair_forward_slots(state_slots)"
    );
    try {
        utd_pair_forward_slots_kernel<<<grid, BLOCK>>>(
            state_index,
            rx_index,
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
            n_pairs,
            k,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_pair_forward_slots_kernel launch");
        throw_cuda(cudaDeviceSynchronize(), "utd_pair_forward_slots_kernel sync");
    } catch (...) {
        cudaFree(device_slots);
        throw;
    }

    cudaFree(device_slots);
}

void utd_pair_jvp_slots(
    const int* state_index,
    const int* rx_index,
    const int* ownership_code,
    const float* const* state_slots,
    const float* rxx,
    const float* rxy,
    const float* rxz,
    const float* const* tangent_slots,
    const float* tangent_rxx,
    const float* tangent_rxy,
    const float* tangent_rxz,
    float* tangent_direct_re,
    float* tangent_direct_im,
    float* tangent_multi_re,
    float* tangent_multi_im,
    float* tangent_direct_vec_x_re,
    float* tangent_direct_vec_x_im,
    float* tangent_direct_vec_y_re,
    float* tangent_direct_vec_y_im,
    float* tangent_direct_vec_z_re,
    float* tangent_direct_vec_z_im,
    float* tangent_multi_vec_x_re,
    float* tangent_multi_vec_x_im,
    float* tangent_multi_vec_y_re,
    float* tangent_multi_vec_y_im,
    float* tangent_multi_vec_z_re,
    float* tangent_multi_vec_z_im,
    int n_pairs,
    float k,
    MaterialParams material)
{
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 86;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    const float** device_state_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_pair_jvp_slots(state_slots)"
    );
    const float** device_tangent_slots = nullptr;
    try {
        device_tangent_slots = copy_state_slots_to_device(
            tangent_slots,
            SLOT_COUNT,
            "utd_pair_jvp_slots(tangent_slots)"
        );
        utd_pair_jvp_slots_kernel<<<grid, BLOCK>>>(
            state_index,
            rx_index,
            ownership_code,
            device_state_slots,
            rxx,
            rxy,
            rxz,
            device_tangent_slots,
            tangent_rxx,
            tangent_rxy,
            tangent_rxz,
            tangent_direct_re,
            tangent_direct_im,
            tangent_multi_re,
            tangent_multi_im,
            tangent_direct_vec_x_re,
            tangent_direct_vec_x_im,
            tangent_direct_vec_y_re,
            tangent_direct_vec_y_im,
            tangent_direct_vec_z_re,
            tangent_direct_vec_z_im,
            tangent_multi_vec_x_re,
            tangent_multi_vec_x_im,
            tangent_multi_vec_y_re,
            tangent_multi_vec_y_im,
            tangent_multi_vec_z_re,
            tangent_multi_vec_z_im,
            n_pairs,
            k,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_pair_jvp_slots_kernel launch");
        throw_cuda(cudaDeviceSynchronize(), "utd_pair_jvp_slots_kernel sync");
    } catch (...) {
        cudaFree(device_tangent_slots);
        cudaFree(device_state_slots);
        throw;
    }

    cudaFree(device_tangent_slots);
    cudaFree(device_state_slots);
}

void utd_accumulate_tiled_vjp_slots(
    const int* state_index,
    const int* rx_index,
    const int* valid_mask,
    const int* ownership_code,
    const float* const* state_slots,
    const float* rxx,
    const float* rxy,
    const float* rxz,
    const float* grad_direct_vec_x_re,
    const float* grad_direct_vec_x_im,
    const float* grad_direct_vec_y_re,
    const float* grad_direct_vec_y_im,
    const float* grad_direct_vec_z_re,
    const float* grad_direct_vec_z_im,
    const float* grad_multi_vec_x_re,
    const float* grad_multi_vec_x_im,
    const float* grad_multi_vec_y_re,
    const float* grad_multi_vec_y_im,
    const float* grad_multi_vec_z_re,
    const float* grad_multi_vec_z_im,
    float* const* grad_state_slots,
    float* grad_rx_x,
    float* grad_rx_y,
    float* grad_rx_z,
    int n_local_states,
    int n_local_receivers,
    float k,
    MaterialParams material)
{
    int n_pairs = n_local_states * n_local_receivers;
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 86;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    const float** device_state_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_accumulate_tiled_vjp_slots(state_slots)"
    );
    float** device_grad_slots = nullptr;
    try {
        device_grad_slots = copy_mutable_state_slots_to_device(
            grad_state_slots,
            SLOT_COUNT,
            "utd_accumulate_tiled_vjp_slots(grad_state_slots)"
        );
        utd_accumulate_tiled_vjp_slots_kernel<<<grid, BLOCK>>>(
            state_index,
            rx_index,
            valid_mask,
            ownership_code,
            device_state_slots,
            rxx,
            rxy,
            rxz,
            grad_direct_vec_x_re,
            grad_direct_vec_x_im,
            grad_direct_vec_y_re,
            grad_direct_vec_y_im,
            grad_direct_vec_z_re,
            grad_direct_vec_z_im,
            grad_multi_vec_x_re,
            grad_multi_vec_x_im,
            grad_multi_vec_y_re,
            grad_multi_vec_y_im,
            grad_multi_vec_z_re,
            grad_multi_vec_z_im,
            device_grad_slots,
            grad_rx_x,
            grad_rx_y,
            grad_rx_z,
            n_local_states,
            n_local_receivers,
            k,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_accumulate_tiled_vjp_slots_kernel launch");
        throw_cuda(cudaDeviceSynchronize(), "utd_accumulate_tiled_vjp_slots_kernel sync");
    } catch (...) {
        cudaFree(device_grad_slots);
        cudaFree(device_state_slots);
        throw;
    }

    cudaFree(device_grad_slots);
    cudaFree(device_state_slots);
}

void utd_pair_vjp_slots(
    const int* state_index,
    const int* rx_index,
    const int* ownership_code,
    const float* const* state_slots,
    const float* rxx,
    const float* rxy,
    const float* rxz,
    const float* grad_direct_re,
    const float* grad_direct_im,
    const float* grad_multi_re,
    const float* grad_multi_im,
    const float* grad_direct_vec_x_re,
    const float* grad_direct_vec_x_im,
    const float* grad_direct_vec_y_re,
    const float* grad_direct_vec_y_im,
    const float* grad_direct_vec_z_re,
    const float* grad_direct_vec_z_im,
    const float* grad_multi_vec_x_re,
    const float* grad_multi_vec_x_im,
    const float* grad_multi_vec_y_re,
    const float* grad_multi_vec_y_im,
    const float* grad_multi_vec_z_re,
    const float* grad_multi_vec_z_im,
    float* const* grad_state_slots,
    float* grad_rx_x,
    float* grad_rx_y,
    float* grad_rx_z,
    int n_pairs,
    float k,
    MaterialParams material)
{
    if (n_pairs <= 0) {
        return;
    }

    constexpr size_t SLOT_COUNT = 86;
    constexpr int BLOCK = 256;
    int grid = (n_pairs + BLOCK - 1) / BLOCK;
    const float** device_state_slots = copy_state_slots_to_device(
        state_slots,
        SLOT_COUNT,
        "utd_pair_vjp_slots(state_slots)"
    );
    float** device_grad_slots = nullptr;
    try {
        device_grad_slots = copy_mutable_state_slots_to_device(
            grad_state_slots,
            SLOT_COUNT,
            "utd_pair_vjp_slots(grad_state_slots)"
        );
        utd_pair_vjp_slots_kernel<<<grid, BLOCK>>>(
            state_index,
            rx_index,
            ownership_code,
            device_state_slots,
            rxx,
            rxy,
            rxz,
            grad_direct_re,
            grad_direct_im,
            grad_multi_re,
            grad_multi_im,
            grad_direct_vec_x_re,
            grad_direct_vec_x_im,
            grad_direct_vec_y_re,
            grad_direct_vec_y_im,
            grad_direct_vec_z_re,
            grad_direct_vec_z_im,
            grad_multi_vec_x_re,
            grad_multi_vec_x_im,
            grad_multi_vec_y_re,
            grad_multi_vec_y_im,
            grad_multi_vec_z_re,
            grad_multi_vec_z_im,
            device_grad_slots,
            grad_rx_x,
            grad_rx_y,
            grad_rx_z,
            n_pairs,
            k,
            material
        );
        throw_cuda(cudaGetLastError(), "utd_pair_vjp_slots_kernel launch");
        throw_cuda(cudaDeviceSynchronize(), "utd_pair_vjp_slots_kernel sync");
    } catch (...) {
        cudaFree(device_grad_slots);
        cudaFree(device_state_slots);
        throw;
    }

    cudaFree(device_grad_slots);
    cudaFree(device_state_slots);
}


} // namespace witwin::channel::native_ext

#undef WITWIN_DEVICE_NOINLINE
