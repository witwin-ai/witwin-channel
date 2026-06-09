#pragma once

#include <utd/utd_types.h>

namespace witwin::channel::native_ext {

// ===================================================================
// Safe length / normalize
// ===================================================================
UTD_DINLINE float safe_length(float3a v) {
    return sqrtf(fmaxf(f3_dot(v,v), 0.f));
}

UTD_DINLINE float3a safe_normalize(float3a v, float3a fallback) {
    float n = safe_length(v);
    if (n > UTD_SMALL_EPS) return f3_div(v, n + UTD_EPS);
    float fn = safe_length(fallback);
    return f3_div(fallback, fn + UTD_EPS);
}

UTD_DINLINE float safe_acos(float v) {
    float c = fminf(fmaxf(v, -1.f), 1.f);
    float s = sqrtf(fmaxf(1.f - c*c, 0.f));
    return atan2f(s, c);
}

UTD_DINLINE float cot_val(float v) {
    float s, c;
#ifdef __CUDACC__
    sincosf(v, &s, &c);
#else
    s = sinf(v);
    c = cosf(v);
#endif
    float d = (fabsf(s) < UTD_SMALL_EPS)
        ? ((s + UTD_SMALL_EPS) >= 0.f ? UTD_SMALL_EPS : -UTD_SMALL_EPS)
        : s;
    float r = c / d;
    return isfinite(r) ? r : 0.f;
}

// ===================================================================
// Wedge geometry helpers
// ===================================================================
UTD_DINLINE float3a project_to_wedge_plane(float3a v, float3a e) {
    return f3_sub(v, f3_mul(e, f3_dot(v,e)));
}

UTD_DINLINE float3a rotate_vector_around_axis(float3a v, float3a axis, float angle) {
    float s, c;
#ifdef __CUDACC__
    sincosf(angle, &s, &c);
#else
    s = sinf(angle);
    c = cosf(angle);
#endif
    float3a term0 = f3_mul(v, c);
    float3a term1 = f3_mul(f3_cross(axis, v), s);
    float3a term2 = f3_mul(axis, f3_dot(axis, v) * (1.f - c));
    return f3_add(f3_add(term0, term1), term2);
}

UTD_DINLINE float3a normalize_in_wedge_plane(float3a v, float3a e) {
    return safe_normalize(project_to_wedge_plane(v,e), make_f3(1,0,0));
}

UTD_DINLINE float3a stable_perp_basis(float3a rayDir, float3a preferred) {
    float3a proj = f3_sub(preferred, f3_mul(rayDir, f3_dot(preferred, rayDir)));
    float3a altAxis = (fabsf(rayDir.z) < 0.9f) ? make_f3(0,0,1) : make_f3(0,1,0);
    float3a altProj = f3_sub(altAxis, f3_mul(rayDir, f3_dot(altAxis, rayDir)));
    return safe_normalize(proj, altProj);
}

UTD_DINLINE Basis3 basis_from_first_vector(float3a rayDir, float3a firstVec, float3a fallback) {
    float3a rayHat = safe_normalize(rayDir, make_f3(0,0,1));
    float3a uVec = f3_sub(firstVec, f3_mul(rayHat, f3_dot(firstVec, rayHat)));
    float3a uHat = safe_normalize(uVec, fallback);
    float3a vFallback = stable_perp_basis(rayHat, make_f3(0,1,0));
    float3a vHat = safe_normalize(f3_cross(rayHat, uHat), vFallback);
    return {uHat, vHat, rayHat};
}

UTD_DINLINE Basis3 diffraction_edge_basis(float3a rayDir, float3a edgeDir, bool outgoing) {
    float3a rayHat = safe_normalize(rayDir, make_f3(0,0,1));
    float3a edgeHat = safe_normalize(edgeDir, make_f3(0,0,1));
    float3a phiHat = f3_cross(rayHat, edgeHat);
    if (outgoing) phiHat = f3_neg(phiHat);
    float3a fallback = stable_perp_basis(rayHat, edgeHat);
    return basis_from_first_vector(rayHat, phiHat, fallback);
}

UTD_DINLINE JonesOperator jop_in_basis(JonesOperator op,
    Basis3 srcIn, Basis3 srcOut, Basis3 dstIn, Basis3 dstOut)
{
    Jones2 unitU = {cplx(1,0), cplx_zero()};
    Jones2 unitV = {cplx_zero(), cplx(1,0)};
    Complex3 fieldU = vector_from_jones(unitU, dstIn);
    Jones2 srcU = jones_from_vector(fieldU, srcIn);
    Jones2 srcOutU = apply_jop(srcU, op);
    Jones2 mappedU = jones_from_vector(vector_from_jones(srcOutU, srcOut), dstOut);
    Complex3 fieldV = vector_from_jones(unitV, dstIn);
    Jones2 srcV = jones_from_vector(fieldV, srcIn);
    Jones2 srcOutV = apply_jop(srcV, op);
    Jones2 mappedV = jones_from_vector(vector_from_jones(srcOutV, srcOut), dstOut);
    return {mappedU.u, mappedV.u, mappedU.v, mappedV.v};
}

UTD_DINLINE Basis3 basis_zero() {
    return {f3_zero(), f3_zero(), f3_zero()};
}

UTD_DINLINE void basis_accum(Basis3& dst, Basis3 src) {
    dst.u = f3_add(dst.u, src.u);
    dst.v = f3_add(dst.v, src.v);
    dst.k = f3_add(dst.k, src.k);
}

UTD_DINLINE void adj_jop_in_basis(
    JonesOperator op,
    Basis3 srcIn,
    Basis3 srcOut,
    Basis3 dstIn,
    Basis3 dstOut,
    JonesOperator gO,
    JonesOperator& gOp,
    Basis3& gSrcIn,
    Basis3& gSrcOut,
    Basis3& gDstIn,
    Basis3& gDstOut)
{
    Jones2 unitU = {cplx(1, 0), cplx_zero()};
    Jones2 unitV = {cplx_zero(), cplx(1, 0)};

    Complex3 fieldU = vector_from_jones(unitU, dstIn);
    Jones2 srcU = jones_from_vector(fieldU, srcIn);
    Jones2 srcOutU = apply_jop(srcU, op);
    Complex3 outFieldU = vector_from_jones(srcOutU, srcOut);
    Jones2 mappedU = jones_from_vector(outFieldU, dstOut);

    Complex3 fieldV = vector_from_jones(unitV, dstIn);
    Jones2 srcV = jones_from_vector(fieldV, srcIn);
    Jones2 srcOutV = apply_jop(srcV, op);
    Complex3 outFieldV = vector_from_jones(srcOutV, srcOut);
    Jones2 mappedV = jones_from_vector(outFieldV, dstOut);

    (void) mappedU;
    (void) mappedV;

    Jones2 gMappedU = {gO.m00, gO.m10};
    Jones2 gMappedV = {gO.m01, gO.m11};

    Complex3 gOutFieldU = c3_zero();
    Basis3 gDstOutLocal = basis_zero();
    adj_jones_from_vector(outFieldU, dstOut, gMappedU, gOutFieldU, gDstOutLocal);
    basis_accum(gDstOut, gDstOutLocal);

    Jones2 gSrcOutU = jones_zero();
    Basis3 gSrcOutLocal = basis_zero();
    adj_vector_from_jones(srcOutU, srcOut, gOutFieldU, gSrcOutU, gSrcOutLocal);
    basis_accum(gSrcOut, gSrcOutLocal);

    Jones2 gSrcU = jones_zero();
    adj_apply_jop(srcU, op, gSrcOutU, gSrcU, gOp);

    Complex3 gFieldU = c3_zero();
    Basis3 gSrcInLocal = basis_zero();
    adj_jones_from_vector(fieldU, srcIn, gSrcU, gFieldU, gSrcInLocal);
    basis_accum(gSrcIn, gSrcInLocal);

    Jones2 gUnitU = jones_zero();
    Basis3 gDstInLocal = basis_zero();
    adj_vector_from_jones(unitU, dstIn, gFieldU, gUnitU, gDstInLocal);
    basis_accum(gDstIn, gDstInLocal);
    (void) gUnitU;

    Complex3 gOutFieldV = c3_zero();
    gDstOutLocal = basis_zero();
    adj_jones_from_vector(outFieldV, dstOut, gMappedV, gOutFieldV, gDstOutLocal);
    basis_accum(gDstOut, gDstOutLocal);

    Jones2 gSrcOutV = jones_zero();
    gSrcOutLocal = basis_zero();
    adj_vector_from_jones(srcOutV, srcOut, gOutFieldV, gSrcOutV, gSrcOutLocal);
    basis_accum(gSrcOut, gSrcOutLocal);

    Jones2 gSrcV = jones_zero();
    adj_apply_jop(srcV, op, gSrcOutV, gSrcV, gOp);

    Complex3 gFieldV = c3_zero();
    gSrcInLocal = basis_zero();
    adj_jones_from_vector(fieldV, srcIn, gSrcV, gFieldV, gSrcInLocal);
    basis_accum(gSrcIn, gSrcInLocal);

    Jones2 gUnitV = jones_zero();
    gDstInLocal = basis_zero();
    adj_vector_from_jones(unitV, dstIn, gFieldV, gUnitV, gDstInLocal);
    basis_accum(gDstIn, gDstInLocal);
    (void) gUnitV;
}

// ===================================================================
// Exterior region / pole safety
// ===================================================================
UTD_DINLINE bool wedge_exterior_mask(float3a dirFromEdge, float3a edgeDir,
                                     float3a n0, float3a nn) {
    float3a dp = project_to_wedge_plane(dirFromEdge, edgeDir);
    float sd0 = f3_dot(dp, n0);
    float sdn = f3_dot(dp, nn);
    return (safe_length(dp) > UTD_SMALL_EPS) &&
           ((sd0 >= -UTD_SMALL_EPS) || (sdn >= -UTD_SMALL_EPS));
}

UTD_DINLINE float distance_to_cot_pole(float v) {
    float np = roundf(v / UTD_PI) * UTD_PI;
    return fabsf(v - np);
}

UTD_DINLINE bool cot_pole_safe_mask(float phi, float phiP, float n, float guard) {
    float twoN = 2.f * n;
    float args[4] = {
        (UTD_PI + phi - phiP) / twoN,
        (UTD_PI - phi + phiP) / twoN,
        (UTD_PI + phi + phiP) / twoN,
        (UTD_PI - phi - phiP) / twoN
    };
    for (int i = 0; i < 4; ++i)
        if (distance_to_cot_pole(args[i]) <= guard) return false;
    return true;
}

UTD_DINLINE bool slope_safe_mask(float phi, float phiP, float n, float step) {
    float npi = n * UTD_PI;
    bool interior = (phi >= step) && (phi <= npi-step) &&
                    (phiP >= step) && (phiP <= npi-step);
    float guard = step / (2.f * n);
    return interior && cot_pole_safe_mask(phi, phiP, n, guard);
}

// ===================================================================
// Boersma Fresnel integral with 1st and 2nd derivatives
// ===================================================================
UTD_DINLINE void poly12(float x,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6, float c7,
    float c8, float c9, float c10, float c11,
    float& val, float& fst, float& snd)
{
    val = c11; fst = 0.f; snd = 0.f;
    #define POLY_STEP(ci) snd = snd*x + 2.f*fst; fst = fst*x + val; val = val*x + ci;
    POLY_STEP(c10) POLY_STEP(c9) POLY_STEP(c8) POLY_STEP(c7)
    POLY_STEP(c6)  POLY_STEP(c5) POLY_STEP(c4) POLY_STEP(c3)
    POLY_STEP(c2)  POLY_STEP(c1) POLY_STEP(c0)
    #undef POLY_STEP
}

UTD_DINLINE void fresnel_boersma(float x, Complex& val, Complex& fst, Complex& snd) {
    const float SE = 1.0e-12f;
    bool xPos = x >= 0.f;
    float xA = fabsf(x);
    float safeX = fmaxf(xA, SE);
    bool cond = xA < 4.f;

    float argS = 0.25f * xA;
    float argL = 4.f / safeX;
    float a1S = 0.25f, a2S = 0.f;
    float a1L = -4.f/(safeX*safeX);
    float a2L = 8.f/(safeX*safeX*safeX);

    float rS, rS1, rS2;
    poly12(argS, +1.595769140f,-0.000001702f,-6.808568854f,-0.000576361f,
           +6.920691902f,-0.016898657f,-3.050485660f,-0.075752419f,
           +0.850663781f,-0.025639041f,-0.150230960f,+0.034404779f, rS,rS1,rS2);
    float iS,iS1,iS2;
    poly12(argS, -0.000000033f,+4.255387524f,-0.000092810f,-7.780020400f,
           -0.009520895f,+5.075161298f,-0.138341947f,-1.363729124f,
           -0.403349276f,+0.702222016f,-0.216195929f,+0.019547031f, iS,iS1,iS2);
    float rL,rL1,rL2;
    poly12(argL, +0.000000000f,-0.024933975f,+0.000003936f,+0.005770956f,
           +0.000689892f,-0.009497136f,+0.011948809f,-0.006748873f,
           +0.000246420f,+0.002102967f,-0.001217930f,+0.000233939f, rL,rL1,rL2);
    float iL,iL1,iL2;
    poly12(argL, +0.199471140f,+0.000000023f,-0.009351341f,+0.000023006f,
           +0.004851466f,+0.001903218f,-0.017122914f,+0.029064067f,
           -0.027928955f,+0.016497308f,-0.005598515f,+0.000838386f, iL,iL1,iL2);

    float rC  = cond ? rS : rL;
    float iC  = cond ? iS : iL;
    float rC1 = cond ? rS1*a1S : rL1*a1L;
    float iC1 = cond ? iS1*a1S : iL1*a1L;
    float rC2 = cond ? rS2*a1S*a1S : rL2*a1L*a1L + rL1*a2L;
    float iC2 = cond ? iS2*a1S*a1S : iL2*a1L*a1L + iL1*a2L;

    float arg = cond ? argS : argL;
    float a1  = cond ? a1S : a1L;
    float a2  = cond ? a2S : a2L;
    float argSafe = fmaxf(arg, SE);
    float aSqrt  = sqrtf(argSafe);
    float aSqrt1 = 0.5f*a1/aSqrt;
    float aSqrt2 = 0.5f*a2/aSqrt - 0.25f*a1*a1/(argSafe*aSqrt);

    float rP  = rC*aSqrt;
    float rP1 = rC1*aSqrt + rC*aSqrt1;
    float rP2 = rC2*aSqrt + 2.f*rC1*aSqrt1 + rC*aSqrt2;
    float iP  = -iC*aSqrt;
    float iP1 = -(iC1*aSqrt + iC*aSqrt1);
    float iP2 = -(iC2*aSqrt + 2.f*iC1*aSqrt1 + iC*aSqrt2);

    float sinX, cosX;
#ifdef __CUDACC__
    sincosf(xA, &sinX, &cosX);
#else
    sinX = sinf(xA);
    cosX = cosf(xA);
#endif
    float vR = cosX*rP - sinX*iP;
    float vI = cosX*iP + sinX*rP;
    float f1R = cosX*(rP1-iP) - sinX*(rP+iP1);
    float f1I = cosX*(iP1+rP) + sinX*(rP1-iP);
    float f2R = cosX*(rP2-rP-2.f*iP1) - sinX*(2.f*rP1-iP+iP2);
    float f2I = cosX*(iP2+2.f*rP1-iP) + sinX*(rP2-rP-2.f*iP1);

    if (!cond) { vR += 0.5f; vI += 0.5f; }

    val = cplx(xPos ? vR : -vR, xPos ? vI : -vI);
    fst = cplx(f1R, f1I);
    snd = cplx(xPos ? f2R : -f2R, xPos ? f2I : -f2I);
}

UTD_DINLINE float first_order_diffraction_parameter(
    float3a sourcePos,
    float3a targetPos,
    float3a edgeOrigin,
    float3a edgeDir)
{
    float3a zeta = safe_normalize(edgeDir, make_f3(0.f, 0.f, 1.f));
    float3a targetOffset = f3_sub(targetPos, edgeOrigin);
    float3a sourceOffset = f3_sub(sourcePos, edgeOrigin);
    float3a targetProjection = f3_mul(zeta, f3_dot(targetOffset, zeta));
    float3a sourceProjection = f3_mul(zeta, f3_dot(sourceOffset, zeta));
    float3a targetRadial = f3_sub(targetOffset, targetProjection);
    float3a sourceRadial = f3_sub(sourceOffset, sourceProjection);
    float targetRadialNorm = safe_length(targetRadial);
    float sourceRadialNorm = safe_length(sourceRadial);
    float3a v1 = f3_div(targetRadial, fmaxf(targetRadialNorm, UTD_SMALL_EPS));
    float3a v2 = f3_div(sourceRadial, fmaxf(sourceRadialNorm, UTD_SMALL_EPS));
    float theta = UTD_PI - safe_acos(f3_dot(v1, v2));
    float3a rotationAxis = f3_cross(sourceRadial, targetRadial);
    float rotationAxisNorm = safe_length(rotationAxis);
    rotationAxis = rotationAxisNorm > UTD_SMALL_EPS
        ? f3_div(rotationAxis, rotationAxisNorm + UTD_EPS)
        : zeta;
    float3a coplanarTarget = rotate_vector_around_axis(targetOffset, rotationAxis, theta);
    float3a sourceToTarget = f3_sub(coplanarTarget, sourceOffset);
    float sourceToTargetNorm = safe_length(sourceToTarget);
    float3a u0 = f3_div(sourceToTarget, fmaxf(sourceToTargetNorm, UTD_SMALL_EPS));
    float3a u1 = f3_cross(sourceOffset, u0);
    float3a u2 = f3_cross(zeta, u0);
    float u2Norm = safe_length(u2);
    float sign = f3_dot(u1, u2) >= 0.f ? 1.f : -1.f;
    return sign * safe_length(u1) / fmaxf(u2Norm, UTD_SMALL_EPS);
}

struct FiniteEdgePointSelection {
    float3a point;
    float edgeLineMin;
    float edgeLineMax;
    bool valid;
    bool inside;
};

UTD_DINLINE FiniteEdgePointSelection finite_edge_diffraction_point(
    PairInputs state,
    float3a targetPos)
{
    float3a edgeHat = safe_normalize(state.edgeDir, make_f3(0.f, 0.f, 1.f));
    float3a edgeOrigin = f3_add(state.edgePos, f3_mul(edgeHat, state.edgeLineMin));
    float edgeLength = state.edgeLineMax - state.edgeLineMin;
    float parameter = first_order_diffraction_parameter(
        state.sourcePos,
        targetPos,
        edgeOrigin,
        edgeHat
    );
    bool valid = (edgeLength > UTD_SMALL_EPS) && isfinite(parameter);
    bool inside = valid && (parameter > 0.f) && (parameter < edgeLength);
    return {
        f3_add(edgeOrigin, f3_mul(edgeHat, parameter)),
        -parameter,
        edgeLength - parameter,
        valid,
        inside,
    };
}

UTD_DINLINE PairInputs pair_state_at_stationary_point(
    PairInputs state,
    float3a targetPos,
    bool& selected,
    bool& inside,
    bool& valid)
{
    selected = false;
    inside = false;
    valid = true;
    if (state.selectStationaryPoint <= 0.5f) {
        return state;
    }
    FiniteEdgePointSelection point = finite_edge_diffraction_point(state, targetPos);
    if (!point.valid) {
        valid = false;
        return state;
    }
    state.edgePos = point.point;
    state.edgeLineMin = point.edgeLineMin;
    state.edgeLineMax = point.edgeLineMax;
    selected = true;
    inside = point.inside;
    return state;
}

UTD_DINLINE Complex direct_source_field(float3a sourcePos, float3a targetPos, float k) {
    float distance = safe_length(f3_sub(targetPos, sourcePos)) + UTD_EPS;
    float fspl = 1.f / (2.f * fmaxf(k, UTD_SMALL_EPS) * distance);
    return cplx_mul_real(cplx_exp_phase(-k * distance), fspl);
}

UTD_DINLINE Complex3 direct_source_vector(
    float3a sourcePos,
    float3a targetPos,
    float k,
    MaterialParams mat)
{
    float3a rayDir = safe_normalize(f3_sub(targetPos, sourcePos), make_f3(0.f, 0.f, 1.f));
    float3a txPol = make_f3(mat.txPolX, mat.txPolY, mat.txPolZ);
    float3a polDir = stable_perp_basis(rayDir, txPol);
    return cplx_scale_real(polDir, direct_source_field(sourcePos, targetPos, k));
}

UTD_DINLINE float poly12_value(float x,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6, float c7,
    float c8, float c9, float c10, float c11)
{
    float val = c11;
    val = val * x + c10;
    val = val * x + c9;
    val = val * x + c8;
    val = val * x + c7;
    val = val * x + c6;
    val = val * x + c5;
    val = val * x + c4;
    val = val * x + c3;
    val = val * x + c2;
    val = val * x + c1;
    val = val * x + c0;
    return val;
}

UTD_DINLINE Complex fresnel_boersma_value(float x) {
    bool xPos = x > 0.f;
    float xA = fabsf(x);
    bool cond = xA < 4.f;
    float arg = cond ? (0.25f * xA) : (4.f / xA);
    float root = sqrtf(arg);

    float rS = poly12_value(arg, +1.595769140f,-0.000001702f,-6.808568854f,-0.000576361f,
        +6.920691902f,-0.016898657f,-3.050485660f,-0.075752419f,
        +0.850663781f,-0.025639041f,-0.150230960f,+0.034404779f);
    float iS = poly12_value(arg, -0.000000033f,+4.255387524f,-0.000092810f,-7.780020400f,
        -0.009520895f,+5.075161298f,-0.138341947f,-1.363729124f,
        -0.403349276f,+0.702222016f,-0.216195929f,+0.019547031f);
    float rL = poly12_value(arg, +0.000000000f,-0.024933975f,+0.000003936f,+0.005770956f,
        +0.000689892f,-0.009497136f,+0.011948809f,-0.006748873f,
        +0.000246420f,+0.002102967f,-0.001217930f,+0.000233939f);
    float iL = poly12_value(arg, +0.199471140f,+0.000000023f,-0.009351341f,+0.000023006f,
        +0.004851466f,+0.001903218f,-0.017122914f,+0.029064067f,
        -0.027928955f,+0.016497308f,-0.005598515f,+0.000838386f);

    float rP = (cond ? rS : rL) * root;
    float iP = -(cond ? iS : iL) * root;
    float sinX, cosX;
#ifdef __CUDACC__
    sincosf(xA, &sinX, &cosX);
#else
    sinX = sinf(xA);
    cosX = cosf(xA);
#endif
    float vR = cosX * rP - sinX * iP;
    float vI = cosX * iP + sinX * rP;
    if (!cond) { vR += 0.5f; vI += 0.5f; }
    return cplx(xPos ? vR : -vR, xPos ? vI : -vI);
}

// ===================================================================
// UTD transition function f(x) with 1st and 2nd derivatives
// ===================================================================
UTD_DINLINE void f_utd_with_derivatives(float x, Complex& val, Complex& fst, Complex& snd) {
    float sx = fmaxf(x, UTD_SMALL_EPS);
    Complex fV, fF, fS;
    fresnel_boersma(x, fV, fF, fS);
    Complex fcV = cplx_conj(fV), fcF = cplx_conj(fF), fcS = cplx_conj(fS);

    float pf  = sqrtf(UTD_PI*sx*0.5f);
    float pf1 = 0.5f*pf/sx;
    float pf2 = -0.25f*pf/(sx*sx);
    Complex ph  = cplx_exp_phase(x);
    Complex ph1 = cplx_mul(cplx(0,1), ph);
    Complex ph2 = cplx(-ph.re, -ph.im);
    Complex br  = cplx_sub(cplx(1,1), cplx_mul(cplx(0,2), fcV));
    Complex br1 = cplx_mul(cplx(0,-2), fcF);
    Complex br2 = cplx_mul(cplx(0,-2), fcS);

    val = cplx_mul_real(cplx_mul(ph, br), pf);
    fst = cplx_add(cplx_add(
        cplx_mul_real(cplx_mul(ph, br), pf1),
        cplx_mul_real(cplx_mul(ph1, br), pf)),
        cplx_mul_real(cplx_mul(ph, br1), pf));
    snd = cplx_add(
        cplx_add(
            cplx_add(
                cplx_mul_real(cplx_mul(ph, br), pf2),
                cplx_mul_real(cplx_mul(ph1, br), 2.f*pf1)),
            cplx_mul_real(cplx_mul(ph, br1), 2.f*pf1)),
        cplx_add(
            cplx_add(
                cplx_mul_real(cplx_mul(ph2, br), pf),
                cplx_mul_real(cplx_mul(ph1, br1), 2.f*pf)),
        cplx_mul_real(cplx_mul(ph, br2), pf)));
}

UTD_DINLINE Complex f_utd_value(float x) {
    float sx = fmaxf(x, 0.0f);
    Complex fV = fresnel_boersma_value(x);
    Complex bracket = cplx_sub(cplx(1.0f, 1.0f), cplx_mul(cplx(0.0f, 2.0f), cplx_conj(fV)));
    Complex phase = cplx_exp_phase(x);
    float prefactor = sqrtf(UTD_PI * sx * 0.5f);
    return cplx_mul_real(cplx_mul(phase, bracket), prefactor);
}

// ===================================================================
// Beta term values + assembly
// ===================================================================
UTD_DINLINE float shadow_a_threshold(float n) {
    return 8.0e-12f * fmaxf(n * n, 1.0f);
}

UTD_DINLINE Complex cot_transition_product_value(
    float cotV,
    Complex transition,
    float x,
    float x1,
    float kL,
    float n,
    float cotSign)
{
    Complex raw = cplx_mul_real(transition, cotV);
    float safeKL = fabsf(kL) > UTD_EPS ? kL : 0.0f;
    float a = safeKL != 0.0f ? fmaxf(x / safeKL, 0.0f) : 0.0f;
    float a1 = safeKL != 0.0f ? x1 / safeKL : 0.0f;
    float threshold = shadow_a_threshold(n);
    if (a > threshold) {
        return raw;
    }

    float fallbackSign = cotV >= 0.0f ? 1.0f : -1.0f;
    float a1Sign = a1 >= 0.0f ? 1.0f : -1.0f;
    float limitSign = fabsf(a1) > UTD_SMALL_EPS ? cotSign * a1Sign : fallbackSign;
    float limitScale = limitSign * n * sqrtf(UTD_PI * fmaxf(kL, 0.0f));
    Complex limit = cplx(limitScale, limitScale);
    float blend = fminf(1.0f, a / fmaxf(threshold, 1.0e-20f));
    return cplx_add(limit, cplx_mul_real(cplx_sub(raw, limit), blend));
}

UTD_DINLINE void beta_term_values(float beta, float n, float kL, float cotSign,
    bool plusBranch, float& cotV, float& c1, float& c2,
    float& xo, float& x1, float& x2)
{
    float twoN = 2.f*n;
    float twoNPi = 2.f*n*UTD_PI;
    float ri = plusBranch ? roundf((beta+UTD_PI)/twoNPi) : roundf((beta-UTD_PI)/twoNPi);
    float po = twoNPi*ri - beta;
    float chp = cosf(0.5f*po);
    float a = 2.f*chp*chp;
    float a1v = sinf(po);
    float a2v = 1.f-a;
    float ca = (UTD_PI + cotSign*beta)/twoN;
    cotV = cot_val(ca);
    c1 = -(cotSign/twoN)*(1.f + cotV*cotV);
    c2 = 0.5f*cotV*(1.f + cotV*cotV)/(n*n);
    xo = kL*a;
    x1 = kL*a1v;
    x2 = kL*a2v;
}

UTD_DINLINE void assemble_beta_term(float cotV, float c1, float c2,
    float x, float x1, float x2, float kL, float n, float cotSign,
    Complex tr, Complex tr1, Complex tr2,
    Complex& val, Complex& fst, Complex& snd)
{
    Complex forwardTransition = f_utd_value(x);
    val = cot_transition_product_value(cotV, forwardTransition, x, x1, kL, n, cotSign);
    fst = cplx_add(cplx_mul_real(tr, c1), cplx_mul_real(tr1, cotV*x1));
    snd = cplx_add(
        cplx_add(cplx_mul_real(tr, c2), cplx_mul_real(tr1, 2.f*c1*x1)),
        cplx_mul_real(cplx_add(cplx_mul_real(tr2, x1*x1), cplx_mul_real(tr1, x2)), cotV));
}

// ===================================================================
// Diffraction beta groups (2D / 3D)
// ===================================================================
UTD_DINLINE float endpoint_unpaired_direct_beta(float beta, float n) {
    float period = fmaxf(2.f * n * UTD_PI, UTD_SMALL_EPS);
    float centered = beta - period * floorf((beta + 0.5f * period) / period);
    if (centered > UTD_PI) {
        return 2.f * UTD_PI - centered;
    }
    if (centered < -UTD_PI) {
        return -2.f * UTD_PI - centered;
    }
    return centered;
}

UTD_DINLINE void diffraction_beta_groups_from_betas(float dP, float sP2, float n, float k,
    float s, float sP, Complex r0, Complex rn,
    Complex& factor, Complex& dG, Complex& dG1,
    Complex& sG, Complex& sG1, Complex& dG2, Complex& sG2)
{
    float l = s*sP/(s+sP+UTD_EPS);
    float kL = k*l;
    factor = cplx_mul_real(cplx_exp_phase(-0.25f*UTD_PI),
                           -1.f/(2.f*n*sqrtf(UTD_TWO_PI*k+UTD_EPS)));
    float cv[4],c1[4],c2[4],xv[4],x1[4],x2[4];
    beta_term_values(dP, n, kL, +1.f, true,  cv[0],c1[0],c2[0],xv[0],x1[0],x2[0]);
    beta_term_values(dP, n, kL, -1.f, false, cv[1],c1[1],c2[1],xv[1],x1[1],x2[1]);
    beta_term_values(sP2,n, kL, +1.f, true,  cv[2],c1[2],c2[2],xv[2],x1[2],x2[2]);
    beta_term_values(sP2,n, kL, -1.f, false, cv[3],c1[3],c2[3],xv[3],x1[3],x2[3]);
    Complex tr[4],tr1[4],tr2[4];
    for (int i=0;i<4;++i) f_utd_with_derivatives(xv[i],tr[i],tr1[i],tr2[i]);
    Complex tv[4],tf[4],ts[4];
    for (int i=0;i<4;++i) {
        float cotSign = (i == 0 || i == 2) ? +1.f : -1.f;
        assemble_beta_term(cv[i],c1[i],c2[i],xv[i],x1[i],x2[i],kL,n,cotSign,tr[i],tr1[i],tr2[i],tv[i],tf[i],ts[i]);
    }
    dG  = cplx_add(tv[0],tv[1]);
    dG1 = cplx_add(tf[0],tf[1]);
    dG2 = cplx_add(ts[0],ts[1]);
    sG  = cplx_add(cplx_mul(rn,tv[2]), cplx_mul(r0,tv[3]));
    sG1 = cplx_add(cplx_mul(rn,tf[2]), cplx_mul(r0,tf[3]));
    sG2 = cplx_add(cplx_mul(rn,ts[2]), cplx_mul(r0,ts[3]));
}

UTD_DINLINE void diffraction_beta_groups(float phi, float phiP, float n, float k,
    float s, float sP, Complex r0, Complex rn,
    Complex& factor, Complex& dG, Complex& dG1,
    Complex& sG, Complex& sG1, Complex& dG2, Complex& sG2)
{
    diffraction_beta_groups_from_betas(phi - phiP, phi + phiP, n, k, s, sP,
                                       r0, rn, factor, dG, dG1, sG, sG1, dG2, sG2);
}

UTD_DINLINE void diffraction_beta_groups_with_direct_beta(float phi, float phiP,
    float directBeta, float n, float k, float s, float sP, Complex r0, Complex rn,
    Complex& factor, Complex& dG, Complex& dG1,
    Complex& sG, Complex& sG1, Complex& dG2, Complex& sG2)
{
    diffraction_beta_groups_from_betas(directBeta, phi + phiP, n, k, s, sP,
                                       r0, rn, factor, dG, dG1, sG, sG1, dG2, sG2);
}

UTD_DINLINE void diffraction_beta_groups_3d_from_betas(float dP, float sP2, float n, float k,
    float s, float sP, float sinBeta0, Complex r0, Complex rn,
    Complex& factor, Complex& dG, Complex& dG1,
    Complex& sG, Complex& sG1, Complex& dG2, Complex& sG2)
{
    float sb = fmaxf(sinBeta0, UTD_SMALL_EPS);
    float l = s*sP/(s+sP+UTD_EPS)*sb*sb;
    float kL = k*l;
    factor = cplx_mul_real(cplx_exp_phase(-0.25f*UTD_PI),
                           -1.f/(2.f*n*sqrtf(UTD_TWO_PI*k+UTD_EPS)*sb));
    float cv[4],c1[4],c2[4],xv[4],x1[4],x2[4];
    beta_term_values(dP, n, kL, +1.f, true,  cv[0],c1[0],c2[0],xv[0],x1[0],x2[0]);
    beta_term_values(dP, n, kL, -1.f, false, cv[1],c1[1],c2[1],xv[1],x1[1],x2[1]);
    beta_term_values(sP2,n, kL, +1.f, true,  cv[2],c1[2],c2[2],xv[2],x1[2],x2[2]);
    beta_term_values(sP2,n, kL, -1.f, false, cv[3],c1[3],c2[3],xv[3],x1[3],x2[3]);
    Complex tr[4],tr1[4],tr2[4];
    for (int i=0;i<4;++i) f_utd_with_derivatives(xv[i],tr[i],tr1[i],tr2[i]);
    Complex tv[4],tf[4],ts[4];
    for (int i=0;i<4;++i) {
        float cotSign = (i == 0 || i == 2) ? +1.f : -1.f;
        assemble_beta_term(cv[i],c1[i],c2[i],xv[i],x1[i],x2[i],kL,n,cotSign,tr[i],tr1[i],tr2[i],tv[i],tf[i],ts[i]);
    }
    dG  = cplx_add(tv[0],tv[1]);
    dG1 = cplx_add(tf[0],tf[1]);
    dG2 = cplx_add(ts[0],ts[1]);
    sG  = cplx_add(cplx_mul(rn,tv[2]), cplx_mul(r0,tv[3]));
    sG1 = cplx_add(cplx_mul(rn,tf[2]), cplx_mul(r0,tf[3]));
    sG2 = cplx_add(cplx_mul(rn,ts[2]), cplx_mul(r0,ts[3]));
}

UTD_DINLINE void diffraction_beta_groups_3d(float phi, float phiP, float n, float k,
    float s, float sP, float sinBeta0, Complex r0, Complex rn,
    Complex& factor, Complex& dG, Complex& dG1,
    Complex& sG, Complex& sG1, Complex& dG2, Complex& sG2)
{
    diffraction_beta_groups_3d_from_betas(phi - phiP, phi + phiP, n, k, s, sP,
                                          sinBeta0, r0, rn, factor, dG, dG1,
                                          sG, sG1, dG2, sG2);
}

UTD_DINLINE void diffraction_beta_groups_3d_with_direct_beta(float phi, float phiP,
    float directBeta, float n, float k, float s, float sP, float sinBeta0,
    Complex r0, Complex rn, Complex& factor, Complex& dG, Complex& dG1,
    Complex& sG, Complex& sG1, Complex& dG2, Complex& sG2)
{
    diffraction_beta_groups_3d_from_betas(directBeta, phi + phiP, n, k, s, sP,
                                          sinBeta0, r0, rn, factor, dG, dG1,
                                          sG, sG1, dG2, sG2);
}

// ===================================================================
// Diffraction coefficients (2D / 3D)
// ===================================================================
UTD_DINLINE Complex diff_coeff_2d(float phi, float phiP, float n, float k,
                                   float s, float sP, Complex r0, Complex rn) {
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups(phi,phiP,n,k,s,sP,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

UTD_DINLINE Complex diff_coeff_2d_endpoint_continued(float phi, float phiP,
    float n, float k, float s, float sP, Complex r0, Complex rn) {
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    float directBeta = endpoint_unpaired_direct_beta(phi - phiP, n);
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,n,k,s,sP,r0,rn,
                                             fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

UTD_DINLINE Complex diff_coeff_3d(float phi, float phiP, float n, float k,
    float s, float sP, float sb, Complex r0, Complex rn) {
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups_3d(phi,phiP,n,k,s,sP,sb,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

UTD_DINLINE Complex diff_coeff_3d_endpoint_continued(float phi, float phiP,
    float n, float k, float s, float sP, float sb, Complex r0, Complex rn) {
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    float directBeta = endpoint_unpaired_direct_beta(phi - phiP, n);
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,n,k,s,sP,sb,
                                                r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    return cplx_mul(fac, cplx_add(dG,sG));
}

UTD_DINLINE Complex diff_coeff_2d_angle_deriv(float phi, float phiP, float n, float k,
    float s, float sP, bool wrtPhi, Complex r0, Complex rn) {
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups(phi,phiP,n,k,s,sP,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    Complex combined = wrtPhi ? cplx_add(dG1,sG1)
                              : cplx_add(cplx_mul_real(dG1,-1.f), sG1);
    return cplx_mul(fac, combined);
}

UTD_DINLINE Complex diff_coeff_3d_angle_deriv(float phi, float phiP, float n, float k,
    float s, float sP, float sb, bool wrtPhi, Complex r0, Complex rn) {
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups_3d(phi,phiP,n,k,s,sP,sb,r0,rn,fac,dG,dG1,sG,sG1,dG2,sG2);
    Complex combined = wrtPhi ? cplx_add(dG1,sG1)
                              : cplx_add(cplx_mul_real(dG1,-1.f), sG1);
    return cplx_mul(fac, combined);
}

UTD_DINLINE Complex slope_diff_2d(float phi, float phiP, float n, float k,
                                  float s, float sP, Complex r0, Complex rn) {
    Complex d = diff_coeff_2d_angle_deriv(phi,phiP,n,k,s,sP,false,r0,rn);
    return cplx_div_real(cplx_mul(cplx(0,-1), d), k);
}

UTD_DINLINE Complex slope_diff_3d(float phi, float phiP, float n, float k,
    float s, float sP, float sb, Complex r0, Complex rn) {
    Complex d = diff_coeff_3d_angle_deriv(phi,phiP,n,k,s,sP,sb,false,r0,rn);
    return cplx_div_real(cplx_mul(cplx(0,-1), d), k);
}

// ===================================================================
// Edge angle computation
// ===================================================================
UTD_DINLINE float oriented_angle_positive(float y, float x) {
    float a = atan2f(y, x);
    return a < 0.f ? a + UTD_TWO_PI : a;
}

UTD_DINLINE void compute_edge_angles(float3a srcPos, float3a edgePos, float3a edgeDir,
    float3a n0, float3a tgtPos,
    float& phi, float& phiP, float& s, float& sP)
{
    float3a srcToEdge = f3_sub(edgePos, srcPos);
    float3a srcProj = project_to_wedge_plane(srcToEdge, edgeDir);
    sP = safe_length(srcProj) + UTD_EPS;
    float3a toHat = safe_normalize(f3_cross(n0, edgeDir), make_f3(0,1,0));
    float3a kiProj = f3_div(srcProj, sP);
    float signP = ((-f3_dot(kiProj, n0)) >= 0.f ? 1.f : -1.f);
    phiP = UTD_PI - safe_acos(-f3_dot(kiProj, toHat));
    phiP = phiP * (-signP) + UTD_PI;

    float3a edgeToTgt = f3_sub(tgtPos, edgePos);
    float3a tgtProj = project_to_wedge_plane(edgeToTgt, edgeDir);
    s = safe_length(tgtProj) + UTD_EPS;
    float3a koProj = f3_div(tgtProj, s);
    float signPhi = (f3_dot(koProj, n0) >= 0.f ? 1.f : -1.f);
    phi = UTD_PI - safe_acos(f3_dot(koProj, toHat));
    phi = phi * (-signPhi) + UTD_PI;
}

UTD_DINLINE void compute_edge_geometry_3d(float3a srcPos, float3a edgePos, float3a edgeDir,
    float3a n0, float3a tgtPos,
    float& phi, float& phiP, float& s, float& sP, float& sinBeta0)
{
    float sProj, sPProj;
    compute_edge_angles(srcPos, edgePos, edgeDir, n0, tgtPos, phi, phiP, sProj, sPProj);
    float3a srcToEdge = f3_sub(edgePos, srcPos);
    float3a edgeToTgt = f3_sub(tgtPos, edgePos);
    sP = safe_length(srcToEdge) + UTD_EPS;
    s  = safe_length(edgeToTgt) + UTD_EPS;
    float sbP = fminf(fmaxf(sPProj/sP, UTD_SMALL_EPS), 1.f);
    float sb  = fminf(fmaxf(sProj/s,  UTD_SMALL_EPS), 1.f);
    sinBeta0 = sqrtf(fmaxf(sb*sbP, UTD_SMALL_EPS));
}

UTD_DINLINE void adj_normalize_branch(float3a v, float3a gO, float3a& gV) {
    float vn = safe_length(v);
    if (vn <= UTD_SMALL_EPS) return;
    float d = vn + UTD_EPS;
    float dg = f3_dot(gO, v);
    gV = f3_add(gV, f3_sub(f3_div(gO, d), f3_mul(v, dg / (vn * d * d))));
}

UTD_DINLINE void adj_safe_normalize(float3a v, float3a fallback, float3a gO,
                                    float3a& gV, float3a& gFallback) {
    float vn = safe_length(v);
    if (vn > UTD_SMALL_EPS) {
        adj_normalize_branch(v, gO, gV);
    } else {
        adj_normalize_branch(fallback, gO, gFallback);
    }
}

UTD_DINLINE void adj_project_to_wedge_plane(float3a v, float3a e, float3a gO,
                                            float3a& gV, float3a& gE) {
    float ve = f3_dot(v, e);
    gV = f3_add(gV, gO);
    gE = f3_sub(gE, f3_mul(gO, ve));
    float gVe = -f3_dot(gO, e);
    gV = f3_add(gV, f3_mul(e, gVe));
    gE = f3_add(gE, f3_mul(v, gVe));
}

UTD_DINLINE void adj_safe_acos(float v, float gO, float& gV) {
    if (v <= -1.f || v >= 1.f) return;
    float denom = sqrtf(fmaxf(1.f - v * v, 0.f) + UTD_SMALL_EPS);
    gV += -gO / denom;
}

UTD_DINLINE void adj_stable_perp_basis(float3a rayDir, float3a preferred, float3a gO,
                                       float3a& gRayDir, float3a& gPreferred) {
    float projDot = f3_dot(preferred, rayDir);
    float3a proj = f3_sub(preferred, f3_mul(rayDir, projDot));
    float3a altAxis = (fabsf(rayDir.z) < 0.9f) ? make_f3(0,0,1) : make_f3(0,1,0);
    float altDot = f3_dot(altAxis, rayDir);
    float3a altProj = f3_sub(altAxis, f3_mul(rayDir, altDot));

    float3a gProj = f3_zero();
    float3a gAltProj = f3_zero();
    adj_safe_normalize(proj, altProj, gO, gProj, gAltProj);

    gPreferred = f3_add(gPreferred, gProj);
    gRayDir = f3_sub(gRayDir, f3_mul(gProj, projDot));
    float gProjDot = -f3_dot(gProj, rayDir);
    gPreferred = f3_add(gPreferred, f3_mul(rayDir, gProjDot));
    gRayDir = f3_add(gRayDir, f3_mul(preferred, gProjDot));

    gRayDir = f3_sub(gRayDir, f3_mul(gAltProj, altDot));
    float gAltDot = -f3_dot(gAltProj, rayDir);
    gRayDir = f3_add(gRayDir, f3_mul(altAxis, gAltDot));
}

UTD_DINLINE void adj_basis_from_first_vector(float3a rayDir, float3a firstVec, float3a fallback,
                                             Basis3 gO,
                                             float3a& gRayDir, float3a& gFirstVec, float3a& gFallback) {
    float3a rayHat = safe_normalize(rayDir, make_f3(0,0,1));
    float firstDot = f3_dot(firstVec, rayHat);
    float3a uVec = f3_sub(firstVec, f3_mul(rayHat, firstDot));
    float3a uHat = safe_normalize(uVec, fallback);
    float3a vFallback = stable_perp_basis(rayHat, make_f3(0,1,0));
    float3a vBase = f3_cross(rayHat, uHat);
    float3a vHat = safe_normalize(vBase, vFallback);

    float3a gRayHat = gO.k;
    float3a gUHat = gO.u;
    float3a gVBase = f3_zero();
    float3a gVFallback = f3_zero();
    adj_safe_normalize(vBase, vFallback, gO.v, gVBase, gVFallback);

    gRayHat = f3_add(gRayHat, f3_cross(uHat, gVBase));
    gUHat = f3_add(gUHat, f3_cross(gVBase, rayHat));
    adj_stable_perp_basis(rayHat, make_f3(0,1,0), gVFallback, gRayHat, gFallback);

    float3a gUVec = f3_zero();
    adj_safe_normalize(uVec, fallback, gUHat, gUVec, gFallback);

    gFirstVec = f3_add(gFirstVec, gUVec);
    gRayHat = f3_sub(gRayHat, f3_mul(gUVec, firstDot));
    float gFirstDot = -f3_dot(gUVec, rayHat);
    gFirstVec = f3_add(gFirstVec, f3_mul(rayHat, gFirstDot));
    gRayHat = f3_add(gRayHat, f3_mul(firstVec, gFirstDot));

    float3a gRayFallback = f3_zero();
    adj_safe_normalize(rayDir, make_f3(0,0,1), gRayHat, gRayDir, gRayFallback);
}

UTD_DINLINE void adj_diffraction_edge_basis(float3a rayDir, float3a edgeDir, bool outgoing,
                                            Basis3 gO,
                                            float3a& gRayDir, float3a& gEdgeDir) {
    float3a rayHat = safe_normalize(rayDir, make_f3(0,0,1));
    float3a edgeHat = safe_normalize(edgeDir, make_f3(0,0,1));
    float3a phiHat = f3_cross(rayHat, edgeHat);
    if (outgoing) phiHat = f3_neg(phiHat);
    float3a fallback = stable_perp_basis(rayHat, edgeHat);

    float3a gRayHat = f3_zero();
    float3a gEdgeHat = f3_zero();
    float3a gPhiHat = f3_zero();
    float3a gFallback = f3_zero();
    adj_basis_from_first_vector(rayHat, phiHat, fallback, gO, gRayHat, gPhiHat, gFallback);
    adj_stable_perp_basis(rayHat, edgeHat, gFallback, gRayHat, gEdgeHat);

    if (outgoing) gPhiHat = f3_neg(gPhiHat);
    gRayHat = f3_add(gRayHat, f3_cross(edgeHat, gPhiHat));
    gEdgeHat = f3_add(gEdgeHat, f3_cross(gPhiHat, rayHat));

    float3a gRayFallback = f3_zero();
    float3a gEdgeFallback = f3_zero();
    adj_safe_normalize(rayDir, make_f3(0,0,1), gRayHat, gRayDir, gRayFallback);
    adj_safe_normalize(edgeDir, make_f3(0,0,1), gEdgeHat, gEdgeDir, gEdgeFallback);
}

UTD_DINLINE void adj_compute_edge_geometry_3d(
    float3a srcPos, float3a edgePos, float3a edgeDir, float3a n0, float3a tgtPos,
    float gPhi, float gPhiP, float gS, float gSP, float gSinBeta0,
    float3a& gSrcPos, float3a& gEdgePos, float3a& gEdgeDir, float3a& gN0, float3a& gTgtPos)
{
    float3a srcToEdge = f3_sub(edgePos, srcPos);
    float3a srcProj = project_to_wedge_plane(srcToEdge, edgeDir);
    float sPProj = safe_length(srcProj) + UTD_EPS;
    float3a toHatBase = f3_cross(n0, edgeDir);
    float3a toHat = safe_normalize(toHatBase, make_f3(0,1,0));
    float3a kiProj = f3_div(srcProj, sPProj);
    float signP = ((-f3_dot(kiProj, n0)) >= 0.f ? 1.f : -1.f);
    float cPhiP = -f3_dot(kiProj, toHat);
    float basePhiP = UTD_PI - safe_acos(cPhiP);
    float phiP = basePhiP * (-signP) + UTD_PI;
    (void) phiP;

    float3a edgeToTgt = f3_sub(tgtPos, edgePos);
    float3a tgtProj = project_to_wedge_plane(edgeToTgt, edgeDir);
    float sProj = safe_length(tgtProj) + UTD_EPS;
    float3a koProj = f3_div(tgtProj, sProj);
    float signPhi = (f3_dot(koProj, n0) >= 0.f ? 1.f : -1.f);
    float cPhi = f3_dot(koProj, toHat);
    float basePhi = UTD_PI - safe_acos(cPhi);
    float phi = basePhi * (-signPhi) + UTD_PI;
    (void) phi;

    float sPFullNorm = safe_length(srcToEdge);
    float sFullNorm = safe_length(edgeToTgt);
    float sPFull = sPFullNorm + UTD_EPS;
    float sFull = sFullNorm + UTD_EPS;
    float sbPUnclamped = sPProj / sPFull;
    float sbUnclamped = sProj / sFull;
    float sbP = fminf(fmaxf(sbPUnclamped, UTD_SMALL_EPS), 1.f);
    float sb = fminf(fmaxf(sbUnclamped, UTD_SMALL_EPS), 1.f);
    float sbProd = sb * sbP;

    float gSb = 0.f;
    float gSbP = 0.f;
    if (sbProd > UTD_SMALL_EPS) {
        float denom = sqrtf(sbProd);
        float gProd = 0.5f * gSinBeta0 / fmaxf(denom, UTD_SMALL_EPS);
        gSb += gProd * sbP;
        gSbP += gProd * sb;
    }

    float gSProj = 0.f;
    float gSPProj = 0.f;
    float gSFull = gS;
    float gSPFull = gSP;
    if (sbUnclamped > UTD_SMALL_EPS && sbUnclamped < 1.f) {
        gSProj += gSb / sFull;
        gSFull -= gSb * sProj / (sFull * sFull);
    }
    if (sbPUnclamped > UTD_SMALL_EPS && sbPUnclamped < 1.f) {
        gSPProj += gSbP / sPFull;
        gSPFull -= gSbP * sPProj / (sPFull * sPFull);
    }

    float3a gSrcToEdge = f3_zero();
    float3a gEdgeToTgt = f3_zero();
    if (sPFullNorm > UTD_SMALL_EPS) {
        gSrcToEdge = f3_add(gSrcToEdge, f3_mul(srcToEdge, gSPFull / sPFullNorm));
    }
    if (sFullNorm > UTD_SMALL_EPS) {
        gEdgeToTgt = f3_add(gEdgeToTgt, f3_mul(edgeToTgt, gSFull / sFullNorm));
    }

    float gBasePhi = -signPhi * gPhi;
    float gBasePhiP = -signP * gPhiP;
    float gCPhi = 0.f;
    float gCPhiP = 0.f;
    gCPhi += (gBasePhi / sqrtf(fmaxf(1.f - cPhi * cPhi, 0.f) + UTD_SMALL_EPS));
    gCPhiP += (gBasePhiP / sqrtf(fmaxf(1.f - cPhiP * cPhiP, 0.f) + UTD_SMALL_EPS));

    float3a gKoProj = f3_zero();
    float3a gKiProj = f3_zero();
    float3a gToHat = f3_zero();
    gKoProj = f3_add(gKoProj, f3_mul(toHat, gCPhi));
    gToHat = f3_add(gToHat, f3_mul(koProj, gCPhi));
    gKiProj = f3_sub(gKiProj, f3_mul(toHat, gCPhiP));
    gToHat = f3_sub(gToHat, f3_mul(kiProj, gCPhiP));

    float gKoNorm = -f3_dot(gKoProj, tgtProj) / (sProj * sProj);
    float gKiNorm = -f3_dot(gKiProj, srcProj) / (sPProj * sPProj);
    gSProj += gKoNorm;
    gSPProj += gKiNorm;

    float3a gTgtProj = f3_div(gKoProj, sProj);
    float3a gSrcProj = f3_div(gKiProj, sPProj);
    float srcProjNorm = safe_length(srcProj);
    float tgtProjNorm = safe_length(tgtProj);
    if (srcProjNorm > UTD_SMALL_EPS) {
        gSrcProj = f3_add(gSrcProj, f3_mul(srcProj, gSPProj / srcProjNorm));
    }
    if (tgtProjNorm > UTD_SMALL_EPS) {
        gTgtProj = f3_add(gTgtProj, f3_mul(tgtProj, gSProj / tgtProjNorm));
    }

    float3a gToHatBase = f3_zero();
    float3a gToHatFallback = f3_zero();
    adj_safe_normalize(toHatBase, make_f3(0,1,0), gToHat, gToHatBase, gToHatFallback);
    (void) gToHatFallback;
    gN0 = f3_add(gN0, f3_cross(edgeDir, gToHatBase));
    gEdgeDir = f3_add(gEdgeDir, f3_cross(gToHatBase, n0));

    float3a gSrcProjVec = f3_zero();
    float3a gTgtProjVec = f3_zero();
    adj_project_to_wedge_plane(srcToEdge, edgeDir, gSrcProj, gSrcProjVec, gEdgeDir);
    adj_project_to_wedge_plane(edgeToTgt, edgeDir, gTgtProj, gTgtProjVec, gEdgeDir);

    gSrcToEdge = f3_add(gSrcToEdge, gSrcProjVec);
    gEdgeToTgt = f3_add(gEdgeToTgt, gTgtProjVec);

    gEdgePos = f3_add(gEdgePos, gSrcToEdge);
    gSrcPos = f3_sub(gSrcPos, gSrcToEdge);
    gTgtPos = f3_add(gTgtPos, gEdgeToTgt);
    gEdgePos = f3_sub(gEdgePos, gEdgeToTgt);
}

// ===================================================================
// Complex sqrt for Fresnel
// ===================================================================
UTD_DINLINE Complex cplx_sqrt(Complex z) {
    float x = z.re, y = z.im;
    float r = sqrtf(x*x + y*y);
    bool nz = r > 0.f;
    bool xnn = x >= 0.f;
    float rMag = sqrtf((xnn && nz) ? 0.5f*(r+x) : 0.f);
    float iMag = sqrtf((!xnn && nz) ? 0.5f*(r-x) : 0.f);
    float srMag = rMag > 0.f ? rMag : 1.f;
    float siMag = iMag > 0.f ? iMag : 1.f;
    float rPart = xnn ? rMag : fabsf(y)/(2.f*siMag);
    float iPart = xnn ? y/(2.f*srMag) : (y < 0.f ? -iMag : iMag);
    return cplx(nz ? rPart : 0.f, nz ? iPart : 0.f);
}

UTD_DINLINE void adj_cplx_sqrt(Complex z, Complex gO, Complex& gZ) {
    Complex y = cplx_sqrt(z);
    float mag2 = cplx_abs_sqr(y);
    if (mag2 <= UTD_EPS)
        return;
    Complex denom = cplx_mul_real(cplx_conj(y), 2.f);
    gZ = cplx_add(gZ, cplx_div(gO, denom));
}

// ===================================================================
// Fresnel reflection
// ===================================================================
UTD_DINLINE void fresnel_reflection_face(float cosTheta, float etaR, float muR, float sigma,
    float omega, Complex& rTE, Complex& rTM)
{
    float ct = fminf(fmaxf(cosTheta, UTD_SMALL_EPS), 1.f);
    float sinSq = 1.f - ct*ct;
    float so = fmaxf(omega, UTD_SMALL_EPS);
    Complex eta = cplx(etaR, -sigma/(so*UTD_EPSILON_0));
    Complex mu = cplx(muR, 0.f);
    Complex a = cplx_sqrt(cplx_sub(cplx_mul(mu, eta), cplx(sinSq, 0)));
    Complex muCt = cplx_mul_real(mu, ct);
    rTE = cplx_div(cplx_sub(muCt, a), cplx_add(muCt, a));
    rTM = cplx_div(cplx_sub(cplx_mul_real(eta,ct), a),
                   cplx_add(cplx_mul_real(eta,ct), a));
}

UTD_DINLINE void adj_fresnel_reflection_face(
    float cosTheta,
    float etaR,
    float muR,
    float sigma,
    float omega,
    Complex gRTE,
    Complex gRTM,
    float& gCosTheta,
    float& gEtaR,
    float& gMuR,
    float& gSigma)
{
    float ct = fminf(fmaxf(cosTheta, UTD_SMALL_EPS), 1.f);
    float sinSq = 1.f - ct * ct;
    float so = fmaxf(omega, UTD_SMALL_EPS);
    Complex eta = cplx(etaR, -sigma / (so * UTD_EPSILON_0));
    Complex mu = cplx(muR, 0.f);
    Complex muEta = cplx_mul(mu, eta);
    Complex tmp = cplx_sub(muEta, cplx(sinSq, 0.f));
    Complex a = cplx_sqrt(tmp);

    Complex muCt = cplx_mul_real(mu, ct);
    Complex numTE = cplx_sub(muCt, a);
    Complex denTE = cplx_add(muCt, a);
    Complex etaCt = cplx_mul_real(eta, ct);
    Complex numTM = cplx_sub(etaCt, a);
    Complex denTM = cplx_add(etaCt, a);

    Complex gNumTE = cplx_zero();
    Complex gDenTE = cplx_zero();
    adj_cplx_div(numTE, denTE, gRTE, gNumTE, gDenTE);
    Complex gMuCt = cplx_add(gNumTE, gDenTE);
    Complex gA = cplx_sub(gDenTE, gNumTE);

    Complex gNumTM = cplx_zero();
    Complex gDenTM = cplx_zero();
    adj_cplx_div(numTM, denTM, gRTM, gNumTM, gDenTM);
    Complex gEtaCt = cplx_add(gNumTM, gDenTM);
    Complex gEta = cplx_zero();
    gA = cplx_add(gA, cplx_sub(gDenTM, gNumTM));

    Complex gMu = cplx_zero();
    float gCt = 0.f;
    adj_cplx_mul_real(mu, ct, gMuCt, gMu, gCt);
    adj_cplx_mul_real(eta, ct, gEtaCt, gEta, gCt);

    Complex gTmp = cplx_zero();
    adj_cplx_sqrt(tmp, gA, gTmp);
    adj_cplx_mul(mu, eta, gTmp, gMu, gEta);
    float gSinSq = -gTmp.re;
    gCt += -2.f * ct * gSinSq;

    if (cosTheta > UTD_SMALL_EPS && cosTheta < 1.f)
        gCosTheta += gCt;
    gEtaR += gEta.re;
    gMuR += gMu.re;
    gSigma += -gEta.im / (so * UTD_EPSILON_0);
}

UTD_DINLINE void adj_face_operator_in_basis(
    JonesOperator localOp,
    float3a normal,
    float3a inHat,
    float3a outHat,
    Basis3 inEdgeBasis,
    Basis3 outEdgeBasis,
    JonesOperator gO,
    JonesOperator& gLocalOp,
    float3a& gNormal,
    float3a& gInHat,
    float3a& gOutHat,
    Basis3& gInEdgeBasis,
    Basis3& gOutEdgeBasis)
{
    float3a faceSIn = f3_cross(normal, inHat);
    float3a faceSOutRaw = f3_cross(normal, outHat);
    float3a fallbackIn = stable_perp_basis(inHat, make_f3(0, 0, 1));
    Basis3 fIn = basis_from_first_vector(inHat, faceSIn, fallbackIn);
    float3a fallbackOut = stable_perp_basis(outHat, faceSIn);
    float faceSOutSign = f3_dot(faceSOutRaw, fallbackOut) < 0.0f ? -1.0f : 1.0f;
    float3a faceSOut = f3_mul(faceSOutRaw, faceSOutSign);
    Basis3 fOut = basis_from_first_vector(outHat, faceSOut, fallbackOut);

    Basis3 gFIn = basis_zero();
    Basis3 gFOut = basis_zero();
    Basis3 gInEdgeLocal = basis_zero();
    Basis3 gOutEdgeLocal = basis_zero();
    adj_jop_in_basis(localOp, fIn, fOut, inEdgeBasis, outEdgeBasis, gO, gLocalOp, gFIn, gFOut, gInEdgeLocal, gOutEdgeLocal);
    basis_accum(gInEdgeBasis, gInEdgeLocal);
    basis_accum(gOutEdgeBasis, gOutEdgeLocal);

    float3a gFaceSIn = f3_zero();
    float3a gFallbackIn = f3_zero();
    adj_basis_from_first_vector(inHat, faceSIn, fallbackIn, gFIn, gInHat, gFaceSIn, gFallbackIn);
    float3a gPreferredIn = f3_zero();
    adj_stable_perp_basis(inHat, make_f3(0, 0, 1), gFallbackIn, gInHat, gPreferredIn);

    float3a gFaceSOut = f3_zero();
    float3a gFallbackOut = f3_zero();
    adj_basis_from_first_vector(outHat, faceSOut, fallbackOut, gFOut, gOutHat, gFaceSOut, gFallbackOut);
    gFaceSOut = f3_mul(gFaceSOut, faceSOutSign);
    float3a gFaceSInFromFallback = f3_zero();
    adj_stable_perp_basis(outHat, faceSIn, gFallbackOut, gOutHat, gFaceSInFromFallback);
    gFaceSIn = f3_add(gFaceSIn, gFaceSInFromFallback);

    gNormal = f3_add(gNormal, f3_cross(inHat, gFaceSIn));
    gInHat = f3_add(gInHat, f3_cross(gFaceSIn, normal));
    gNormal = f3_add(gNormal, f3_cross(outHat, gFaceSOut));
    gOutHat = f3_add(gOutHat, f3_cross(gFaceSOut, normal));
    (void) gPreferredIn;
}

UTD_DINLINE JonesOperator face_reflection_operator(FaceMaterialParams fm,
    float cosTheta, float3a normal, float3a inHat, float3a outHat,
    Basis3 inEdgeBasis, Basis3 outEdgeBasis, float omega)
{
    Complex gain = cplx(fm.gain, 0);
    bool useFr = fm.useFresnel > 0.5f;
    Complex rTE, rTM;
    fresnel_reflection_face(cosTheta, fm.etaR, fm.muR, fm.sigma, omega, rTE, rTM);
    JonesOperator diagOp = useFr
        ? JonesOperator{cplx_mul(gain,rTE), cplx_zero(), cplx_zero(), cplx_mul(gain,rTM)}
        : JonesOperator{cplx(-fm.gain,0), cplx_zero(), cplx_zero(), cplx(-fm.gain,0)};
    float3a faceSIn = f3_cross(normal, inHat);
    float3a faceSOutRaw = f3_cross(normal, outHat);
    float3a fallbackOut = stable_perp_basis(outHat, faceSIn);
    float3a faceSOut = f3_dot(faceSOutRaw, fallbackOut) < 0.0f
        ? f3_neg(faceSOutRaw)
        : faceSOutRaw;
    Basis3 fIn  = basis_from_first_vector(inHat,  faceSIn, stable_perp_basis(inHat,  make_f3(0,0,1)));
    Basis3 fOut = basis_from_first_vector(outHat, faceSOut, fallbackOut);
    return jop_in_basis(diagOp, fIn, fOut, inEdgeBasis, outEdgeBasis);
}

UTD_DINLINE JonesOperator fallback_face_operator(JonesOperator stored,
    float3a normal, float3a inHat, float3a outHat,
    Basis3 inEdgeBasis, Basis3 outEdgeBasis)
{
    (void) normal;
    (void) inHat;
    (void) outHat;
    (void) inEdgeBasis;
    (void) outEdgeBasis;
    // Stored face operators in the state are already represented in the
    // diffraction edge basis. Re-basing them again corrupts both the forward
    // value and the operator gradients.
    return stored;
}

UTD_DINLINE void adj_face_reflection_operator(
    FaceMaterialParams fm,
    float cosTheta,
    float3a normal,
    float3a inHat,
    float3a outHat,
    Basis3 inEdgeBasis,
    Basis3 outEdgeBasis,
    float omega,
    JonesOperator gO,
    float& gCosTheta,
    float3a& gNormal,
    float3a& gInHat,
    float3a& gOutHat,
    Basis3& gInEdgeBasis,
    Basis3& gOutEdgeBasis,
    FaceMaterialParams& gFm)
{
    Complex gain = cplx(fm.gain, 0.f);
    bool useFr = fm.useFresnel > 0.5f;
    Complex rTE, rTM;
    fresnel_reflection_face(cosTheta, fm.etaR, fm.muR, fm.sigma, omega, rTE, rTM);
    JonesOperator diagOp = useFr
        ? JonesOperator{cplx_mul(gain, rTE), cplx_zero(), cplx_zero(), cplx_mul(gain, rTM)}
        : JonesOperator{cplx(-fm.gain, 0.f), cplx_zero(), cplx_zero(), cplx(-fm.gain, 0.f)};

    JonesOperator gDiagOp = jop_zero();
    adj_face_operator_in_basis(diagOp, normal, inHat, outHat, inEdgeBasis, outEdgeBasis, gO, gDiagOp, gNormal, gInHat, gOutHat, gInEdgeBasis, gOutEdgeBasis);

    if (!useFr) {
        gFm.gain += -(gDiagOp.m00.re + gDiagOp.m11.re);
        return;
    }

    Complex gGain0 = cplx_zero();
    Complex gRTE = cplx_zero();
    Complex gRTM = cplx_zero();
    adj_cplx_mul(gain, rTE, gDiagOp.m00, gGain0, gRTE);
    Complex gGain1 = cplx_zero();
    adj_cplx_mul(gain, rTM, gDiagOp.m11, gGain1, gRTM);
    Complex gGain = cplx_add(gGain0, gGain1);
    gFm.gain += gGain.re;
    adj_fresnel_reflection_face(
        cosTheta,
        fm.etaR,
        fm.muR,
        fm.sigma,
        omega,
        gRTE,
        gRTM,
        gCosTheta,
        gFm.etaR,
        gFm.muR,
        gFm.sigma
    );
}

UTD_DINLINE void adj_fallback_face_operator(
    JonesOperator stored,
    float3a normal,
    float3a inHat,
    float3a outHat,
    Basis3 inEdgeBasis,
    Basis3 outEdgeBasis,
    JonesOperator gO,
    JonesOperator& gStored,
    float3a& gNormal,
    float3a& gInHat,
    float3a& gOutHat,
    Basis3& gInEdgeBasis,
    Basis3& gOutEdgeBasis)
{
    (void) stored;
    (void) normal;
    (void) inHat;
    (void) outHat;
    (void) inEdgeBasis;
    (void) outEdgeBasis;
    (void) gNormal;
    (void) gInHat;
    (void) gOutHat;
    (void) gInEdgeBasis;
    (void) gOutEdgeBasis;
    gStored = jop_add(gStored, gO);
}

// ===================================================================
// Operator term computation (3D / 2D)
// ===================================================================
UTD_DINLINE DiffractionOperatorTerms compute_op_terms_3d(float phi, float phiP,
    float wedgeN, float k, float s, float sP, float sinBeta0)
{
    Complex z = cplx_zero(), one = cplx(1,0);
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    diffraction_beta_groups_3d(phi,phiP,wedgeN,k,s,sP,sinBeta0,z,z,fac,dG,dG1,sG,sG1,dG2,sG2);
    Complex fac0,dF0,dF01,sF0,sF01,dF02,sF02;
    diffraction_beta_groups_3d(phi,phiP,wedgeN,k,s,sP,sinBeta0,one,z,fac0,dF0,dF01,sF0,sF01,dF02,sF02);
    Complex fac1,dF1,dF11,sF1,sF11,dF12,sF12;
    diffraction_beta_groups_3d(phi,phiP,wedgeN,k,s,sP,sinBeta0,z,one,fac1,dF1,dF11,sF1,sF11,dF12,sF12);
    return {
        cplx_mul(fac, dG),
        cplx_mul(fac0, sF0),
        cplx_mul(fac1, sF1),
        cplx_mul(fac, cplx_mul_real(dG1, -1.f)),
        cplx_mul(fac0, sF01),
        cplx_mul(fac1, sF11)
    };
}

UTD_DINLINE DiffractionOperatorTerms compute_op_terms_3d_endpoint_continued(float phi, float phiP,
    float wedgeN, float k, float s, float sP, float sinBeta0)
{
    Complex z = cplx_zero(), one = cplx(1,0);
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    float directBeta = endpoint_unpaired_direct_beta(phi - phiP, wedgeN);
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,sinBeta0,z,z,fac,dG,dG1,sG,sG1,dG2,sG2);
    Complex fac0,dF0,dF01,sF0,sF01,dF02,sF02;
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,sinBeta0,one,z,fac0,dF0,dF01,sF0,sF01,dF02,sF02);
    Complex fac1,dF1,dF11,sF1,sF11,dF12,sF12;
    diffraction_beta_groups_3d_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,sinBeta0,z,one,fac1,dF1,dF11,sF1,sF11,dF12,sF12);
    return {
        cplx_mul(fac, dG),
        cplx_mul(fac0, sF0),
        cplx_mul(fac1, sF1),
        cplx_mul(fac, cplx_mul_real(dG1, -1.f)),
        cplx_mul(fac0, sF01),
        cplx_mul(fac1, sF11)
    };
}

UTD_DINLINE DiffractionOperatorTerms compute_op_terms_2d(float phi, float phiP,
    float wedgeN, float k, float s, float sP)
{
    float l = s*sP/(s+sP+UTD_EPS);
    float kL = k*l;
    float dPhi = phi - phiP;
    float sPhi = phi + phiP;
    // Build beta term caches inline
    float cv[4],c1v[4],c2v[4],xv[4],x1v[4],x2v[4];
    beta_term_values(dPhi, wedgeN, kL, +1.f, true,  cv[0],c1v[0],c2v[0],xv[0],x1v[0],x2v[0]);
    beta_term_values(dPhi, wedgeN, kL, -1.f, false, cv[1],c1v[1],c2v[1],xv[1],x1v[1],x2v[1]);
    beta_term_values(sPhi, wedgeN, kL, +1.f, true,  cv[2],c1v[2],c2v[2],xv[2],x1v[2],x2v[2]);
    beta_term_values(sPhi, wedgeN, kL, -1.f, false, cv[3],c1v[3],c2v[3],xv[3],x1v[3],x2v[3]);
    Complex tr[4],tr1[4],tr2[4];
    for (int i=0;i<4;++i) f_utd_with_derivatives(xv[i],tr[i],tr1[i],tr2[i]);
    Complex tv[4],tf[4],ts[4];
    for (int i=0;i<4;++i) {
        float cotSign = (i == 0 || i == 2) ? +1.f : -1.f;
        assemble_beta_term(cv[i],c1v[i],c2v[i],xv[i],x1v[i],x2v[i],kL,wedgeN,cotSign,tr[i],tr1[i],tr2[i],tv[i],tf[i],ts[i]);
    }
    Complex factor = cplx_mul_real(cplx_exp_phase(-0.25f*UTD_PI),
                     -1.f/(2.f*wedgeN*sqrtf(UTD_TWO_PI*k+UTD_EPS)));
    Complex difV = cplx_add(tv[0],tv[1]);
    Complex difF = cplx_add(tf[0],tf[1]);
    return {
        cplx_mul(factor, difV),
        cplx_mul(factor, tv[3]),
        cplx_mul(factor, tv[2]),
        cplx_mul(factor, cplx_mul_real(difF, -1.f)),
        cplx_mul(factor, tf[3]),
        cplx_mul(factor, tf[2])
    };
}

UTD_DINLINE DiffractionOperatorTerms compute_op_terms_2d_endpoint_continued(float phi, float phiP,
    float wedgeN, float k, float s, float sP)
{
    Complex z = cplx_zero(), one = cplx(1,0);
    Complex fac,dG,dG1,sG,sG1,dG2,sG2;
    float directBeta = endpoint_unpaired_direct_beta(phi - phiP, wedgeN);
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,z,z,fac,dG,dG1,sG,sG1,dG2,sG2);
    Complex fac0,dF0,dF01,sF0,sF01,dF02,sF02;
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,one,z,fac0,dF0,dF01,sF0,sF01,dF02,sF02);
    Complex fac1,dF1,dF11,sF1,sF11,dF12,sF12;
    diffraction_beta_groups_with_direct_beta(phi,phiP,directBeta,wedgeN,k,s,sP,z,one,fac1,dF1,dF11,sF1,sF11,dF12,sF12);
    return {
        cplx_mul(fac, dG),
        cplx_mul(fac0, sF0),
        cplx_mul(fac1, sF1),
        cplx_mul(fac, cplx_mul_real(dG1, -1.f)),
        cplx_mul(fac0, sF01),
        cplx_mul(fac1, sF11)
    };
}

UTD_DINLINE void adj_compute_op_terms_3d_direct(
    float phi, float phiP, float wedgeN, float k, float s, float sP, float sinBeta0,
    Complex gDirect, Complex gFace0, Complex gFace1,
    float& gPhi, float& gPhiP, float& gWedgeN, float& gS, float& gSP, float& gSinBeta0)
{
    float sb = fmaxf(sinBeta0, UTD_SMALL_EPS);
    float den = s + sP + UTD_EPS;
    float frac = s * sP / den;
    float l = frac * sb * sb;
    float kL = k * l;
    float dP = phi - phiP;
    float sP2 = phi + phiP;

    Complex factorPhase = cplx_exp_phase(-0.25f * UTD_PI);
    float factorScale = -1.f / (2.f * wedgeN * sqrtf(UTD_TWO_PI * k + UTD_EPS) * sb);
    Complex factor = cplx_mul_real(factorPhase, factorScale);

    Complex tv[4];
    Complex tr[4];
    Complex tr1[4];
    float cotV[4];
    float a[4];
    float betaVals[4];
    float cotSigns[4];
    float riVals[4];
    for (int i = 0; i < 4; ++i) {
        bool plusBranch = (i == 0) || (i == 2);
        float beta = (i < 2) ? dP : sP2;
        float cotSign = plusBranch ? +1.f : -1.f;
        betaVals[i] = beta;
        cotSigns[i] = cotSign;

        float twoNPi = 2.f * wedgeN * UTD_PI;
        float ri = plusBranch
            ? roundf((beta + UTD_PI) / twoNPi)
            : roundf((beta - UTD_PI) / twoNPi);
        riVals[i] = ri;

        float po = twoNPi * ri - beta;
        float chp = cosf(0.5f * po);
        a[i] = 2.f * chp * chp;
        float c1Dummy, c2Dummy, x0Dummy, x1Dummy, x2Dummy;
        beta_term_values(beta, wedgeN, kL, cotSign, plusBranch, cotV[i], c1Dummy, c2Dummy, x0Dummy, x1Dummy, x2Dummy);
        Complex tr2Dummy;
        f_utd_with_derivatives(x0Dummy, tr[i], tr1[i], tr2Dummy);
        tv[i] = cplx_mul_real(tr[i], cotV[i]);
    }

    Complex gFactor = cplx_zero();
    Complex gTv[4] = {cplx_zero(), cplx_zero(), cplx_zero(), cplx_zero()};
    Complex directSum = cplx_add(tv[0], tv[1]);
    Complex gDirectSum = cplx_zero();
    adj_cplx_mul(factor, directSum, gDirect, gFactor, gDirectSum);
    gTv[0] = cplx_add(gTv[0], gDirectSum);
    gTv[1] = cplx_add(gTv[1], gDirectSum);
    adj_cplx_mul(factor, tv[3], gFace0, gFactor, gTv[3]);
    adj_cplx_mul(factor, tv[2], gFace1, gFactor, gTv[2]);

    float gFactorScale = cplx_adj_dot(gFactor, factorPhase);
    gWedgeN += -gFactorScale * factorScale / wedgeN;
    gSinBeta0 += -gFactorScale * factorScale / sb;

    float gKL = 0.f;
    for (int i = 0; i < 4; ++i) {
        float beta = betaVals[i];
        float cotSign = cotSigns[i];
        float ri = riVals[i];
        float twoN = 2.f * wedgeN;
        float po = 2.f * wedgeN * UTD_PI * ri - beta;
        float a1 = sinf(po);
        float cotDerivArg = -(1.f + cotV[i] * cotV[i]);

        Complex gTr = cplx_zero();
        float gCot = 0.f;
        adj_cplx_mul_real(tr[i], cotV[i], gTv[i], gTr, gCot);
        float gX = cplx_adj_dot(gTr, tr1[i]);
        gKL += gX * a[i];
        float gA = gX * kL;

        float gPo = -gA * a1;
        float gBetaLocal = gCot * cotDerivArg * cotSign / twoN - gPo;
        float gNLocal = gCot * cotDerivArg * (-(UTD_PI + cotSign * beta) / (2.f * wedgeN * wedgeN));
        gNLocal += gPo * (2.f * UTD_PI * ri);
        gWedgeN += gNLocal;
        if (i < 2) {
            gPhi += gBetaLocal;
            gPhiP -= gBetaLocal;
        } else {
            gPhi += gBetaLocal;
            gPhiP += gBetaLocal;
        }
    }

    float gL = gKL * k;
    float sbSq = sb * sb;
    gSinBeta0 += gL * (2.f * sb * frac);
    gS += gL * sbSq * (sP * (sP + UTD_EPS) / (den * den));
    gSP += gL * sbSq * (s * (s + UTD_EPS) / (den * den));
}

UTD_DINLINE void adj_compute_op_terms_2d_direct(
    float phi, float phiP, float wedgeN, float k, float s, float sP,
    Complex gDirect, Complex gFace0, Complex gFace1,
    float& gPhi, float& gPhiP, float& gWedgeN, float& gS, float& gSP)
{
    float den = s + sP + UTD_EPS;
    float frac = s * sP / den;
    float kL = k * frac;
    float dP = phi - phiP;
    float sP2 = phi + phiP;

    Complex factorPhase = cplx_exp_phase(-0.25f * UTD_PI);
    float factorScale = -1.f / (2.f * wedgeN * sqrtf(UTD_TWO_PI * k + UTD_EPS));
    Complex factor = cplx_mul_real(factorPhase, factorScale);

    Complex tv[4];
    Complex tr[4];
    Complex tr1[4];
    float cotV[4];
    float a[4];
    float betaVals[4];
    float cotSigns[4];
    float riVals[4];
    for (int i = 0; i < 4; ++i) {
        bool plusBranch = (i == 0) || (i == 2);
        float beta = (i < 2) ? dP : sP2;
        float cotSign = plusBranch ? +1.f : -1.f;
        betaVals[i] = beta;
        cotSigns[i] = cotSign;

        float twoNPi = 2.f * wedgeN * UTD_PI;
        float ri = plusBranch
            ? roundf((beta + UTD_PI) / twoNPi)
            : roundf((beta - UTD_PI) / twoNPi);
        riVals[i] = ri;

        float po = twoNPi * ri - beta;
        float chp = cosf(0.5f * po);
        a[i] = 2.f * chp * chp;
        float c1Dummy, c2Dummy, x0Dummy, x1Dummy, x2Dummy;
        beta_term_values(beta, wedgeN, kL, cotSign, plusBranch, cotV[i], c1Dummy, c2Dummy, x0Dummy, x1Dummy, x2Dummy);
        Complex tr2Dummy;
        f_utd_with_derivatives(kL * a[i], tr[i], tr1[i], tr2Dummy);
        tv[i] = cplx_mul_real(tr[i], cotV[i]);
    }

    Complex gFactor = cplx_zero();
    Complex gTv[4] = {cplx_zero(), cplx_zero(), cplx_zero(), cplx_zero()};
    Complex directSum = cplx_add(tv[0], tv[1]);
    Complex gDirectSum = cplx_zero();
    adj_cplx_mul(factor, directSum, gDirect, gFactor, gDirectSum);
    gTv[0] = cplx_add(gTv[0], gDirectSum);
    gTv[1] = cplx_add(gTv[1], gDirectSum);
    adj_cplx_mul(factor, tv[3], gFace0, gFactor, gTv[3]);
    adj_cplx_mul(factor, tv[2], gFace1, gFactor, gTv[2]);

    float gFactorScale = cplx_adj_dot(gFactor, factorPhase);
    gWedgeN += -gFactorScale * factorScale / wedgeN;

    float gKL = 0.f;
    for (int i = 0; i < 4; ++i) {
        float beta = betaVals[i];
        float cotSign = cotSigns[i];
        float ri = riVals[i];
        float twoN = 2.f * wedgeN;
        float po = 2.f * wedgeN * UTD_PI * ri - beta;
        float a1 = sinf(po);
        float cotDerivArg = -(1.f + cotV[i] * cotV[i]);

        Complex gTr = cplx_zero();
        float gCot = 0.f;
        adj_cplx_mul_real(tr[i], cotV[i], gTv[i], gTr, gCot);
        float gX = cplx_adj_dot(gTr, tr1[i]);
        gKL += gX * a[i];
        float gA = gX * kL;

        float gPo = -gA * a1;
        float gBetaLocal = gCot * cotDerivArg * cotSign / twoN - gPo;
        float gNLocal = gCot * cotDerivArg * (-(UTD_PI + cotSign * beta) / (2.f * wedgeN * wedgeN));
        gNLocal += gPo * (2.f * UTD_PI * ri);
        gWedgeN += gNLocal;
        if (i < 2) {
            gPhi += gBetaLocal;
            gPhiP -= gBetaLocal;
        } else {
            gPhi += gBetaLocal;
            gPhiP += gBetaLocal;
        }
    }

    float gFrac = gKL * k;
    gS += gFrac * (sP * (sP + UTD_EPS) / (den * den));
    gSP += gFrac * (s * (s + UTD_EPS) / (den * den));
}

// ===================================================================
// Assemble diffraction operator (Jones) from terms
// ===================================================================
UTD_DINLINE JonesOperator assemble_diff_operator(Complex free_term,
    Complex face0_term, Complex face1_term,
    JonesOperator face0Op, JonesOperator face1Op)
{
    JonesOperator total = jop_scale(jop_identity(), free_term);
    total = jop_add(total, jop_scale(face0Op, face0_term));
    total = jop_add(total, jop_scale(face1Op, face1_term));
    return total;
}

// ===================================================================
// Scalar field terms (for computePairFieldTerms)
// ===================================================================
UTD_DINLINE Complex finite_wedge_truncation_factor_bounds(
    PairInputs state,
    float3a tgtPos,
    float k,
    float lineMin,
    float lineMax,
    bool stationaryAtOrigin)
{
    float3a edgeHat = safe_normalize(state.edgeDir, make_f3(0.f, 0.f, 1.f));
    float3a edgePos = state.edgePos;
    float3a sourcePos = state.sourcePos;

    float sourceAxial = f3_dot(f3_sub(sourcePos, edgePos), edgeHat);
    float targetAxial = f3_dot(f3_sub(tgtPos, edgePos), edgeHat);

    float3a sourceToEdge = f3_sub(edgePos, sourcePos);
    float3a edgeToTarget = f3_sub(tgtPos, edgePos);
    float sPrimeProj = safe_length(project_to_wedge_plane(sourceToEdge, edgeHat)) + UTD_EPS;
    float sProj = safe_length(project_to_wedge_plane(edgeToTarget, edgeHat)) + UTD_EPS;

    float stationaryU = stationaryAtOrigin
        ? 0.f
        : (sPrimeProj * targetAxial + sProj * sourceAxial) / (sProj + sPrimeProj + UTD_EPS);
    float sourceOffset = stationaryU - sourceAxial;
    float targetOffset = targetAxial - stationaryU;
    float sourceRange =
        sqrtf(sPrimeProj * sPrimeProj + sourceOffset * sourceOffset + UTD_EPS);
    float targetRange =
        sqrtf(sProj * sProj + targetOffset * targetOffset + UTD_EPS);
    float curvature =
        sPrimeProj * sPrimeProj / (sourceRange * sourceRange * sourceRange + UTD_EPS)
        + sProj * sProj / (targetRange * targetRange * targetRange + UTD_EPS);
    float scale = sqrtf(fmaxf(k * curvature, UTD_EPS) / UTD_PI);

    Complex fMin, fMin1, fMin2;
    Complex fMax, fMax1, fMax2;
    fresnel_boersma(scale * (lineMin - stationaryU), fMin, fMin1, fMin2);
    fresnel_boersma(scale * (lineMax - stationaryU), fMax, fMax1, fMax2);
    Complex delta = cplx_sub(fMax, fMin);
    return cplx_mul(cplx(0.5f, 0.5f), cplx_conj(delta));
}

UTD_DINLINE Complex finite_wedge_truncation_factor_bounds(
    PairInputs state,
    float3a tgtPos,
    float k,
    float lineMin,
    float lineMax)
{
    return finite_wedge_truncation_factor_bounds(
        state,
        tgtPos,
        k,
        lineMin,
        lineMax,
        false
    );
}

UTD_DINLINE Complex finite_wedge_truncation_factor(PairInputs state, float3a tgtPos, float k) {
    return finite_wedge_truncation_factor_bounds(
        state,
        tgtPos,
        k,
        state.edgeLineMin,
        state.edgeLineMax
    );
}

UTD_DINLINE Complex finite_wedge_stationary_completion_factor(
    PairInputs state,
    float3a tgtPos,
    float k,
    bool inside)
{
    if (inside) {
        return cplx(1.f, 0.f);
    }
    float edgeLength = state.edgeLineMax - state.edgeLineMin;
    float outsideDistance = fmaxf(fmaxf(state.edgeLineMin, -state.edgeLineMax), 0.f);
    float wavelength = (2.f * UTD_PI) / fmaxf(k, UTD_SMALL_EPS);
    float taperLength = fminf(0.25f * edgeLength, fmaxf(0.5f * wavelength, UTD_EPS));
    float endpointU = fmaxf(outsideDistance / fmaxf(taperLength, UTD_EPS), 0.f);
    float endpointWeight = expf(-endpointU * endpointU);
    Complex raw = finite_wedge_truncation_factor_bounds(
        state,
        tgtPos,
        k,
        state.edgeLineMin,
        state.edgeLineMax,
        true
    );
    Complex boundary = state.edgeLineMin >= 0.f
        ? finite_wedge_truncation_factor_bounds(state, tgtPos, k, 0.f, edgeLength, true)
        : finite_wedge_truncation_factor_bounds(state, tgtPos, k, -edgeLength, 0.f, true);
    float boundaryPower = cplx_abs_sqr(boundary);
    if (boundaryPower <= UTD_EPS) {
        return cplx_mul_real(raw, endpointWeight);
    }
    return cplx_mul_real(cplx_div(raw, boundary), endpointWeight);
}

UTD_DINLINE void compute_pair_field_terms(PairInputs state, float3a tgtPos, float k,
    MaterialParams mat, bool& geomValid, Complex& field,
    Complex& directGain, Complex& derivativeGain)
{
    geomValid = false;
    field = cplx_zero(); directGain = cplx_zero(); derivativeGain = cplx_zero();

    bool selectedStationary = false;
    bool selectedInside = false;
    bool selectedValid = true;
    state = pair_state_at_stationary_point(
        state,
        tgtPos,
        selectedStationary,
        selectedInside,
        selectedValid
    );
    if (!selectedValid) return;

    bool srcExt = wedge_exterior_mask(f3_sub(state.sourcePos, state.edgePos), state.edgeDir, state.n0, state.nn);
    bool tgtExt = wedge_exterior_mask(f3_sub(tgtPos, state.edgePos), state.edgeDir, state.n0, state.nn);
    float phi,phiP,s,sP,sb;
    compute_edge_geometry_3d(state.sourcePos, state.edgePos, state.edgeDir, state.n0, tgtPos, phi,phiP,s,sP,sb);

    geomValid = srcExt && (sP > UTD_MIN_DISTANCE) && (s > UTD_MIN_DISTANCE);
    if (!geomValid) return;

    Complex r0 = state.r0, rn = state.rn;
    float w = state.wedgeN;
    bool poleSafe = cot_pole_safe_mask(phi,phiP,w,1.0e-6f);
    float safePhi  = poleSafe ? phi  : 0.5f*w*UTD_PI;
    float safePhiP = poleSafe ? phiP : 0.5f*w*UTD_PI;
    bool slopeSafe = slope_safe_mask(safePhi,safePhiP,w,UTD_SLOPE_STEP);
    bool useFace = (state.face0Material.present > 0.5f) || (state.face1Material.present > 0.5f);
    bool endpointContinuation = selectedStationary && !tgtExt;
    Complex d = endpointContinuation
        ? (useFace ? diff_coeff_3d_endpoint_continued(phi,phiP,w,k,s,sP,sb,r0,rn)
                   : diff_coeff_2d_endpoint_continued(phi,phiP,w,k,s,sP,r0,rn))
        : (useFace ? diff_coeff_3d(phi,phiP,w,k,s,sP,sb,r0,rn)
                   : diff_coeff_2d(phi,phiP,w,k,s,sP,r0,rn));
    if (!poleSafe) { d.re = d.re; d.im = d.im; } // detach (no AD in CUDA anyway)
    Complex dSlope = cplx_zero();
    bool hasSlope = (cplx_abs_sqr(state.incidentNormalDerivative) > 1.0e-24f) && slopeSafe;
    if (hasSlope) {
        dSlope = useFace ? slope_diff_3d(safePhi,safePhiP,w,k,s,sP,sb,r0,rn)
                         : slope_diff_2d(safePhi,safePhiP,w,k,s,sP,r0,rn);
    }
    float ls = sqrtf(sP/(s*(s+sP)+UTD_EPS));
    Complex phase = cplx_exp_phase(-k*s);
    directGain = cplx_mul_real(cplx_mul(d,phase), ls);
    derivativeGain = cplx_mul_real(cplx_mul(dSlope,phase), ls);
    Complex finiteFactor = selectedStationary
        ? finite_wedge_stationary_completion_factor(state, tgtPos, k, selectedInside)
        : finite_wedge_truncation_factor(state, tgtPos, k);
    directGain = cplx_mul(directGain, finiteFactor);
    derivativeGain = cplx_mul(derivativeGain, finiteFactor);
    Complex incidentField = selectedStationary
        ? direct_source_field(state.sourcePos, state.edgePos, k)
        : state.incidentField;
    Complex incidentNormalDerivative = selectedStationary
        ? cplx_zero()
        : state.incidentNormalDerivative;
    field = cplx_add(cplx_mul(incidentField, directGain),
                     cplx_mul(incidentNormalDerivative, derivativeGain));
}

// ===================================================================
// Vector field contribution (mega-kernel core)
// ===================================================================
UTD_DINLINE Complex3 c3_scale_real(Complex3 value, float scale) {
    Complex s = cplx(scale, 0.0f);
    return c3_scale(value, s);
}

UTD_DINLINE Complex3 compute_pair_vector_at_angles(
    PairInputs state,
    float3a tgtPos,
    float k,
    MaterialParams mat,
    float phi,
    float phiP,
    float s,
    float sP,
    float sb,
    Basis3 inEB,
    Basis3 outEB,
    Complex finiteFactor,
    bool endpointContinuation)
{
    bool selectedStationary = state.selectStationaryPoint > 0.5f;
    Complex3 incidentVector = selectedStationary
        ? direct_source_vector(state.sourcePos, state.edgePos, k, mat)
        : vector_from_jones(state.incidentJones, state.incidentBasis);
    Complex3 incidentDerivativeVector = selectedStationary
        ? c3_zero()
        : vector_from_jones(state.incidentDerivativeJones, state.incidentBasis);
    Jones2 incJE  = jones_from_vector(incidentVector, inEB);
    Jones2 incDJE = jones_from_vector(incidentDerivativeVector, inEB);
    bool poleSafe = cot_pole_safe_mask(phi, phiP, state.wedgeN, 1.0e-6f);
    float safePhi = poleSafe ? phi : 0.5f * state.wedgeN * UTD_PI;
    float safePhiP = poleSafe ? phiP : 0.5f * state.wedgeN * UTD_PI;
    bool slopeSafe = slope_safe_mask(safePhi, safePhiP, state.wedgeN, UTD_SLOPE_STEP);
    float derivativePower = cplx_abs_sqr(incDJE.u) + cplx_abs_sqr(incDJE.v);
    bool hasSlope = (derivativePower > 1.0e-24f) && slopeSafe;

    bool useFace = (state.face0Material.present > 0.5f) || (state.face1Material.present > 0.5f);
    bool f0HasMat = state.face0Material.present > 0.5f;
    bool f1HasMat = state.face1Material.present > 0.5f;
    bool useStoredFaceOps = mat.omega <= 0.f;

    JonesOperator f0Op = (f0HasMat && !useStoredFaceOps)
        ? face_reflection_operator(state.face0Material,
            fminf(fmaxf(fabsf(sinf(phiP)), 1.0e-6f), 1.f),
            state.n0, inEB.k, outEB.k, inEB, outEB, mat.omega)
        : fallback_face_operator(state.face0Operator, state.n0, inEB.k, outEB.k, inEB, outEB);
    JonesOperator f1Op = (f1HasMat && !useStoredFaceOps)
        ? face_reflection_operator(state.face1Material,
            fminf(fmaxf(fabsf(sinf(state.wedgeN*UTD_PI - phi)), 1.0e-6f), 1.f),
            state.nn, inEB.k, outEB.k, inEB, outEB, mat.omega)
        : fallback_face_operator(state.face1Operator, state.nn, inEB.k, outEB.k, inEB, outEB);

    DiffractionOperatorTerms terms = endpointContinuation
        ? (useFace
            ? compute_op_terms_3d_endpoint_continued(phi,phiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d_endpoint_continued(phi,phiP,state.wedgeN,k,s,sP))
        : (useFace
            ? compute_op_terms_3d(phi,phiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d(phi,phiP,state.wedgeN,k,s,sP));
    DiffractionOperatorTerms slopeTerms = endpointContinuation
        ? (useFace
            ? compute_op_terms_3d_endpoint_continued(safePhi,safePhiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d_endpoint_continued(safePhi,safePhiP,state.wedgeN,k,s,sP))
        : (useFace
            ? compute_op_terms_3d(safePhi,safePhiP,state.wedgeN,k,s,sP,sb)
            : compute_op_terms_2d(safePhi,safePhiP,state.wedgeN,k,s,sP));
    JonesOperator directOp = assemble_diff_operator(
        cplx_mul_real(terms.direct, -1.f),
        terms.face0,
        terms.face1,
        f0Op,
        f1Op
    );
    Complex slopeFactor = cplx(0, -1.f/k);
    JonesOperator slopeOp = hasSlope
        ? assemble_diff_operator(
            cplx_mul(slopeFactor, cplx_mul_real(slopeTerms.directDphiPrime, -1.f)),
            cplx_mul(slopeFactor, slopeTerms.face0DphiPrime),
            cplx_mul(slopeFactor, slopeTerms.face1DphiPrime),
            f0Op, f1Op)
        : jop_zero();

    Jones2 slopeFieldJ = hasSlope ? apply_jop(incDJE, slopeOp) : jones_zero();
    Jones2 fieldJ = jones_add(apply_jop(incJE, directOp), slopeFieldJ);
    fieldJ = jones_scale(fieldJ, finiteFactor);
    float ls = sqrtf(sP/(s*(s+sP)+UTD_EPS));
    Complex scale = cplx_mul_real(cplx_exp_phase(-k*s), ls);
    return c3_scale(vector_from_jones(fieldJ, outEB), scale);
}

UTD_DINLINE Complex3 compute_pair_vector_contribution_no_completion(PairInputs state, float3a tgtPos,
    float k, MaterialParams mat)
{
    bool selectedStationary = false;
    bool selectedInside = false;
    bool selectedValid = true;
    state = pair_state_at_stationary_point(
        state,
        tgtPos,
        selectedStationary,
        selectedInside,
        selectedValid
    );
    if (!selectedValid) return c3_zero();

    bool srcExt = wedge_exterior_mask(f3_sub(state.sourcePos, state.edgePos), state.edgeDir, state.n0, state.nn);
    float phi,phiP,s,sP,sb;
    compute_edge_geometry_3d(state.sourcePos, state.edgePos, state.edgeDir, state.n0, tgtPos, phi,phiP,s,sP,sb);
    bool geomValid = srcExt && (sP > UTD_MIN_DISTANCE) && (s > UTD_MIN_DISTANCE);
    if (!geomValid) return c3_zero();

    Basis3 inEB  = diffraction_edge_basis(f3_sub(state.edgePos, state.sourcePos), state.edgeDir, false);
    Basis3 outEB = diffraction_edge_basis(f3_sub(tgtPos, state.edgePos), state.edgeDir, true);
    Complex finiteFactor = selectedStationary
        ? finite_wedge_stationary_completion_factor(state, tgtPos, k, selectedInside)
        : finite_wedge_truncation_factor(state, tgtPos, k);
    bool tgtExt = wedge_exterior_mask(f3_sub(tgtPos, state.edgePos), state.edgeDir, state.n0, state.nn);
    bool endpointContinuation = selectedStationary && !tgtExt;
    return compute_pair_vector_at_angles(
        state, tgtPos, k, mat, phi, phiP, s, sP, sb, inEB, outEB, finiteFactor,
        endpointContinuation);
}

UTD_DINLINE Complex3 compute_pair_vector_contribution(PairInputs state, float3a tgtPos,
    float k, MaterialParams mat)
{
    return compute_pair_vector_contribution_no_completion(state, tgtPos, k, mat);
}

// ===================================================================
// Vector-output VJP for one source-edge/receiver pair
// ===================================================================
UTD_DINLINE void pair_vector_output_vjp(
    PairInputs pi,
    float3a tgt,
    float k,
    MaterialParams mat,
    Complex3 vecGrad,
    PairInputsGrad& sg,
    float3a& gTgt)
{
    if (!c3_grad_any_nonzero(vecGrad))
        return;

    bool srcExt = wedge_exterior_mask(f3_sub(pi.sourcePos, pi.edgePos), pi.edgeDir, pi.n0, pi.nn);
    float phi, phiP, s, sP, sb;
    compute_edge_geometry_3d(pi.sourcePos, pi.edgePos, pi.edgeDir, pi.n0, tgt, phi, phiP, s, sP, sb);
    bool geomValid = srcExt && (sP > UTD_MIN_DISTANCE) && (s > UTD_MIN_DISTANCE);
    if (!geomValid)
        return;

    Basis3 inEB  = diffraction_edge_basis(f3_sub(pi.edgePos, pi.sourcePos), pi.edgeDir, false);
    Basis3 outEB = diffraction_edge_basis(f3_sub(tgt, pi.edgePos), pi.edgeDir, true);
    Complex3 incidentVector = vector_from_jones(pi.incidentJones, pi.incidentBasis);
    Complex3 incidentDerivativeVector =
        vector_from_jones(pi.incidentDerivativeJones, pi.incidentBasis);
    Jones2 incJE  = jones_from_vector(incidentVector, inEB);
    Jones2 incDJE = jones_from_vector(incidentDerivativeVector, inEB);
    bool poleSafe = cot_pole_safe_mask(phi, phiP, pi.wedgeN, 1.0e-6f);
    float safePhi = poleSafe ? phi : 0.5f * pi.wedgeN * UTD_PI;
    float safePhiP = poleSafe ? phiP : 0.5f * pi.wedgeN * UTD_PI;
    bool slopeSafe = slope_safe_mask(safePhi, safePhiP, pi.wedgeN, UTD_SLOPE_STEP);
    float derivativePower = cplx_abs_sqr(incDJE.u) + cplx_abs_sqr(incDJE.v);
    bool hasSlope = (derivativePower > 1.0e-24f) && slopeSafe;

    bool useFace = (pi.face0Material.present > 0.5f) || (pi.face1Material.present > 0.5f);
    bool f0HasMat = pi.face0Material.present > 0.5f;
    bool f1HasMat = pi.face1Material.present > 0.5f;
    bool useStoredFaceOps = mat.omega <= 0.f;

    JonesOperator f0Op = (f0HasMat && !useStoredFaceOps)
        ? face_reflection_operator(
            pi.face0Material,
            fminf(fmaxf(fabsf(sinf(phiP)), 1.0e-6f), 1.f),
            pi.n0,
            inEB.k,
            outEB.k,
            inEB,
            outEB,
            mat.omega
        )
        : fallback_face_operator(pi.face0Operator, pi.n0, inEB.k, outEB.k, inEB, outEB);
    JonesOperator f1Op = (f1HasMat && !useStoredFaceOps)
        ? face_reflection_operator(
            pi.face1Material,
            fminf(fmaxf(fabsf(sinf(pi.wedgeN * UTD_PI - phi)), 1.0e-6f), 1.f),
            pi.nn,
            inEB.k,
            outEB.k,
            inEB,
            outEB,
            mat.omega
        )
        : fallback_face_operator(pi.face1Operator, pi.nn, inEB.k, outEB.k, inEB, outEB);

    DiffractionOperatorTerms terms = useFace
        ? compute_op_terms_3d(phi, phiP, pi.wedgeN, k, s, sP, sb)
        : compute_op_terms_2d(phi, phiP, pi.wedgeN, k, s, sP);
    DiffractionOperatorTerms slopeTerms = useFace
        ? compute_op_terms_3d(safePhi, safePhiP, pi.wedgeN, k, s, sP, sb)
        : compute_op_terms_2d(safePhi, safePhiP, pi.wedgeN, k, s, sP);
    JonesOperator directOp = assemble_diff_operator(
        cplx_mul_real(terms.direct, -1.f),
        terms.face0,
        terms.face1,
        f0Op,
        f1Op
    );
    Complex slopeFactor = cplx(0.f, -1.f / k);
    JonesOperator slopeOp = hasSlope
        ? assemble_diff_operator(
            cplx_mul(slopeFactor, cplx_mul_real(slopeTerms.directDphiPrime, -1.f)),
            cplx_mul(slopeFactor, slopeTerms.face0DphiPrime),
            cplx_mul(slopeFactor, slopeTerms.face1DphiPrime),
            f0Op,
            f1Op
        )
        : jop_zero();

    Jones2 directFieldJ = apply_jop(incJE, directOp);
    Jones2 slopeFieldJ = hasSlope ? apply_jop(incDJE, slopeOp) : jones_zero();
    Jones2 fieldJ = jones_add(directFieldJ, slopeFieldJ);
    Complex finiteFactor = finite_wedge_truncation_factor(pi, tgt, k);
    Jones2 scaledFieldJ = jones_scale(fieldJ, finiteFactor);
    float ls = sqrtf(sP / (s * (s + sP) + UTD_EPS));
    Complex phase = cplx_exp_phase(-k * s);
    Complex scale = cplx_mul_real(phase, ls);
    Complex3 transport = vector_from_jones(scaledFieldJ, outEB);

    Complex3 gTransport = c3_zero();
    Complex gScale = cplx_zero();
    adj_c3_scale(transport, scale, vecGrad, gTransport, gScale);

    Jones2 gScaledFieldJ = jones_zero();
    Basis3 gOutEB = {f3_zero(), f3_zero(), f3_zero()};
    adj_vector_from_jones(scaledFieldJ, outEB, gTransport, gScaledFieldJ, gOutEB);

    Jones2 gFieldJ = jones_zero();
    Complex gFiniteFactor = cplx_zero();
    adj_jones_scale(fieldJ, finiteFactor, gScaledFieldJ, gFieldJ, gFiniteFactor);
    (void) gFiniteFactor;

    Jones2 gDirectFieldJ = jones_zero();
    Jones2 gSlopeFieldJ = jones_zero();
    adj_jones_add(gFieldJ, gDirectFieldJ, gSlopeFieldJ);

    Jones2 gIncJE = jones_zero();
    Jones2 gIncDJE = jones_zero();
    JonesOperator gDirectOp = jop_zero();
    JonesOperator gSlopeOp = jop_zero();
    adj_apply_jop(incJE, directOp, gDirectFieldJ, gIncJE, gDirectOp);
    if (hasSlope) {
        adj_apply_jop(incDJE, slopeOp, gSlopeFieldJ, gIncDJE, gSlopeOp);
    }

    Basis3 gInEB = {f3_zero(), f3_zero(), f3_zero()};
    Complex3 gIncidentVector = c3_zero();
    Complex3 gIncidentDerivativeVector = c3_zero();
    adj_jones_from_vector(incidentVector, inEB, gIncJE, gIncidentVector, gInEB);
    adj_jones_from_vector(incidentDerivativeVector, inEB, gIncDJE, gIncidentDerivativeVector, gInEB);
    Basis3 gIncidentBasis = basis_zero();
    adj_vector_from_jones(pi.incidentJones, pi.incidentBasis, gIncidentVector, sg.incidentJones, gIncidentBasis);
    adj_vector_from_jones(
        pi.incidentDerivativeJones,
        pi.incidentBasis,
        gIncidentDerivativeVector,
        sg.incidentDerivativeJones,
        gIncidentBasis
    );
    basis_accum(sg.incidentBasis, gIncidentBasis);

    Complex gDirectTerm = cplx_zero();
    Complex gFace0Term = cplx_zero();
    Complex gFace1Term = cplx_zero();
    JonesOperator gFace0Op = jop_zero();
    JonesOperator gFace1Op = jop_zero();
    adj_assemble_diff_operator(
        cplx_mul_real(terms.direct, -1.f),
        terms.face0,
        terms.face1,
        f0Op,
        f1Op,
        gDirectOp,
        gDirectTerm,
        gFace0Term,
        gFace1Term,
        gFace0Op,
        gFace1Op
    );
    Complex slopeDirectCoeff = cplx_mul(slopeFactor, cplx_mul_real(slopeTerms.directDphiPrime, -1.f));
    Complex slopeFace0Coeff = cplx_mul(slopeFactor, slopeTerms.face0DphiPrime);
    Complex slopeFace1Coeff = cplx_mul(slopeFactor, slopeTerms.face1DphiPrime);
    Complex gSlopeDirectIgnored = cplx_zero();
    Complex gSlopeFace0Ignored = cplx_zero();
    Complex gSlopeFace1Ignored = cplx_zero();
    JonesOperator gSlopeFace0Op = jop_zero();
    JonesOperator gSlopeFace1Op = jop_zero();
    if (hasSlope) {
        adj_assemble_diff_operator(
            slopeDirectCoeff,
            slopeFace0Coeff,
            slopeFace1Coeff,
            f0Op,
            f1Op,
            gSlopeOp,
            gSlopeDirectIgnored,
            gSlopeFace0Ignored,
            gSlopeFace1Ignored,
            gSlopeFace0Op,
            gSlopeFace1Op
        );
        gFace0Op = jop_add(gFace0Op, gSlopeFace0Op);
        gFace1Op = jop_add(gFace1Op, gSlopeFace1Op);
    }
    (void) gSlopeDirectIgnored;
    (void) gSlopeFace0Ignored;
    (void) gSlopeFace1Ignored;

    float gPhi = 0.f;
    float gPhiP = 0.f;
    float gWedgeN = 0.f;
    float gTermsS = 0.f;
    float gTermsSP = 0.f;
    float gTermsSb = 0.f;
    Complex gRawDirect = cplx_mul_real(gDirectTerm, -1.f);
    if (useFace) {
        adj_compute_op_terms_3d_direct(
            phi,
            phiP,
            pi.wedgeN,
            k,
            s,
            sP,
            sb,
            gRawDirect,
            gFace0Term,
            gFace1Term,
            gPhi,
            gPhiP,
            gWedgeN,
            gTermsS,
            gTermsSP,
            gTermsSb
        );
    } else {
        adj_compute_op_terms_2d_direct(
            phi,
            phiP,
            pi.wedgeN,
            k,
            s,
            sP,
            gRawDirect,
            gFace0Term,
            gFace1Term,
            gPhi,
            gPhiP,
            gWedgeN,
            gTermsS,
            gTermsSP
        );
    }

    float gS = 0.f;
    float gSP = 0.f;
    float gSb = gTermsSb;
    sg.wedgeN += gWedgeN;
    Complex gPhase = cplx_zero();
    float gLs = 0.f;
    adj_cplx_mul_real(phase, ls, gScale, gPhase, gLs);
    Complex dPhaseDs = cplx_mul(phase, cplx(0.f, -k));
    gS += cplx_adj_dot(gPhase, dPhaseDs);

    float den = s * (s + sP) + UTD_EPS;
    float den2 = den * den;
    if (ls > UTD_SMALL_EPS) {
        float dLsDs = -0.5f * sP * (2.f * s + sP) / (ls * den2);
        float dLsDSP = 0.5f * (s * s + UTD_EPS) / (ls * den2);
        gS += gLs * dLsDs;
        gSP += gLs * dLsDSP;
    }
    gS += gTermsS;
    gSP += gTermsSP;

    float3a gFace0Normal = f3_zero();
    float3a gFace1Normal = f3_zero();
    float3a gFace0InHat = f3_zero();
    float3a gFace1InHat = f3_zero();
    float3a gFace0OutHat = f3_zero();
    float3a gFace1OutHat = f3_zero();
    Basis3 gFace0InEdgeBasis = basis_zero();
    Basis3 gFace1InEdgeBasis = basis_zero();
    Basis3 gFace0OutEdgeBasis = basis_zero();
    Basis3 gFace1OutEdgeBasis = basis_zero();
    float gCosTheta0 = 0.f;
    float gCosTheta1 = 0.f;
    float rawCosTheta0 = fabsf(sinf(phiP));
    float theta1 = pi.wedgeN * UTD_PI - phi;
    float rawCosTheta1 = fabsf(sinf(theta1));
    if (f0HasMat && !useStoredFaceOps) {
        adj_face_reflection_operator(
            pi.face0Material,
            fminf(fmaxf(rawCosTheta0, 1.0e-6f), 1.f),
            pi.n0,
            inEB.k,
            outEB.k,
            inEB,
            outEB,
            mat.omega,
            gFace0Op,
            gCosTheta0,
            gFace0Normal,
            gFace0InHat,
            gFace0OutHat,
            gFace0InEdgeBasis,
            gFace0OutEdgeBasis,
            sg.face0Material
        );
    } else {
        adj_fallback_face_operator(
            pi.face0Operator,
            pi.n0,
            inEB.k,
            outEB.k,
            inEB,
            outEB,
            gFace0Op,
            sg.face0Operator,
            gFace0Normal,
            gFace0InHat,
            gFace0OutHat,
            gFace0InEdgeBasis,
            gFace0OutEdgeBasis
        );
    }
    if (f1HasMat && !useStoredFaceOps) {
        adj_face_reflection_operator(
            pi.face1Material,
            fminf(fmaxf(rawCosTheta1, 1.0e-6f), 1.f),
            pi.nn,
            inEB.k,
            outEB.k,
            inEB,
            outEB,
            mat.omega,
            gFace1Op,
            gCosTheta1,
            gFace1Normal,
            gFace1InHat,
            gFace1OutHat,
            gFace1InEdgeBasis,
            gFace1OutEdgeBasis,
            sg.face1Material
        );
    } else {
        adj_fallback_face_operator(
            pi.face1Operator,
            pi.nn,
            inEB.k,
            outEB.k,
            inEB,
            outEB,
            gFace1Op,
            sg.face1Operator,
            gFace1Normal,
            gFace1InHat,
            gFace1OutHat,
            gFace1InEdgeBasis,
            gFace1OutEdgeBasis
        );
    }
    sg.n0 = f3_add(sg.n0, gFace0Normal);
    sg.nn = f3_add(sg.nn, gFace1Normal);
    gInEB.k = f3_add(gInEB.k, f3_add(gFace0InHat, gFace1InHat));
    gOutEB.k = f3_add(gOutEB.k, f3_add(gFace0OutHat, gFace1OutHat));
    basis_accum(gInEB, gFace0InEdgeBasis);
    basis_accum(gInEB, gFace1InEdgeBasis);
    basis_accum(gOutEB, gFace0OutEdgeBasis);
    basis_accum(gOutEB, gFace1OutEdgeBasis);
    if (rawCosTheta0 > 1.0e-6f && rawCosTheta0 < 1.f) {
        float sinPhiP = sinf(phiP);
        float signSinPhiP = sinPhiP >= 0.f ? 1.f : -1.f;
        gPhiP += gCosTheta0 * signSinPhiP * cosf(phiP);
    }
    if (rawCosTheta1 > 1.0e-6f && rawCosTheta1 < 1.f) {
        float sinTheta1 = sinf(theta1);
        float signSinTheta1 = sinTheta1 >= 0.f ? 1.f : -1.f;
        float gTheta1 = gCosTheta1 * signSinTheta1 * cosf(theta1);
        gPhi -= gTheta1;
        sg.wedgeN += UTD_PI * gTheta1;
    }

    float3a gInRayDir = f3_zero();
    float3a gInEdgeDir = f3_zero();
    adj_diffraction_edge_basis(
        f3_sub(pi.edgePos, pi.sourcePos),
        pi.edgeDir,
        false,
        gInEB,
        gInRayDir,
        gInEdgeDir
    );
    sg.edgePos = f3_add(sg.edgePos, gInRayDir);
    sg.sourcePos = f3_sub(sg.sourcePos, gInRayDir);
    sg.edgeDir = f3_add(sg.edgeDir, gInEdgeDir);

    float3a gOutRayDir = f3_zero();
    float3a gOutEdgeDir = f3_zero();
    adj_diffraction_edge_basis(
        f3_sub(tgt, pi.edgePos),
        pi.edgeDir,
        true,
        gOutEB,
        gOutRayDir,
        gOutEdgeDir
    );
    gTgt = f3_add(gTgt, gOutRayDir);
    sg.edgePos = f3_sub(sg.edgePos, gOutRayDir);
    sg.edgeDir = f3_add(sg.edgeDir, gOutEdgeDir);

    adj_compute_edge_geometry_3d(
        pi.sourcePos,
        pi.edgePos,
        pi.edgeDir,
        pi.n0,
        tgt,
        gPhi,
        gPhiP,
        gS,
        gSP,
        gSb,
        sg.sourcePos,
        sg.edgePos,
        sg.edgeDir,
        sg.n0,
        gTgt
    );
}

UTD_DINLINE Complex complex_add_scaled(Complex value, Complex tangent, float scale)
{
    return cplx_add(value, cplx_mul_real(tangent, scale));
}

UTD_DINLINE Complex3 complex3_add_scaled(Complex3 value, Complex3 tangent, float scale)
{
    return {
        complex_add_scaled(value.x, tangent.x, scale),
        complex_add_scaled(value.y, tangent.y, scale),
        complex_add_scaled(value.z, tangent.z, scale)
    };
}

UTD_DINLINE Jones2 jones_add_scaled(Jones2 value, Jones2 tangent, float scale)
{
    return {
        complex_add_scaled(value.u, tangent.u, scale),
        complex_add_scaled(value.v, tangent.v, scale)
    };
}

UTD_DINLINE JonesOperator jop_add_scaled(JonesOperator value, JonesOperator tangent, float scale)
{
    return {
        complex_add_scaled(value.m00, tangent.m00, scale),
        complex_add_scaled(value.m01, tangent.m01, scale),
        complex_add_scaled(value.m10, tangent.m10, scale),
        complex_add_scaled(value.m11, tangent.m11, scale)
    };
}

UTD_DINLINE Basis3 basis_add_scaled(Basis3 value, Basis3 tangent, float scale)
{
    return {
        f3_add(value.u, f3_mul(tangent.u, scale)),
        f3_add(value.v, f3_mul(tangent.v, scale)),
        f3_add(value.k, f3_mul(tangent.k, scale))
    };
}

UTD_DINLINE FaceMaterialParams face_material_add_scaled(
    FaceMaterialParams value,
    FaceMaterialParams tangent,
    float scale)
{
    return {
        value.etaR + scale * tangent.etaR,
        value.muR + scale * tangent.muR,
        value.sigma + scale * tangent.sigma,
        value.gain + scale * tangent.gain,
        value.useFresnel,
        value.present
    };
}

UTD_DINLINE PairInputs pair_inputs_add_scaled(PairInputs value, PairInputsGrad tangent, float scale)
{
    PairInputs out = value;
    out.edgePos = f3_add(value.edgePos, f3_mul(tangent.edgePos, scale));
    out.edgeDir = f3_add(value.edgeDir, f3_mul(tangent.edgeDir, scale));
    out.n0 = f3_add(value.n0, f3_mul(tangent.n0, scale));
    out.nn = f3_add(value.nn, f3_mul(tangent.nn, scale));
    out.wedgeN = value.wedgeN + scale * tangent.wedgeN;
    out.sourcePos = f3_add(value.sourcePos, f3_mul(tangent.sourcePos, scale));
    out.incidentField = complex_add_scaled(value.incidentField, tangent.incidentField, scale);
    out.incidentNormalDerivative = complex_add_scaled(
        value.incidentNormalDerivative,
        tangent.incidentNormalDerivative,
        scale);
    out.r0 = complex_add_scaled(value.r0, tangent.r0, scale);
    out.rn = complex_add_scaled(value.rn, tangent.rn, scale);
    out.incidentVector = complex3_add_scaled(value.incidentVector, tangent.incidentVector, scale);
    out.incidentDerivativeVector = complex3_add_scaled(
        value.incidentDerivativeVector,
        tangent.incidentDerivativeVector,
        scale);
    out.incidentJones = jones_add_scaled(value.incidentJones, tangent.incidentJones, scale);
    out.incidentDerivativeJones = jones_add_scaled(
        value.incidentDerivativeJones,
        tangent.incidentDerivativeJones,
        scale);
    out.incidentBasis = basis_add_scaled(value.incidentBasis, tangent.incidentBasis, scale);
    out.face0Operator = jop_add_scaled(value.face0Operator, tangent.face0Operator, scale);
    out.face1Operator = jop_add_scaled(value.face1Operator, tangent.face1Operator, scale);
    out.face0Material = face_material_add_scaled(
        value.face0Material,
        tangent.face0Material,
        scale);
    out.face1Material = face_material_add_scaled(
        value.face1Material,
        tangent.face1Material,
        scale);
    return out;
}

UTD_DINLINE Complex3 complex3_sub(Complex3 a, Complex3 b)
{
    return {cplx_sub(a.x, b.x), cplx_sub(a.y, b.y), cplx_sub(a.z, b.z)};
}

UTD_DINLINE Complex3 pair_vector_output_jvp_completion(
    PairInputs pi,
    PairInputsGrad tangentState,
    float3a tgt,
    float3a tangentTgt,
    float k,
    MaterialParams mat)
{
    constexpr float eps = 1.0e-3f;
    PairInputs plusState = pair_inputs_add_scaled(pi, tangentState, eps);
    PairInputs minusState = pair_inputs_add_scaled(pi, tangentState, -eps);
    float3a plusTgt = f3_add(tgt, f3_mul(tangentTgt, eps));
    float3a minusTgt = f3_add(tgt, f3_mul(tangentTgt, -eps));
    Complex3 plusValue = compute_pair_vector_contribution(plusState, plusTgt, k, mat);
    Complex3 minusValue = compute_pair_vector_contribution(minusState, minusTgt, k, mat);
    return c3_scale_real(complex3_sub(plusValue, minusValue), 0.5f / eps);
}

// ===================================================================
// Full pair contribution (scalar field + vector field)
// ===================================================================
UTD_DINLINE PairOutputs compute_pair_contribution(PairInputs state, float3a tgtPos,
    float k, MaterialParams mat)
{
    PairOutputs out;
    out.field = cplx_zero(); out.vectorField = c3_zero();
    bool gv; Complex dg, dvg;
    compute_pair_field_terms(state, tgtPos, k, mat, gv, out.field, dg, dvg);
    out.vectorField = compute_pair_vector_contribution(state, tgtPos, k, mat);
    return out;
}

} // namespace witwin::channel::native_ext
