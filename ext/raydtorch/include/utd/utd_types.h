#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define UTD_DEVICE   __device__
#define UTD_DINLINE  __device__ __forceinline__
#define UTD_GLOBAL   __global__
#else
#define UTD_DEVICE
#define UTD_DINLINE  inline
#define UTD_GLOBAL
#endif

namespace witwin::channel::native_ext {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr float UTD_PI             = 3.14159265358979323846f;
constexpr float UTD_TWO_PI         = 6.28318530717958647692f;
constexpr float UTD_EPS            = 1.0e-10f;
constexpr float UTD_SMALL_EPS      = 1.0e-6f;
constexpr float UTD_EPSILON_0      = 8.8541878128e-12f;
constexpr float UTD_MIN_DISTANCE   = 5.0e-2f;
constexpr float UTD_SLOPE_STEP     = 1.0e-4f;

constexpr int OWNERSHIP_DIRECT   = 0;
constexpr int OWNERSHIP_MIXED    = 1;
constexpr int UTD_PAIR_VALID_FLAG = 1;

// ---------------------------------------------------------------------------
// Primitive types
// ---------------------------------------------------------------------------
struct float3a {
    float x, y, z;
};

struct Complex {
    float re, im;
};

struct Complex3 {
    Complex x, y, z;
};

struct Jones2 {
    Complex u, v;
};

struct JonesOperator {
    Complex m00, m01, m10, m11;
};

struct Basis3 {
    float3a u, v, k;
};

struct FaceMaterialParams {
    float etaR;
    float muR;
    float sigma;
    float gain;
    float useFresnel;
    float present;
};

// Full state for a source-edge pair (SoA ??loaded per thread)
struct PairInputs {
    float3a edgePos;
    float3a edgeDir;
    float3a n0;
    float3a nn;
    float   wedgeN;
    float   edgeLineMin;
    float   edgeLineMax;
    float3a sourcePos;
    Complex incidentField;
    Complex incidentNormalDerivative;
    Complex r0;
    Complex rn;
    Complex3 incidentVector;
    Complex3 incidentDerivativeVector;
    Jones2  incidentJones;
    Jones2  incidentDerivativeJones;
    Basis3  incidentBasis;
    JonesOperator face0Operator;
    JonesOperator face1Operator;
    FaceMaterialParams face0Material;
    FaceMaterialParams face1Material;
    float   selectStationaryPoint;
};

struct PairOutputs {
    Complex  field;
    Complex3 vectorField;
};

struct DiffractionOperatorTerms {
    Complex direct;
    Complex face0;
    Complex face1;
    Complex directDphiPrime;
    Complex face0DphiPrime;
    Complex face1DphiPrime;
};

struct EdgeAngleCache {
    float3a sourceToEdge;
    float3a sourceToEdgeProj;
    float   sourceToEdgeProjNorm;
    float3a edgeToTarget;
    float3a edgeToTargetProj;
    float   edgeToTargetProjNorm;
    float3a toHatBase;
    float3a toHat;
    float   toHatBaseNorm;
    float3a kiProj;
    float3a koProj;
    float   phi;
    float   phiPrime;
    float   s;
    float   sPrime;
};

struct BetaTermCache {
    float n, kL, cotSign, cotArg;
    float cotValue, cot1, cot2;
    float a, a1, a2, aN, a1N;
    Complex transition, transition1, transition2;
    Complex value, first, second;
};

struct PairScalarInputs {
    float phi, phiPrime, s, sPrime, wedgeN;
    Complex incidentField;
    Complex incidentNormalDerivative;
    Complex r0, rn;
};

struct MaterialParams {
    int   useFresnel;
    float etaR;
    float muR;
    float sigma;
    float gain;
    float omega;
    float txPolX;
    float txPolY;
    float txPolZ;
};

// Gradient accumulator for PairInputs (mirrors the differentiable fields)
struct PairInputsGrad {
    float3a edgePos;
    float3a edgeDir;
    float3a n0;
    float3a nn;
    float   wedgeN;
    float3a sourcePos;
    Complex incidentField;
    Complex incidentNormalDerivative;
    Complex r0;
    Complex rn;
    Complex3 incidentVector;
    Complex3 incidentDerivativeVector;
    Jones2  incidentJones;
    Jones2  incidentDerivativeJones;
    Basis3  incidentBasis;
    JonesOperator face0Operator;
    JonesOperator face1Operator;
    FaceMaterialParams face0Material;
    FaceMaterialParams face1Material;
};

// ---------------------------------------------------------------------------
// float3a inline helpers
// ---------------------------------------------------------------------------
UTD_DINLINE float3a make_f3(float x, float y, float z) { return {x, y, z}; }
UTD_DINLINE float3a f3_zero() { return {0.f, 0.f, 0.f}; }
UTD_DINLINE float3a f3_add(float3a a, float3a b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
UTD_DINLINE float3a f3_sub(float3a a, float3a b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
UTD_DINLINE float3a f3_mul(float3a a, float s) { return {a.x*s, a.y*s, a.z*s}; }
UTD_DINLINE float3a f3_neg(float3a a) { return {-a.x, -a.y, -a.z}; }
UTD_DINLINE float f3_dot(float3a a, float3a b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
UTD_DINLINE float3a f3_cross(float3a a, float3a b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
UTD_DINLINE float f3_len(float3a a) { return sqrtf(fmaxf(f3_dot(a,a), 0.f)); }
UTD_DINLINE float3a f3_div(float3a a, float s) { return {a.x/s, a.y/s, a.z/s}; }

// ---------------------------------------------------------------------------
// Complex inline helpers
// ---------------------------------------------------------------------------
UTD_DINLINE Complex cplx(float re, float im) { return {re, im}; }
UTD_DINLINE Complex cplx_zero() { return {0.f, 0.f}; }
UTD_DINLINE Complex cplx_add(Complex a, Complex b) { return {a.re+b.re, a.im+b.im}; }
UTD_DINLINE Complex cplx_sub(Complex a, Complex b) { return {a.re-b.re, a.im-b.im}; }
UTD_DINLINE Complex cplx_mul(Complex a, Complex b) {
    return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}
UTD_DINLINE Complex cplx_mul_real(Complex a, float b) { return {a.re*b, a.im*b}; }
UTD_DINLINE Complex cplx_div_real(Complex a, float b) { return {a.re/b, a.im/b}; }
UTD_DINLINE Complex cplx_div(Complex a, Complex b) {
    float d = b.re*b.re + b.im*b.im + UTD_EPS;
    return {(a.re*b.re + a.im*b.im)/d, (a.im*b.re - a.re*b.im)/d};
}
UTD_DINLINE Complex cplx_conj(Complex a) { return {a.re, -a.im}; }
UTD_DINLINE Complex cplx_exp_phase(float p) {
#ifdef __CUDACC__
    float s, c; sincosf(p, &s, &c);
#else
    float s = sinf(p), c = cosf(p);
#endif
    return {c, s};
}
UTD_DINLINE float   cplx_abs_sqr(Complex a) { return a.re*a.re + a.im*a.im; }
UTD_DINLINE float   cplx_adj_dot(Complex g, Complex v) { return g.re*v.re + g.im*v.im; }
UTD_DINLINE bool    cplx_any_nonzero(Complex v) { return fabsf(v.re)>0.f || fabsf(v.im)>0.f; }

// ---------------------------------------------------------------------------
// Complex3 inline helpers
// ---------------------------------------------------------------------------
UTD_DINLINE Complex3 c3_zero() { return {cplx_zero(), cplx_zero(), cplx_zero()}; }
UTD_DINLINE Complex3 c3_add(Complex3 a, Complex3 b) {
    return {cplx_add(a.x,b.x), cplx_add(a.y,b.y), cplx_add(a.z,b.z)};
}
UTD_DINLINE Complex3 c3_scale(Complex3 v, Complex c) {
    return {cplx_mul(v.x,c), cplx_mul(v.y,c), cplx_mul(v.z,c)};
}
UTD_DINLINE Complex cplx_dot_real(Complex3 v, float3a b) {
    Complex s = cplx_zero();
    s = cplx_add(s, cplx_mul_real(v.x, b.x));
    s = cplx_add(s, cplx_mul_real(v.y, b.y));
    s = cplx_add(s, cplx_mul_real(v.z, b.z));
    return s;
}
UTD_DINLINE Complex3 cplx_scale_real(float3a b, Complex c) {
    return {cplx_mul_real(c, b.x), cplx_mul_real(c, b.y), cplx_mul_real(c, b.z)};
}
UTD_DINLINE bool c3_grad_any_nonzero(Complex3 v) {
    return cplx_any_nonzero(v.x) || cplx_any_nonzero(v.y) || cplx_any_nonzero(v.z);
}

// ---------------------------------------------------------------------------
// Jones inline helpers
// ---------------------------------------------------------------------------
UTD_DINLINE Jones2 jones_zero() { return {cplx_zero(), cplx_zero()}; }
UTD_DINLINE Jones2 jones_add(Jones2 a, Jones2 b) { return {cplx_add(a.u,b.u), cplx_add(a.v,b.v)}; }
UTD_DINLINE Jones2 jones_scale(Jones2 v, Complex c) { return {cplx_mul(v.u,c), cplx_mul(v.v,c)}; }

UTD_DINLINE JonesOperator jop_zero() { return {cplx_zero(),cplx_zero(),cplx_zero(),cplx_zero()}; }
UTD_DINLINE JonesOperator jop_identity() { return {cplx(1,0),cplx_zero(),cplx_zero(),cplx(1,0)}; }
UTD_DINLINE JonesOperator jop_add(JonesOperator a, JonesOperator b) {
    return {cplx_add(a.m00,b.m00), cplx_add(a.m01,b.m01),
            cplx_add(a.m10,b.m10), cplx_add(a.m11,b.m11)};
}
UTD_DINLINE JonesOperator jop_scale(JonesOperator v, Complex c) {
    return {cplx_mul(v.m00,c), cplx_mul(v.m01,c),
            cplx_mul(v.m10,c), cplx_mul(v.m11,c)};
}
UTD_DINLINE Jones2 apply_jop(Jones2 v, JonesOperator op) {
    return {cplx_add(cplx_mul(op.m00,v.u), cplx_mul(op.m01,v.v)),
            cplx_add(cplx_mul(op.m10,v.u), cplx_mul(op.m11,v.v))};
}
UTD_DINLINE Complex3 vector_from_jones(Jones2 v, Basis3 b) {
    return c3_add(cplx_scale_real(b.u, v.u), cplx_scale_real(b.v, v.v));
}
UTD_DINLINE Jones2 jones_from_vector(Complex3 v, Basis3 b) {
    return {cplx_dot_real(v, b.u), cplx_dot_real(v, b.v)};
}

// ---------------------------------------------------------------------------
// Gradient helper: zero-initialise a PairInputsGrad
// ---------------------------------------------------------------------------
UTD_DINLINE PairInputsGrad pig_zero() {
    PairInputsGrad g{};
    g.edgePos = f3_zero(); g.edgeDir = f3_zero();
    g.n0 = f3_zero(); g.nn = f3_zero(); g.wedgeN = 0.f;
    g.sourcePos = f3_zero();
    g.incidentField = cplx_zero(); g.incidentNormalDerivative = cplx_zero();
    g.r0 = cplx_zero(); g.rn = cplx_zero();
    g.incidentVector = c3_zero(); g.incidentDerivativeVector = c3_zero();
    g.incidentJones = jones_zero(); g.incidentDerivativeJones = jones_zero();
    g.incidentBasis = {f3_zero(), f3_zero(), f3_zero()};
    g.face0Operator = jop_zero(); g.face1Operator = jop_zero();
    g.face0Material = {0,0,0,0,0,0}; g.face1Material = {0,0,0,0,0,0};
    return g;
}

// ---------------------------------------------------------------------------
// Adjoint helper macros for complex mul
// ---------------------------------------------------------------------------
UTD_DINLINE void adj_cplx_mul(Complex a, Complex b, Complex gO,
                              Complex& gA, Complex& gB) {
    gA.re += gO.re*b.re + gO.im*b.im;
    gA.im += -gO.re*b.im + gO.im*b.re;
    gB.re += gO.re*a.re + gO.im*a.im;
    gB.im += -gO.re*a.im + gO.im*a.re;
}
UTD_DINLINE void adj_cplx_mul_real(Complex a, float b, Complex gO,
                                   Complex& gA, float& gB) {
    gA.re += gO.re*b;
    gA.im += gO.im*b;
    gB += cplx_adj_dot(gO, a);
}
UTD_DINLINE void adj_cplx_div(Complex a, Complex b, Complex gO,
                              Complex& gA, Complex& gB) {
    Complex invB = cplx_div(cplx(1.f, 0.f), b);
    Complex coeffB = cplx_mul_real(cplx_mul(a, cplx_mul(invB, invB)), -1.f);
    gA = cplx_add(gA, cplx_div(gO, cplx_conj(b)));
    gB.re += gO.re * coeffB.re + gO.im * coeffB.im;
    gB.im += -gO.re * coeffB.im + gO.im * coeffB.re;
}
UTD_DINLINE void adj_cplx_scale_real(float3a basis, Complex coeff,
                                     Complex3 gO, float3a& gBasis, Complex& gCoeff) {
    gBasis.x += cplx_adj_dot(gO.x, coeff);
    gBasis.y += cplx_adj_dot(gO.y, coeff);
    gBasis.z += cplx_adj_dot(gO.z, coeff);
    gCoeff.re += gO.x.re*basis.x + gO.y.re*basis.y + gO.z.re*basis.z;
    gCoeff.im += gO.x.im*basis.x + gO.y.im*basis.y + gO.z.im*basis.z;
}
UTD_DINLINE void adj_cplx_dot_real(Complex3 v, float3a b, Complex gO,
                                   Complex3& gV, float3a& gB) {
    gV.x.re += gO.re*b.x; gV.x.im += gO.im*b.x;
    gV.y.re += gO.re*b.y; gV.y.im += gO.im*b.y;
    gV.z.re += gO.re*b.z; gV.z.im += gO.im*b.z;
    gB.x += cplx_adj_dot(gO, v.x);
    gB.y += cplx_adj_dot(gO, v.y);
    gB.z += cplx_adj_dot(gO, v.z);
}
UTD_DINLINE void adj_c3_scale(Complex3 v, Complex c, Complex3 gO,
                              Complex3& gV, Complex& gC) {
    Complex gVx=cplx_zero(), gCx=cplx_zero();
    adj_cplx_mul(v.x, c, gO.x, gVx, gCx);
    gV.x = cplx_add(gV.x, gVx); gC = cplx_add(gC, gCx);
    Complex gVy=cplx_zero(), gCy=cplx_zero();
    adj_cplx_mul(v.y, c, gO.y, gVy, gCy);
    gV.y = cplx_add(gV.y, gVy); gC = cplx_add(gC, gCy);
    Complex gVz=cplx_zero(), gCz=cplx_zero();
    adj_cplx_mul(v.z, c, gO.z, gVz, gCz);
    gV.z = cplx_add(gV.z, gVz); gC = cplx_add(gC, gCz);
}

UTD_DINLINE void adj_jones_add(Jones2 gO, Jones2& gA, Jones2& gB) {
    gA.u = cplx_add(gA.u, gO.u);
    gA.v = cplx_add(gA.v, gO.v);
    gB.u = cplx_add(gB.u, gO.u);
    gB.v = cplx_add(gB.v, gO.v);
}

UTD_DINLINE void adj_jones_scale(Jones2 v, Complex c, Jones2 gO,
                                 Jones2& gV, Complex& gC) {
    Complex gVu = cplx_zero(), gCu = cplx_zero();
    adj_cplx_mul(v.u, c, gO.u, gVu, gCu);
    gV.u = cplx_add(gV.u, gVu);
    gC = cplx_add(gC, gCu);

    Complex gVv = cplx_zero(), gCv = cplx_zero();
    adj_cplx_mul(v.v, c, gO.v, gVv, gCv);
    gV.v = cplx_add(gV.v, gVv);
    gC = cplx_add(gC, gCv);
}

UTD_DINLINE void adj_apply_jop(Jones2 v, JonesOperator op, Jones2 gO,
                               Jones2& gV, JonesOperator& gOp) {
    Complex gVU = cplx_zero(), gVV = cplx_zero();

    Complex gM00 = cplx_zero(), gM01 = cplx_zero();
    adj_cplx_mul(op.m00, v.u, gO.u, gM00, gVU);
    gOp.m00 = cplx_add(gOp.m00, gM00);
    Complex gM01b = cplx_zero(), gVVb = cplx_zero();
    adj_cplx_mul(op.m01, v.v, gO.u, gM01b, gVVb);
    gOp.m01 = cplx_add(gOp.m01, gM01b);
    gVV = cplx_add(gVV, gVVb);

    Complex gM10 = cplx_zero(), gM11 = cplx_zero();
    Complex gVUb = cplx_zero(), gVVc = cplx_zero();
    adj_cplx_mul(op.m10, v.u, gO.v, gM10, gVUb);
    adj_cplx_mul(op.m11, v.v, gO.v, gM11, gVVc);
    gOp.m10 = cplx_add(gOp.m10, gM10);
    gOp.m11 = cplx_add(gOp.m11, gM11);
    gVU = cplx_add(gVU, gVUb);
    gVV = cplx_add(gVV, gVVc);

    gV.u = cplx_add(gV.u, gVU);
    gV.v = cplx_add(gV.v, gVV);
}

UTD_DINLINE void adj_vector_from_jones(Jones2 v, Basis3 b, Complex3 gO,
                                       Jones2& gV, Basis3& gB) {
    Complex gCoeff = cplx_zero();
    adj_cplx_scale_real(b.u, v.u, gO, gB.u, gCoeff);
    gV.u = cplx_add(gV.u, gCoeff);
    gCoeff = cplx_zero();
    adj_cplx_scale_real(b.v, v.v, gO, gB.v, gCoeff);
    gV.v = cplx_add(gV.v, gCoeff);
}

UTD_DINLINE void adj_jones_from_vector(Complex3 v, Basis3 b, Jones2 gO,
                                       Complex3& gV, Basis3& gB) {
    adj_cplx_dot_real(v, b.u, gO.u, gV, gB.u);
    adj_cplx_dot_real(v, b.v, gO.v, gV, gB.v);
}

UTD_DINLINE void adj_jop_add(JonesOperator gO, JonesOperator& gA, JonesOperator& gB) {
    gA.m00 = cplx_add(gA.m00, gO.m00);
    gA.m01 = cplx_add(gA.m01, gO.m01);
    gA.m10 = cplx_add(gA.m10, gO.m10);
    gA.m11 = cplx_add(gA.m11, gO.m11);
    gB.m00 = cplx_add(gB.m00, gO.m00);
    gB.m01 = cplx_add(gB.m01, gO.m01);
    gB.m10 = cplx_add(gB.m10, gO.m10);
    gB.m11 = cplx_add(gB.m11, gO.m11);
}

UTD_DINLINE void adj_jop_scale(JonesOperator v, Complex c, JonesOperator gO,
                               JonesOperator& gV, Complex& gC) {
    Complex gElem = cplx_zero(), gCoeff = cplx_zero();
    adj_cplx_mul(v.m00, c, gO.m00, gElem, gCoeff);
    gV.m00 = cplx_add(gV.m00, gElem);
    gC = cplx_add(gC, gCoeff);

    gElem = cplx_zero(); gCoeff = cplx_zero();
    adj_cplx_mul(v.m01, c, gO.m01, gElem, gCoeff);
    gV.m01 = cplx_add(gV.m01, gElem);
    gC = cplx_add(gC, gCoeff);

    gElem = cplx_zero(); gCoeff = cplx_zero();
    adj_cplx_mul(v.m10, c, gO.m10, gElem, gCoeff);
    gV.m10 = cplx_add(gV.m10, gElem);
    gC = cplx_add(gC, gCoeff);

    gElem = cplx_zero(); gCoeff = cplx_zero();
    adj_cplx_mul(v.m11, c, gO.m11, gElem, gCoeff);
    gV.m11 = cplx_add(gV.m11, gElem);
    gC = cplx_add(gC, gCoeff);
}

UTD_DINLINE void adj_assemble_diff_operator(
    Complex freeTerm,
    Complex face0Term,
    Complex face1Term,
    JonesOperator face0Op,
    JonesOperator face1Op,
    JonesOperator gO,
    Complex& gFreeTerm,
    Complex& gFace0Term,
    Complex& gFace1Term,
    JonesOperator& gFace0Op,
    JonesOperator& gFace1Op)
{
    gFreeTerm = cplx_add(gFreeTerm, gO.m00);
    gFreeTerm = cplx_add(gFreeTerm, gO.m11);

    Complex gCoeff = cplx_zero();
    adj_jop_scale(face0Op, face0Term, gO, gFace0Op, gCoeff);
    gFace0Term = cplx_add(gFace0Term, gCoeff);

    gCoeff = cplx_zero();
    adj_jop_scale(face1Op, face1Term, gO, gFace1Op, gCoeff);
    gFace1Term = cplx_add(gFace1Term, gCoeff);
}

} // namespace witwin::channel::native_ext
