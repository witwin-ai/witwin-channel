implementing utd_accumulate;

static const float PI = 3.14159265358979323846f;
static const float TWO_PI = 6.28318530717958647692f;
static const float EPS = 1.0e-10f;
static const float SMALL_EPS = 1.0e-6f;
static const float EPSILON_0 = 8.8541878128e-12f;
static const float DIFFRACTION_MIN_DISTANCE = 5.0e-2f;
static const float SLOPE_DERIVATIVE_STEP = 1.0e-4f;
static const int OWNERSHIP_DIRECT_DIFFRACTION = 0;
static const int OWNERSHIP_MIXED_DIFFRACTION = 1;

struct Complex : IDifferentiable
{
    float re;
    float im;
};

struct Complex3 : IDifferentiable
{
    Complex x;
    Complex y;
    Complex z;
};

struct Jones2 : IDifferentiable
{
    Complex u;
    Complex v;
};

struct JonesOperator : IDifferentiable
{
    Complex m00;
    Complex m01;
    Complex m10;
    Complex m11;
};

struct Basis3 : IDifferentiable
{
    float3 u;
    float3 v;
    float3 k;
};

struct FaceMaterialParams : IDifferentiable
{
    float etaR;
    float sigma;
    float gain;
    float useFresnel;
    float present;
};

struct PairInputs : IDifferentiable
{
    float3 edgePos;
    float3 edgeDir;
    float3 n0;
    float3 nn;
    float wedgeN;
    float3 sourcePos;
    Complex incidentField;
    Complex incidentNormalDerivative;
    Complex r0;
    Complex rn;
    Complex3 incidentVector;
    Complex3 incidentDerivativeVector;
    Jones2 incidentJones;
    Jones2 incidentDerivativeJones;
    Basis3 incidentBasis;
    JonesOperator face0Operator;
    JonesOperator face1Operator;
    FaceMaterialParams face0Material;
    FaceMaterialParams face1Material;
};

struct PairOutputs : IDifferentiable
{
    Complex field;
    Complex3 vectorField;
};

struct PairScalarInputs : IDifferentiable
{
    float phi;
    float phiPrime;
    float s;
    float sPrime;
    float wedgeN;
    Complex incidentField;
    Complex incidentNormalDerivative;
    Complex r0;
    Complex rn;
};

struct FaceReflectionOutputs : IDifferentiable
{
    Complex r0;
    Complex rn;
};

struct EdgeAngleCache
{
    float3 sourceToEdge;
    float3 sourceToEdgeProj;
    float sourceToEdgeProjNorm;
    float3 edgeToTarget;
    float3 edgeToTargetProj;
    float edgeToTargetProjNorm;
    float3 toHatBase;
    float3 toHat;
    float toHatBaseNorm;
    float3 kiProj;
    float3 koProj;
    float phi;
    float phiPrime;
    float s;
    float sPrime;
};

struct BetaTermCache
{
    float n;
    float kL;
    float cotSign;
    float cotArg;
    float cotValue;
    float cot1;
    float cot2;
    float a;
    float a1;
    float a2;
    float aN;
    float a1N;
    Complex transition;
    Complex transition1;
    Complex transition2;
    Complex value;
    Complex first;
    Complex second;
};

struct StateTensors
{
    TensorView<float> edgePosX;
    TensorView<float> edgePosY;
    TensorView<float> edgePosZ;
    TensorView<float> edgeDirX;
    TensorView<float> edgeDirY;
    TensorView<float> edgeDirZ;
    TensorView<float> n0X;
    TensorView<float> n0Y;
    TensorView<float> n0Z;
    TensorView<float> nnX;
    TensorView<float> nnY;
    TensorView<float> nnZ;
    TensorView<float> wedgeN;
    TensorView<float> sourcePosX;
    TensorView<float> sourcePosY;
    TensorView<float> sourcePosZ;
    TensorView<float> incidentFieldReal;
    TensorView<float> incidentFieldImag;
    TensorView<float> incidentNormalDerivativeReal;
    TensorView<float> incidentNormalDerivativeImag;
    TensorView<float> r0Real;
    TensorView<float> r0Imag;
    TensorView<float> rnReal;
    TensorView<float> rnImag;
    TensorView<float> incidentVectorXReal;
    TensorView<float> incidentVectorXImag;
    TensorView<float> incidentVectorYReal;
    TensorView<float> incidentVectorYImag;
    TensorView<float> incidentVectorZReal;
    TensorView<float> incidentVectorZImag;
    TensorView<float> incidentDerivativeVectorXReal;
    TensorView<float> incidentDerivativeVectorXImag;
    TensorView<float> incidentDerivativeVectorYReal;
    TensorView<float> incidentDerivativeVectorYImag;
    TensorView<float> incidentDerivativeVectorZReal;
    TensorView<float> incidentDerivativeVectorZImag;
    TensorView<float> incidentJonesUReal;
    TensorView<float> incidentJonesUImag;
    TensorView<float> incidentJonesVReal;
    TensorView<float> incidentJonesVImag;
    TensorView<float> incidentDerivativeJonesUReal;
    TensorView<float> incidentDerivativeJonesUImag;
    TensorView<float> incidentDerivativeJonesVReal;
    TensorView<float> incidentDerivativeJonesVImag;
    TensorView<float> incidentBasisUX;
    TensorView<float> incidentBasisUY;
    TensorView<float> incidentBasisUZ;
    TensorView<float> incidentBasisVX;
    TensorView<float> incidentBasisVY;
    TensorView<float> incidentBasisVZ;
    TensorView<float> incidentBasisKX;
    TensorView<float> incidentBasisKY;
    TensorView<float> incidentBasisKZ;
    TensorView<float> face0OperatorM00Real;
    TensorView<float> face0OperatorM00Imag;
    TensorView<float> face0OperatorM01Real;
    TensorView<float> face0OperatorM01Imag;
    TensorView<float> face0OperatorM10Real;
    TensorView<float> face0OperatorM10Imag;
    TensorView<float> face0OperatorM11Real;
    TensorView<float> face0OperatorM11Imag;
    TensorView<float> face1OperatorM00Real;
    TensorView<float> face1OperatorM00Imag;
    TensorView<float> face1OperatorM01Real;
    TensorView<float> face1OperatorM01Imag;
    TensorView<float> face1OperatorM10Real;
    TensorView<float> face1OperatorM10Imag;
    TensorView<float> face1OperatorM11Real;
    TensorView<float> face1OperatorM11Imag;
    TensorView<float> face0EtaR;
    TensorView<float> face0Sigma;
    TensorView<float> face0Gain;
    TensorView<float> face0UseFresnel;
    TensorView<float> face0MaterialPresent;
    TensorView<float> face1EtaR;
    TensorView<float> face1Sigma;
    TensorView<float> face1Gain;
    TensorView<float> face1UseFresnel;
    TensorView<float> face1MaterialPresent;
};

struct RxTensors
{
    TensorView<float> x;
    TensorView<float> y;
    TensorView<float> z;
};

struct OutputTensors
{
    TensorView<float> directReal;
    TensorView<float> directImag;
    TensorView<float> multiReal;
    TensorView<float> multiImag;
    TensorView<float> directVectorXReal;
    TensorView<float> directVectorXImag;
    TensorView<float> directVectorYReal;
    TensorView<float> directVectorYImag;
    TensorView<float> directVectorZReal;
    TensorView<float> directVectorZImag;
    TensorView<float> multiVectorXReal;
    TensorView<float> multiVectorXImag;
    TensorView<float> multiVectorYReal;
    TensorView<float> multiVectorYImag;
    TensorView<float> multiVectorZReal;
    TensorView<float> multiVectorZImag;
};

struct MaterialParams
{
    int useFresnel;
    float etaR;
    float sigma;
    float gain;
    float omega;
};

Complex complexZero()
{
    return Complex(0.0f, 0.0f);
}

[Differentiable]
Complex complexAdd(Complex a, Complex b)
{
    return Complex(a.re + b.re, a.im + b.im);
}

[Differentiable]
Complex complexSub(Complex a, Complex b)
{
    return Complex(a.re - b.re, a.im - b.im);
}

[Differentiable]
Complex complexMul(Complex a, Complex b)
{
    return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

[Differentiable]
Complex complexMulReal(Complex a, float b)
{
    return Complex(a.re * b, a.im * b);
}

float complexAdjointDot(Complex grad, Complex value)
{
    return grad.re * value.re + grad.im * value.im;
}

void adjComplexMul(Complex a, Complex b, Complex gradOut, inout Complex gradA, inout Complex gradB)
{
    gradA.re += gradOut.re * b.re + gradOut.im * b.im;
    gradA.im += -gradOut.re * b.im + gradOut.im * b.re;
    gradB.re += gradOut.re * a.re + gradOut.im * a.im;
    gradB.im += -gradOut.re * a.im + gradOut.im * a.re;
}

void adjComplexMulReal(Complex a, float b, Complex gradOut, inout Complex gradA, inout float gradB)
{
    gradA.re += gradOut.re * b;
    gradA.im += gradOut.im * b;
    gradB += complexAdjointDot(gradOut, a);
}

[Differentiable]
Complex complexDivReal(Complex a, float b)
{
    return Complex(a.re / b, a.im / b);
}

[Differentiable]
Complex complexDiv(Complex a, Complex b)
{
    float denom = b.re * b.re + b.im * b.im + EPS;
    return Complex(
        (a.re * b.re + a.im * b.im) / denom,
        (a.im * b.re - a.re * b.im) / denom
    );
}

[Differentiable]
Complex complexConj(Complex a)
{
    return Complex(a.re, -a.im);
}

[Differentiable]
Complex complexExpPhase(float phase)
{
    float s, c;
    sincos(phase, s, c);
    return Complex(c, s);
}

[Differentiable]
float complexAbsSqr(Complex a)
{
    return a.re * a.re + a.im * a.im;
}

Complex3 complex3Zero()
{
    return Complex3(complexZero(), complexZero(), complexZero());
}

Jones2 jonesZero()
{
    return Jones2(complexZero(), complexZero());
}

JonesOperator jonesOperatorZero()
{
    return JonesOperator(complexZero(), complexZero(), complexZero(), complexZero());
}

Complex adjointZero()
{
    return Complex(0.0f, 0.0f);
}

float3 zeroFloat3()
{
    return float3(0.0f, 0.0f, 0.0f);
}

PairInputs.Differential zeroPairInputsDifferential()
{
    return PairInputs.Differential(
        zeroFloat3(),
        zeroFloat3(),
        zeroFloat3(),
        zeroFloat3(),
        0.0f,
        zeroFloat3(),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex3.Differential(
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f)
        ),
        Complex3.Differential(
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f)
        ),
        Jones2.Differential(
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f)
        ),
        Jones2.Differential(
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f)
        ),
        Basis3.Differential(
            zeroFloat3(),
            zeroFloat3(),
            zeroFloat3()
        ),
        JonesOperator.Differential(
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f)
        ),
        JonesOperator.Differential(
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f)
        ),
        FaceMaterialParams.Differential(
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f
        ),
        FaceMaterialParams.Differential(
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f
        )
    );
}

void addPairInputsDifferential(inout PairInputs.Differential dst, PairInputs.Differential src)
{
    dst.edgePos += src.edgePos;
    dst.edgeDir += src.edgeDir;
    dst.n0 += src.n0;
    dst.nn += src.nn;
    dst.wedgeN += src.wedgeN;
    dst.sourcePos += src.sourcePos;
    dst.incidentField.re += src.incidentField.re;
    dst.incidentField.im += src.incidentField.im;
    dst.incidentNormalDerivative.re += src.incidentNormalDerivative.re;
    dst.incidentNormalDerivative.im += src.incidentNormalDerivative.im;
    dst.r0.re += src.r0.re;
    dst.r0.im += src.r0.im;
    dst.rn.re += src.rn.re;
    dst.rn.im += src.rn.im;
    dst.incidentVector.x.re += src.incidentVector.x.re;
    dst.incidentVector.x.im += src.incidentVector.x.im;
    dst.incidentVector.y.re += src.incidentVector.y.re;
    dst.incidentVector.y.im += src.incidentVector.y.im;
    dst.incidentVector.z.re += src.incidentVector.z.re;
    dst.incidentVector.z.im += src.incidentVector.z.im;
    dst.incidentDerivativeVector.x.re += src.incidentDerivativeVector.x.re;
    dst.incidentDerivativeVector.x.im += src.incidentDerivativeVector.x.im;
    dst.incidentDerivativeVector.y.re += src.incidentDerivativeVector.y.re;
    dst.incidentDerivativeVector.y.im += src.incidentDerivativeVector.y.im;
    dst.incidentDerivativeVector.z.re += src.incidentDerivativeVector.z.re;
    dst.incidentDerivativeVector.z.im += src.incidentDerivativeVector.z.im;
    dst.incidentJones.u.re += src.incidentJones.u.re;
    dst.incidentJones.u.im += src.incidentJones.u.im;
    dst.incidentJones.v.re += src.incidentJones.v.re;
    dst.incidentJones.v.im += src.incidentJones.v.im;
    dst.incidentDerivativeJones.u.re += src.incidentDerivativeJones.u.re;
    dst.incidentDerivativeJones.u.im += src.incidentDerivativeJones.u.im;
    dst.incidentDerivativeJones.v.re += src.incidentDerivativeJones.v.re;
    dst.incidentDerivativeJones.v.im += src.incidentDerivativeJones.v.im;
    dst.incidentBasis.u += src.incidentBasis.u;
    dst.incidentBasis.v += src.incidentBasis.v;
    dst.incidentBasis.k += src.incidentBasis.k;
    dst.face0Operator.m00.re += src.face0Operator.m00.re;
    dst.face0Operator.m00.im += src.face0Operator.m00.im;
    dst.face0Operator.m01.re += src.face0Operator.m01.re;
    dst.face0Operator.m01.im += src.face0Operator.m01.im;
    dst.face0Operator.m10.re += src.face0Operator.m10.re;
    dst.face0Operator.m10.im += src.face0Operator.m10.im;
    dst.face0Operator.m11.re += src.face0Operator.m11.re;
    dst.face0Operator.m11.im += src.face0Operator.m11.im;
    dst.face1Operator.m00.re += src.face1Operator.m00.re;
    dst.face1Operator.m00.im += src.face1Operator.m00.im;
    dst.face1Operator.m01.re += src.face1Operator.m01.re;
    dst.face1Operator.m01.im += src.face1Operator.m01.im;
    dst.face1Operator.m10.re += src.face1Operator.m10.re;
    dst.face1Operator.m10.im += src.face1Operator.m10.im;
    dst.face1Operator.m11.re += src.face1Operator.m11.re;
    dst.face1Operator.m11.im += src.face1Operator.m11.im;
    dst.face0Material.etaR += src.face0Material.etaR;
    dst.face0Material.sigma += src.face0Material.sigma;
    dst.face0Material.gain += src.face0Material.gain;
    dst.face0Material.useFresnel += src.face0Material.useFresnel;
    dst.face0Material.present += src.face0Material.present;
    dst.face1Material.etaR += src.face1Material.etaR;
    dst.face1Material.sigma += src.face1Material.sigma;
    dst.face1Material.gain += src.face1Material.gain;
    dst.face1Material.useFresnel += src.face1Material.useFresnel;
    dst.face1Material.present += src.face1Material.present;
}

bool complexAnyNonzero(Complex value)
{
    return (abs(value.re) > 0.0f) || (abs(value.im) > 0.0f);
}

[Differentiable]
Complex3 complex3Add(Complex3 a, Complex3 b)
{
    return Complex3(
        complexAdd(a.x, b.x),
        complexAdd(a.y, b.y),
        complexAdd(a.z, b.z)
    );
}

[Differentiable]
Complex3 complex3Scale(Complex3 value, Complex coeff)
{
    return Complex3(
        complexMul(value.x, coeff),
        complexMul(value.y, coeff),
        complexMul(value.z, coeff)
    );
}

[Differentiable]
Jones2 jonesAdd(Jones2 a, Jones2 b)
{
    return Jones2(
        complexAdd(a.u, b.u),
        complexAdd(a.v, b.v)
    );
}

[Differentiable]
Jones2 jonesScale(Jones2 value, Complex coeff)
{
    return Jones2(
        complexMul(value.u, coeff),
        complexMul(value.v, coeff)
    );
}

[Differentiable]
JonesOperator jonesOperatorIdentity()
{
    return JonesOperator(
        Complex(1.0f, 0.0f),
        complexZero(),
        complexZero(),
        Complex(1.0f, 0.0f)
    );
}

[Differentiable]
JonesOperator jonesOperatorAdd(JonesOperator a, JonesOperator b)
{
    return JonesOperator(
        complexAdd(a.m00, b.m00),
        complexAdd(a.m01, b.m01),
        complexAdd(a.m10, b.m10),
        complexAdd(a.m11, b.m11)
    );
}

[Differentiable]
JonesOperator jonesOperatorScale(JonesOperator value, Complex coeff)
{
    return JonesOperator(
        complexMul(value.m00, coeff),
        complexMul(value.m01, coeff),
        complexMul(value.m10, coeff),
        complexMul(value.m11, coeff)
    );
}

[Differentiable]
Jones2 applyJonesOperator(Jones2 value, JonesOperator op)
{
    return Jones2(
        complexAdd(complexMul(op.m00, value.u), complexMul(op.m01, value.v)),
        complexAdd(complexMul(op.m10, value.u), complexMul(op.m11, value.v))
    );
}

[Differentiable]
Complex complexDotReal(Complex3 value, float3 basis)
{
    Complex sum = complexZero();
    sum = complexAdd(sum, complexMulReal(value.x, basis.x));
    sum = complexAdd(sum, complexMulReal(value.y, basis.y));
    sum = complexAdd(sum, complexMulReal(value.z, basis.z));
    return sum;
}

[Differentiable]
Complex3 complexScaleReal(float3 basis, Complex coeff)
{
    return Complex3(
        complexMulReal(coeff, basis.x),
        complexMulReal(coeff, basis.y),
        complexMulReal(coeff, basis.z)
    );
}

[Differentiable]
Complex3 vectorFromJones(Jones2 value, Basis3 basis)
{
    return complex3Add(
        complexScaleReal(basis.u, value.u),
        complexScaleReal(basis.v, value.v)
    );
}

[Differentiable]
Jones2 jonesFromVector(Complex3 value, Basis3 basis)
{
    return Jones2(
        complexDotReal(value, basis.u),
        complexDotReal(value, basis.v)
    );
}

void adjComplexDotReal(
    Complex3 value,
    float3 basis,
    Complex gradOut,
    inout Complex3.Differential gradValue,
    inout float3 gradBasis
)
{
    gradValue.x.re += gradOut.re * basis.x;
    gradValue.x.im += gradOut.im * basis.x;
    gradValue.y.re += gradOut.re * basis.y;
    gradValue.y.im += gradOut.im * basis.y;
    gradValue.z.re += gradOut.re * basis.z;
    gradValue.z.im += gradOut.im * basis.z;
    gradBasis.x += complexAdjointDot(gradOut, value.x);
    gradBasis.y += complexAdjointDot(gradOut, value.y);
    gradBasis.z += complexAdjointDot(gradOut, value.z);
}

void adjComplexScaleReal(
    float3 basis,
    Complex coeff,
    Complex3.Differential gradOut,
    inout float3 gradBasis,
    inout Complex gradCoeff
)
{
    gradBasis.x += complexAdjointDot(Complex(gradOut.x.re, gradOut.x.im), coeff);
    gradBasis.y += complexAdjointDot(Complex(gradOut.y.re, gradOut.y.im), coeff);
    gradBasis.z += complexAdjointDot(Complex(gradOut.z.re, gradOut.z.im), coeff);
    gradCoeff.re +=
        gradOut.x.re * basis.x +
        gradOut.y.re * basis.y +
        gradOut.z.re * basis.z;
    gradCoeff.im +=
        gradOut.x.im * basis.x +
        gradOut.y.im * basis.y +
        gradOut.z.im * basis.z;
}

void adjComplex3Scale(
    Complex3 value,
    Complex coeff,
    Complex3.Differential gradOut,
    inout Complex3.Differential gradValue,
    inout Complex gradCoeff
)
{
    Complex gradValueX = complexZero();
    Complex gradCoeffX = complexZero();
    adjComplexMul(value.x, coeff, Complex(gradOut.x.re, gradOut.x.im), gradValueX, gradCoeffX);
    gradValue.x.re += gradValueX.re;
    gradValue.x.im += gradValueX.im;
    gradCoeff = complexAdd(gradCoeff, gradCoeffX);

    Complex gradValueY = complexZero();
    Complex gradCoeffY = complexZero();
    adjComplexMul(value.y, coeff, Complex(gradOut.y.re, gradOut.y.im), gradValueY, gradCoeffY);
    gradValue.y.re += gradValueY.re;
    gradValue.y.im += gradValueY.im;
    gradCoeff = complexAdd(gradCoeff, gradCoeffY);

    Complex gradValueZ = complexZero();
    Complex gradCoeffZ = complexZero();
    adjComplexMul(value.z, coeff, Complex(gradOut.z.re, gradOut.z.im), gradValueZ, gradCoeffZ);
    gradValue.z.re += gradValueZ.re;
    gradValue.z.im += gradValueZ.im;
    gradCoeff = complexAdd(gradCoeff, gradCoeffZ);
}

bool complex3DifferentialAnyNonzero(Complex3.Differential value)
{
    return
        complexAnyNonzero(Complex(value.x.re, value.x.im)) ||
        complexAnyNonzero(Complex(value.y.re, value.y.im)) ||
        complexAnyNonzero(Complex(value.z.re, value.z.im));
}

[Differentiable]
float safeLength(float3 value)
{
    return sqrt(max(dot(value, value), 0.0f));
}

[Differentiable]
float3 safeNormalize(float3 value, float3 fallback)
{
    float valueNorm = safeLength(value);
    if (valueNorm > SMALL_EPS)
    {
        return value / (valueNorm + EPS);
    }
    float fallbackNorm = safeLength(fallback);
    return fallback / (fallbackNorm + EPS);
}

[Differentiable]
float3 stablePerpendicularBasis(float3 rayDir, float3 preferred)
{
    float3 proj = preferred - dot(preferred, rayDir) * rayDir;
    float3 altAxis = abs(rayDir.z) < 0.9f ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);
    float3 altProj = altAxis - dot(altAxis, rayDir) * rayDir;
    return safeNormalize(proj, altProj);
}

[Differentiable]
Basis3 basisFromFirstVector(float3 rayDir, float3 firstVector, float3 fallback)
{
    float3 rayHat = safeNormalize(rayDir, float3(0.0f, 0.0f, 1.0f));
    float3 uVector = firstVector - dot(firstVector, rayHat) * rayHat;
    float3 uHat = safeNormalize(uVector, fallback);
    float3 vFallback = stablePerpendicularBasis(rayHat, float3(0.0f, 1.0f, 0.0f));
    float3 vHat = safeNormalize(cross(rayHat, uHat), vFallback);
    return Basis3(uHat, vHat, rayHat);
}

[Differentiable]
Basis3 diffractionEdgeBasis(float3 rayDir, float3 edgeDir, no_diff bool outgoing)
{
    float3 rayHat = safeNormalize(rayDir, float3(0.0f, 0.0f, 1.0f));
    float3 edgeHat = safeNormalize(edgeDir, float3(0.0f, 0.0f, 1.0f));
    float3 phiHat = cross(rayHat, edgeHat);
    if (outgoing)
    {
        phiHat = -phiHat;
    }
    float3 fallback = stablePerpendicularBasis(rayHat, edgeHat);
    return basisFromFirstVector(rayHat, phiHat, fallback);
}

[Differentiable]
JonesOperator jonesOperatorInBasis(
    JonesOperator op,
    Basis3 srcInBasis,
    Basis3 srcOutBasis,
    Basis3 dstInBasis,
    Basis3 dstOutBasis
)
{
    Jones2 unitU = Jones2(Complex(1.0f, 0.0f), complexZero());
    Jones2 unitV = Jones2(complexZero(), Complex(1.0f, 0.0f));

    Complex3 fieldU = vectorFromJones(unitU, dstInBasis);
    Jones2 srcU = jonesFromVector(fieldU, srcInBasis);
    Jones2 srcOutU = applyJonesOperator(srcU, op);
    Jones2 mappedU = jonesFromVector(vectorFromJones(srcOutU, srcOutBasis), dstOutBasis);

    Complex3 fieldV = vectorFromJones(unitV, dstInBasis);
    Jones2 srcV = jonesFromVector(fieldV, srcInBasis);
    Jones2 srcOutV = applyJonesOperator(srcV, op);
    Jones2 mappedV = jonesFromVector(vectorFromJones(srcOutV, srcOutBasis), dstOutBasis);

    return JonesOperator(mappedU.u, mappedV.u, mappedU.v, mappedV.v);
}

[Differentiable]
float3 projectToWedgePlane(float3 value, float3 edgeDir)
{
    return value - dot(value, edgeDir) * edgeDir;
}

[Differentiable]
float3 normalizeInWedgePlane(float3 value, float3 edgeDir)
{
    return safeNormalize(projectToWedgePlane(value, edgeDir), float3(1.0f, 0.0f, 0.0f));
}

[Differentiable]
float clip1(float value)
{
    return clamp(value, -1.0f, 1.0f);
}

[Differentiable]
float safeAcos(float value)
{
    float clipped = clip1(value);
    float sinTheta = sqrt(max(1.0f - clipped * clipped, 0.0f) + SMALL_EPS);
    return atan2(sinTheta, clipped);
}

[Differentiable]
float cot(float value)
{
    float s, c;
    sincos(value, s, c);
    float denom = abs(s) < SMALL_EPS ? (s >= 0.0f ? SMALL_EPS : -SMALL_EPS) : s;
    float result = c / denom;
    return isfinite(result) ? result : 0.0f;
}

float distanceToCotPole(float value)
{
    float nearestPole = round(value / PI) * PI;
    return abs(value - nearestPole);
}

bool cotangentPoleSafeMask(float phi, float phiPrime, float wedgeN, float poleGuard)
{
    float twoN = 2.0f * wedgeN;
    float difPhi = phi - phiPrime;
    float sumPhi = phi + phiPrime;
    float args[4] = {
        (PI + difPhi) / twoN,
        (PI - difPhi) / twoN,
        (PI + sumPhi) / twoN,
        (PI - sumPhi) / twoN
    };
    for (int i = 0; i < 4; ++i)
    {
        if (distanceToCotPole(args[i]) <= poleGuard)
        {
            return false;
        }
    }
    return true;
}

bool slopeDerivativeSafeMask(float phi, float phiPrime, float wedgeN, float step)
{
    float nPi = wedgeN * PI;
    bool interior =
        (phi >= step) && (phi <= (nPi - step)) &&
        (phiPrime >= step) && (phiPrime <= (nPi - step));
    float poleGuard = step / (2.0f * wedgeN);
    return interior && cotangentPoleSafeMask(phi, phiPrime, wedgeN, poleGuard);
}

bool wedgeExteriorRegionMask(float3 directionFromEdge, float3 edgeDir, float3 n0, float3 nn)
{
    float3 directionProj = projectToWedgePlane(directionFromEdge, edgeDir);
    float signedDistance0 = dot(directionProj, n0);
    float signedDistanceN = dot(directionProj, nn);
    return
        (safeLength(directionProj) > SMALL_EPS) &&
        ((signedDistance0 >= -SMALL_EPS) || (signedDistanceN >= -SMALL_EPS));
}

[Differentiable]
Complex complexSqrt(Complex z)
{
    float x = z.re;
    float y = z.im;
    float r = sqrt(x * x + y * y);
    bool nonzero = r > 0.0f;
    bool xNonnegative = x >= 0.0f;
    float realMag = sqrt((xNonnegative && nonzero) ? (0.5f * (r + x)) : 0.0f);
    float imagMag = sqrt(((!xNonnegative) && nonzero) ? (0.5f * (r - x)) : 0.0f);
    float safeRealMag = realMag > 0.0f ? realMag : 1.0f;
    float safeImagMag = imagMag > 0.0f ? imagMag : 1.0f;
    float realPart = xNonnegative ? realMag : abs(y) / (2.0f * safeImagMag);
    float imagPart = xNonnegative ? (y / (2.0f * safeRealMag)) : (y < 0.0f ? -imagMag : imagMag);
    return Complex(nonzero ? realPart : 0.0f, nonzero ? imagPart : 0.0f);
}

[Differentiable]
Complex complexRelativePermittivity(no_diff float etaR, no_diff float sigma, no_diff float omega)
{
    return Complex(etaR, -sigma / (omega * EPSILON_0));
}

[Differentiable]
void fresnelReflection(float cosTheta, no_diff MaterialParams material, out Complex rTE, out Complex rTM)
{
    float clippedCosTheta = clamp(cosTheta, SMALL_EPS, 1.0f);
    float sinThetaSqr = 1.0f - clippedCosTheta * clippedCosTheta;
    Complex eta = complexRelativePermittivity(material.etaR, material.sigma, material.omega);
    Complex a = complexSqrt(complexSub(eta, Complex(sinThetaSqr, 0.0f)));
    rTE = complexDiv(
        complexSub(Complex(clippedCosTheta, 0.0f), a),
        complexAdd(Complex(clippedCosTheta, 0.0f), a)
    );
    rTM = complexDiv(
        complexSub(complexMulReal(eta, clippedCosTheta), a),
        complexAdd(complexMulReal(eta, clippedCosTheta), a)
    );
}

[Differentiable]
Complex scalarFresnelReflection(float cosTheta, no_diff MaterialParams material)
{
    Complex rTE, rTM;
    fresnelReflection(cosTheta, material, rTE, rTM);
    Complex coeff = complexMulReal(complexAdd(rTE, rTM), 0.5f * material.gain);
    return Complex(
        isfinite(coeff.re) ? coeff.re : 0.0f,
        isfinite(coeff.im) ? coeff.im : 0.0f
    );
}

[Differentiable]
void computeFaceReflectionCoefficients(PairInputs state, no_diff MaterialParams material, out Complex r0, out Complex rn)
{
    r0 = state.r0;
    rn = state.rn;
}

[Differentiable]
FaceReflectionOutputs computeFaceReflectionOutputs(PairInputs state, no_diff MaterialParams material)
{
    FaceReflectionOutputs outputs;
    computeFaceReflectionCoefficients(state, material, outputs.r0, outputs.rn);
    return outputs;
}
