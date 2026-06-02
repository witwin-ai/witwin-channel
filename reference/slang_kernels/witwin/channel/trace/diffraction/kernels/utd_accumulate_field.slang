implementing utd_accumulate;

[Differentiable]
void computeEdgeGeometry3D(
    float3 sourcePos,
    float3 edgePos,
    float3 edgeDir,
    float3 n0,
    float3 targetPos,
    out float phi,
    out float phiPrime,
    out float s,
    out float sPrime,
    out float sinBeta0
)
{
    float sProj, sPrimeProj;
    computeEdgeAngles(sourcePos, edgePos, edgeDir, n0, targetPos, phi, phiPrime, sProj, sPrimeProj);

    float3 sourceToEdge = edgePos - sourcePos;
    float3 edgeToTarget = targetPos - edgePos;
    sPrime = safeLength(sourceToEdge) + EPS;
    s = safeLength(edgeToTarget) + EPS;

    float sinBetaPrime = clamp(sPrimeProj / sPrime, SMALL_EPS, 1.0f);
    float sinBeta = clamp(sProj / s, SMALL_EPS, 1.0f);
    sinBeta0 = sqrt(max(sinBeta * sinBetaPrime, SMALL_EPS));
}

[Differentiable]
bool stateHasFaceMaterialParams(PairInputs state)
{
    return no_diff ((state.face0Material.present > 0.5f) || (state.face1Material.present > 0.5f));
}

[Differentiable]
void fresnelReflectionFace(
    float cosTheta,
    float etaR,
    float sigma,
    no_diff float omega,
    out Complex rTE,
    out Complex rTM
)
{
    float clippedCosTheta = clamp(cosTheta, SMALL_EPS, 1.0f);
    float sinThetaSqr = 1.0f - clippedCosTheta * clippedCosTheta;
    float safeOmega = max(omega, SMALL_EPS);
    Complex eta = Complex(etaR, -sigma / (safeOmega * EPSILON_0));
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
JonesOperator faceReflectionOperatorFromMaterial(
    FaceMaterialParams faceMaterial,
    float cosTheta,
    float3 normal,
    float3 incomingHat,
    float3 outgoingHat,
    Basis3 incomingEdgeBasis,
    Basis3 outgoingEdgeBasis,
    no_diff float omega
)
{
    Complex gain = Complex(faceMaterial.gain, 0.0f);
    no_diff bool useFresnel = faceMaterial.useFresnel > 0.5f;

    Complex rTE;
    Complex rTM;
    fresnelReflectionFace(cosTheta, faceMaterial.etaR, faceMaterial.sigma, omega, rTE, rTM);
    JonesOperator diagOperator = useFresnel
        ? JonesOperator(complexMul(gain, rTE), complexZero(), complexZero(), complexMul(gain, rTM))
        : JonesOperator(Complex(-faceMaterial.gain, 0.0f), complexZero(), complexZero(), Complex(-faceMaterial.gain, 0.0f));

    float3 faceS = cross(incomingHat, normal);
    Basis3 faceInBasis = basisFromFirstVector(
        incomingHat,
        faceS,
        stablePerpendicularBasis(incomingHat, float3(0.0f, 1.0f, 0.0f))
    );
    Basis3 faceOutBasis = basisFromFirstVector(
        outgoingHat,
        faceS,
        stablePerpendicularBasis(outgoingHat, float3(0.0f, 1.0f, 0.0f))
    );
    return jonesOperatorInBasis(
        diagOperator,
        faceInBasis,
        faceOutBasis,
        incomingEdgeBasis,
        outgoingEdgeBasis
    );
}

[Differentiable]
JonesOperator fallbackFaceOperator(
    JonesOperator storedOperator,
    float3 normal,
    float3 incomingHat,
    float3 outgoingHat,
    Basis3 incomingEdgeBasis,
    Basis3 outgoingEdgeBasis
)
{
    float3 faceS = cross(incomingHat, normal);
    Basis3 faceInBasis = basisFromFirstVector(
        incomingHat,
        faceS,
        stablePerpendicularBasis(incomingHat, float3(0.0f, 1.0f, 0.0f))
    );
    Basis3 faceOutBasis = basisFromFirstVector(
        outgoingHat,
        faceS,
        stablePerpendicularBasis(outgoingHat, float3(0.0f, 1.0f, 0.0f))
    );
    return jonesOperatorInBasis(
        storedOperator,
        faceInBasis,
        faceOutBasis,
        incomingEdgeBasis,
        outgoingEdgeBasis
    );
}

[Differentiable]
void computePairFieldTerms(
    PairInputs state,
    float3 targetPos,
    no_diff float k,
    no_diff MaterialParams material,
    out bool geometryValid,
    out Complex field,
    out Complex directGain,
    out Complex derivativeGain
)
{
    geometryValid = false;
    field = complexZero();
    directGain = complexZero();
    derivativeGain = complexZero();

    bool sourceExterior = no_diff wedgeExteriorRegionMask(state.sourcePos - state.edgePos, state.edgeDir, state.n0, state.nn);
    bool targetExterior = no_diff wedgeExteriorRegionMask(targetPos - state.edgePos, state.edgeDir, state.n0, state.nn);

    float phi, phiPrime, s, sPrime, sinBeta0;
    computeEdgeGeometry3D(state.sourcePos, state.edgePos, state.edgeDir, state.n0, targetPos, phi, phiPrime, s, sPrime, sinBeta0);

    geometryValid =
        sourceExterior &&
        targetExterior &&
        (sPrime > DIFFRACTION_MIN_DISTANCE) &&
        (s > DIFFRACTION_MIN_DISTANCE);
    if (!geometryValid)
    {
        return;
    }

    Complex r0, rn;
    computeFaceReflectionCoefficients(state, material, r0, rn);

    float wedge = state.wedgeN;
    bool poleSafe = no_diff cotangentPoleSafeMask(phi, phiPrime, wedge, 1.0e-6f);
    float safePhi = poleSafe ? phi : (0.5f * wedge * PI);
    float safePhiPrime = poleSafe ? phiPrime : (0.5f * wedge * PI);
    bool slopeSafe = no_diff slopeDerivativeSafeMask(safePhi, safePhiPrime, wedge, SLOPE_DERIVATIVE_STEP);

    no_diff bool useFaceMaterialParams = stateHasFaceMaterialParams(state);
    Complex dForward = useFaceMaterialParams
        ? diffractionCoefficient3D(phi, phiPrime, wedge, k, s, sPrime, sinBeta0, r0, rn)
        : diffractionCoefficient2D(phi, phiPrime, wedge, k, s, sPrime, r0, rn);
    Complex d = poleSafe ? dForward : detach(dForward);
    Complex dSlope = complexZero();
    bool hasSlope = (complexAbsSqr(state.incidentNormalDerivative) > 1.0e-24f) && slopeSafe;
    if (hasSlope)
    {
        dSlope = useFaceMaterialParams
            ? slopeDiffractionCoefficient3D(safePhi, safePhiPrime, wedge, k, s, sPrime, sinBeta0, r0, rn)
            : slopeDiffractionCoefficient2D(safePhi, safePhiPrime, wedge, k, s, sPrime, r0, rn);
    }

    float localScale = sqrt(sPrime / (s * (s + sPrime) + EPS));
    Complex phase = complexExpPhase(-k * s);
    directGain = complexMulReal(complexMul(d, phase), localScale);
    derivativeGain = complexMulReal(complexMul(dSlope, phase), localScale);
    field = complexAdd(
        complexMul(state.incidentField, directGain),
        complexMul(state.incidentNormalDerivative, derivativeGain)
    );
}

[Differentiable]
Complex computePairFieldContribution(PairInputs state, float3 targetPos, no_diff float k, no_diff MaterialParams material)
{
    bool geometryValid;
    Complex field;
    Complex directGain;
    Complex derivativeGain;
    computePairFieldTerms(state, targetPos, k, material, geometryValid, field, directGain, derivativeGain);
    return geometryValid ? field : complexZero();
}

[Differentiable]
Complex computePairFieldContributionAssumeSafe(
    PairInputs state,
    float3 targetPos,
    no_diff float k,
    no_diff MaterialParams material
)
{
    float phi, phiPrime, s, sPrime, sinBeta0;
    computeEdgeGeometry3D(state.sourcePos, state.edgePos, state.edgeDir, state.n0, targetPos, phi, phiPrime, s, sPrime, sinBeta0);

    Complex r0, rn;
    computeFaceReflectionCoefficients(state, material, r0, rn);

    float wedge = state.wedgeN;
    no_diff bool useFaceMaterialParams = stateHasFaceMaterialParams(state);
    Complex d = useFaceMaterialParams
        ? diffractionCoefficient3D(phi, phiPrime, wedge, k, s, sPrime, sinBeta0, r0, rn)
        : diffractionCoefficient2D(phi, phiPrime, wedge, k, s, sPrime, r0, rn);
    Complex dSlope = complexZero();
    bool hasSlope = complexAbsSqr(state.incidentNormalDerivative) > 1.0e-24f;
    if (hasSlope)
    {
        dSlope = useFaceMaterialParams
            ? slopeDiffractionCoefficient3D(phi, phiPrime, wedge, k, s, sPrime, sinBeta0, r0, rn)
            : slopeDiffractionCoefficient2D(phi, phiPrime, wedge, k, s, sPrime, r0, rn);
    }

    float localScale = sqrt(sPrime / (s * (s + sPrime) + EPS));
    Complex phase = complexExpPhase(-k * s);
    return complexAdd(
        complexMul(state.incidentField, complexMulReal(complexMul(d, phase), localScale)),
        complexMul(state.incidentNormalDerivative, complexMulReal(complexMul(dSlope, phase), localScale))
    );
}

struct DiffractionOperatorTerms : IDifferentiable
{
    Complex direct;
    Complex face0;
    Complex face1;
    Complex directDphiPrime;
    Complex face0DphiPrime;
    Complex face1DphiPrime;
};

[Differentiable]
DiffractionOperatorTerms computeDiffractionOperatorTerms2D(
    float phi,
    float phiPrime,
    float wedgeN,
    no_diff float k,
    float s,
    float sPrime
)
{
    float l = s * sPrime / (s + sPrime + EPS);
    float kL = k * l;
    BetaTermCache difPlus;
    BetaTermCache difMinus;
    BetaTermCache sumPlus;
    BetaTermCache sumMinus;
    buildBetaTermCache(phi - phiPrime, wedgeN, kL, +1.0f, true, difPlus);
    buildBetaTermCache(phi - phiPrime, wedgeN, kL, -1.0f, false, difMinus);
    buildBetaTermCache(phi + phiPrime, wedgeN, kL, +1.0f, true, sumPlus);
    buildBetaTermCache(phi + phiPrime, wedgeN, kL, -1.0f, false, sumMinus);

    Complex factor = complexMulReal(complexExpPhase(-0.25f * PI), -1.0f / (2.0f * wedgeN * sqrt(TWO_PI * k + EPS)));
    Complex difValue = complexAdd(difPlus.value, difMinus.value);
    Complex difFirst = complexAdd(difPlus.first, difMinus.first);
    return DiffractionOperatorTerms(
        complexMul(factor, difValue),
        complexMul(factor, sumPlus.value),
        complexMul(factor, sumMinus.value),
        complexMul(factor, complexMulReal(difFirst, -1.0f)),
        complexMul(factor, sumPlus.first),
        complexMul(factor, sumMinus.first)
    );
}

[Differentiable]
DiffractionOperatorTerms computeDiffractionOperatorTerms3D(
    float phi,
    float phiPrime,
    float wedgeN,
    no_diff float k,
    float s,
    float sPrime,
    float sinBeta0
)
{
    Complex zero = complexZero();
    Complex one = Complex(1.0f, 0.0f);
    Complex factor;
    Complex difGroup;
    Complex difGroup1;
    Complex sumGroup;
    Complex sumGroup1;
    Complex difGroup2;
    Complex sumGroup2;
    diffractionBetaGroups3D(
        phi, phiPrime, wedgeN, k, s, sPrime, sinBeta0, zero, zero,
        factor, difGroup, difGroup1, sumGroup, sumGroup1, difGroup2, sumGroup2
    );

    Complex factorFace0;
    Complex difFace0;
    Complex difFace01;
    Complex sumFace0;
    Complex sumFace01;
    Complex difFace02;
    Complex sumFace02;
    diffractionBetaGroups3D(
        phi, phiPrime, wedgeN, k, s, sPrime, sinBeta0, one, zero,
        factorFace0, difFace0, difFace01, sumFace0, sumFace01, difFace02, sumFace02
    );

    Complex factorFace1;
    Complex difFace1;
    Complex difFace11;
    Complex sumFace1;
    Complex sumFace11;
    Complex difFace12;
    Complex sumFace12;
    diffractionBetaGroups3D(
        phi, phiPrime, wedgeN, k, s, sPrime, sinBeta0, zero, one,
        factorFace1, difFace1, difFace11, sumFace1, sumFace11, difFace12, sumFace12
    );

    return DiffractionOperatorTerms(
        complexMul(factor, difGroup),
        complexMul(factorFace0, sumFace0),
        complexMul(factorFace1, sumFace1),
        complexMul(factor, complexMulReal(difGroup1, -1.0f)),
        complexMul(factorFace0, sumFace01),
        complexMul(factorFace1, sumFace11)
    );
}

[Differentiable]
JonesOperator assembleDiffractionOperator(
    Complex freeTerm,
    Complex face0Term,
    Complex face1Term,
    JonesOperator face0Operator,
    JonesOperator face1Operator
)
{
    JonesOperator total = jonesOperatorScale(jonesOperatorIdentity(), freeTerm);
    total = jonesOperatorAdd(total, jonesOperatorScale(face0Operator, face0Term));
    total = jonesOperatorAdd(total, jonesOperatorScale(face1Operator, face1Term));
    return total;
}

[Differentiable]
Complex3 computePairVectorContribution(PairInputs state, float3 targetPos, no_diff float k, no_diff MaterialParams material)
{
    bool sourceExterior = no_diff wedgeExteriorRegionMask(state.sourcePos - state.edgePos, state.edgeDir, state.n0, state.nn);
    bool targetExterior = no_diff wedgeExteriorRegionMask(targetPos - state.edgePos, state.edgeDir, state.n0, state.nn);

    float phi, phiPrime, s, sPrime, sinBeta0;
    computeEdgeGeometry3D(state.sourcePos, state.edgePos, state.edgeDir, state.n0, targetPos, phi, phiPrime, s, sPrime, sinBeta0);

    bool geometryValid =
        sourceExterior &&
        targetExterior &&
        (sPrime > DIFFRACTION_MIN_DISTANCE) &&
        (s > DIFFRACTION_MIN_DISTANCE);
    if (!geometryValid)
    {
        return complex3Zero();
    }

    Basis3 incomingEdgeBasis = diffractionEdgeBasis(state.edgePos - state.sourcePos, state.edgeDir, false);
    Basis3 outgoingEdgeBasis = diffractionEdgeBasis(targetPos - state.edgePos, state.edgeDir, true);

    Jones2 incidentJonesEdge = jonesFromVector(vectorFromJones(state.incidentJones, state.incidentBasis), incomingEdgeBasis);
    Jones2 incidentDerivativeJonesEdge = jonesFromVector(
        vectorFromJones(state.incidentDerivativeJones, state.incidentBasis),
        incomingEdgeBasis
    );

    no_diff bool useFaceMaterialParams = stateHasFaceMaterialParams(state);
    no_diff bool face0HasMaterial = state.face0Material.present > 0.5f;
    no_diff bool face1HasMaterial = state.face1Material.present > 0.5f;

    JonesOperator face0Operator = face0HasMaterial
        ? faceReflectionOperatorFromMaterial(
            state.face0Material,
            clamp(abs(sin(phiPrime)), 1.0e-6f, 1.0f),
            state.n0,
            incomingEdgeBasis.k,
            outgoingEdgeBasis.k,
            incomingEdgeBasis,
            outgoingEdgeBasis,
            material.omega
        )
        : fallbackFaceOperator(
            state.face0Operator,
            state.n0,
            incomingEdgeBasis.k,
            outgoingEdgeBasis.k,
            incomingEdgeBasis,
            outgoingEdgeBasis
        );
    JonesOperator face1Operator = face1HasMaterial
        ? faceReflectionOperatorFromMaterial(
            state.face1Material,
            clamp(abs(sin(state.wedgeN * PI - phi)), 1.0e-6f, 1.0f),
            state.nn,
            incomingEdgeBasis.k,
            outgoingEdgeBasis.k,
            incomingEdgeBasis,
            outgoingEdgeBasis,
            material.omega
        )
        : fallbackFaceOperator(
            state.face1Operator,
            state.nn,
            incomingEdgeBasis.k,
            outgoingEdgeBasis.k,
            incomingEdgeBasis,
            outgoingEdgeBasis
        );

    DiffractionOperatorTerms terms = useFaceMaterialParams
        ? computeDiffractionOperatorTerms3D(phi, phiPrime, state.wedgeN, k, s, sPrime, sinBeta0)
        : computeDiffractionOperatorTerms2D(phi, phiPrime, state.wedgeN, k, s, sPrime);
    JonesOperator directOperator = assembleDiffractionOperator(
        terms.direct,
        terms.face0,
        terms.face1,
        face0Operator,
        face1Operator
    );
    Complex slopeFactor = Complex(0.0f, -1.0f / k);
    JonesOperator slopeOperator = assembleDiffractionOperator(
        complexMul(slopeFactor, terms.directDphiPrime),
        complexMul(slopeFactor, terms.face0DphiPrime),
        complexMul(slopeFactor, terms.face1DphiPrime),
        face0Operator,
        face1Operator
    );

    Jones2 fieldJones = jonesAdd(
        applyJonesOperator(incidentJonesEdge, directOperator),
        applyJonesOperator(incidentDerivativeJonesEdge, slopeOperator)
    );
    float localScale = sqrt(sPrime / (s * (s + sPrime) + EPS));
    Complex scale = complexMulReal(complexExpPhase(-k * s), localScale);
    return complex3Scale(vectorFromJones(fieldJones, outgoingEdgeBasis), scale);
}

[Differentiable]
Complex computeScalarFieldContributionSafe(PairScalarInputs state, no_diff float k)
{
    Complex d = diffractionCoefficient2D(state.phi, state.phiPrime, state.wedgeN, k, state.s, state.sPrime, state.r0, state.rn);
    Complex dSlope = complexZero();
    bool hasSlope = complexAbsSqr(state.incidentNormalDerivative) > 1.0e-24f;
    if (hasSlope)
    {
        dSlope = slopeDiffractionCoefficient2D(
            state.phi, state.phiPrime, state.wedgeN, k, state.s, state.sPrime, state.r0, state.rn
        );
    }

    float localScale = sqrt(state.sPrime / (state.s * (state.s + state.sPrime) + EPS));
    Complex phase = complexExpPhase(-k * state.s);
    return complexAdd(
        complexMul(state.incidentField, complexMulReal(complexMul(d, phase), localScale)),
        complexMul(state.incidentNormalDerivative, complexMulReal(complexMul(dSlope, phase), localScale))
    );
}

void buildBetaTermCache(
    float beta,
    float n,
    float kL,
    float cotSign,
    bool plusBranch,
    out BetaTermCache cache
)
{
    float twoN = 2.0f * n;
    float twoNPi = 2.0f * n * PI;
    float roundIndex = plusBranch
        ? round((beta + PI) / twoNPi)
        : round((beta - PI) / twoNPi);
    float phaseOffset = twoNPi * roundIndex - beta;
    float cosHalfOffset = cos(0.5f * phaseOffset);

    cache.n = n;
    cache.kL = kL;
    cache.cotSign = cotSign;
    cache.a = 2.0f * cosHalfOffset * cosHalfOffset;
    cache.a1 = sin(phaseOffset);
    cache.a2 = 1.0f - cache.a;
    float phaseOffsetN = 2.0f * PI * roundIndex;
    cache.aN = -cache.a1 * phaseOffsetN;
    cache.a1N = (cache.a - 1.0f) * phaseOffsetN;

    cache.cotArg = (PI + cotSign * beta) / twoN;
    cache.cotValue = cot(cache.cotArg);
    cache.cot1 = -(cotSign / twoN) * (1.0f + cache.cotValue * cache.cotValue);
    cache.cot2 = 0.5f * cache.cotValue * (1.0f + cache.cotValue * cache.cotValue) / (n * n);

    float x = kL * cache.a;
    float x1 = kL * cache.a1;
    fUTDWithDerivatives(x, cache.transition, cache.transition1, cache.transition2);
    assembleBetaTermState(
        cache.cotValue,
        cache.cot1,
        cache.cot2,
        x1,
        kL * cache.a2,
        cache.transition,
        cache.transition1,
        cache.transition2,
        cache.value,
        cache.first,
        cache.second
    );
}

void adjointBetaTermSafe(
    BetaTermCache cache,
    float k,
    Complex gradValue,
    Complex gradFirst,
    inout float gradBeta,
    inout float gradN,
    inout float gradL
)
{
    Complex gradTransition = complexZero();
    Complex gradTransition1 = complexZero();
    float gradCot = 0.0f;
    float gradCot1 = 0.0f;
    float gradScale = 0.0f;
    float gradX1 = 0.0f;
    float gradX = 0.0f;

    adjComplexMulReal(cache.transition, cache.cotValue, gradValue, gradTransition, gradCot);
    adjComplexMulReal(cache.transition, cache.cot1, gradFirst, gradTransition, gradCot1);
    float firstScale = cache.cotValue * cache.kL * cache.a1;
    adjComplexMulReal(cache.transition1, firstScale, gradFirst, gradTransition1, gradScale);
    gradCot += gradScale * (cache.kL * cache.a1);
    gradX1 += gradScale * cache.cotValue;

    gradX += complexAdjointDot(gradTransition, cache.transition1);
    gradX += complexAdjointDot(gradTransition1, cache.transition2);

    float cotH = 1.0f + cache.cotValue * cache.cotValue;
    float uBeta = cache.cotSign / (2.0f * cache.n);
    float uN = -cache.cotArg / cache.n;
    float cotN = -cotH * uN;
    float cot1N = (uBeta / cache.n) * cotH - 2.0f * uBeta * cache.cotValue * cotN;

    gradBeta += gradCot * cache.cot1;
    gradBeta += gradCot1 * cache.cot2;
    gradBeta += gradX * (cache.kL * cache.a1);
    gradBeta += gradX1 * (cache.kL * cache.a2);

    gradN += gradCot * cotN;
    gradN += gradCot1 * cot1N;
    gradN += gradX * (cache.kL * cache.aN);
    gradN += gradX1 * (cache.kL * cache.a1N);

    float gradKL = gradX * cache.a + gradX1 * cache.a1;
    gradL += gradKL * k;
}

void computeScalarFieldAdjointSafe(
    PairScalarInputs state,
    no_diff float k,
    Complex fieldGrad,
    out PairScalarInputs.Differential stateGrad
)
{
    stateGrad = PairScalarInputs.Differential(
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f)
    );

    float L = state.s * state.sPrime / (state.s + state.sPrime + EPS);
    float kL = k * L;
    BetaTermCache difPlus;
    BetaTermCache difMinus;
    BetaTermCache sumPlus;
    BetaTermCache sumMinus;
    buildBetaTermCache(state.phi - state.phiPrime, state.wedgeN, kL, +1.0f, true, difPlus);
    buildBetaTermCache(state.phi - state.phiPrime, state.wedgeN, kL, -1.0f, false, difMinus);
    buildBetaTermCache(state.phi + state.phiPrime, state.wedgeN, kL, +1.0f, true, sumPlus);
    buildBetaTermCache(state.phi + state.phiPrime, state.wedgeN, kL, -1.0f, false, sumMinus);

    Complex factor = complexMulReal(complexExpPhase(-0.25f * PI), -1.0f / (2.0f * state.wedgeN * sqrt(TWO_PI * k + EPS)));
    Complex combinedValue = complexAdd(
        complexAdd(difPlus.value, difMinus.value),
        complexAdd(
            complexMul(state.r0, sumPlus.value),
            complexMul(state.rn, sumMinus.value)
        )
    );
    Complex combinedPhiPrime = complexAdd(
        complexMulReal(complexAdd(difPlus.first, difMinus.first), -1.0f),
        complexAdd(
            complexMul(state.r0, sumPlus.first),
            complexMul(state.rn, sumMinus.first)
        )
    );

    Complex d = complexMul(factor, combinedValue);
    Complex dSlope = complexZero();
    bool hasSlope = complexAbsSqr(state.incidentNormalDerivative) > 1.0e-24f;
    if (hasSlope)
    {
        dSlope = complexMul(Complex(0.0f, -1.0f / k), complexMul(factor, combinedPhiPrime));
    }

    float localScale = sqrt(state.sPrime / (state.s * (state.s + state.sPrime) + EPS));
    Complex phase = complexExpPhase(-k * state.s);
    Complex baseGain = complexMul(d, phase);
    Complex directGain = complexMulReal(baseGain, localScale);
    Complex slopeBaseGain = complexMul(dSlope, phase);
    Complex derivativeGain = complexMulReal(slopeBaseGain, localScale);
    Complex directField = complexMul(state.incidentField, directGain);
    Complex derivativeField = complexMul(state.incidentNormalDerivative, derivativeGain);

    Complex gradIncidentField = complexZero();
    Complex gradIncidentNormalDerivative = complexZero();
    Complex gradDirectGain = complexZero();
    Complex gradDerivativeGain = complexZero();
    adjComplexMul(state.incidentField, directGain, fieldGrad, gradIncidentField, gradDirectGain);
    adjComplexMul(state.incidentNormalDerivative, derivativeGain, fieldGrad, gradIncidentNormalDerivative, gradDerivativeGain);

    Complex gradBaseGain = complexZero();
    Complex gradSlopeBaseGain = complexZero();
    float gradLocalScale = 0.0f;
    adjComplexMulReal(baseGain, localScale, gradDirectGain, gradBaseGain, gradLocalScale);
    adjComplexMulReal(slopeBaseGain, localScale, gradDerivativeGain, gradSlopeBaseGain, gradLocalScale);

    Complex gradD = complexZero();
    Complex gradDSlope = complexZero();
    Complex gradPhase = complexZero();
    adjComplexMul(d, phase, gradBaseGain, gradD, gradPhase);
    adjComplexMul(dSlope, phase, gradSlopeBaseGain, gradDSlope, gradPhase);

    float gradS = 0.0f;
    gradS += k * (gradPhase.re * phase.im - gradPhase.im * phase.re);

    Complex gradFactor = complexZero();
    Complex gradCombinedValue = complexZero();
    adjComplexMul(factor, combinedValue, gradD, gradFactor, gradCombinedValue);

    Complex gradCombinedPhiPrime = complexZero();
    if (hasSlope)
    {
        Complex slopeFactor = Complex(0.0f, -1.0f / k);
        Complex factorPhiPrime = complexMul(factor, combinedPhiPrime);
        Complex gradFactorPhiPrime = complexZero();
        Complex gradSlopeFactor = complexZero();
        adjComplexMul(slopeFactor, factorPhiPrime, gradDSlope, gradSlopeFactor, gradFactorPhiPrime);
        adjComplexMul(factor, combinedPhiPrime, gradFactorPhiPrime, gradFactor, gradCombinedPhiPrime);
    }

    float gradWedgeN = complexAdjointDot(gradFactor, complexMulReal(factor, -1.0f / state.wedgeN));

    Complex gradDifPlusValue = gradCombinedValue;
    Complex gradDifMinusValue = gradCombinedValue;
    Complex gradSumPlusValue = complexZero();
    Complex gradSumMinusValue = complexZero();
    Complex gradR0 = complexZero();
    Complex gradRn = complexZero();
    adjComplexMul(state.r0, sumPlus.value, gradCombinedValue, gradR0, gradSumPlusValue);
    adjComplexMul(state.rn, sumMinus.value, gradCombinedValue, gradRn, gradSumMinusValue);

    Complex gradDifPlusFirst = complexMulReal(gradCombinedPhiPrime, -1.0f);
    Complex gradDifMinusFirst = complexMulReal(gradCombinedPhiPrime, -1.0f);
    Complex gradSumPlusFirst = complexZero();
    Complex gradSumMinusFirst = complexZero();
    Complex gradR0First = complexZero();
    Complex gradRnFirst = complexZero();
    if (hasSlope)
    {
        adjComplexMul(state.r0, sumPlus.first, gradCombinedPhiPrime, gradR0First, gradSumPlusFirst);
        adjComplexMul(state.rn, sumMinus.first, gradCombinedPhiPrime, gradRnFirst, gradSumMinusFirst);
        gradR0 = complexAdd(gradR0, gradR0First);
        gradRn = complexAdd(gradRn, gradRnFirst);
    }

    float gradPhi = 0.0f;
    float gradPhiPrime = 0.0f;
    float gradL = 0.0f;
    float gradNTerms = 0.0f;
    float gradDifBeta = 0.0f;
    adjointBetaTermSafe(difPlus, k, gradDifPlusValue, gradDifPlusFirst, gradDifBeta, gradNTerms, gradL);
    adjointBetaTermSafe(difMinus, k, gradDifMinusValue, gradDifMinusFirst, gradDifBeta, gradNTerms, gradL);
    gradPhi += gradDifBeta;
    gradPhiPrime -= gradDifBeta;

    float gradSumBeta = 0.0f;
    adjointBetaTermSafe(sumPlus, k, gradSumPlusValue, gradSumPlusFirst, gradSumBeta, gradNTerms, gradL);
    adjointBetaTermSafe(sumMinus, k, gradSumMinusValue, gradSumMinusFirst, gradSumBeta, gradNTerms, gradL);
    gradPhi += gradSumBeta;
    gradPhiPrime += gradSumBeta;
    gradWedgeN += gradNTerms;

    float denom = state.s * (state.s + state.sPrime) + EPS;
    float gradRatio = 0.0f;
    if (localScale > 0.0f)
    {
        gradRatio = gradLocalScale * 0.5f / localScale;
    }
    gradS += gradRatio * (-state.sPrime * (2.0f * state.s + state.sPrime) / (denom * denom));
    float gradSPrime = gradRatio * ((state.s * state.s + EPS) / (denom * denom));

    float lDenom = state.s + state.sPrime + EPS;
    gradS += gradL * (state.sPrime * (state.sPrime + EPS) / (lDenom * lDenom));
    gradSPrime += gradL * (state.s * (state.s + EPS) / (lDenom * lDenom));

    stateGrad.phi += gradPhi;
    stateGrad.phiPrime += gradPhiPrime;
    stateGrad.s += gradS;
    stateGrad.sPrime += gradSPrime;
    stateGrad.wedgeN += gradWedgeN;
    stateGrad.incidentField = Complex.Differential(gradIncidentField.re, gradIncidentField.im);
    stateGrad.incidentNormalDerivative = Complex.Differential(
        gradIncidentNormalDerivative.re,
        gradIncidentNormalDerivative.im
    );
    stateGrad.r0 = Complex.Differential(gradR0.re, gradR0.im);
    stateGrad.rn = Complex.Differential(gradRn.re, gradRn.im);
}

void addPairScalarAdjoint(
    inout PairScalarInputs.Differential dst,
    PairScalarInputs.Differential src
)
{
    dst.phi += src.phi;
    dst.phiPrime += src.phiPrime;
    dst.s += src.s;
    dst.sPrime += src.sPrime;
    dst.wedgeN += src.wedgeN;
    dst.incidentField.re += src.incidentField.re;
    dst.incidentField.im += src.incidentField.im;
    dst.incidentNormalDerivative.re += src.incidentNormalDerivative.re;
    dst.incidentNormalDerivative.im += src.incidentNormalDerivative.im;
    dst.r0.re += src.r0.re;
    dst.r0.im += src.r0.im;
    dst.rn.re += src.rn.re;
    dst.rn.im += src.rn.im;
}

void computeGainAdjointSafe(
    float phi,
    float phiPrime,
    float s,
    float sPrime,
    float wedgeN,
    Complex r0,
    Complex rn,
    bool hasSlope,
    no_diff float k,
    Complex gradDirectGain,
    Complex gradDerivativeGain,
    out PairScalarInputs.Differential stateGrad
)
{
    stateGrad = PairScalarInputs.Differential(
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f)
    );

    if (complexAnyNonzero(gradDirectGain))
    {
        PairScalarInputs directState = PairScalarInputs(
            phi,
            phiPrime,
            s,
            sPrime,
            wedgeN,
            Complex(1.0f, 0.0f),
            complexZero(),
            r0,
            rn
        );
        PairScalarInputs.Differential directGrad;
        computeScalarFieldAdjointSafe(directState, k, gradDirectGain, directGrad);
        addPairScalarAdjoint(stateGrad, directGrad);
    }

    if (hasSlope && complexAnyNonzero(gradDerivativeGain))
    {
        PairScalarInputs slopeState = PairScalarInputs(
            phi,
            phiPrime,
            s,
            sPrime,
            wedgeN,
            complexZero(),
            Complex(1.0f, 0.0f),
            r0,
            rn
        );
        PairScalarInputs.Differential slopeGrad;
        computeScalarFieldAdjointSafe(slopeState, k, gradDerivativeGain, slopeGrad);
        addPairScalarAdjoint(stateGrad, slopeGrad);
    }
}

[Differentiable]
PairOutputs computePairContribution(PairInputs state, float3 targetPos, no_diff float k, no_diff MaterialParams material)
{
    PairOutputs result;
    result.field = complexZero();
    result.vectorField = complex3Zero();

    bool geometryValid;
    Complex directGain;
    Complex derivativeGain;
    computePairFieldTerms(state, targetPos, k, material, geometryValid, result.field, directGain, derivativeGain);
    if (!geometryValid)
    {
        return result;
    }
    result.vectorField = computePairVectorContribution(state, targetPos, k, material);
    return result;
}
