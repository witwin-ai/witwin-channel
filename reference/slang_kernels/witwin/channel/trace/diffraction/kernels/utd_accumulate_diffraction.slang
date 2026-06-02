implementing utd_accumulate;

[Differentiable]
void betaTermMetadata(
    float beta,
    float n,
    no_diff float kL,
    no_diff float cotSign,
    no_diff bool plusBranch,
    out float cotValue,
    out float cot1,
    out float cot2,
    out float x,
    out float x1,
    out float x2
)
{
    float twoN = 2.0f * n;
    float twoNPi = 2.0f * n * PI;
    float roundIndex = plusBranch
        ? no_diff round((beta + PI) / twoNPi)
        : no_diff round((beta - PI) / twoNPi);
    float phaseOffset = twoNPi * roundIndex - beta;
    float cosHalfOffset = cos(0.5f * phaseOffset);
    float a = 2.0f * cosHalfOffset * cosHalfOffset;
    float a1 = sin(phaseOffset);
    float a2 = 1.0f - a;

    float cotArg = (PI + cotSign * beta) / twoN;
    cotValue = cot(cotArg);
    cot1 = -(cotSign / twoN) * (1.0f + cotValue * cotValue);
    cot2 = 0.5f * cotValue * (1.0f + cotValue * cotValue) / (n * n);

    x = kL * a;
    x1 = kL * a1;
    x2 = kL * a2;
}

[Differentiable]
void assembleBetaTermState(
    float cotValue,
    float cot1,
    float cot2,
    float x1,
    float x2,
    Complex transition,
    Complex transition1,
    Complex transition2,
    out Complex value,
    out Complex first,
    out Complex second
)
{
    value = complexMulReal(transition, cotValue);
    first = complexAdd(
        complexMulReal(transition, cot1),
        complexMulReal(transition1, cotValue * x1)
    );
    second = complexAdd(
        complexAdd(
            complexMulReal(transition, cot2),
            complexMulReal(transition1, 2.0f * cot1 * x1)
        ),
        complexMulReal(
            complexAdd(
                complexMulReal(transition2, x1 * x1),
                complexMulReal(transition1, x2)
            ),
            cotValue
        )
    );
}

[Differentiable]
void diffractionBetaGroups(
    float phi,
    float phiPrime,
    float n,
    no_diff float k,
    float s,
    float sPrime,
    Complex r0,
    Complex rn,
    out Complex factor,
    out Complex difGroup,
    out Complex difGroup1,
    out Complex sumGroup,
    out Complex sumGroup1,
    out Complex difGroup2,
    out Complex sumGroup2
)
{
    float l = s * sPrime / (s + sPrime + EPS);
    float kL = k * l;
    float difPhi = phi - phiPrime;
    float sumPhi = phi + phiPrime;
    factor = complexMulReal(complexExpPhase(-0.25f * PI), -1.0f / (2.0f * n * sqrt(TWO_PI * k + EPS)));

    float cotValue[4];
    float cot1[4];
    float cot2[4];
    float x[4];
    float x1[4];
    float x2[4];
    betaTermMetadata(difPhi, n, kL, +1.0f, true, cotValue[0], cot1[0], cot2[0], x[0], x1[0], x2[0]);
    betaTermMetadata(difPhi, n, kL, -1.0f, false, cotValue[1], cot1[1], cot2[1], x[1], x1[1], x2[1]);
    betaTermMetadata(sumPhi, n, kL, +1.0f, true, cotValue[2], cot1[2], cot2[2], x[2], x1[2], x2[2]);
    betaTermMetadata(sumPhi, n, kL, -1.0f, false, cotValue[3], cot1[3], cot2[3], x[3], x1[3], x2[3]);

    Complex transition[4];
    Complex transition1[4];
    Complex transition2[4];
    for (int i = 0; i < 4; ++i)
    {
        fUTDWithDerivatives(x[i], transition[i], transition1[i], transition2[i]);
    }

    Complex termValue[4];
    Complex termFirst[4];
    Complex termSecond[4];
    for (int i = 0; i < 4; ++i)
    {
        assembleBetaTermState(
            cotValue[i], cot1[i], cot2[i], x1[i], x2[i],
            transition[i], transition1[i], transition2[i],
            termValue[i], termFirst[i], termSecond[i]
        );
    }

    difGroup = complexAdd(termValue[0], termValue[1]);
    difGroup1 = complexAdd(termFirst[0], termFirst[1]);
    difGroup2 = complexAdd(termSecond[0], termSecond[1]);
    sumGroup = complexAdd(complexMul(r0, termValue[2]), complexMul(rn, termValue[3]));
    sumGroup1 = complexAdd(complexMul(r0, termFirst[2]), complexMul(rn, termFirst[3]));
    sumGroup2 = complexAdd(complexMul(r0, termSecond[2]), complexMul(rn, termSecond[3]));
}

[Differentiable]
void diffractionBetaGroups3D(
    float phi,
    float phiPrime,
    float n,
    no_diff float k,
    float s,
    float sPrime,
    float sinBeta0,
    Complex r0,
    Complex rn,
    out Complex factor,
    out Complex difGroup,
    out Complex difGroup1,
    out Complex sumGroup,
    out Complex sumGroup1,
    out Complex difGroup2,
    out Complex sumGroup2
)
{
    float safeSinBeta0 = max(sinBeta0, SMALL_EPS);
    float l = s * sPrime / (s + sPrime + EPS) * safeSinBeta0 * safeSinBeta0;
    float kL = k * l;
    float difPhi = phi - phiPrime;
    float sumPhi = phi + phiPrime;
    factor = complexMulReal(
        complexExpPhase(-0.25f * PI),
        -1.0f / (2.0f * n * sqrt(TWO_PI * k + EPS) * safeSinBeta0)
    );

    float cotValue[4];
    float cot1[4];
    float cot2[4];
    float x[4];
    float x1[4];
    float x2[4];
    betaTermMetadata(difPhi, n, kL, +1.0f, true, cotValue[0], cot1[0], cot2[0], x[0], x1[0], x2[0]);
    betaTermMetadata(difPhi, n, kL, -1.0f, false, cotValue[1], cot1[1], cot2[1], x[1], x1[1], x2[1]);
    betaTermMetadata(sumPhi, n, kL, +1.0f, true, cotValue[2], cot1[2], cot2[2], x[2], x1[2], x2[2]);
    betaTermMetadata(sumPhi, n, kL, -1.0f, false, cotValue[3], cot1[3], cot2[3], x[3], x1[3], x2[3]);

    Complex transition[4];
    Complex transition1[4];
    Complex transition2[4];
    for (int i = 0; i < 4; ++i)
    {
        fUTDWithDerivatives(x[i], transition[i], transition1[i], transition2[i]);
    }

    Complex termValue[4];
    Complex termFirst[4];
    Complex termSecond[4];
    for (int i = 0; i < 4; ++i)
    {
        assembleBetaTermState(
            cotValue[i], cot1[i], cot2[i], x1[i], x2[i],
            transition[i], transition1[i], transition2[i],
            termValue[i], termFirst[i], termSecond[i]
        );
    }

    difGroup = complexAdd(termValue[0], termValue[1]);
    difGroup1 = complexAdd(termFirst[0], termFirst[1]);
    difGroup2 = complexAdd(termSecond[0], termSecond[1]);
    sumGroup = complexAdd(complexMul(r0, termValue[2]), complexMul(rn, termValue[3]));
    sumGroup1 = complexAdd(complexMul(r0, termFirst[2]), complexMul(rn, termFirst[3]));
    sumGroup2 = complexAdd(complexMul(r0, termSecond[2]), complexMul(rn, termSecond[3]));
}

[Differentiable]
Complex diffractionCoefficient2D(float phi, float phiPrime, float n, no_diff float k, float s, float sPrime, Complex r0, Complex rn)
{
    Complex factor, difGroup, difGroup1, sumGroup, sumGroup1, difGroup2, sumGroup2;
    diffractionBetaGroups(phi, phiPrime, n, k, s, sPrime, r0, rn, factor, difGroup, difGroup1, sumGroup, sumGroup1, difGroup2, sumGroup2);
    return complexMul(factor, complexAdd(difGroup, sumGroup));
}

[Differentiable]
Complex diffractionCoefficient3D(
    float phi,
    float phiPrime,
    float n,
    no_diff float k,
    float s,
    float sPrime,
    float sinBeta0,
    Complex r0,
    Complex rn
)
{
    Complex factor, difGroup, difGroup1, sumGroup, sumGroup1, difGroup2, sumGroup2;
    diffractionBetaGroups3D(
        phi,
        phiPrime,
        n,
        k,
        s,
        sPrime,
        sinBeta0,
        r0,
        rn,
        factor,
        difGroup,
        difGroup1,
        sumGroup,
        sumGroup1,
        difGroup2,
        sumGroup2
    );
    return complexMul(factor, complexAdd(difGroup, sumGroup));
}

[Differentiable]
Complex diffractionCoefficient2DAngleDerivative(
    float phi,
    float phiPrime,
    float n,
    no_diff float k,
    float s,
    float sPrime,
    no_diff bool derivativeWrtPhi,
    Complex r0,
    Complex rn
)
{
    Complex factor, difGroup, difGroup1, sumGroup, sumGroup1, difGroup2, sumGroup2;
    diffractionBetaGroups(phi, phiPrime, n, k, s, sPrime, r0, rn, factor, difGroup, difGroup1, sumGroup, sumGroup1, difGroup2, sumGroup2);
    Complex combined = derivativeWrtPhi ? complexAdd(difGroup1, sumGroup1) : complexAdd(complexMulReal(difGroup1, -1.0f), sumGroup1);
    return complexMul(factor, combined);
}

[Differentiable]
Complex diffractionCoefficient3DAngleDerivative(
    float phi,
    float phiPrime,
    float n,
    no_diff float k,
    float s,
    float sPrime,
    float sinBeta0,
    no_diff bool derivativeWrtPhi,
    Complex r0,
    Complex rn
)
{
    Complex factor, difGroup, difGroup1, sumGroup, sumGroup1, difGroup2, sumGroup2;
    diffractionBetaGroups3D(
        phi,
        phiPrime,
        n,
        k,
        s,
        sPrime,
        sinBeta0,
        r0,
        rn,
        factor,
        difGroup,
        difGroup1,
        sumGroup,
        sumGroup1,
        difGroup2,
        sumGroup2
    );
    Complex combined = derivativeWrtPhi ? complexAdd(difGroup1, sumGroup1) : complexAdd(complexMulReal(difGroup1, -1.0f), sumGroup1);
    return complexMul(factor, combined);
}

[Differentiable]
Complex slopeDiffractionCoefficient2D(float phi, float phiPrime, float n, no_diff float k, float s, float sPrime, Complex r0, Complex rn)
{
    Complex derivative = diffractionCoefficient2DAngleDerivative(phi, phiPrime, n, k, s, sPrime, false, r0, rn);
    return complexDivReal(complexMul(Complex(0.0f, -1.0f), derivative), k);
}

[Differentiable]
Complex slopeDiffractionCoefficient3D(
    float phi,
    float phiPrime,
    float n,
    no_diff float k,
    float s,
    float sPrime,
    float sinBeta0,
    Complex r0,
    Complex rn
)
{
    Complex derivative = diffractionCoefficient3DAngleDerivative(
        phi,
        phiPrime,
        n,
        k,
        s,
        sPrime,
        sinBeta0,
        false,
        r0,
        rn
    );
    return complexDivReal(complexMul(Complex(0.0f, -1.0f), derivative), k);
}

[Differentiable]
void computeEdgeAngles(
    float3 sourcePos,
    float3 edgePos,
    float3 edgeDir,
    float3 n0,
    float3 targetPos,
    out float phi,
    out float phiPrime,
    out float s,
    out float sPrime
)
{
    float3 sourceToEdge = edgePos - sourcePos;
    float3 sourceToEdgeProj = projectToWedgePlane(sourceToEdge, edgeDir);
    sPrime = safeLength(sourceToEdgeProj) + EPS;
    float3 toHat = safeNormalize(cross(n0, edgeDir), float3(0.0f, 1.0f, 0.0f));
    float3 kiProj = sourceToEdgeProj / sPrime;

    float signPrime = no_diff sign(-dot(kiProj, n0));
    phiPrime = PI - safeAcos(-dot(kiProj, toHat));
    phiPrime = phiPrime * (-signPrime);
    phiPrime = phiPrime + PI;

    float3 edgeToTarget = targetPos - edgePos;
    float3 edgeToTargetProj = projectToWedgePlane(edgeToTarget, edgeDir);
    s = safeLength(edgeToTargetProj) + EPS;
    float3 koProj = edgeToTargetProj / s;

    float signPhi = no_diff sign(dot(koProj, n0));
    phi = PI - safeAcos(dot(koProj, toHat));
    phi = phi * (-signPhi);
    phi = phi + PI;
}

float orientedAnglePositive(float y, float x)
{
    float angle = atan2(y, x);
    if (angle < 0.0f)
    {
        angle += TWO_PI;
    }
    return angle;
}

void buildEdgeAngleCache(
    float3 sourcePos,
    float3 edgePos,
    float3 edgeDir,
    float3 n0,
    float3 targetPos,
    out EdgeAngleCache cache
)
{
    cache.sourceToEdge = edgePos - sourcePos;
    cache.sourceToEdgeProj = projectToWedgePlane(cache.sourceToEdge, edgeDir);
    cache.sourceToEdgeProjNorm = safeLength(cache.sourceToEdgeProj);
    cache.sPrime = cache.sourceToEdgeProjNorm + EPS;
    cache.toHatBase = cross(n0, edgeDir);
    cache.toHatBaseNorm = safeLength(cache.toHatBase);
    cache.toHat = safeNormalize(cache.toHatBase, float3(0.0f, 1.0f, 0.0f));
    cache.kiProj = cache.sourceToEdgeProj / cache.sPrime;

    cache.edgeToTarget = targetPos - edgePos;
    cache.edgeToTargetProj = projectToWedgePlane(cache.edgeToTarget, edgeDir);
    cache.edgeToTargetProjNorm = safeLength(cache.edgeToTargetProj);
    cache.s = cache.edgeToTargetProjNorm + EPS;
    cache.koProj = cache.edgeToTargetProj / cache.s;

    float phiPrimeX = -dot(cache.kiProj, cache.toHat);
    float phiPrimeY = -dot(cache.kiProj, n0);
    cache.phiPrime = orientedAnglePositive(phiPrimeY, phiPrimeX);

    float phiX = dot(cache.koProj, cache.toHat);
    float phiY = dot(cache.koProj, n0);
    cache.phi = orientedAnglePositive(phiY, phiX);
}

void adjProjectToWedgePlane(
    float3 value,
    float3 edgeDir,
    float3 gradProj,
    inout float3 gradValue,
    inout float3 gradEdgeDir
)
{
    float dotValueEdge = dot(value, edgeDir);
    float gradDot = dot(gradProj, edgeDir);
    gradValue += gradProj - edgeDir * gradDot;
    gradEdgeDir += -value * gradDot - gradProj * dotValueEdge;
}

void adjSafeNormalizeConstFallback(
    float3 value,
    float valueNorm,
    float3 gradOutput,
    inout float3 gradValue
)
{
    if (valueNorm <= SMALL_EPS)
    {
        return;
    }

    float denom = valueNorm + EPS;
    float dotGradValue = dot(gradOutput, value);
    gradValue += gradOutput / denom;
    gradValue -= value * (dotGradValue / (valueNorm * denom * denom));
}

void adjOrientedAnglePositive(
    float y,
    float x,
    float gradAngle,
    inout float gradY,
    inout float gradX
)
{
    float denom = x * x + y * y;
    if (denom <= SMALL_EPS)
    {
        return;
    }

    gradY += gradAngle * (x / denom);
    gradX -= gradAngle * (y / denom);
}

void adjEdgeAnglesFromCache(
    EdgeAngleCache cache,
    float3 sourcePos,
    float3 edgePos,
    float3 edgeDir,
    float3 n0,
    float3 targetPos,
    float gradPhi,
    float gradPhiPrime,
    float gradS,
    float gradSPrime,
    out float3 gradSourcePos,
    out float3 gradEdgePos,
    out float3 gradEdgeDir,
    out float3 gradN0,
    out float3 gradTargetPos
)
{
    gradSourcePos = zeroFloat3();
    gradEdgePos = zeroFloat3();
    gradEdgeDir = zeroFloat3();
    gradN0 = zeroFloat3();
    gradTargetPos = zeroFloat3();

    float3 gradSourceProj = zeroFloat3();
    float3 gradTargetProj = zeroFloat3();
    float3 gradToHat = zeroFloat3();

    float phiX = dot(cache.koProj, cache.toHat);
    float phiY = dot(cache.koProj, n0);
    float gradPhiX = 0.0f;
    float gradPhiY = 0.0f;
    adjOrientedAnglePositive(phiY, phiX, gradPhi, gradPhiY, gradPhiX);
    gradTargetProj += cache.toHat * gradPhiX + n0 * gradPhiY;
    gradToHat += cache.koProj * gradPhiX;
    gradN0 += cache.koProj * gradPhiY;

    float phiPrimeX = -dot(cache.kiProj, cache.toHat);
    float phiPrimeY = -dot(cache.kiProj, n0);
    float gradPhiPrimeX = 0.0f;
    float gradPhiPrimeY = 0.0f;
    adjOrientedAnglePositive(phiPrimeY, phiPrimeX, gradPhiPrime, gradPhiPrimeY, gradPhiPrimeX);
    gradSourceProj -= cache.toHat * gradPhiPrimeX + n0 * gradPhiPrimeY;
    gradToHat -= cache.kiProj * gradPhiPrimeX;
    gradN0 -= cache.kiProj * gradPhiPrimeY;

    float gradSLocal = gradS - dot(gradTargetProj, cache.edgeToTargetProj) / (cache.s * cache.s);
    gradTargetProj /= cache.s;
    if (cache.edgeToTargetProjNorm > SMALL_EPS)
    {
        gradTargetProj += cache.edgeToTargetProj * (gradSLocal / cache.edgeToTargetProjNorm);
    }

    float gradSPrimeLocal = gradSPrime - dot(gradSourceProj, cache.sourceToEdgeProj) / (cache.sPrime * cache.sPrime);
    gradSourceProj /= cache.sPrime;
    if (cache.sourceToEdgeProjNorm > SMALL_EPS)
    {
        gradSourceProj += cache.sourceToEdgeProj * (gradSPrimeLocal / cache.sourceToEdgeProjNorm);
    }

    float3 gradEdgeToTarget = zeroFloat3();
    adjProjectToWedgePlane(cache.edgeToTarget, edgeDir, gradTargetProj, gradEdgeToTarget, gradEdgeDir);
    gradTargetPos += gradEdgeToTarget;
    gradEdgePos -= gradEdgeToTarget;

    float3 gradSourceToEdge = zeroFloat3();
    adjProjectToWedgePlane(cache.sourceToEdge, edgeDir, gradSourceProj, gradSourceToEdge, gradEdgeDir);
    gradEdgePos += gradSourceToEdge;
    gradSourcePos -= gradSourceToEdge;

    float3 gradToHatBase = zeroFloat3();
    adjSafeNormalizeConstFallback(cache.toHatBase, cache.toHatBaseNorm, gradToHat, gradToHatBase);
    gradN0 += cross(edgeDir, gradToHatBase);
    gradEdgeDir += cross(gradToHatBase, n0);
}

[Differentiable]
Complex3 transportDiffractionVector(
    Complex3 incidentVector,
    Complex3 incidentDerivativeVector,
    float3 sourcePos,
    float3 edgePos,
    float3 edgeDir,
    float3 targetPos,
    Complex directGain,
    Complex derivativeGain
)
{
    float3 edgeHat = edgeDir / (safeLength(edgeDir) + EPS);
    float3 incomingHat = (edgePos - sourcePos) / (safeLength(edgePos - sourcePos) + EPS);
    float3 outgoingHat = (targetPos - edgePos) / (safeLength(targetPos - edgePos) + EPS);

    float3 phiInHat = safeNormalize(cross(incomingHat, edgeHat), stablePerpendicularBasis(incomingHat, edgeHat));
    float3 phiOutHat = safeNormalize(-cross(outgoingHat, edgeHat), stablePerpendicularBasis(outgoingHat, edgeHat));

    Complex incPhi = complexDotReal(incidentVector, phiInHat);
    Complex incEdge = complexDotReal(incidentVector, edgeHat);
    Complex3 outDirect = complex3Scale(
        complex3Add(
            complexScaleReal(phiOutHat, incPhi),
            complexScaleReal(edgeHat, incEdge)
        ),
        directGain
    );

    Complex derPhi = complexDotReal(incidentDerivativeVector, phiInHat);
    Complex derEdge = complexDotReal(incidentDerivativeVector, edgeHat);
    Complex3 outDerivative = complex3Scale(
        complex3Add(
            complexScaleReal(phiOutHat, derPhi),
            complexScaleReal(edgeHat, derEdge)
        ),
        derivativeGain
    );
    return complex3Add(outDirect, outDerivative);
}

void computeVectorTransportAdjoint(
    PairInputs state,
    float3 targetPos,
    Complex directGain,
    Complex derivativeGain,
    Complex3.Differential vectorGrad,
    out PairInputs.Differential stateGrad,
    out float3 gradTargetPos,
    out Complex gradDirectGain,
    out Complex gradDerivativeGain
)
{
    stateGrad = zeroPairInputsDifferential();
    gradTargetPos = zeroFloat3();
    gradDirectGain = complexZero();
    gradDerivativeGain = complexZero();

    if (!complex3DifferentialAnyNonzero(vectorGrad))
    {
        return;
    }

    float edgeDirNorm = safeLength(state.edgeDir);
    float3 edgeHat = state.edgeDir / (edgeDirNorm + EPS);

    float3 incomingDelta = state.edgePos - state.sourcePos;
    float incomingNorm = safeLength(incomingDelta);
    float3 incomingHat = incomingDelta / (incomingNorm + EPS);

    float3 outgoingDelta = targetPos - state.edgePos;
    float outgoingNorm = safeLength(outgoingDelta);
    float3 outgoingHat = outgoingDelta / (outgoingNorm + EPS);

    float3 phiInBase = cross(incomingHat, edgeHat);
    float phiInBaseNorm = safeLength(phiInBase);
    float3 phiInHat = safeNormalize(phiInBase, stablePerpendicularBasis(incomingHat, edgeHat));

    float3 phiOutBase = cross(edgeHat, outgoingHat);
    float phiOutBaseNorm = safeLength(phiOutBase);
    float3 phiOutHat = safeNormalize(phiOutBase, stablePerpendicularBasis(outgoingHat, edgeHat));

    Complex incPhi = complexDotReal(state.incidentVector, phiInHat);
    Complex incEdge = complexDotReal(state.incidentVector, edgeHat);
    Complex3 directBasis = complex3Add(
        complexScaleReal(phiOutHat, incPhi),
        complexScaleReal(edgeHat, incEdge)
    );

    Complex derPhi = complexDotReal(state.incidentDerivativeVector, phiInHat);
    Complex derEdge = complexDotReal(state.incidentDerivativeVector, edgeHat);
    Complex3 derivativeBasis = complex3Add(
        complexScaleReal(phiOutHat, derPhi),
        complexScaleReal(edgeHat, derEdge)
    );

    Complex3.Differential gradDirectBasis = Complex3.Differential(
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f)
    );
    Complex3.Differential gradDerivativeBasis = Complex3.Differential(
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f),
        Complex.Differential(0.0f, 0.0f)
    );
    adjComplex3Scale(directBasis, directGain, vectorGrad, gradDirectBasis, gradDirectGain);
    adjComplex3Scale(derivativeBasis, derivativeGain, vectorGrad, gradDerivativeBasis, gradDerivativeGain);

    float3 gradPhiOutHat = zeroFloat3();
    float3 gradEdgeHat = zeroFloat3();
    Complex gradIncPhi = complexZero();
    Complex gradIncEdge = complexZero();
    adjComplexScaleReal(phiOutHat, incPhi, gradDirectBasis, gradPhiOutHat, gradIncPhi);
    adjComplexScaleReal(edgeHat, incEdge, gradDirectBasis, gradEdgeHat, gradIncEdge);

    Complex gradDerPhi = complexZero();
    Complex gradDerEdge = complexZero();
    adjComplexScaleReal(phiOutHat, derPhi, gradDerivativeBasis, gradPhiOutHat, gradDerPhi);
    adjComplexScaleReal(edgeHat, derEdge, gradDerivativeBasis, gradEdgeHat, gradDerEdge);

    float3 gradPhiInHat = zeroFloat3();
    adjComplexDotReal(state.incidentVector, phiInHat, gradIncPhi, stateGrad.incidentVector, gradPhiInHat);
    adjComplexDotReal(state.incidentVector, edgeHat, gradIncEdge, stateGrad.incidentVector, gradEdgeHat);
    adjComplexDotReal(
        state.incidentDerivativeVector,
        phiInHat,
        gradDerPhi,
        stateGrad.incidentDerivativeVector,
        gradPhiInHat
    );
    adjComplexDotReal(
        state.incidentDerivativeVector,
        edgeHat,
        gradDerEdge,
        stateGrad.incidentDerivativeVector,
        gradEdgeHat
    );

    float3 gradPhiInBase = zeroFloat3();
    adjSafeNormalizeConstFallback(phiInBase, phiInBaseNorm, gradPhiInHat, gradPhiInBase);

    float3 gradPhiOutBase = zeroFloat3();
    adjSafeNormalizeConstFallback(phiOutBase, phiOutBaseNorm, gradPhiOutHat, gradPhiOutBase);

    float3 gradIncomingHat = zeroFloat3();
    float3 gradOutgoingHat = zeroFloat3();
    gradIncomingHat += cross(edgeHat, gradPhiInBase);
    gradEdgeHat += cross(gradPhiInBase, incomingHat);
    gradEdgeHat += cross(outgoingHat, gradPhiOutBase);
    gradOutgoingHat += cross(gradPhiOutBase, edgeHat);

    float3 gradIncomingDelta = zeroFloat3();
    adjSafeNormalizeConstFallback(incomingDelta, incomingNorm, gradIncomingHat, gradIncomingDelta);
    stateGrad.edgePos += gradIncomingDelta;
    stateGrad.sourcePos -= gradIncomingDelta;

    float3 gradOutgoingDelta = zeroFloat3();
    adjSafeNormalizeConstFallback(outgoingDelta, outgoingNorm, gradOutgoingHat, gradOutgoingDelta);
    gradTargetPos += gradOutgoingDelta;
    stateGrad.edgePos -= gradOutgoingDelta;

    float3 gradEdgeDir = zeroFloat3();
    adjSafeNormalizeConstFallback(state.edgeDir, edgeDirNorm, gradEdgeHat, gradEdgeDir);
    stateGrad.edgeDir += gradEdgeDir;
}
