implementing utd_accumulate;

[Differentiable]
void poly12(
    float x,
    no_diff float c0, no_diff float c1, no_diff float c2, no_diff float c3,
    no_diff float c4, no_diff float c5, no_diff float c6, no_diff float c7,
    no_diff float c8, no_diff float c9, no_diff float c10, no_diff float c11,
    out float value,
    out float first,
    out float second
)
{
    value = c11;
    first = 0.0f;
    second = 0.0f;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c10;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c9;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c8;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c7;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c6;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c5;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c4;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c3;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c2;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c1;

    second = second * x + 2.0f * first;
    first = first * x + value;
    value = value * x + c0;
}

[Differentiable]
void fresnelBoersmaValueFirstSecond(
    float x,
    out Complex value,
    out Complex first,
    out Complex second
)
{
    const float smallEps = 1.0e-12f;

    bool xPos = x >= 0.0f;
    float xAbs = abs(x);
    float safeX = max(xAbs, smallEps);
    bool cond = xAbs < 4.0f;

    float argSmall = 0.25f * xAbs;
    float argLarge = 4.0f / safeX;
    float arg = cond ? argSmall : argLarge;
    float argSafe = max(arg, smallEps);

    float arg1Small = 0.25f;
    float arg2Small = 0.0f;
    float arg1Large = -4.0f / (safeX * safeX);
    float arg2Large = 8.0f / (safeX * safeX * safeX);

    float rSmall, rSmall1, rSmall2;
    poly12(
        argSmall,
        +1.595769140f, -0.000001702f, -6.808568854f, -0.000576361f,
        +6.920691902f, -0.016898657f, -3.050485660f, -0.075752419f,
        +0.850663781f, -0.025639041f, -0.150230960f, +0.034404779f,
        rSmall, rSmall1, rSmall2
    );

    float iSmall, iSmall1, iSmall2;
    poly12(
        argSmall,
        -0.000000033f, +4.255387524f, -0.000092810f, -7.780020400f,
        -0.009520895f, +5.075161298f, -0.138341947f, -1.363729124f,
        -0.403349276f, +0.702222016f, -0.216195929f, +0.019547031f,
        iSmall, iSmall1, iSmall2
    );

    float rLarge, rLarge1, rLarge2;
    poly12(
        argLarge,
        +0.000000000f, -0.024933975f, +0.000003936f, +0.005770956f,
        +0.000689892f, -0.009497136f, +0.011948809f, -0.006748873f,
        +0.000246420f, +0.002102967f, -0.001217930f, +0.000233939f,
        rLarge, rLarge1, rLarge2
    );

    float iLarge, iLarge1, iLarge2;
    poly12(
        argLarge,
        +0.199471140f, +0.000000023f, -0.009351341f, +0.000023006f,
        +0.004851466f, +0.001903218f, -0.017122914f, +0.029064067f,
        -0.027928955f, +0.016497308f, -0.005598515f, +0.000838386f,
        iLarge, iLarge1, iLarge2
    );

    float rCoef = cond ? rSmall : rLarge;
    float iCoef = cond ? iSmall : iLarge;
    float rCoef1 = cond ? (rSmall1 * arg1Small) : (rLarge1 * arg1Large);
    float iCoef1 = cond ? (iSmall1 * arg1Small) : (iLarge1 * arg1Large);
    float rCoef2 = cond
        ? (rSmall2 * arg1Small * arg1Small)
        : (rLarge2 * arg1Large * arg1Large + rLarge1 * arg2Large);
    float iCoef2 = cond
        ? (iSmall2 * arg1Small * arg1Small)
        : (iLarge2 * arg1Large * arg1Large + iLarge1 * arg2Large);

    float arg1 = cond ? arg1Small : arg1Large;
    float arg2 = cond ? arg2Small : arg2Large;

    float argSqrt = sqrt(argSafe);
    float argSqrt1 = 0.5f * arg1 / argSqrt;
    float argSqrt2 = 0.5f * arg2 / argSqrt - 0.25f * arg1 * arg1 / (argSafe * argSqrt);

    float rPart = rCoef * argSqrt;
    float rPart1 = rCoef1 * argSqrt + rCoef * argSqrt1;
    float rPart2 = rCoef2 * argSqrt + 2.0f * rCoef1 * argSqrt1 + rCoef * argSqrt2;
    float iPart = -iCoef * argSqrt;
    float iPart1 = -(iCoef1 * argSqrt + iCoef * argSqrt1);
    float iPart2 = -(iCoef2 * argSqrt + 2.0f * iCoef1 * argSqrt1 + iCoef * argSqrt2);

    float sinX, cosX;
    sincos(xAbs, sinX, cosX);

    float valueReal = cosX * rPart - sinX * iPart;
    float valueImag = cosX * iPart + sinX * rPart;
    float firstReal = cosX * (rPart1 - iPart) - sinX * (rPart + iPart1);
    float firstImag = cosX * (iPart1 + rPart) + sinX * (rPart1 - iPart);
    float secondReal = cosX * (rPart2 - rPart - 2.0f * iPart1) - sinX * (2.0f * rPart1 - iPart + iPart2);
    float secondImag = cosX * (iPart2 + 2.0f * rPart1 - iPart) + sinX * (rPart2 - rPart - 2.0f * iPart1);

    if (!cond)
    {
        valueReal += 0.5f;
        valueImag += 0.5f;
    }

    value = Complex(xPos ? valueReal : -valueReal, xPos ? valueImag : -valueImag);
    first = Complex(firstReal, firstImag);
    second = Complex(xPos ? secondReal : -secondReal, xPos ? secondImag : -secondImag);
}

[Differentiable]
void fUTDWithDerivatives(float x, out Complex value, out Complex first, out Complex second)
{
    float safeX = max(x, SMALL_EPS);
    Complex fresnelValue, fresnelFirst, fresnelSecond;
    fresnelBoersmaValueFirstSecond(x, fresnelValue, fresnelFirst, fresnelSecond);

    Complex fresnelConj = complexConj(fresnelValue);
    Complex fresnelFirstConj = complexConj(fresnelFirst);
    Complex fresnelSecondConj = complexConj(fresnelSecond);

    float prefactor = sqrt(PI * safeX * 0.5f);
    float prefactor1 = 0.5f * prefactor / safeX;
    float prefactor2 = -0.25f * prefactor / (safeX * safeX);
    Complex phase = complexExpPhase(x);
    Complex phase1 = complexMul(Complex(0.0f, 1.0f), phase);
    Complex phase2 = Complex(-phase.re, -phase.im);
    Complex bracket = complexSub(Complex(1.0f, 1.0f), complexMul(Complex(0.0f, 2.0f), fresnelConj));
    Complex bracket1 = complexMul(Complex(0.0f, -2.0f), fresnelFirstConj);
    Complex bracket2 = complexMul(Complex(0.0f, -2.0f), fresnelSecondConj);

    value = complexMulReal(complexMul(phase, bracket), prefactor);
    first = complexAdd(
        complexAdd(
            complexMulReal(complexMul(phase, bracket), prefactor1),
            complexMulReal(complexMul(phase1, bracket), prefactor)
        ),
        complexMulReal(complexMul(phase, bracket1), prefactor)
    );
    second = complexAdd(
        complexAdd(
            complexAdd(
                complexMulReal(complexMul(phase, bracket), prefactor2),
                complexMulReal(complexMul(phase1, bracket), 2.0f * prefactor1)
            ),
            complexMulReal(complexMul(phase, bracket1), 2.0f * prefactor1)
        ),
        complexAdd(
            complexAdd(
                complexMulReal(complexMul(phase2, bracket), prefactor),
                complexMulReal(complexMul(phase1, bracket1), 2.0f * prefactor)
            ),
            complexMulReal(complexMul(phase, bracket2), prefactor)
        )
    );
}
