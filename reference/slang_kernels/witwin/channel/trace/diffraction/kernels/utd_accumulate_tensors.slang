implementing utd_accumulate;

PairInputs loadPairInputs(StateTensors state, int sIdx)
{
    return PairInputs(
        float3(state.edgePosX[sIdx], state.edgePosY[sIdx], state.edgePosZ[sIdx]),
        float3(state.edgeDirX[sIdx], state.edgeDirY[sIdx], state.edgeDirZ[sIdx]),
        float3(state.n0X[sIdx], state.n0Y[sIdx], state.n0Z[sIdx]),
        float3(state.nnX[sIdx], state.nnY[sIdx], state.nnZ[sIdx]),
        state.wedgeN[sIdx],
        float3(state.sourcePosX[sIdx], state.sourcePosY[sIdx], state.sourcePosZ[sIdx]),
        Complex(state.incidentFieldReal[sIdx], state.incidentFieldImag[sIdx]),
        Complex(state.incidentNormalDerivativeReal[sIdx], state.incidentNormalDerivativeImag[sIdx]),
        Complex(state.r0Real[sIdx], state.r0Imag[sIdx]),
        Complex(state.rnReal[sIdx], state.rnImag[sIdx]),
        Complex3(
            Complex(state.incidentVectorXReal[sIdx], state.incidentVectorXImag[sIdx]),
            Complex(state.incidentVectorYReal[sIdx], state.incidentVectorYImag[sIdx]),
            Complex(state.incidentVectorZReal[sIdx], state.incidentVectorZImag[sIdx])
        ),
        Complex3(
            Complex(state.incidentDerivativeVectorXReal[sIdx], state.incidentDerivativeVectorXImag[sIdx]),
            Complex(state.incidentDerivativeVectorYReal[sIdx], state.incidentDerivativeVectorYImag[sIdx]),
            Complex(state.incidentDerivativeVectorZReal[sIdx], state.incidentDerivativeVectorZImag[sIdx])
        ),
        Jones2(
            Complex(state.incidentJonesUReal[sIdx], state.incidentJonesUImag[sIdx]),
            Complex(state.incidentJonesVReal[sIdx], state.incidentJonesVImag[sIdx])
        ),
        Jones2(
            Complex(state.incidentDerivativeJonesUReal[sIdx], state.incidentDerivativeJonesUImag[sIdx]),
            Complex(state.incidentDerivativeJonesVReal[sIdx], state.incidentDerivativeJonesVImag[sIdx])
        ),
        Basis3(
            float3(state.incidentBasisUX[sIdx], state.incidentBasisUY[sIdx], state.incidentBasisUZ[sIdx]),
            float3(state.incidentBasisVX[sIdx], state.incidentBasisVY[sIdx], state.incidentBasisVZ[sIdx]),
            float3(state.incidentBasisKX[sIdx], state.incidentBasisKY[sIdx], state.incidentBasisKZ[sIdx])
        ),
        JonesOperator(
            Complex(state.face0OperatorM00Real[sIdx], state.face0OperatorM00Imag[sIdx]),
            Complex(state.face0OperatorM01Real[sIdx], state.face0OperatorM01Imag[sIdx]),
            Complex(state.face0OperatorM10Real[sIdx], state.face0OperatorM10Imag[sIdx]),
            Complex(state.face0OperatorM11Real[sIdx], state.face0OperatorM11Imag[sIdx])
        ),
        JonesOperator(
            Complex(state.face1OperatorM00Real[sIdx], state.face1OperatorM00Imag[sIdx]),
            Complex(state.face1OperatorM01Real[sIdx], state.face1OperatorM01Imag[sIdx]),
            Complex(state.face1OperatorM10Real[sIdx], state.face1OperatorM10Imag[sIdx]),
            Complex(state.face1OperatorM11Real[sIdx], state.face1OperatorM11Imag[sIdx])
        ),
        FaceMaterialParams(
            state.face0EtaR[sIdx],
            state.face0Sigma[sIdx],
            state.face0Gain[sIdx],
            state.face0UseFresnel[sIdx],
            state.face0MaterialPresent[sIdx]
        ),
        FaceMaterialParams(
            state.face1EtaR[sIdx],
            state.face1Sigma[sIdx],
            state.face1Gain[sIdx],
            state.face1UseFresnel[sIdx],
            state.face1MaterialPresent[sIdx]
        )
    );
}

PairInputs.Differential loadPairInputDifferential(StateTensors state, int sIdx)
{
    return PairInputs.Differential(
        float3(state.edgePosX[sIdx], state.edgePosY[sIdx], state.edgePosZ[sIdx]),
        float3(state.edgeDirX[sIdx], state.edgeDirY[sIdx], state.edgeDirZ[sIdx]),
        float3(state.n0X[sIdx], state.n0Y[sIdx], state.n0Z[sIdx]),
        float3(state.nnX[sIdx], state.nnY[sIdx], state.nnZ[sIdx]),
        state.wedgeN[sIdx],
        float3(state.sourcePosX[sIdx], state.sourcePosY[sIdx], state.sourcePosZ[sIdx]),
        Complex.Differential(state.incidentFieldReal[sIdx], state.incidentFieldImag[sIdx]),
        Complex.Differential(state.incidentNormalDerivativeReal[sIdx], state.incidentNormalDerivativeImag[sIdx]),
        Complex.Differential(state.r0Real[sIdx], state.r0Imag[sIdx]),
        Complex.Differential(state.rnReal[sIdx], state.rnImag[sIdx]),
        Complex3.Differential(
            Complex.Differential(state.incidentVectorXReal[sIdx], state.incidentVectorXImag[sIdx]),
            Complex.Differential(state.incidentVectorYReal[sIdx], state.incidentVectorYImag[sIdx]),
            Complex.Differential(state.incidentVectorZReal[sIdx], state.incidentVectorZImag[sIdx])
        ),
        Complex3.Differential(
            Complex.Differential(state.incidentDerivativeVectorXReal[sIdx], state.incidentDerivativeVectorXImag[sIdx]),
            Complex.Differential(state.incidentDerivativeVectorYReal[sIdx], state.incidentDerivativeVectorYImag[sIdx]),
            Complex.Differential(state.incidentDerivativeVectorZReal[sIdx], state.incidentDerivativeVectorZImag[sIdx])
        ),
        Jones2.Differential(
            Complex.Differential(state.incidentJonesUReal[sIdx], state.incidentJonesUImag[sIdx]),
            Complex.Differential(state.incidentJonesVReal[sIdx], state.incidentJonesVImag[sIdx])
        ),
        Jones2.Differential(
            Complex.Differential(state.incidentDerivativeJonesUReal[sIdx], state.incidentDerivativeJonesUImag[sIdx]),
            Complex.Differential(state.incidentDerivativeJonesVReal[sIdx], state.incidentDerivativeJonesVImag[sIdx])
        ),
        Basis3.Differential(
            float3(state.incidentBasisUX[sIdx], state.incidentBasisUY[sIdx], state.incidentBasisUZ[sIdx]),
            float3(state.incidentBasisVX[sIdx], state.incidentBasisVY[sIdx], state.incidentBasisVZ[sIdx]),
            float3(state.incidentBasisKX[sIdx], state.incidentBasisKY[sIdx], state.incidentBasisKZ[sIdx])
        ),
        JonesOperator.Differential(
            Complex.Differential(state.face0OperatorM00Real[sIdx], state.face0OperatorM00Imag[sIdx]),
            Complex.Differential(state.face0OperatorM01Real[sIdx], state.face0OperatorM01Imag[sIdx]),
            Complex.Differential(state.face0OperatorM10Real[sIdx], state.face0OperatorM10Imag[sIdx]),
            Complex.Differential(state.face0OperatorM11Real[sIdx], state.face0OperatorM11Imag[sIdx])
        ),
        JonesOperator.Differential(
            Complex.Differential(state.face1OperatorM00Real[sIdx], state.face1OperatorM00Imag[sIdx]),
            Complex.Differential(state.face1OperatorM01Real[sIdx], state.face1OperatorM01Imag[sIdx]),
            Complex.Differential(state.face1OperatorM10Real[sIdx], state.face1OperatorM10Imag[sIdx]),
            Complex.Differential(state.face1OperatorM11Real[sIdx], state.face1OperatorM11Imag[sIdx])
        ),
        FaceMaterialParams.Differential(
            state.face0EtaR[sIdx],
            state.face0Sigma[sIdx],
            state.face0Gain[sIdx],
            state.face0UseFresnel[sIdx],
            state.face0MaterialPresent[sIdx]
        ),
        FaceMaterialParams.Differential(
            state.face1EtaR[sIdx],
            state.face1Sigma[sIdx],
            state.face1Gain[sIdx],
            state.face1UseFresnel[sIdx],
            state.face1MaterialPresent[sIdx]
        )
    );
}

float3 loadRx(RxTensors rx, int rIdx)
{
    return float3(rx.x[rIdx], rx.y[rIdx], rx.z[rIdx]);
}

void atomicAdd1D(TensorView<float> tensor, int index, float value)
{
    float oldValue;
    tensor.InterlockedAdd(uint3(uint(index), 0, 0), value, oldValue);
}

void atomicAddComplex(OutputTensors output, int ownership, int rIdx, Complex value)
{
    if (ownership == OWNERSHIP_DIRECT_DIFFRACTION)
    {
        atomicAdd1D(output.directReal, rIdx, value.re);
        atomicAdd1D(output.directImag, rIdx, value.im);
    }
    else if (ownership == OWNERSHIP_MIXED_DIFFRACTION)
    {
        atomicAdd1D(output.multiReal, rIdx, value.re);
        atomicAdd1D(output.multiImag, rIdx, value.im);
    }
}

void atomicAddComplex3(OutputTensors output, int ownership, int rIdx, Complex3 value)
{
    if (ownership == OWNERSHIP_DIRECT_DIFFRACTION)
    {
        atomicAdd1D(output.directVectorXReal, rIdx, value.x.re);
        atomicAdd1D(output.directVectorXImag, rIdx, value.x.im);
        atomicAdd1D(output.directVectorYReal, rIdx, value.y.re);
        atomicAdd1D(output.directVectorYImag, rIdx, value.y.im);
        atomicAdd1D(output.directVectorZReal, rIdx, value.z.re);
        atomicAdd1D(output.directVectorZImag, rIdx, value.z.im);
    }
    else if (ownership == OWNERSHIP_MIXED_DIFFRACTION)
    {
        atomicAdd1D(output.multiVectorXReal, rIdx, value.x.re);
        atomicAdd1D(output.multiVectorXImag, rIdx, value.x.im);
        atomicAdd1D(output.multiVectorYReal, rIdx, value.y.re);
        atomicAdd1D(output.multiVectorYImag, rIdx, value.y.im);
        atomicAdd1D(output.multiVectorZReal, rIdx, value.z.re);
        atomicAdd1D(output.multiVectorZImag, rIdx, value.z.im);
    }
}

void atomicAddPairOutput(OutputTensors output, int ownership, int rIdx, PairOutputs value)
{
    atomicAddComplex(output, ownership, rIdx, value.field);
    atomicAddComplex3(output, ownership, rIdx, value.vectorField);
}

PairOutputs.Differential loadPairOutputDifferential(OutputTensors outputGrad, int ownership, int rIdx)
{
    if (ownership == OWNERSHIP_DIRECT_DIFFRACTION)
    {
        return PairOutputs.Differential(
            Complex.Differential(outputGrad.directReal[rIdx], outputGrad.directImag[rIdx]),
            Complex3.Differential(
                Complex.Differential(outputGrad.directVectorXReal[rIdx], outputGrad.directVectorXImag[rIdx]),
                Complex.Differential(outputGrad.directVectorYReal[rIdx], outputGrad.directVectorYImag[rIdx]),
                Complex.Differential(outputGrad.directVectorZReal[rIdx], outputGrad.directVectorZImag[rIdx])
            )
        );
    }
    if (ownership == OWNERSHIP_MIXED_DIFFRACTION)
    {
        return PairOutputs.Differential(
            Complex.Differential(outputGrad.multiReal[rIdx], outputGrad.multiImag[rIdx]),
            Complex3.Differential(
                Complex.Differential(outputGrad.multiVectorXReal[rIdx], outputGrad.multiVectorXImag[rIdx]),
                Complex.Differential(outputGrad.multiVectorYReal[rIdx], outputGrad.multiVectorYImag[rIdx]),
                Complex.Differential(outputGrad.multiVectorZReal[rIdx], outputGrad.multiVectorZImag[rIdx])
            )
        );
    }
    return PairOutputs.Differential(
        Complex.Differential(0.0f, 0.0f),
        Complex3.Differential(
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f),
            Complex.Differential(0.0f, 0.0f)
        )
    );
}

void atomicAddPairInputDifferential(StateTensors stateGrad, int sIdx, PairInputs.Differential value)
{
    atomicAdd1D(stateGrad.edgePosX, sIdx, value.edgePos.x);
    atomicAdd1D(stateGrad.edgePosY, sIdx, value.edgePos.y);
    atomicAdd1D(stateGrad.edgePosZ, sIdx, value.edgePos.z);
    atomicAdd1D(stateGrad.edgeDirX, sIdx, value.edgeDir.x);
    atomicAdd1D(stateGrad.edgeDirY, sIdx, value.edgeDir.y);
    atomicAdd1D(stateGrad.edgeDirZ, sIdx, value.edgeDir.z);
    atomicAdd1D(stateGrad.n0X, sIdx, value.n0.x);
    atomicAdd1D(stateGrad.n0Y, sIdx, value.n0.y);
    atomicAdd1D(stateGrad.n0Z, sIdx, value.n0.z);
    atomicAdd1D(stateGrad.nnX, sIdx, value.nn.x);
    atomicAdd1D(stateGrad.nnY, sIdx, value.nn.y);
    atomicAdd1D(stateGrad.nnZ, sIdx, value.nn.z);
    atomicAdd1D(stateGrad.wedgeN, sIdx, value.wedgeN);
    atomicAdd1D(stateGrad.sourcePosX, sIdx, value.sourcePos.x);
    atomicAdd1D(stateGrad.sourcePosY, sIdx, value.sourcePos.y);
    atomicAdd1D(stateGrad.sourcePosZ, sIdx, value.sourcePos.z);
    atomicAdd1D(stateGrad.incidentFieldReal, sIdx, value.incidentField.re);
    atomicAdd1D(stateGrad.incidentFieldImag, sIdx, value.incidentField.im);
    atomicAdd1D(stateGrad.incidentNormalDerivativeReal, sIdx, value.incidentNormalDerivative.re);
    atomicAdd1D(stateGrad.incidentNormalDerivativeImag, sIdx, value.incidentNormalDerivative.im);
    atomicAdd1D(stateGrad.r0Real, sIdx, value.r0.re);
    atomicAdd1D(stateGrad.r0Imag, sIdx, value.r0.im);
    atomicAdd1D(stateGrad.rnReal, sIdx, value.rn.re);
    atomicAdd1D(stateGrad.rnImag, sIdx, value.rn.im);
    atomicAdd1D(stateGrad.incidentVectorXReal, sIdx, value.incidentVector.x.re);
    atomicAdd1D(stateGrad.incidentVectorXImag, sIdx, value.incidentVector.x.im);
    atomicAdd1D(stateGrad.incidentVectorYReal, sIdx, value.incidentVector.y.re);
    atomicAdd1D(stateGrad.incidentVectorYImag, sIdx, value.incidentVector.y.im);
    atomicAdd1D(stateGrad.incidentVectorZReal, sIdx, value.incidentVector.z.re);
    atomicAdd1D(stateGrad.incidentVectorZImag, sIdx, value.incidentVector.z.im);
    atomicAdd1D(stateGrad.incidentDerivativeVectorXReal, sIdx, value.incidentDerivativeVector.x.re);
    atomicAdd1D(stateGrad.incidentDerivativeVectorXImag, sIdx, value.incidentDerivativeVector.x.im);
    atomicAdd1D(stateGrad.incidentDerivativeVectorYReal, sIdx, value.incidentDerivativeVector.y.re);
    atomicAdd1D(stateGrad.incidentDerivativeVectorYImag, sIdx, value.incidentDerivativeVector.y.im);
    atomicAdd1D(stateGrad.incidentDerivativeVectorZReal, sIdx, value.incidentDerivativeVector.z.re);
    atomicAdd1D(stateGrad.incidentDerivativeVectorZImag, sIdx, value.incidentDerivativeVector.z.im);
    atomicAdd1D(stateGrad.incidentJonesUReal, sIdx, value.incidentJones.u.re);
    atomicAdd1D(stateGrad.incidentJonesUImag, sIdx, value.incidentJones.u.im);
    atomicAdd1D(stateGrad.incidentJonesVReal, sIdx, value.incidentJones.v.re);
    atomicAdd1D(stateGrad.incidentJonesVImag, sIdx, value.incidentJones.v.im);
    atomicAdd1D(stateGrad.incidentDerivativeJonesUReal, sIdx, value.incidentDerivativeJones.u.re);
    atomicAdd1D(stateGrad.incidentDerivativeJonesUImag, sIdx, value.incidentDerivativeJones.u.im);
    atomicAdd1D(stateGrad.incidentDerivativeJonesVReal, sIdx, value.incidentDerivativeJones.v.re);
    atomicAdd1D(stateGrad.incidentDerivativeJonesVImag, sIdx, value.incidentDerivativeJones.v.im);
    atomicAdd1D(stateGrad.incidentBasisUX, sIdx, value.incidentBasis.u.x);
    atomicAdd1D(stateGrad.incidentBasisUY, sIdx, value.incidentBasis.u.y);
    atomicAdd1D(stateGrad.incidentBasisUZ, sIdx, value.incidentBasis.u.z);
    atomicAdd1D(stateGrad.incidentBasisVX, sIdx, value.incidentBasis.v.x);
    atomicAdd1D(stateGrad.incidentBasisVY, sIdx, value.incidentBasis.v.y);
    atomicAdd1D(stateGrad.incidentBasisVZ, sIdx, value.incidentBasis.v.z);
    atomicAdd1D(stateGrad.incidentBasisKX, sIdx, value.incidentBasis.k.x);
    atomicAdd1D(stateGrad.incidentBasisKY, sIdx, value.incidentBasis.k.y);
    atomicAdd1D(stateGrad.incidentBasisKZ, sIdx, value.incidentBasis.k.z);
    atomicAdd1D(stateGrad.face0OperatorM00Real, sIdx, value.face0Operator.m00.re);
    atomicAdd1D(stateGrad.face0OperatorM00Imag, sIdx, value.face0Operator.m00.im);
    atomicAdd1D(stateGrad.face0OperatorM01Real, sIdx, value.face0Operator.m01.re);
    atomicAdd1D(stateGrad.face0OperatorM01Imag, sIdx, value.face0Operator.m01.im);
    atomicAdd1D(stateGrad.face0OperatorM10Real, sIdx, value.face0Operator.m10.re);
    atomicAdd1D(stateGrad.face0OperatorM10Imag, sIdx, value.face0Operator.m10.im);
    atomicAdd1D(stateGrad.face0OperatorM11Real, sIdx, value.face0Operator.m11.re);
    atomicAdd1D(stateGrad.face0OperatorM11Imag, sIdx, value.face0Operator.m11.im);
    atomicAdd1D(stateGrad.face1OperatorM00Real, sIdx, value.face1Operator.m00.re);
    atomicAdd1D(stateGrad.face1OperatorM00Imag, sIdx, value.face1Operator.m00.im);
    atomicAdd1D(stateGrad.face1OperatorM01Real, sIdx, value.face1Operator.m01.re);
    atomicAdd1D(stateGrad.face1OperatorM01Imag, sIdx, value.face1Operator.m01.im);
    atomicAdd1D(stateGrad.face1OperatorM10Real, sIdx, value.face1Operator.m10.re);
    atomicAdd1D(stateGrad.face1OperatorM10Imag, sIdx, value.face1Operator.m10.im);
    atomicAdd1D(stateGrad.face1OperatorM11Real, sIdx, value.face1Operator.m11.re);
    atomicAdd1D(stateGrad.face1OperatorM11Imag, sIdx, value.face1Operator.m11.im);
    atomicAdd1D(stateGrad.face0EtaR, sIdx, value.face0Material.etaR);
    atomicAdd1D(stateGrad.face0Sigma, sIdx, value.face0Material.sigma);
    atomicAdd1D(stateGrad.face0Gain, sIdx, value.face0Material.gain);
    atomicAdd1D(stateGrad.face0UseFresnel, sIdx, value.face0Material.useFresnel);
    atomicAdd1D(stateGrad.face0MaterialPresent, sIdx, value.face0Material.present);
    atomicAdd1D(stateGrad.face1EtaR, sIdx, value.face1Material.etaR);
    atomicAdd1D(stateGrad.face1Sigma, sIdx, value.face1Material.sigma);
    atomicAdd1D(stateGrad.face1Gain, sIdx, value.face1Material.gain);
    atomicAdd1D(stateGrad.face1UseFresnel, sIdx, value.face1Material.useFresnel);
    atomicAdd1D(stateGrad.face1MaterialPresent, sIdx, value.face1Material.present);
}

void atomicAddRxDifferential(RxTensors rxGrad, int rIdx, float3 value)
{
    atomicAdd1D(rxGrad.x, rIdx, value.x);
    atomicAdd1D(rxGrad.y, rIdx, value.y);
    atomicAdd1D(rxGrad.z, rIdx, value.z);
}
