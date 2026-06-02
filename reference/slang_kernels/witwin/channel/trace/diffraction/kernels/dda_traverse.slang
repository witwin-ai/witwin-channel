static const float EPS = 1.0e-10f;
static const float MIN_PROPAGATION_DISTANCE = 1.0e-2f;
static const float PI = 3.14159265358979323846f;

int clampInt(int value, int lo, int hi)
{
    return min(max(value, lo), hi);
}

void atomicAdd1D(TensorView<float> tensor, int index, float value)
{
    float oldValue;
    tensor.InterlockedAdd(uint3(uint(index), 0, 0), value, oldValue);
}

[AutoPyBindCUDA]
[CUDAKernel]
void suffixDdaTraverseForward(
    TensorView<int> activeMask,
    TensorView<int> stateIndex,
    TensorView<float> segOriginX,
    TensorView<float> segOriginY,
    TensorView<float> segOriginZ,
    TensorView<float> segDirX,
    TensorView<float> segDirY,
    TensorView<float> segDirZ,
    TensorView<float> blockerDist,
    TensorView<float> segFieldReal,
    TensorView<float> segFieldImag,
    TensorView<float> segVectorXReal,
    TensorView<float> segVectorXImag,
    TensorView<float> segVectorYReal,
    TensorView<float> segVectorYImag,
    TensorView<float> segVectorZReal,
    TensorView<float> segVectorZImag,
    TensorView<float> xCoords,
    TensorView<float> yCoords,
    TensorView<float> resultReal,
    TensorView<float> resultImag,
    TensorView<float> resultCount,
    TensorView<float> resultVectorRealX,
    TensorView<float> resultVectorImagX,
    TensorView<float> resultVectorRealY,
    TensorView<float> resultVectorImagY,
    TensorView<float> resultVectorRealZ,
    TensorView<float> resultVectorImagZ,
    float xMin,
    float xMax,
    float yMin,
    float yMax,
    float cellSizeX,
    float cellSizeY,
    int nx,
    int nRx,
    int maxSteps,
    float rxZ,
    float wavelength,
    float k
)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int rayIdx = int(dispatchIdx.x);
    if (rayIdx >= activeMask.size(0))
    {
        return;
    }
    if (activeMask[rayIdx] == 0)
    {
        return;
    }

    float dirX = segDirX[rayIdx];
    float dirY = segDirY[rayIdx];
    float originX = segOriginX[rayIdx];
    float originY = segOriginY[rayIdx];
    float originZ = segOriginZ[rayIdx];

    float t = 0.0f;
    float dtX = abs(cellSizeX / max(abs(dirX), EPS));
    float dtY = abs(cellSizeY / max(abs(dirY), EPS));
    float stepX = dirX > 0.0f ? cellSizeX : -cellSizeX;
    float stepY = dirY > 0.0f ? cellSizeY : -cellSizeY;
    float nextX = dirX > 0.0f
        ? (floor((originX - xMin) / cellSizeX) + 1.0f) * cellSizeX + xMin
        : floor((originX - xMin) / cellSizeX) * cellSizeX + xMin;
    float nextY = dirY > 0.0f
        ? (floor((originY - yMin) / cellSizeY) + 1.0f) * cellSizeY + yMin
        : floor((originY - yMin) / cellSizeY) * cellSizeY + yMin;
    float tMaxX = abs((nextX - originX) / max(abs(dirX), EPS));
    float tMaxY = abs((nextY - originY) / max(abs(dirY), EPS));

    float curX = originX;
    float curY = originY;
    bool loopActive =
        (curX >= xMin) && (curX < xMax) &&
        (curY >= yMin) && (curY < yMax);
    int ny = int(yCoords.size(0));

    for (int step = 0; step < maxSteps && loopActive; ++step)
    {
        bool inBounds =
            loopActive &&
            (curX >= xMin) && (curX < xMax) &&
            (curY >= yMin) && (curY < yMax) &&
            (t < blockerDist[rayIdx]);

        if (inBounds)
        {
            int ix = clampInt(int((curX - xMin) / cellSizeX), 0, nx - 1);
            int iy = clampInt(int((curY - yMin) / cellSizeY), 0, ny - 1);
            int cellIdx = iy * nx + ix;
            float cellX = xCoords[ix];
            float cellY = yCoords[iy];

            float dx = cellX - originX;
            float dy = cellY - originY;
            float dz = rxZ - originZ;
            float dSeg = sqrt(max(dx * dx + dy * dy + dz * dz, 0.0f)) + EPS;

            float fspl = (wavelength / (4.0f * PI)) / max(dSeg, MIN_PROPAGATION_DISTANCE);
            float phase = -k * dSeg;
            float sinP, cosP;
            sincos(phase, sinP, cosP);

            float realContrib = fspl * (segFieldReal[rayIdx] * cosP - segFieldImag[rayIdx] * sinP);
            float imagContrib = fspl * (segFieldReal[rayIdx] * sinP + segFieldImag[rayIdx] * cosP);
            float vectorRealX = fspl * (segVectorXReal[rayIdx] * cosP - segVectorXImag[rayIdx] * sinP);
            float vectorImagX = fspl * (segVectorXReal[rayIdx] * sinP + segVectorXImag[rayIdx] * cosP);
            float vectorRealY = fspl * (segVectorYReal[rayIdx] * cosP - segVectorYImag[rayIdx] * sinP);
            float vectorImagY = fspl * (segVectorYReal[rayIdx] * sinP + segVectorYImag[rayIdx] * cosP);
            float vectorRealZ = fspl * (segVectorZReal[rayIdx] * cosP - segVectorZImag[rayIdx] * sinP);
            float vectorImagZ = fspl * (segVectorZReal[rayIdx] * sinP + segVectorZImag[rayIdx] * cosP);

            int flatIdx = stateIndex[rayIdx] * nRx + cellIdx;
            atomicAdd1D(resultReal, flatIdx, realContrib);
            atomicAdd1D(resultImag, flatIdx, imagContrib);
            atomicAdd1D(resultCount, flatIdx, 1.0f);
            atomicAdd1D(resultVectorRealX, flatIdx, vectorRealX);
            atomicAdd1D(resultVectorImagX, flatIdx, vectorImagX);
            atomicAdd1D(resultVectorRealY, flatIdx, vectorRealY);
            atomicAdd1D(resultVectorImagY, flatIdx, vectorImagY);
            atomicAdd1D(resultVectorRealZ, flatIdx, vectorRealZ);
            atomicAdd1D(resultVectorImagZ, flatIdx, vectorImagZ);
        }

        bool moveX = tMaxX < tMaxY;
        t = moveX ? tMaxX : tMaxY;
        curX = moveX ? (curX + stepX) : curX;
        curY = moveX ? curY : (curY + stepY);
        tMaxX = moveX ? (tMaxX + dtX) : tMaxX;
        tMaxY = moveX ? tMaxY : (tMaxY + dtY);
        loopActive = inBounds;
    }
}
