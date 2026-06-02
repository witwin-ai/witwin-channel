from __future__ import annotations

import drjit.cuda.ad as cuda_ad


Float = cuda_ad.Float
UInt32 = cuda_ad.UInt32
Int32 = cuda_ad.Int32
Bool = cuda_ad.Bool
Point2f = cuda_ad.Array2f
Point3f = cuda_ad.Array3f
Vector2f = cuda_ad.Array2f
Vector3f = cuda_ad.Array3f
Vector3u = cuda_ad.Array3u
Complex2f = cuda_ad.Complex2f
Matrix4f = cuda_ad.Matrix4f


class InteractionType:
    NONE = 0
    REFLECTION = 1
    DIFFRACTION = 2
    TRANSMISSION = 4
    SCATTERING = 8


__all__ = [
    "Bool",
    "Complex2f",
    "Float",
    "Int32",
    "InteractionType",
    "Matrix4f",
    "Point2f",
    "Point3f",
    "UInt32",
    "Vector2f",
    "Vector3f",
    "Vector3u",
]
