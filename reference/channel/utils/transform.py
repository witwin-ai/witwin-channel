from __future__ import annotations

import drjit as dr

from ..types import Float, Point3f, Vector3f


class Transform4f:
    """Minimal DrJit-compatible transform helper for local mesh transforms."""

    def __init__(self):
        self._r00 = Float(1.0)
        self._r01 = Float(0.0)
        self._r02 = Float(0.0)
        self._r10 = Float(0.0)
        self._r11 = Float(1.0)
        self._r12 = Float(0.0)
        self._r20 = Float(0.0)
        self._r21 = Float(0.0)
        self._r22 = Float(1.0)
        self._tx = Float(0.0)
        self._ty = Float(0.0)
        self._tz = Float(0.0)

    def _compose(
        self,
        r00,
        r01,
        r02,
        r10,
        r11,
        r12,
        r20,
        r21,
        r22,
        tx,
        ty,
        tz,
    ):
        old_r00, old_r01, old_r02 = self._r00, self._r01, self._r02
        old_r10, old_r11, old_r12 = self._r10, self._r11, self._r12
        old_r20, old_r21, old_r22 = self._r20, self._r21, self._r22
        old_tx, old_ty, old_tz = self._tx, self._ty, self._tz

        self._r00 = old_r00 * r00 + old_r01 * r10 + old_r02 * r20
        self._r01 = old_r00 * r01 + old_r01 * r11 + old_r02 * r21
        self._r02 = old_r00 * r02 + old_r01 * r12 + old_r02 * r22
        self._r10 = old_r10 * r00 + old_r11 * r10 + old_r12 * r20
        self._r11 = old_r10 * r01 + old_r11 * r11 + old_r12 * r21
        self._r12 = old_r10 * r02 + old_r11 * r12 + old_r12 * r22
        self._r20 = old_r20 * r00 + old_r21 * r10 + old_r22 * r20
        self._r21 = old_r20 * r01 + old_r21 * r11 + old_r22 * r21
        self._r22 = old_r20 * r02 + old_r21 * r12 + old_r22 * r22
        self._tx = old_r00 * tx + old_r01 * ty + old_r02 * tz + old_tx
        self._ty = old_r10 * tx + old_r11 * ty + old_r12 * tz + old_ty
        self._tz = old_r20 * tx + old_r21 * ty + old_r22 * tz + old_tz
        return self

    def translate(self, vec):
        if type(vec) is not Vector3f:
            raise TypeError(f"Transform4f.translate expects Vector3f, got {type(vec).__name__}.")
        return self._compose(
            Float(1.0),
            Float(0.0),
            Float(0.0),
            Float(0.0),
            Float(1.0),
            Float(0.0),
            Float(0.0),
            Float(0.0),
            Float(1.0),
            vec.x,
            vec.y,
            vec.z,
        )

    def rotate(self, axis, angle_degrees):
        if type(axis) is not Vector3f:
            raise TypeError(f"Transform4f.rotate expects Vector3f axis, got {type(axis).__name__}.")
        if type(angle_degrees) is not Float:
            raise TypeError(
                f"Transform4f.rotate expects Float angle_degrees, got {type(angle_degrees).__name__}."
            )
        ax = axis.x
        ay = axis.y
        az = axis.z
        inv_norm = dr.rcp(dr.sqrt(ax * ax + ay * ay + az * az) + Float(1e-12))
        ax = ax * inv_norm
        ay = ay * inv_norm
        az = az * inv_norm
        angle = angle_degrees * Float(dr.pi / 180.0)
        c = dr.cos(angle)
        s = dr.sin(angle)
        one_minus_c = Float(1.0) - c

        return self._compose(
            c + ax * ax * one_minus_c,
            ax * ay * one_minus_c - az * s,
            ax * az * one_minus_c + ay * s,
            ay * ax * one_minus_c + az * s,
            c + ay * ay * one_minus_c,
            ay * az * one_minus_c - ax * s,
            az * ax * one_minus_c - ay * s,
            az * ay * one_minus_c + ax * s,
            c + az * az * one_minus_c,
            Float(0.0),
            Float(0.0),
            Float(0.0),
        )

    def __matmul__(self, points):
        if type(points) is not Point3f:
            raise TypeError(f"Transform4f.__matmul__ expects Point3f, got {type(points).__name__}.")
        return Point3f(
            self._r00 * points.x + self._r01 * points.y + self._r02 * points.z + self._tx,
            self._r10 * points.x + self._r11 * points.y + self._r12 * points.z + self._ty,
            self._r20 * points.x + self._r21 * points.y + self._r22 * points.z + self._tz,
        )


__all__ = ["Transform4f"]
