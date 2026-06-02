之前 trace/reflection/field.py 计算反射场时用的是 cell center 坐标，但实际网格用的是 edge point 坐标，导致每个点的距离计算有偏差，进而导致相位错误
