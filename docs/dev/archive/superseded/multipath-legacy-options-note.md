  方案 A：Per-RX 精确计算（准确但慢）

  对每个 RX 点:
      1. 找到该 RX 的有效反射路径
      2. 计算 Image Source Chain
      3. 检测 Secondary Visibility
      4. 如果被边缘遮挡，计算 R-D 衍射

  优点：物理准确
  缺点：计算量大（O(n_rx * n_bounces)）

  方案 B：基于 Monte Carlo 的近似（快但需要设计）

  1. Monte Carlo 射线追踪时记录：
     - 每条射线的反射序列
     - 每次反射的 Image Source

  2. 对每个 bounce 层:
     - 收集该 bounce 的所有 Image Source
     - 做 Visibility 检测（批量）
     - 计算 R-D 衍射（批量）

  优点：可以 GPU 并行
  缺点：需要修改当前的 DDA 遍历逻辑

  方案 C：混合方案（推荐）

  1. 保持 Monte Carlo 反射计算
  2. 对 shadow boundary 区域（场强突变处）：
     - 检测这些 RX 点
     - 只对它们做精确的 R-D 计算

  优点：平衡准确性和性能
  缺点：需要 shadow boundary 检测算法