#pragma once

#ifdef __CUDACC__
#define WITWIN_KERNEL_DINLINE __device__ __forceinline__
#define WITWIN_KERNEL_HD_INLINE __host__ __device__ __forceinline__
#else
#define WITWIN_KERNEL_DINLINE inline
#define WITWIN_KERNEL_HD_INLINE inline
#endif
