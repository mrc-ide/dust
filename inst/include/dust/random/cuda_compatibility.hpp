#ifndef DUST_RANDOM_CUDA_COMPATIBILITY_HPP
#define DUST_RANDOM_CUDA_COMPATIBILITY_HPP

// There are 3 different ways we hit this file:
//
// *  __CUDA_ARCH__ is defined: we're compiling under nvcc generating
//        device code. In this case __NVCC__ is always defined.
//
// * __NVCC__: we're compiling under nvcc, either generating host or
//        device code
//
// * Neither is defined: we're compiling under gcc/clang etc and need
//   to inject our stubs

#ifdef __NVCC__
// Compiling under nvcc, but could be either host or device code
#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__
#define ALIGN(n) __align__(n)

// This is necessary due to templates which are __host__ __device__;
// whenever a HOSTDEVICE function is called from another HOSTDEVICE
// function the compiler gets confused as it can't tell which one it's
// going to use. This suppresses the warning as it is ok here.
#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#else
// Compiling under gcc/clang
#define HOST
#define HOSTDEVICE
#define DEVICE
#define KERNEL
#define ALIGN(n)
#define __nv_exec_check_disable__

#endif

#ifdef __CUDA_ARCH__
// Compiling under nvcc for the device
#define CONSTANT __constant__
#define SYNCWARP __syncwarp();
#else
// gcc/clang or nvcc for the host
#define CONSTANT const
#define SYNCWARP
#endif

#endif
