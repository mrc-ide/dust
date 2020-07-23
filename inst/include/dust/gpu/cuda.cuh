#ifndef DUST_CUDA_CUH
#define DUST_CUDA_CUH
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Cuda call
static void HandleCUDAError(const char *file,
	int line,
	cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
	cudaDeviceSynchronize();
#endif
    if (status != CUDA_SUCCESS || (status = cudaGetLastError()) != CUDA_SUCCESS)
    {
        if (status == cudaErrorUnknown)
        {
            printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
            getchar();
            exit(1);
        }
        printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
        getchar();
        exit(1);
    }
}
#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))


#endif