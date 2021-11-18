#pragma once

#include <stdio.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaError : public std::runtime_error 
{
public:
  CudaError(cudaError_t err, const std::string& source) 
    : std::runtime_error(cudaGetErrorString(err))
    , m_source(source)
  {}

  const std::string& source() const
  {
    return m_source;
  }

private:
  const std::string m_source;
};
#define CUDA_CHECK(x) \
  x; \
  { \
    auto err = cudaGetLastError(); \
    if(err != cudaSuccess) \
      throw CudaError(err, std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
  }

#ifndef __INTELLISENSE__
#define CUDA_LAUNCH(func, block_sz, n_threads, ... ) func<<<(n_threads + block_sz - 1) / block_sz, block_sz>>>(__VA_ARGS__); CUDA_CHECK({}) 
#else
//Stop intellisense squigglies because of <<<, >>>
#define CUDA_LAUNCH(func, block_sz, n_threads, ... )
#endif // !__INTELLISENSE__

void exercise_1();
void exercise_2();
void exercise_3();
void exercise_4();