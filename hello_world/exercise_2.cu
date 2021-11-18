#include <random>
#include <array>
#include <algorithm>
#include <memory>
#include <limits>
#include <exception>

#include "exercises.cuh"
#include "device_array.cuh"
#include "profiling_timer.cuh"

constexpr unsigned int ARRAY_SIZE = 10'000;
constexpr unsigned int BLOCK_SIZE = 16;



void saxpy_cpu(const std::array<float, ARRAY_SIZE>& x, std::array<float, ARRAY_SIZE>& y, const float a)
{
  for (size_t i = 0; i < y.size(); ++i)
  {
    y[i] += a * x[i];
  }
}

__global__ void saxpy_gpu(float* x, float* y, float a)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[idx] += a * x[idx];
}


void exercise_2()
{
  try 
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.f, 1.f);
    std::array<float, ARRAY_SIZE> x, y;
    std::generate(x.begin(), x.end(), [&] {return dist(gen); });
    std::generate(y.begin(), y.end(), [&] {return dist(gen); });
    auto a = dist(gen);
    std::array<float, ARRAY_SIZE> gpu_y;
    {
      ProfilingTimer timer("GPU");
      DeviceArray<float, ARRAY_SIZE, BLOCK_SIZE> d_x(x);
      DeviceArray<float, ARRAY_SIZE, BLOCK_SIZE> d_y(y);
      saxpy_gpu << <(ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_x.device_get(), d_y.device_get(), a);
      CUDA_CHECK({});
      CUDA_CHECK(cudaDeviceSynchronize());

      gpu_y = d_y.host_get();
    }
    {
      ProfilingTimer timer("CPU");
      saxpy_cpu(x, y, a);
    }
    int n_diff = 0;
    for (size_t i = 0; i < ARRAY_SIZE; ++i)
    {
      if (std::fabs(y[i] - gpu_y[i]) >  10.f * std::numeric_limits<float>::epsilon())
      {
        ++n_diff;
      }
    }
    printf("Number of different : %d \n", n_diff);
  }
  catch (const CudaError&)
  {
    printf("Cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
  }
}