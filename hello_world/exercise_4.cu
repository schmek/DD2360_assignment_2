#include <curand_kernel.h>
#include <curand.h>

#include "exercises.cuh"
#include "device_array.cuh"
#include "profiling_timer.cuh"

#include <fstream>
#include <utility>
#include <tuple>

template<size_t BLOCK_SZ, bool use_single=true>
__global__ void calculate_pi(unsigned long long int* result, curandState* states, const size_t trials)
{
  __shared__ unsigned long data[BLOCK_SZ];
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(idx, idx, 0, &states[idx]);
  
  data[threadIdx.x] = 0;

  for (size_t i = 0; i < trials; ++i)
  {
    if constexpr(use_single)
    {
      auto x = curand_uniform(&states[idx]);
      auto y = curand_uniform(&states[idx]);
      if (x * x + y * y <= 1.f)
      {
        ++data[threadIdx.x];
      }
    }
    else
    {
      auto x = curand_uniform_double(&states[idx]);
      auto y = curand_uniform_double(&states[idx]);
      if (x * x + y * y <= 1.)
      {
        ++data[threadIdx.x];
      }
    }
    
  }
  __syncthreads();
  harris_reduce(data);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    atomicAdd(result, data[0]);
  }
}


template<typename T>
__device__ void harris_reduce(T* data)
{
  // The code in this function is essentially a verbatim copy of the code in the following reference:
  // Harris, M. (2007). Optimizing parallel reduction in CUDA. Nvidia developer technology, 2(4), 70.
  // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s) 
    {
      data[threadIdx.x] += data[threadIdx.x + s];
    }
    __syncthreads();
  }
}

namespace 
{
  template<size_t N_ITER, size_t BLOCK_SZ, bool use_single=true>
  class Simulation
  {
    
  public:
    void run(int n_trials, std::ofstream& file, bool write_file)
    {
      printf("N_ITER: %d\n", N_ITER);
      ProfilingTimer timer("Sim " + std::to_string(n_trials));

      DeviceOnlyArray<curandState, N_ITER, BLOCK_SZ> d_random;

      std::array<unsigned long long int, 1> result;
      result[0] = 0;
      DeviceArray<unsigned long long int, 1, BLOCK_SZ> d_result(result);

      calculate_pi<BLOCK_SZ, use_single> << <(N_ITER + BLOCK_SZ - 1) / BLOCK_SZ, BLOCK_SZ >> > (d_result.device_get(), d_random.device_get(), n_trials);

      result = d_result.host_get();

      auto frac = ((N_ITER + BLOCK_SZ - 1) / BLOCK_SZ) * BLOCK_SZ * static_cast<float>(n_trials);
      auto pi = (static_cast<float>(result[0]) * 4.f) / static_cast<float>(frac);
      auto ms = timer.lap_time();
      if (write_file)
      {
        file << pi << ", " << frac << ", " << BLOCK_SZ << ", " << ms << "\n";
      }
      printf("The result is: %f\n", pi);
    }
  };
}

void exercise_4() 
{

  std::ofstream pi_file("pi_estimates.csv", std::ios::out);
  pi_file << "pi, iterations, block size, time \n";
  Simulation<1, 1>().run(1, pi_file, false);
   

  Simulation<128, 128>().run(1, pi_file, true);
  Simulation<256, 128>().run(1, pi_file, true);
  Simulation<1024, 128>().run(1, pi_file, true);
  Simulation<8192, 128>().run(1, pi_file, true);
  Simulation<8192 << 2, 128>().run(1, pi_file, true);
  Simulation<8192 << 4, 128>().run(1, pi_file, true);
  Simulation<8192 << 6, 128>().run(1, pi_file, true);
  Simulation<8192 << 8, 128>().run(1, pi_file, true);
  Simulation<8192 << 10, 128>().run(1, pi_file, true);

  Simulation<8192 << 10, 16, true>().run(1, pi_file, true);
  Simulation<8192 << 10, 16, false>().run(1, pi_file, true);

  Simulation<8192 << 10, 16, true>().run(1, pi_file, true);
  Simulation<8192 << 10, 32, true>().run(1, pi_file, true);
  Simulation<8192 << 10, 64, true>().run(1, pi_file, true);
  Simulation<8192 << 10, 128, true>().run(1, pi_file, true);
  Simulation<8192 << 10, 256, true>().run(1, pi_file, true);
};
