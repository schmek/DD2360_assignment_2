#include <random>
#include <array>
#include <algorithm>
#include <memory>
#include <fstream>

#include "exercises.cuh"
#include "device_array.cuh"
#include "profiling_timer.cuh"
 
template<typename Distribution>
float3 get_random_vector(std::mt19937& gen,  Distribution& distribution)
{
  float3 vec;
  vec.x = distribution(gen);
  vec.y = distribution(gen);
  vec.z = distribution(gen);
  return vec;
}
 
__device__ void iterate_particle(float3* particle, float3* velocity, const float dt, const size_t idx)
{
  particle->x += velocity->x * dt;
  particle->y += velocity->y * dt;
  particle->z += velocity->z * dt;
}
 
__global__ void evovle_particle_gpu(float3 particles[], float3 velocities[], const float dt, const size_t n_iter)
{
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  float3* particle = &particles[idx];
  float3* velocity = &velocities[idx];
  for (size_t i = 0; i < n_iter; ++i)
  {
    iterate_particle(particle, velocity, dt, idx);
  }
}

template<size_t ARRAY_SIZE>
__host__ void evolve_particle_cpu(float3 particles[], float3 velocities[], const float dt, const size_t n_iter)
{
  for (size_t j = 0; j < ARRAY_SIZE; ++j)
  {
    for (size_t i = 0; i < n_iter; ++i)
    {
      particles[j].x += velocities[j].x * dt;
      particles[j].y += velocities[j].y * dt;
      particles[j].z += velocities[j].z * dt;
    }
  }
}


namespace 
{
  template <size_t N_PARTICLES, size_t N_ITER, size_t BLOCK_SZ> 
  class Simulation
  {
  public:

    void run(std::ofstream& gpu_file, std::ofstream& cpu_file, bool run_cpu = true)
    {
      try
      {
        printf("==== Running simulation with %d particles over %d iterations\n\n", N_PARTICLES, N_ITER);
        auto particles = generate_data();
        auto velocities = generate_data();

        const float dt = 1.f;
        
        //GPU
        {
          ProfilingTimer timer("Evolving particles on GPU");
          DeviceArray<float3, N_PARTICLES, BLOCK_SZ> d_particles(*particles);
          DeviceArray<float3, N_PARTICLES, BLOCK_SZ> d_velocities(*velocities);

          CUDA_LAUNCH(evovle_particle_gpu, BLOCK_SZ, N_PARTICLES,
            d_particles.device_get(), d_velocities.device_get(), dt, N_ITER);
          CUDA_CHECK(cudaDeviceSynchronize());
          
          auto ms = timer.lap_time();
          gpu_file << N_PARTICLES << ", " << N_ITER << ", " << BLOCK_SZ << ", " << ms << '\n';
        }

        // CPU
        if (run_cpu)
        {
          ProfilingTimer timer("Evolving particles on CPU");
          evolve_particle_cpu<N_PARTICLES>(particles->data(), velocities->data(), dt, N_ITER);
          auto ms = timer.lap_time();
          cpu_file << N_PARTICLES << ", " << N_ITER << ", " << ms << '\n';
        }

      }
      catch (const CudaError& ex)
      {
        printf("Cuda error: %s\n source: %s\n", ex.what(), ex.source().c_str());
      }
    }

  private:
    auto generate_data()
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<float> dist(0.f, 1.f);

      auto data = std::make_unique<std::array<float3, N_PARTICLES>>();
      std::generate(data->begin(), data->end(), [&] { return get_random_vector(gen, dist); });
      return data;
    }
  };
}


void exercise_3()
{
  try 
  {
    std::ofstream cpu_file("cpu.csv", std::ios::out);
    std::ofstream gpu_file("gpu.csv", std::ios::out);

    cpu_file << "particles, iterations, time\n";
    gpu_file << "particles, iterations, block_size, time\n";

    //Cache warmup.
    Simulation<1, 1, 1>().run(gpu_file, cpu_file);
    
    //Fixed iterations, fixed block size.
    Simulation<8'192, 8'192, 2 << 4>().run(gpu_file, cpu_file);
    Simulation<8'192, 8'192, 2 << 5>().run(gpu_file, cpu_file);
    Simulation<8'192, 8'192, 2 << 6>().run(gpu_file, cpu_file);
    Simulation<8'192, 8'192, 2 << 7>().run(gpu_file, cpu_file);
    Simulation<8'192, 8'192, 2 << 8>().run(gpu_file, cpu_file);


    Simulation < 131'072, 8'192, 2 << 4 > ().run(gpu_file, cpu_file);
    Simulation < 131'072, 8'192, 2 << 5 >().run(gpu_file, cpu_file);
    Simulation < 131'072, 8'192, 2 << 6 >().run(gpu_file, cpu_file);
    Simulation < 131'072, 8'192, 2 << 7 >().run(gpu_file, cpu_file);
    Simulation < 131'072, 8'192, 2 << 8 >().run(gpu_file, cpu_file);


    Simulation<1'048'576, 8'192, 2 << 4>().run(gpu_file, cpu_file);
    Simulation<1'048'576, 8'192, 2 << 5>().run(gpu_file, cpu_file);
    Simulation<1'048'576, 8'192, 2 << 6>().run(gpu_file, cpu_file);
    Simulation<1'048'576, 8'192, 2 << 7>().run(gpu_file, cpu_file);
    Simulation<1'048'576, 8'192, 2 << 8>().run(gpu_file, cpu_file);





    //Fixed Particles, fixed block size.
    Simulation<8'192, 8'192, 128>().run(gpu_file, cpu_file);
    Simulation<8'192, 131'072, 128>().run(gpu_file, cpu_file);
    Simulation<8'192, 1'048'576, 128>().run(gpu_file, cpu_file);

    //Fixed Particles, fixed iterations.
    Simulation<131'072, 131'072, 2>().run(gpu_file, cpu_file, false);
    Simulation<131'072, 131'072, 4>().run(gpu_file, cpu_file, false);
    Simulation<131'072, 131'072, 8>().run(gpu_file, cpu_file, false);
    Simulation<131'072, 131'072, 16>().run(gpu_file, cpu_file, false);
    Simulation<131'072, 131'072, 32>().run(gpu_file, cpu_file, false);
    Simulation<131'072, 131'072, 64>().run(gpu_file, cpu_file, false);
    Simulation<131'072, 131'072, 128>().run(gpu_file, cpu_file, false);
    Simulation<131'072, 131'072, 256>().run(gpu_file, cpu_file, false);

  }
  catch (const CudaError& ex)
  {
    printf("Cuda error: %s\n", ex.what());
  }
}

