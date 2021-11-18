#pragma once
#include "exercises.cuh"

#include <array>

template<typename T, size_t N, size_t BLOCK_SZ>
class DeviceArray
{
private:
  T* m_device_data = nullptr;

public:

  DeviceArray(const std::array<T, N>& x)
  {
    auto M = BLOCK_SZ * ((N + BLOCK_SZ - 1) / BLOCK_SZ);
    CUDA_CHECK(cudaMalloc(&m_device_data, M * sizeof(T))); //Allocating enough memory to fill out each block. 
    CUDA_CHECK(cudaMemcpy(m_device_data, x.data(), N * sizeof(T), cudaMemcpyHostToDevice));
  }

  ~DeviceArray()
  {
    if (!m_device_data)
    {
      return;
    }

    auto err = cudaFree(m_device_data);
    if (err != cudaSuccess)
    {
      printf("Error while deallocating device memory: %s\n", cudaGetErrorString(err));
    }
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  T* device_get() const
  {
    return m_device_data;
  }

  std::array<T, N> host_get() const
  {
    std::array<T, N> ret;
    CUDA_CHECK(cudaMemcpy(ret.data(), m_device_data, N * sizeof(T), cudaMemcpyDeviceToHost));
    return ret;
  }
};

template<typename T, size_t N, size_t BLOCK_SZ>
class DeviceOnlyArray
{
private:
  T* m_device_data = nullptr;

public:

  DeviceOnlyArray()
  {
    auto M = BLOCK_SZ * ((N + BLOCK_SZ - 1) / BLOCK_SZ);
    CUDA_CHECK(cudaMalloc(&m_device_data, M * sizeof(T))); //Allocating enough memory to fill out each block. 
  }

  ~DeviceOnlyArray()
  {
    if (!m_device_data)
    {
      return;
    }

    auto err = cudaFree(m_device_data);
    if (err != cudaSuccess)
    {
      printf("Error while deallocating device memory: %s\n", cudaGetErrorString(err));
    }
  }

  DeviceOnlyArray(const DeviceOnlyArray&) = delete;
  DeviceOnlyArray& operator=(const DeviceOnlyArray&) = delete;

  T* device_get() const
  {
    return m_device_data;
  }
};