
#include "exercises.cuh"




__global__ void print_hw() 
{
  printf("Hello World! My threadId is %d\n", threadIdx.x);
}

void exercise_1()
{
  print_hw << <1, 256 >> > ();
  cudaDeviceSynchronize();
}

