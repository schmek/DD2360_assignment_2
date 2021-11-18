
#include "exercises.cuh"
#include <functional>

void run_exercise(const char* name, std::function<void()> fn)
{
  printf("==================== Running exercise %s ================ \n\n", name);
  fn();
}


int main()
{
  //run_exercise("1", &exercise_1);
  //run_exercise("2", &exercise_2);
  run_exercise("3", &exercise_3);
  //run_exercise("4", &exercise_4);
  return EXIT_SUCCESS;
}
