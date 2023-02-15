
#ifndef __CEC2023_TESTS__
#define __CEC2023_TESTS__

#include <Zakharov.cuh>
#include <Rosenbrock.cuh>
#include <StepRastrigin.cuh>
#include <SchafferF7.cuh>
#include <Levy.cuh>
#include <Hybrid01.cuh>
#include <Hybrid02.cuh>
#include <Hybrid03.cuh>
#include <Composition01.cuh>
#include <Composition02.cuh>
#include <Composition03.cuh>
#include <Composition04.cuh>
#include <Benchmark.cuh>

template <typename T>
Benchmark<T> *createBenchmark(int n, int pop_size, int func_id, char *ShuffleFileName, char *ShiftFileName, char *MatrixFileName)
{

      printf("Evaluating ");
      switch (func_id)
      {
      case F_ZAKHAROV:
            printf("Zakharov Function\n");
            return new Zakharov<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_ROSENBROCK:
            printf("Rosenbrock Function\n");
            return new Rosenbrock<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_SCHAFFER_F7:
            printf("Schaffer F7 Function\n");
            return new SchafferF7<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_STEP_RASTRIGIN:
            printf("Step Rastrigin Function\n");
            return new StepRastrigin<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_LEVY:
            printf("Levy Function\n");
            return new Levy<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_HYBRID1:
            printf("Hybrid Function 01\n");
            return new Hybrid01<T>(n, pop_size, ShuffleFileName, ShiftFileName, MatrixFileName);
      case F_HYBRID2:
            printf("Hybrid Function 02\n");
            return new Hybrid02<T>(n, pop_size, ShuffleFileName, ShiftFileName, MatrixFileName);
      case F_HYBRID3:
            printf("Hybrid Function 03\n");
            return new Hybrid03<T>(n, pop_size, ShuffleFileName, ShiftFileName, MatrixFileName);
      case F_COMPOSITION1:
            printf("Composition Function 01\n");
            return new Composition01<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_COMPOSITION2:
            printf("Composition Function 02\n");
            return new Composition02<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_COMPOSITION3:
            printf("Composition Function 03\n");
            return new Composition03<T>(n, pop_size, ShiftFileName, MatrixFileName);
      case F_COMPOSITION4:
            printf("Composition Function 04\n");
            return new Composition04<T>(n, pop_size, ShiftFileName, MatrixFileName);
      default:
            return NULL;
      }
}

int get_optimum(int func_id){
      switch(func_id){
            case F_ZAKHAROV:
                  return C_ZAKHAROV;
            case F_ROSENBROCK:
                  return C_ROSENBROCK;
            case F_SCHAFFER_F7:
                  return C_SCHAFFER_F7;
            case F_STEP_RASTRIGIN:
                  return C_RASTRIGIN;
            case F_LEVY:
                  return C_LEVY;
            case F_HYBRID1:
                  return C_HYBRID1;
            case F_HYBRID2:
                  return C_HYBRID2;
            case F_HYBRID3:
                  return C_HYBRID3;
            case F_COMPOSITION1:
                  return C_COMPOSITION1;
            case F_COMPOSITION2:
                  return C_COMPOSITION2;
            case F_COMPOSITION3:
                  return C_COMPOSITION3;
            case F_COMPOSITION4:
                  return C_COMPOSITION4;
            default:
                  return -1;
            }
}

#endif
