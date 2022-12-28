#include <stdio.h>
#include <random>
#include <type_traits>
#include <time.h>

#ifndef UTILS
#define UTILS
int next_pows(int n);
int read_dimensions(int argc, char *argv[]);
int read_population(int argc, char *argv[]);
int read_function(int argc, char *argv[]);
int read_shift(int argc, char *argv[]);
int read_rotate(int argc, char *argv[]);
int read_ipb(int argc, char *argv[]);

void print_time(clock_t t, const char msg[]);
void mul_matrix_vector_cpu(double *M, double *V, double *out, int n);
#endif

