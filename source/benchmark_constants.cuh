#ifndef BENCH_CONSTANTS
#define BENCH_CONSTANTS

#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029

// o = [oi1, oi2, ... , oiD]: shifted global optimum randomly distributed ind [-80, 80]^D
#define OPT_BOUND 80.0

// [-100, 100]^D search range restriction
#define X_BOUND   100.0

#define ROSENBROCK_BOUND    2.048
#define SCHAFFER_F7_BOUND   100.0
#define SCHAFFER_F6_BOUND   100.0
#define RASTRIGIN_BOUND     5.12
// #define ACKLEY_BOUND        32.768 
#define ACKLEY_BOUND        100.0
#define ZAKHAROV_BOUND      100.0
#define BENT_CIGAR_BOUND    100.0
#define HGBAT_BOUND         5.0
#define HAPPYCAT_BOUND      5.0
#define KATSUURA_BOUND      5.0
#define GRIE_ROSEN_BOUND    5.0
#define LEVY_BOUND          100.0
#define ELLIPSIS_BOUND      100.0
#define DISCUS_BOUND        100.0
#define GRIEWANK_BOUND      600.0
#define STEP_RASTRIGIN_BOUND 5.12
#define SCHWEFEL_BOUND      1000.0
#define ESCAFFER6_BOUND   100.0

enum func{
    F_ZAKHAROV,
    F_ROSENBROCK,
    F_SCHAFFER_F6,
    F_STEP_RASTRIGIN,
    F_LEVY,
    F_HYBRID1,
    F_HYBRID2,
    F_HYBRID3,
    F_COMPOSITION1,
    F_COMPOSITION2,
    F_COMPOSITION3,
    F_COMPOSITION4
};

#endif