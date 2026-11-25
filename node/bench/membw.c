// membw.c -- STREAM-style host memory bandwidth (copy + triad), OpenMP, first-touch init (31.10., gpu02).
// gcc -O3 -fopenmp -march=native -o membw membw.c && OMP_PROC_BIND=spread ./membw [GiB_per_array=2] [iters=10]
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  size_t gib = argc > 1 ? atoi(argv[1]) : 2; int iters = argc > 2 ? atoi(argv[2]) : 10;
  size_t n = (gib << 30) / sizeof(double);
  double *a = malloc(n * 8), *b = malloc(n * 8), *c = malloc(n * 8);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }
  double best_copy = 0, best_triad = 0;
  for (int it = 0; it < iters; it++) {
    double t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) c[i] = a[i];
    double copy = 2.0 * n * 8 / (omp_get_wtime() - t) / 1e9;
    t = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
    double triad = 3.0 * n * 8 / (omp_get_wtime() - t) / 1e9;
    if (copy > best_copy) best_copy = copy;
    if (triad > best_triad) best_triad = triad;
  }
  printf("threads %d, %zu GiB x 3 arrays: copy %.1f GB/s, triad %.1f GB/s (best of %d)\n",
         omp_get_max_threads(), gib, best_copy, best_triad, iters);
  return a[n / 2] == 0;  // keep the loops
}
