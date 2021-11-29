#include <math.h>
/* #include <mpi.h> */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void residualc(void *R, void *A, void *B, unsigned int nx, double h) {
  double *r = (double *)R;
  double *a = (double *)A;
  double *b = (double *)B;
  unsigned int i, j;
  h = (double)1 / (h * h);
  for (j = 1; j < nx - 1; j++) {
    for (i = 1; i < nx - 1; i++) {
      r[i + nx * j] =
          b[i + nx * j] -
          (a[i + nx * j - 1] + a[i + nx * j + 1] + a[i + nx * (j + 1)] +
           a[i + nx * (j - 1)] - 4 * a[i + nx * j]) *
              h;
    }
  }
}

int main(void) {
  unsigned int nx = 1000;
  double *r = (double *)calloc(nx * nx, sizeof(double));
  double *a = (double *)calloc(nx * nx, sizeof(double));
  double *b = (double *)calloc(nx * nx, sizeof(double));
  unsigned int i, j, k;
  double h = 0.215156;
  double end, begin, t;

  begin = clock();
  for (k = 0; k < 500; k++) {
    for (j = 1; j < nx - 1; j++) {
      for (i = 1; i < nx - 1; i++) {
        r[i + nx * j] =
            b[i + nx * j] -
            (a[i + nx * j - 1] + a[i + nx * j + 1] + a[i + nx * (j + 1)] +
             a[i + nx * (j - 1)] - 4 * a[i + nx * j]) *
                h;
      }
    }
  }
  end = clock();
  t = (end - begin) / CLOCKS_PER_SEC;
  printf("t = %f \n", t);
  return 0;
}
