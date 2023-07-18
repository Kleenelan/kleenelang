#include <algorithm>
#include <stdio.h>
#include <random>

using namespace std;

void init_vector(float* A, int n);
template<bool ROW_MAJOR>
void print_matrix(float* A, int m, int n, int lda);




typedef float (Array_load)[32][2 *64];
typedef float (Array_calc)[32][2][64];

void cpu_add_A_B_test(int m, int n, float *A, float *B)
{
  float* C = nullptr;
  C = (float*)malloc(m*n*sizeof(float));
  for(int i=0; i < m; i++)
  {
    for(int j=0; j<n; j++)
    {
      C[i*n + j] = A[ i*n + j] + B[i*n + j];
    }
  }

  printf("\nC(cpu) = A+B =\n");
  print_matrix<true>(C, m, n, n);

  ////////////////////////////////////////////////////////////////////////////////////////
  float* C2 = nullptr;
  C2 = (float*)malloc(m*n*sizeof(float));
  Array_load* C_ld, *A_ld, *B_ld;
  C_ld = (Array_load*)C2;
  A_ld = (Array_load*)A;
  B_ld = (Array_load*)B;

  for(int i=0; i < m; i++)
  {
    for(int j=0; j<n; j++)
    {
      (*C_ld)[i][j] =(*A_ld)[i][j] + (*B_ld)[i][j];
    }
  }

  printf("\nC2(cpu) = A_ld + B_ld =\n");
  print_matrix<true>(C2, m, n, n);
  ////////////////////////////////////////////////////////////////////////////////////////
  float* C3 = nullptr;
  C3 = (float*)malloc(m*n*sizeof(float));
  Array_calc* C_cl, *A_cl, *B_cl;
  C_cl = (Array_calc*)C3;
  A_cl = (Array_calc*)A;
  B_cl = (Array_calc*)B;

  for(int i=0; i < m; i++)
  {
    for(int j=0; j<2; j++)
    {
      for(int k=0; k<n/2; k++)
      {
        (*C_cl)[i][j][k] =(*A_cl)[i][j][k] + (*B_cl)[i][j][k];
      }
    }
  }

  printf("\nC3(cpu) = A_cl + B_cl =\n");
  print_matrix<true>(C3, m, n, n);
    
}


__global__
void add_A_B_kernel(int m, int n, float *A, float *B, float *C)
{
  __shared__ Array_load sA_ld;
  __shared__ Array_load sB_ld;


  //__shared__ float sA_cl[32][2][64];
  //__shared__ float sB_cl[32][2][64];
  unsigned i = threadIdx.z;
  unsigned j = threadIdx.y*blockDim.x + threadIdx.x;//0~127

  sA_ld[i][j] = A[ i*n + j];
  sB_ld[i][j] = B[ i*n + j];

  Array_calc* sA_cl = (Array_calc*)sA_ld;
  Array_calc* sB_cl = (Array_calc*)sB_ld;
  Array_calc* gC_cl = (Array_calc*)C;

  float a, b;
  a = (*sA_cl)[threadIdx.z][threadIdx.y][threadIdx.x];
  b = (*sB_cl)[threadIdx.z][threadIdx.y][threadIdx.x];
  if(threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0)
    printf("\nLL:: A[0] = %7.3f, B[0] = %7.3f, %lu, %lu, %lu, %lu\n", a, b, sA_cl, sB_cl, sA_ld, sB_ld);  

  (*gC_cl)[threadIdx.z][threadIdx.y][threadIdx.x] = a + b;

}


int main(void)
{

  int m = 32;
  int n = 128;
  float* A_h = nullptr;
  float* B_h = nullptr;
  float* C_h = nullptr;

  A_h = (float*)malloc(m*n*sizeof(float));
  B_h = (float*)malloc(m*n*sizeof(float));
  C_h = (float*)malloc(m*n*sizeof(float));

  float* A_d = nullptr;
  float* B_d = nullptr;
  float* C_d = nullptr;

  cudaMalloc((void**)&A_d, m*n*sizeof(float)); 
  cudaMalloc((void**)&B_d, m*n*sizeof(float));
  cudaMalloc((void**)&C_d, m*n*sizeof(float)); 

  init_vector(A_h, m*n);    printf("A_h = \n"); print_matrix<true>(A_h, m, n, n);
  init_vector(B_h, m*n);    printf("B_h = \n"); print_matrix<true>(B_h, m, n, n);
  //init_vector(C_h, m*n);
  //void cpu_add_A_B_test(int m, int n, float *A, float *B)
  cpu_add_A_B_test(m, n, A_h, B_h);


  cudaMemcpy(A_d, A_h, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, m*n*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(C_d, A_h, m*n*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid_;
  dim3 block_;

  block_.x = 64;
  block_.y = 2;
  block_.z = 32;

  // Perform C = (A + B)
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  
//__global__void add_A_B_kernel(int m, int n, float *A, float *B, float *C)
  add_A_B_kernel<<<grid_, block_, 0, NULL>>>(m, n, A_d, B_d, C_d);

#if 1
  cudaMemcpy(C_h, C_d, m*n*sizeof(float), cudaMemcpyDeviceToHost);  printf("C_h result =\n");  print_matrix<true>(C_h, m, n, n);
#endif

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}



























void init_vector(float* A, int n){
    random_device rd;
    mt19937 gen(rd());
    float range_start =-1;
    float range_end = 1;
    uniform_real_distribution<> dis(range_start, range_end);

    for(int i=0; i<n; i++)
        A[i] = dis(gen);
}

template<bool ROW_MAJOR>
void print_matrix(float* A, int m, int n, int lda)
{
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            int idx;

            idx = ROW_MAJOR? i*lda+j: i + j*lda;
            printf("%7.3f ", A[idx]);
        }
        printf("\n");
    }
}

