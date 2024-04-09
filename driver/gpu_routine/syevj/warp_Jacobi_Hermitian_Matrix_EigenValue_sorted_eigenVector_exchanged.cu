
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <iostream>
#include <string>


#define ZERO_IJ 1
using namespace std;

__host__
__device__
float2 zero__ = {0.0f, 0.0f};

__host__ __device__
float2  zero_f2 = {0.0f, 0.0f};



#if PRINT_DEBUG

template <typename T1>
void print_vector(T1 *A, int N)
{
        for (int idx = 0; idx < N; idx++)
        {
                // printf("%f\n", A[idx]);
                cout << A[idx] << endl;
        }
        cout << endl;
}

void print_Int_vector(int *A, int N)
{
        for (int idx = 0; idx < N; idx++)
        {
                printf("%5d ", A[idx]);
        }
}

template <typename T1>
void print_matrix(T1 *A, int M, int N, int lda)
{
        for (int i = 0; i < M; i++)
        {
                for (int j = 0; j < N; j++)
                {
                        // printf("%7.4f ", A[i + j*lda]);
                        cout << A[i + j * lda] << " ";
                }
                // printf("\n");
                cout << endl;
        }
}
template <typename T2>
void print_matrix_complex(T2 *A, int M, int N, int lda)
{
        for (int i = 0; i < M; i++)
        {
                for (int j = 0; j < N; j++)
                { //*(sizeof(complex<float>)/sizeof(float))
                        T2 tmp;
                        tmp = A[i + j * lda];
                        cout << "(" << tmp.x << " + " << tmp.y << "*j) "
                        // printf("(%7.4f, %7.4f)", tmp.x, tmp.y);
                }
                cout << endl;
                // printf("\n");
        }
}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float signf_thread(float a)
{
        union f_i
        {
                float a;
                int x;
        };
        union f_i tmp;
        tmp.a = a;

        return (tmp.x >> 31) ? -1.0f : 1.0f;
}

__device__ void Schur2_Hermite_thread(float Aii, float Ajj, float2 Aij, float norm_Aij, float2 &cC, float2 &s_plus, float2 &s_neg, float &t)
{
        float Aii_Ajj = Aii - Ajj;
        t = 2.0f * norm_Aij * signf_thread(Aii_Ajj) / (fabs(Aii_Ajj) + sqrt(Aii_Ajj * Aii_Ajj + 4.0f * norm_Aij * norm_Aij));
        float c = 1.0f / sqrt(1 + t * t);
        float s = c * t;
        float sDNAij; // s/|Aij|

        sDNAij = s / norm_Aij;
        s_plus = make_cuFloatComplex(sDNAij * Aij.x, sDNAij * Aij.y);
        s_neg = cuConjf(s_plus);
        cC = make_cuFloatComplex(c, 0.0f);
}

__device__ void right_rotate_H_thread(float2 *A, int n, int lda, float2 cC, float2 s_neg, float2 s_plus, int i, int j)
{
        // for (int r = 0; r < n; r++) {// right_rotate  i-th column  j-th column  changed
        unsigned r = threadIdx.x;
        float2 Ari, Arj;

        Ari = A[r + i * lda];
        Arj = A[r + j * lda];
        // if(r!=i)
        A[r + i * lda] = cuCaddf(cuCmulf(Ari, cC), cuCmulf(Arj, s_neg)); // Ari * c    + Arj * S-
        // if(r!=j)
        A[r + j * lda] = cuCsubf(cuCmulf(Arj, cC), cuCmulf(Ari, s_plus)); // Ari *(-S+) + Arj * c
        //}
}

__device__ void left_rotate_H_thread(float2 *A, int n, int lda, float2 cC, float2 s_neg, float2 s_plus, int i, int j)
{
        // for (int r = 0; r < n; r++) {// left_rotate i-th row   j-th row changed
        unsigned r = threadIdx.x;
        float2 Air, Ajr;

        Air = A[i + r * lda];
        Ajr = A[j + r * lda];

        A[i + r * lda] = cuCaddf(cuCmulf(cC, Air), cuCmulf(s_plus, Ajr)); //  c * Air   + S+ * Ajr
        A[j + r * lda] = cuCsubf(cuCmulf(cC, Ajr), cuCmulf(s_neg, Air));  // -S-* Air   +  c * Ajr
        //}
}

__device__ void set_V_2_E_thread(unsigned n, float2 *V_d, int ldv)
{
        float2 one_ = make_cuFloatComplex(1.0f, 0.0f);
        float2 zero_ = make_cuFloatComplex(0.0f, 0.0f);

        unsigned tid = threadIdx.x;

        for (unsigned j = 0; j < n; j++)
        {
                V_d[tid + j * ldv] = zero_;
        }

        V_d[tid + tid * ldv] = one_;
}

__device__ void sort_thread(float *sig, int N)
{

        int x = threadIdx.x;
        extern __shared__ float s_a[];
        if (x < N)
                s_a[x] = sig[x];
        __syncthreads();

        for (int i = 0; i < N; i++)
        {

                int j = i % 2;
                int idx = 2 * x + j;
                if (idx + 1 < N && s_a[idx] > s_a[idx + 1])
                {
                        float tmp = s_a[idx];
                        s_a[idx] = s_a[idx + 1];
                        s_a[idx + 1] = tmp;
                }
                __syncthreads();
        }
        if (x < N)
                sig[x] = s_a[x];
        __syncthreads();
}

__device__ void sort_value_id_thread(float *sig, int N, float *s_a, unsigned *ids)
{

        int x = threadIdx.x;
        // extern __shared__ float s_a[];
        if (x < N)
        {
                s_a[x] = sig[x];
                ids[x] = x;
        }
        __syncthreads();

        for (int i = 0; i < N; i++)
        {

                int j = i % 2;
                int idx = 2 * x + j;
                if (idx + 1 < N && s_a[idx] > s_a[idx + 1])
                {
                        float tmp = s_a[idx];
                        s_a[idx] = s_a[idx + 1];
                        s_a[idx + 1] = tmp;
                        unsigned tmpid = ids[idx];
                        ids[idx] = ids[idx + 1];
                        ids[idx + 1] = tmpid;
                }
                __syncthreads();
        }
        if (x < N)
                sig[x] = s_a[x];
        __syncthreads();
}
__global__ void hermite_jacobi_kernel(float2 *A, int n, int lda, bool cal_Vectors, float2 *V, int ldv, float *Sig, float eps, unsigned jt)
{
        unsigned tid = threadIdx.x;
        if (tid < n)                set_V_2_E_thread(n, V, ldv);

        unsigned count = 0;

        while (count <jt )//jt
        {
                count++;
                for (int i = 0; i < n - 1; i++)
                {
                        for (int j = i + 1; j < n; j++)
                        {

                                float s, c, t;
                                float2 Aij;
                                float norm_Aij;
                                float Aii, Ajj;
                                float2 s_plus, s_neg;
                                float2 cC;

                                Aii = A[i + i * lda].x;
                                Ajj = A[j + j * lda].x;
                                Aij = A[i + j * lda];

                                norm_Aij = cuCabsf(Aij);
                                if (norm_Aij < eps)
                                        continue;

                                Schur2_Hermite_thread(Aii, Ajj, Aij, norm_Aij, cC, s_plus, s_neg, t);

                                if (tid < n)
                                {
                                        right_rotate_H_thread(A, n, lda, cC, s_neg, s_plus, i, j);
                                        left_rotate_H_thread(A, n, lda, cC, s_neg, s_plus, i, j);
                                }
#if ZERO_IJ
                                if (tid == 63)
                                { // third warp
                                        A[i + j * lda] = zero__;
                                        A[j + i * lda] = zero__;
                                        // float A_i_i = c * c * Aii + 2.0f * c * s * norm_Aij + s * s * Ajj;
                                        //  printf("A[i,i] %d= %f\n", count, A[i+i*lda].x);
                                        //  printf("A[i,i] %d= %f\n", count, A_i_i);
                                        float a_ii_look = Aii + t * norm_Aij;
                                        // printf("a_ii_ %d = %f\n", count, a_ii_look);

                                        // float A_j_j = s * s * Aii - 2.0f * c * s * norm_Aij + c * c * Ajj;
                                        //  printf("A[j,j] %d= %f\n", count, A[j + j * lda].x);
                                        //  printf("A[j,j] %d= %f\n", count, A_j_j);
                                        float a_jj_look = Ajj - t * norm_Aij;
                                        // printf("a_jj_ %d = %f\n", count, a_jj_look);

                                        // A[i + i * lda].x = A_i_i;        A[i + i * lda].y = 0.0f;
                                        // A[j + j * lda].x = A_j_j;        A[j + j * lda].y = 0.0f;
                                        A[i + i * lda].x = a_ii_look;
                                        A[i + i * lda].y = 0.0f;
                                        A[j + j * lda].x = a_jj_look;
                                        A[j + j * lda].y = 0.0f;
                                }
                //__syncthreads();
//                if(threadIdx.x == 0) printf("count = %d\n", count);
#endif
                                if (cal_Vectors)
                                { // another warp do this
                                        if (tid < n)
                                                right_rotate_H_thread(V, n, ldv, cC, s_neg, s_plus, i, j);
                                }
                //__syncthreads();
                        }

                }
        }

        // for (int dia = 0; dia < n; dia++) {
        if (tid < n)
        {
                Sig[tid] = A[tid + tid * lda].x;
        }

#if 1 
        // sort Sigma////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        int N = n;
        unsigned x = tid; // threadIdx.x;
        __shared__ unsigned ids[64];
        __shared__ float s_a[64];

        sort_value_id_thread(Sig, N, s_a, ids);

        for (int col = 0; col < n; col++)
        {
                if (tid < n)
                {
                        A[tid + col * lda] = V[tid + ids[col] * ldv];
                }

        }
#endif
}

void gpu_hermite_jacobi(float2 *A_d, int n, int lda, bool cal_Vectors, float2 *V_d, int ldv, float *Sig, float eps, unsigned jt)
{
        dim3 grid_;
        dim3 block_;

        block_.x = 32;
        //__global__ void hermite_jacobi_kernel(float2* A, int n, int lda, bool cal_Vectors, float2* V, int ldv, float* Sig, float eps, unsigned jt)
#if MEASURE_TIME
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        //
        cudaEventRecord(start);
#endif
        hermite_jacobi_kernel<<<grid_, block_, 0, NULL>>>(A_d, n, lda, cal_Vectors, V_d, ldv, Sig, eps, jt);
    cudaDeviceSynchronize();
#if MASURE_TIME
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        //
        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, end);
        std::cout << "CUDA Kernel time: " << time_ms << " ms" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(end);
        std::cout << "end gpu 1" << std::endl;
#endif
}





cusolverStatus_t cusolverDnCheevj_(
        cusolverEigMode_t jobz, // CUSOLVER_EIG_MODE_NOVECTOR     CUSOLVER_EIG_MODE_VECTOR
        cublasFillMode_t uplo,  // CUBLAS_FILL_MODE_LOWER    CUBLAS_FILL_MODE_UPPER     CUBLAS_FILL_MODE_FULL
        int N,                                  // n
        cuComplex *A,                   // A
        int lda,                                // lda
        float *W,                               // float *W= nullptr;                   cudaMalloc((void**)&W, N*sizeof(float));
        cuComplex *work,                // cuComplex *work=nullptr;             cudaMalloc((void**)&work, lwork);
        int lwork,                              // 3*N*(N+5)
        int *info)                              // 0: fine;   -i: the i-th parameter failed;   i:  i-th off-diagonal element cannot converge to zero;
{
        // return cusolverDnXXevd<float, cuComplex>(handle, jobz, uplo, N, A, lda, W, work, lwork, info);
        cudaMemset(info, 0x00, sizeof(int));

        bool cal_Vectors = (jobz == CUSOLVER_EIG_MODE_VECTOR);
    printf("LL::0000001 cal_Vectors = %d\n", cal_Vectors);
        int ldv = N;
    //void gpu_hermite_jacobi(float2 *A_d, int n, int lda, bool cal_Vectors, float2 *V_d, int ldv, float *Sig, float eps, unsigned jt)
    gpu_hermite_jacobi(A, N, lda, cal_Vectors, work, ldv, W, 0.00000001f, 7);
        //new gpu_Cheevd_Jacobi(N, A, lda, cal_Vectors, work, ldv, W, 0.00001f,70);
    cudaDeviceSynchronize();
printf("LL::0000100\n");
        return CUSOLVER_STATUS_SUCCESS;
}









// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)



void print_complex_matrix(cuComplex* A, int m, int n, int lda)
{
    for(int i=0; i<(m<12? m:12); i++){
        for(int j=0; j<(n<12? n:12); j++){
            printf("(%6.3f+(", A[i + j*lda].x);
            printf("%6.3f*j)), ", A[i + j*lda].y);
        }
        printf("; ...\n");
    }
}

void init_Hermitian_matrix(cuComplex *A, int m, int n ,int lda, int seed)
{
    srand(seed);

    for(int i=0; i<lda; i++){
        for(int j=0; j<m; j++){
            if(i<=j){
                A[i + j*lda].x = (rand()%2000)/1000.0f;
                A[i + j*lda].y = (rand()%2000)/1000.0f;
                if(i==j)
                    A[i + j*lda].y = 0.0f;
            }else{
                A[i + j*lda].x = A[j + i*lda].x;
                A[i + j*lda].y = -A[j + i*lda].y;
            }
        }
    }
}

void complex_gemm_NT(cuComplex *A, int lda, cuComplex *B, int ldb, cuComplex *C, int ldc, int M, int N, int K)
{
    cuComplex zero_c;
    zero_c.x = 0.0f;
    zero_c.y = 0.0f;

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            cuComplex sigma = zero_c;
            for(int k=0; k<K; k++){
                sigma = cuCaddf(sigma, cuCmulf(A[i + k*lda], cuConjf(B[j+ k*ldb])));
            }
            C[i + j*ldc] = sigma;
        }
    }
}










int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    syevjInfo_t syevj_params = NULL;

    const int m = 7;
    const int lda = m;

    cuComplex *A = nullptr;
    A = (cuComplex*)malloc(lda*m*sizeof(cuComplex));
    cuComplex *V = nullptr;
    V = (cuComplex*)malloc(lda*m*sizeof(cuComplex));
    float *W = nullptr;
    W = (float*)malloc(m*sizeof(float));

    init_Hermitian_matrix(A, m, m, lda, 2024);

    cuComplex *d_A = nullptr;
    float *d_W = nullptr;
    int *devInfo = nullptr;
    cuComplex *d_work = nullptr;
    int lwork = 4096;
    int info_gpu = 0;

    /* configuration of syevj  */
    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    //const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    /* numerical results of syevj  */
    double residual = 0;
    int executed_sweeps = 0;

    printf("tol = %E, default value is machine zero \n", tol);
    printf("max. sweeps = %d, default value is 100\n", max_sweeps);

    printf("A = (matlab base-1)\n");
    //print_matrix(m, m, A, lda);
    print_complex_matrix(A, m, m, lda);
    printf("=====\n");

    /* step 3: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cuComplex) * lda * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(float) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(cuComplex) * lda * m, cudaMemcpyHostToDevice));

    printf("LL:: lwork = %d\n", lwork);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(cuComplex) * lwork));

    /* step 5: compute eigen-pair   */
    CUSOLVER_CHECK(cusolverDnCheevj_( jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo));
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(V, d_A, sizeof(cuComplex) * lda * m, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(W, d_W, sizeof(float) * m, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    //CUDA_CHECK(cudaStreamSynchronize(stream));

    if (0 == info_gpu) {
        printf("syevj converges \n");
    } else if (0 > info_gpu) {
        printf("%d-th parameter is wrong \n", -info_gpu);
        exit(1);
    } else {
        printf("WARNING: info = %d : syevj does not converge \n", info_gpu);
    }

    printf("Eigenvalue = (matlab base-1), ascending order\n");
    for (int i = 0; i < m; i++) {
        printf("W[%d] = %E\n", i + 1, W[i]);
    }
#if 1
    printf("V = (matlab base-1)\n");
    print_complex_matrix(V, m, m, lda);
    printf("=====\n");
#endif

    cuComplex *E = nullptr;
    E = (cuComplex*)malloc(m*m*sizeof(cuComplex));

    //void complex_gemm_NT(cuComplex *A, int lda, cuComplex *B, int ldb, cuComplex *C, int ldc, int M, int N, int K)
    complex_gemm_NT(V, lda, V, lda, E, m, m, m, m);
    printf("E =\n");
    print_complex_matrix(E, m, m, m);

    printf("residual |A - V*W*V**H|_F = %E \n", residual);
    printf("number of executed sweeps = %d \n", executed_sweeps);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}






