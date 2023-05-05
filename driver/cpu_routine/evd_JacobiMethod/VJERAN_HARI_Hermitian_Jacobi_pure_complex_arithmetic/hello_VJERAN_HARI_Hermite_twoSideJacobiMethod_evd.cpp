// hello_HJ_world.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <random>
#include <iostream>
#include <string.h>
#include "cuComplex.h"

using namespace std;

#define NA 5

void print_matrix(float* A, int M, int N, int lda);
void print_vector(float* A, int n);
void print_complex_matrix(int M, int N, float2* A, int lda);
void print_complex_vector(float2* A, int n);


void complex_gemm(int M, int N, int K, float2* A, int lda, float2* B, int ldb, float2* C, int ldc);
void complex_matrix_conjugate_transposition(int n, float2* A, int lda, float2* B, int ldb);

void set_V_2_E(unsigned n, float2* V, int ldv)
{
    float2 one_ = make_cuFloatComplex(1.0f, 0.0f);
    float2 zero_ = make_cuFloatComplex(0.0f, 0.0f);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            V[i + j * ldv] = zero_;
        }
    }

    for (int dia = 0; dia < n; dia++)
    {
        V[dia + dia * ldv] = one_;
    }
}











void norm_max_ij(float2* A, int n, int lda, int &p, int &q)
{
    float max = 0.0f;
    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            float tmp;
            tmp = cuCabsf(A[i + j * lda]);
            if (tmp > max){
                max = tmp;
                p = i;
                q = j;
            }
        }
    }
}

float signf(float a)
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
void hermite_jacobi(float2* A, int n, int lda, bool cal_Vectors, float2* V, int ldv, float* Sig, float eps, unsigned jt)
{
    set_V_2_E(n, V, ldv);
    printf("V=E=\n");
    print_complex_matrix(n, n, V, ldv);
    unsigned count = 0;

    while (count < jt) {
        count++;
        int i, j;

        i = 0;
        j = 1;
        norm_max_ij(A, n, lda, i, j);

        float s, c, t;
        float2 Aij;
        float norm_Aij;
        float Aii, Ajj;
        float Aii_Ajj;
        float2 s_plus, s_neg;

        Aii = A[i + i * lda].x;
        Ajj = A[j + j * lda].x;
        Aii_Ajj = Aii - Ajj;
        Aij = A[i + j * lda];
        norm_Aij = cuCabsf(Aij);

        t = 2.0f * norm_Aij * signf(Aii_Ajj) / (fabs(Aii_Ajj) + sqrt(Aii_Ajj * Aii_Ajj + 4.0f * norm_Aij * norm_Aij));
        c = 1.0f / sqrt(1 + t * t);
        s = c * t;
        float sDNAij;//s/|Aij|
        sDNAij = s / norm_Aij;
        s_plus = make_cuFloatComplex(sDNAij * Aij.x, sDNAij * Aij.y);
        s_neg = cuConjf(s_plus);

        //A[i + i * lda].x = Aii + t * norm_Aij; A[j + j * lda].x = Ajj - t * norm_Aij;
        float2 cC, sC;
        cC = make_cuFloatComplex(c, 0.0f);
        sC = make_cuFloatComplex(s, 0.0f);

        for (int r = 0; r < n; r++) {// right_rotate  i-th column  j-th column  changed
            float2 Ari, Arj;
            Ari = A[r + i * lda];
            Arj = A[r + j * lda];
            //if(r!=i)
            A[r + i * lda] = cuCaddf(cuCmulf(Ari, cC), cuCmulf(Arj, s_neg));//Ari * c    + Arj * S-
            //if(r!=j)
            A[r + j * lda] = cuCsubf(cuCmulf(Arj, cC), cuCmulf(Ari, s_plus));//Ari *(-S+) + Arj * c
        }
        for (int r = 0; r < n; r++) {// left_rotate i-th row   j-th row changed
            float2 Air, Ajr;
            Air = A[i + r * lda];
            Ajr = A[j + r * lda];

            A[i + r * lda] = cuCaddf(cuCmulf(cC, Air), cuCmulf(s_plus, Ajr));//  c * Air   + S+ * Ajr
            A[j + r * lda] = cuCsubf(cuCmulf(cC, Ajr), cuCmulf(s_neg, Air));// -S-* Air   +  c * Ajr
        }

        if (cal_Vectors) {
            for (int r = 0; r < n; r++) {// right_rotate  i-th column  j-th column  changed
                float2 Vri, Vrj;
                Vri = V[r + i * lda];
                Vrj = V[r + j * lda];
                V[r + i * lda] = cuCaddf(cuCmulf(Vri, cC), cuCmulf(Vrj, s_neg));//Ari * c    + Arj * S-
                V[r + j * lda] = cuCsubf(cuCmulf(Vrj, cC), cuCmulf(Vri, s_plus));//Ari *(-S+) + Arj * c
            }
        }
    }
}


void init_complex_matrix(int M, int N, float2* A, int lda) {
    random_device rd;
    mt19937 gen(rd());
    float range_start = -3.3f;
    float range_end = 3.3f;

    uniform_real_distribution<> dis(range_start, range_end);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i + j * lda].x = dis(gen);
            A[i + j * lda].y = dis(gen);
        }
    }
}


void init_Hermitian_matrix(int N, float2* A, int lda) {

    float2 *a;
    a = (float2*)malloc(N * N * sizeof(float2));
    init_complex_matrix(N, N, a, N);
    float2* at;// [NA * NA] ;
    at = (float2*)malloc(N * N * sizeof(float2));
    complex_matrix_conjugate_transposition(N, a, N, at, N);
    complex_gemm(N, N, N, at, N, a, N, A, N);

}

int main()
{
    float2 A[NA * NA];/** =
    {
            {1.8937,  0.0000}, {1.4574, 0.2006}, {1.6041, -0.2653}, {2.3225, -0.3367},
            {1.4574, -0.2006}, {1.6985, 0.0000}, {1.6143, -0.2756}, {2.4423, -0.4715},
            {1.6041,  0.2653}, {1.6143, 0.2756}, {2.2980,  0.0000}, {2.3027,  0.2985},
            {2.3225,  0.3367}, {2.4423, 0.4715}, {2.3027, -0.2985}, {4.2158,  0.0000}
    };*/

    init_Hermitian_matrix(NA, A, NA);

    float2 A_con[NA * NA];
    memcpy(A_con, A, NA * NA * sizeof(float2));

    int n = NA;
    int lda = n;
    float2 V[NA * NA];

    int ldv = n;

    bool cal_eiV = true;
    float Sig[NA];
    float eps = 1.0e-8;
    unsigned jt = 300;
    printf("A=\n");
    print_complex_matrix(n, n, A, lda);

    hermite_jacobi(A, n, lda, cal_eiV, V, ldv, Sig, eps, jt);

    printf("Sigma(A)=\n");
    print_complex_matrix(n, n, A, lda);
    printf("EigenVectors=\n");
    print_complex_matrix(n, n, V, ldv);

    //void complex_gemm(int M, int N, int K, float2 * A, int lda, float2 * B, int ldb, float2 * C, int ldc);
    //void complex_matrix_conjugate_transposition(int n, float2 * A, int lda, float2 * B, int ldb);
    float2 Vt[NA * NA];
    int ldvt = n;
    complex_matrix_conjugate_transposition(n, V, ldv, Vt, ldvt);
    printf("V'=\n");
    print_complex_matrix(n, n, Vt, ldvt);

    float2 VtxA[NA * NA];
    int ldvtxa = n;
    float2 VtxAxV[NA * NA];
    int ldvtxaxv = n;

    complex_gemm(n, n, n, Vt, ldvt, A_con, lda, VtxA, ldvtxa);
    complex_gemm(n, n, n, VtxA, ldvtxa, V, ldv, VtxAxV, ldvtxaxv);

    printf("VtxAxV=\n");
    print_complex_matrix(n, n, VtxAxV, ldvtxaxv);


    float2 E[NA * NA];
    int lde = n;
    complex_gemm(n, n, n, Vt, ldvt, V, ldv, E, lde);
    printf("E=VtxV=\n");
    print_complex_matrix(n, n, E, lde);

    std::cout << "Hello World!\n";
}










void print_vector(float* A, int n) {
    for (int i = 0; i < n; i++)
        printf("%8.5f", A[i]);
}

void print_matrix(float* A, int M, int N, int lda) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf(" %8.5f", A[i + j * lda]);
        printf("\n");
    }
    printf("\n");
}

void print_complex_vector(float2* A, int n)
{
    for (int i = 0; i < n; i++) {
        printf("%7.4f + %7.4f*i  ", A[i].x, A[i].y);
    }
}

void print_complex_matrix(int M, int N, float2* A, int lda)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%9.6f + %9.6f*i ,", A[i + j * lda].x, A[i + j * lda].y);
        }
        printf("\n");
    }

}


void complex_matrix_conjugate_transposition(int n, float2* A, int lda, float2* B, int ldb)
{
    for (int i = 0; i < n; i++) {
        for(int j=0; j<n; j++)
        {
            B[i + j * ldb] = cuConjf(A[j + i * lda]);
        }
    }
}

void complex_gemm(int M, int N, int K, float2* A, int lda, float2* B, int ldb, float2* C, int ldc)
{
    float2 zero_ = make_cuFloatComplex(0.0f, 0.0f);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float2 sigma = zero_;

            for (int k = 0; k < K; k++) {
                sigma = cuCaddf(sigma, cuCmulf(A[i + k * lda], B[k + j * ldb]));
            }
            C[i + j * ldc] = sigma;
        }
    }
}





/*
octave:37> A
A =

   1.8937 + 0.0000i   1.4574 - 0.2006i   1.6041 + 0.2653i   2.3225 + 0.3367i
   1.4574 + 0.2006i   1.6985 + 0.0000i   1.6143 + 0.2756i   2.4423 + 0.4715i
   1.6041 - 0.2653i   1.6143 - 0.2756i   2.2980 + 0.0000i   2.3027 - 0.2985i
   2.3225 - 0.3367i   2.4423 - 0.4715i   2.3027 + 0.2985i   4.2158 + 0.0000i

octave:38> [u d]=eig(A)
u =

  -0.04847 - 0.31989i  -0.82406 + 0.09126i  -0.10042 + 0.14573i   0.41784 + 0.04485i
  -0.75933 - 0.14197i   0.31451 - 0.31403i  -0.06196 + 0.15423i   0.41544 + 0.07428i
   0.20242 + 0.28441i   0.16901 + 0.16300i  -0.75165 - 0.24049i   0.44600 - 0.03567i
   0.42038 + 0.00000i   0.24482 + 0.00000i   0.56413 + 0.00000i   0.66716 + 0.00000i

d =

Diagonal Matrix

   0.027868          0          0          0
          0   0.447434          0          0
          0          0   0.809093          0
          0          0          0   8.821602

octave:39>




*/

