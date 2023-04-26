// Jacobi_hermitian_matrix_evd.cpp


#include <iostream>
#include <stdio.h>
#include "cuComplex.h"

using namespace std;

#define NA 4
#define PRINT_DEBUG 1
#define VERI_GEMM 1

/*
float cuCrealf (cuFloatComplex x)
float cuCimagf (cuFloatComplex x)
cuFloatComplex make_cuFloatComplex(float r, float i)
cuFloatComplex cuConjf (cuFloatComplex x)
cuFloatComplex cuCaddf (cuFloatComplex x, cuFloatComplex y)
cuFloatComplex cuCsubf (cuFloatComplex x, cuFloatComplex y)
cuFloatComplex cuCmulf (cuFloatComplex x, cuFloatComplex y)
cuFloatComplex cuCdivf (cuFloatComplex x, cuFloatComplex y)
float cuCabsf (cuFloatComplex x)
*/


#if PRINT_DEBUG

template<typename T1>
void print_vector(T1* A, int N) {
    for (int idx = 0; idx < N; idx++) {
        //printf("%f\n", A[idx]);
        cout << A[idx] << endl;
    }
    cout << endl;
}

void print_Int_vector(int* A, int N) {
    for (int idx = 0; idx < N; idx++) {
        printf("%5d ", A[idx]);
    }
}

template<typename T1>
void print_matrix(T1* A, int M, int N, int lda)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            //printf("%7.4f ", A[i + j*lda]);
            cout << A[i + j * lda] << " ";
        }
        //printf("\n");
        cout << endl;
    }
}
template<typename T2>
void print_matrix_complex(T2* A, int M, int N, int lda) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {//*(sizeof(complex<float>)/sizeof(float))
            T2 tmp;
            tmp = A[i*lda + j];
            //cout<<"("<<tmp.x<<" + "<<tmp.y<<"*j) ";
            printf("(%7.4f, %7.4f)", tmp.x, tmp.y);
        }
        cout << endl;
        //printf("\n");
    }
}

#endif
void set_to_E(float2* V, int n, int ldv)
{
    float2 zero_ = make_cuFloatComplex(0.0f, 0.0f);
    float2 one_ = make_cuFloatComplex(1.0f, 0.0f);


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i + j * ldv] = zero_;
            if (i == j)
                V[i + j * ldv] = one_;
        }
    }


}

void complexGemm(int M, int N, int K, float2* A, int lda, float2* B, int ldb, float2* C, int ldc) 
{
    float2 zero_ = make_cuFloatComplex(0.0f, 0.0f);
    float2 sigma;

    for (int i = 0; i < M; i++) {
        
        for (int j = 0; j < N; j++) {
            sigma = zero_;
            for (int k = 0; k < K; k++) {
                sigma = cuCaddf(sigma, cuCmulf(A[i*lda + k], B[k*ldb + j]));
            }
            C[i*ldc + j] = sigma;//cuCaddf(sigma, C[i+j]);
        }
    }
}

int  evd_Hermite_Jacobi(float2* A, int n, int lda, float2* V, int ldv)
{
    set_to_E(V, n, ldv);
    float2 R[NA * NA];
    int ldr = NA;

#if PRINT_DEBUG
    printf("V(origin)=\n");
    print_matrix_complex<float2>(V, n, n, ldv);
#endif

    while (1 == 1) {
        for (int j = 1; j < n; j++) {
            for (int i = 0; i < j; i++) {
                float x1, y1, x2, y2;
                float2 Hpq, Hpp, Hqq;
                float tan_fe1, tan_fe2;
                float fe1, fe2, th1, th2;

                Hpq = A[i + j * lda];
                Hpp = A[i + i * lda];
                Hqq = A[j + j * lda];

                //tan_fe1 = Hpq.y/Hpq.x;
                fe1 = atan2(Hpq.y, Hpq.x);
                th1 = (2.0f * fe1 - 3.1415926) / 4.0f;
                //tan_fe2 = 2.0f*cuCabsf(Hpq)/(Hpp.x-Hqq.x);//double atan2(double y, double x)返回以弧度表示的 y/x 的反正切。y 和 x 的值的符号决定了正确的象限。
                fe2 = atan2(2.0f * cuCabsf(Hpq), (Hpp.x - Hqq.x));
                th2 = fe2 / 2.0f;

                float2 Rpp, Rpq, Rqp, Rqq;
                float sin_th1, cos_th1, sin_th2, cos_th2;

                sin_th1 = sin(th1);
                cos_th1 = cos(th1);

                sin_th2 = sin(th2);
                cos_th2 = cos(th2);

                Rpp = make_cuFloatComplex(-sin_th1 * sin_th2, -cos_th1 * sin_th2);
                Rpq = make_cuFloatComplex(-cos_th1 * cos_th2, -sin_th1 * cos_th2);
                Rqp = make_cuFloatComplex(cos_th1 * cos_th2, -sin_th1 * cos_th2);
                Rqq = make_cuFloatComplex(-sin_th1 * sin_th2, cos_th2 * sin_th2);

                //make R
                set_to_E(R, n, n);
                R[i + i * ldr] = Rpp;
                R[i + j * ldr] = Rpq;
                R[j + i * ldr] = Rqp;
                R[j + j * ldr] = Rqq;











            }
        }
    }
    return 0;
}


int main()
{





    // verify complex_gemm
#if VERI_GEMM
    float2 A[4 * 3] = {
                                {0.987599, 0.577728}, {0.105383, 0.051115}, {0.416711, 0.957729},
                                {0.231463, 0.725252}, {0.466096, 0.921880}, {0.539319, 0.788323},
                                {0.087162, 0.716588}, {0.351281, 0.384556}, {0.968724, 0.003711},
                                {0.344171, 0.603070}, {0.345267, 0.466700}, {0.780139, 0.070330}
    };
    float2 B[3 * 3] = {
                                {0.000533, 0.070620}, {0.713952, 0.686842}, {0.395393, 0.518629},
                                {0.753261, 0.512059}, {0.372614, 0.297314}, {0.135508, 0.406991},
                                {0.578717, 0.315435}, {0.956620, 0.202800}, {0.198674, 0.829188}
    };

    float2 C[3 * 4];

    int M = 4;
    int N = 3;
    int K = 3;
    int lda = K;
    int ldb = N;
    int ldc = N;

    //void print_matrix_complex(T2 * A, int M, int N, int lda) {
    printf("A=\n");
    print_matrix_complex<float2>(A, M, K, lda);
    printf("B=\n");
    print_matrix_complex<float2>(B, K, N, ldb);

    complexGemm(M, N, K, A, lda, B, ldb, C, ldc);

    printf("C=\n");
    print_matrix_complex<float2>(C, M, N, ldc);
#else
    int n = NA;
    int ldv = n;
    int lda = n;
    float2 A_h[NA * NA] = {
                        (1.9412 , 0.0000), (1.5117 , 0.1657), (1.9372 , 0.3868), (1.3532 , 0.2145),
                        (1.5117 , 0.1657), (2.5386 , 0.0000), (2.5796 , 0.4703), (2.4599 , 0.3037),
                        (1.9372 , 0.3868), (2.5796 , 0.4703), (3.2516 , 0.0000), (2.5948 , 0.5017),
                        (1.3532 , 0.2145), (2.4599 , 0.3037), (2.5948 , 0.5017), (2.8660 , 0.0000)
    };

    float2 V_h[NA * NA];


    evd_Hermite_Jacobi(A_h, n, lda, V_h, ldv);



#endif
    



    return 0;
}


/*
 *
 *
 *octave:7> A'
ans =

   0.000533 - 0.070620i   0.713952 - 0.686842i   0.395393 - 0.518629i
   0.753261 - 0.512059i   0.372614 - 0.297314i   0.135508 - 0.406991i
   0.578717 - 0.315435i   0.956620 - 0.202800i   0.198674 - 0.829188i

octave:8> B'
ans =

   0.987599 - 0.577728i   0.105383 - 0.051115i   0.416711 - 0.957729i
   0.231463 - 0.725252i   0.466096 - 0.921880i   0.539319 - 0.788323i
   0.087162 - 0.716588i   0.351281 - 0.384556i   0.968724 - 0.003711i
   0.344171 - 0.603070i   0.345267 - 0.466700i   0.780139 - 0.070330i

octave:9> B'*A'
ans =

  -0.04801 - 0.84822i   0.53677 - 2.14186i  -0.62701 - 1.32625i
  -0.10861 - 1.57615i  -0.07724 - 2.02236i  -1.14317 - 1.32524i
   0.57658 - 0.78380i   0.51255 - 1.01921i  -0.25671 - 1.32761i
   0.40799 - 0.83976i   0.55344 - 1.16900i  -0.22317 - 1.28156i

 * */

