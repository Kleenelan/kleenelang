#include <stdio.h>
#include <iostream>

#define NA 16
#define PRINT_CPU 1
#define PRINT_GPU 1


#define ACCURACY_MORE 0
using namespace std;




void print_matrix(float* A, int n, int lda)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            printf("%7.4f ", A[i+j*lda]);
        }
        printf("\n\n");
    }
    printf("\n\n");
}

void set_mat_2_E(float* V, int n, int ldv)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            V[i+j*ldv] = 0.0f;
        }
        V[i+i*ldv] = 1.0f;
    }
}

void symSchur2(float App, float Aqq, float Apq, float &c, float &s, float &t)
{
    if(Apq!=0.0f){
        float tau;

        tau = (Aqq-App)/(2.0f* Apq);
        if(tau>= 0.0f){
            t = 1.0f/(tau + sqrt(1.0f+tau*tau));
        }else{
            t = 1.0f/(tau - sqrt(1.0f+tau*tau));
        }

        c = 1.0f/(sqrt(1.0f+t*t));
        s = c*t;
    }else{
        c = 1.0f;
        s = 0.0f;
    }
}

/*
         |  c  -s |         | c    s |
    J' = |  s   c |     J = | -s   c |
*/


void left_rotation(float* A, int n, int lda, int p, int q, float c, float s)
{// p-th row      q-th row

    for(int j=0; j<n; j++){
        float Apj, Aqj;

        Apj = A[p+j*lda];
        Aqj = A[q+j*lda];
        // A[p j]  =  c*A[p j] - s*A[q j]
        A[p + j*lda] = c*Apj - s*Aqj;// p-th row 
        // A[q j]  =  s*A[p j] + c*A[q j]
        A[q + j*lda] = s*Apj + c*Aqj;// q-th row 
    }
}

void right_rotation(float* A, int n, int lda, int p, int q, float c, float s)
{// p-th col      q-th col
    for(int j=0; j<n; j++){
        float Ajp, Ajq;

        Ajp = A[j + p*lda];
        Ajq = A[j + q*lda];
        // A[p j]  =  c*A[p j] + s*A[q j]
        A[j + p*lda] = c*Ajp - s*Ajq;
        // A[q j]  = -s*A[p j] + c*A[q j]
        A[j + q*lda] = s*Ajp + c*Ajq;
    }
}

void cpu_cycle_Jac_method(float* A, int n, int lda, float* Sig, bool cal_vector, float* V, int ldv, float eps, unsigned jt)
{
    set_mat_2_E(V, n, ldv);
    int count=0;
    while(count<jt){
        count++;
        for(int p=0; p<n-1; p++)
        {
            for(int q=p+1; q<n; q++)
            {
                float c, s, t;
                float App, Aqq, Apq;

                App = A[p + p*lda];
                Aqq = A[q + q*lda];
                Apq = A[p + q*lda];
                if(fabs(Apq)<eps) continue;

                printf("App = %f, Aqq=%f, Apq=%f\n", App, Aqq, Apq);
                symSchur2(App, Aqq, Apq, c, s, t);
                printf("c=%f, s=%f\n", c, s);
                // A = J'(p,q,0)*A
                left_rotation(A, n, lda, p, q, c, s);
                // A = A*J(p,q,0)
                right_rotation(A, n, lda, p, q, c, s);
#if ACCURACY_MORE
                A[p + q*lda] = 0.0f;
                A[q + p*lda] = 0.0f;
                A[p + p*lda] = c*c*App - 2.0f*c*s*Apq + s*s*Aqq;
                A[q + q*lda] = s*s*App + 2.0f*c*s*Apq + c*c*Aqq;

                //A[p + p*lda] = App - t*fabs(Apq); //c*c*App - 2.0f*c*s*Apq + s*s*Aqq;
                //A[q + q*lda] = Aqq + t*fabs(Apq); //s*s*App + 2.0f*c*s*Apq + c*c*Aqq;
#endif
                // V = V*J(p,q,0)
                right_rotation(V, n, ldv, p, q, c, s);
            }
        }
    }
}



void test_cpu_cycle_Jac_method(float* A_origin, int n, int lda)
{
    float* A = nullptr;

    A = (float*)malloc(lda*n*sizeof(float));
    memcpy(A, A_origin, lda*n*sizeof(float));
#if PRINT_CPU
    printf("\nA_cpu(origin2)=\n\n");
    print_matrix(A, n, lda);
#endif

    
    int ldv = n;
    float* V = nullptr;
    float* Sig = nullptr;

    V = (float*)malloc(ldv*n*sizeof(float));
    Sig = (float*)malloc(n*sizeof(float));

    bool cal_vector = true;
    float eps = 1.e-9;
    unsigned jt = n*10;

    cpu_cycle_Jac_method(A, n, lda, Sig, cal_vector, V, ldv, eps, jt);

#if PRINT_CPU
    printf("\nA_cpu(sig)=\n\n");
    print_matrix(A, n, lda);
    printf("\nV_cpu(eigVector)=\n");
    print_matrix(V, n, ldv);

#endif


}

__device__
void set_mat_2_E_thread(float* V, int n, int ldv)
{
    unsigned tid = threadIdx.x;

    if(tid<n){
        for(int j=0; j<n; j++){
            V[tid + j*ldv] = 0.0f;
        }
        V[tid + tid*ldv] = 1.0f;
    }
}

__device__
void symSchur2_thread(float App, float Aqq, float Apq, float &c, float &s, float &t)
{
    if(Apq!=0.0f){
        float tau;

        tau = (Aqq-App)/(2.0f* Apq);
        if(tau>= 0.0f){
            t = 1.0f/(tau + sqrt(1.0f+tau*tau));
        }else{
            t = 1.0f/(tau - sqrt(1.0f+tau*tau));
        }

        c = 1.0f/(sqrt(1.0f+t*t));
        s = c*t;
    }else{
        c = 1.0f;
        s = 0.0f;
    }
}

__device__
void left_rotation_thread(float* A, int n, int lda, int p, int q, float c, float s)
{// p-th row      q-th row
    unsigned j = threadIdx.x;
    //for(int j=0; j<n; j++){
    if(j<n){
        float Apj, Aqj;

        Apj = A[p+j*lda];
        Aqj = A[q+j*lda];
        // A[p j]  =  c*A[p j] - s*A[q j]
        A[p + j*lda] = c*Apj - s*Aqj;// p-th row 
        // A[q j]  =  s*A[p j] + c*A[q j]
        A[q + j*lda] = s*Apj + c*Aqj;// q-th row 
    }
}

__device__
void right_rotation_thread(float* A, int n, int lda, int p, int q, float c, float s)
{// p-th col      q-th col
    unsigned i = threadIdx.x;
    //for(int i=0; i<n; i++){
    if(i<n){
        float Aip, Aiq;

        Aip = A[i + p*lda];
        Aiq = A[i + q*lda];
        // A[i p]  =  c*A[i p] + s*A[i q]
        A[i + p*lda] = c*Aip - s*Aiq;
        // A[i q]  = -s*A[i p] + c*A[i q]
        A[i + q*lda] = s*Aip + c*Aiq;
    }
}




__global__
void cycle_warps_evdj_batched_kernel(float* A, int n, int lda, float* Sig_d, bool cal_vector, float* V_d, int ldv, float eps, unsigned jt)
{
    set_mat_2_E_thread(V_d, n, ldv);
    int count=0;
    while(count<jt){
        count++;
        for(int p=0; p<n-1; p++)
        {
            for(int q=p+1; q<n; q++)
            {
                float c, s, t;
                float App, Aqq, Apq;

                App = A[p + p*lda];
                Aqq = A[q + q*lda];
                Apq = A[p + q*lda];
                if(fabs(Apq)<eps) continue;

                //printf("App = %f, Aqq=%f, Apq=%f\n", App, Aqq, Apq);
                symSchur2_thread(App, Aqq, Apq, c, s, t);
                //printf("c=%f, s=%f\n", c, s);
                // A = J'(p,q,0)*A
                left_rotation_thread(A, n, lda, p, q, c, s);
                // A = A*J(p,q,0)
                right_rotation_thread(A, n, lda, p, q, c, s);
#if ACCURACY_MORE
                A[p + q*lda] = 0.0f;
                A[q + p*lda] = 0.0f;
                A[p + p*lda] = c*c*App - 2.0f*c*s*Apq + s*s*Aqq;
                A[q + q*lda] = s*s*App + 2.0f*c*s*Apq + c*c*Aqq;

                //A[p + p*lda] = App - t*fabs(Apq); //c*c*App - 2.0f*c*s*Apq + s*s*Aqq;
                //A[q + q*lda] = Aqq + t*fabs(Apq); //s*s*App + 2.0f*c*s*Apq + c*c*Aqq;
#endif
                // V = V*J(p,q,0)
                right_rotation_thread(V_d, n, ldv, p, q, c, s);
            }
        }
    }
}

void cycle_warps_evdj_batched_gpu(float* A_d, int n, int lda, float* Sig_d, bool cal_vector, float* V_d, int ldv, float eps, unsigned jt)
{
    dim3 grid_;
    dim3 block_;

    block_.x = 64;
    cycle_warps_evdj_batched_kernel<<<grid_, block_, 0, NULL >>>(A_d, n, lda, Sig_d, cal_vector, V_d, ldv, eps, jt);
}

void test_gpu_cycle_Jac_method_warps_batched(float* A_h, int n, int lda)
{
    float* A_d = nullptr;
    float* V_d = nullptr;
    float* Sig_d = nullptr;
    int ldv = n;

    cudaMalloc((void**)&A_d, lda*n*sizeof(float));
    cudaMemcpy(A_d, A_h, lda*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&V_d, ldv*n*sizeof(float));
    cudaMalloc((void**)&Sig_d, n*sizeof(float));

    bool cal_vector = true;
    float eps = 1.e-9;
    unsigned jt = 1000;


    cycle_warps_evdj_batched_gpu(A_d, n, lda, Sig_d, cal_vector, V_d, ldv, eps, jt);

#if PRINT_GPU
    float* A_hh =nullptr;
    A_hh = (float*)malloc(lda*n*sizeof(float));
    float* V_h =nullptr;
    V_h = (float*)malloc(ldv*n*sizeof(float));

    cudaMemcpy(A_hh, A_d, lda*n*sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n\nA_gpu(sigma)=\n\n");
    print_matrix(A_hh, n, lda);
    cudaMemcpy(V_h, V_d, ldv*n*sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n\nV_gpu(eigVect)=\n\n");
    print_matrix(V_h, n, ldv);
    
#endif

    cudaFree(A_d);
}

int main()
{

    float a[NA*NA]=
    {
        6.1612,  4.5364,  3.7169,  3.9139,  4.0917,  4.3700,  3.4825,  3.2865,  4.3305,  4.0457,  3.6477,  4.1974,  5.3102,  5.1594,  4.2083,  4.4336,
        4.5364,  6.5672,  3.2867,  3.7167,  4.2094,  4.8452,  3.2699,  2.8016,  4.1077,  4.4042,  3.4640,  4.0221,  5.0456,  5.2967,  4.6162,  5.1724,
        3.7169,  3.2867,  4.7818,  3.3130,  2.7890,  2.4994,  3.0040,  2.2576,  3.1962,  3.3201,  3.0118,  3.0770,  3.6078,  4.3167,  3.3635,  3.1406,
        3.9139,  3.7167,  3.3130,  6.4944,  4.1474,  3.9002,  4.0854,  2.6071,  4.3111,  3.3132,  3.7818,  4.1045,  4.5588,  5.1275,  3.5897,  4.7889,
        4.0917,  4.2094,  2.7890,  4.1474,  6.6871,  4.7111,  3.2987,  3.3974,  4.4687,  4.1372,  3.2490,  4.2709,  4.4797,  5.0426,  4.9907,  5.0486,
        4.3700,  4.8452,  2.4994,  3.9002,  4.7111,  7.3204,  2.9767,  3.3971,  4.7700,  3.9394,  3.9120,  4.7997,  4.5839,  5.4332,  4.9513,  5.6693,
        3.4825,  3.2699,  3.0040,  4.0854,  3.2987,  2.9767,  4.9697,  2.3518,  3.9516,  3.0647,  3.1183,  3.2647,  4.1501,  4.4729,  3.3198,  4.1061,
        3.2865,  2.8016,  2.2576,  2.6071,  3.3974,  3.3971,  2.3518,  4.4785,  3.5613,  3.1857,  2.3775,  3.4221,  3.3039,  3.6282,  3.7115,  3.2597,
        4.3305,  4.1077,  3.1962,  4.3111,  4.4687,  4.7700,  3.9516,  3.5613,  7.2725,  4.2487,  4.0750,  5.0149,  4.8895,  5.0421,  5.0779,  5.2965,
        4.0457,  4.4042,  3.3201,  3.3132,  4.1372,  3.9394,  3.0647,  3.1857,  4.2487,  6.0102,  3.2056,  4.0181,  4.4608,  5.0873,  4.7124,  4.6422,
        3.6477,  3.4640,  3.0118,  3.7818,  3.2490,  3.9120,  3.1183,  2.3775,  4.0750,  3.2056,  5.4860,  3.5059,  3.6975,  4.0835,  3.4928,  4.2831,
        4.1974,  4.0221,  3.0770,  4.1045,  4.2709,  4.7997,  3.2647,  3.4221,  5.0149,  4.0181,  3.5059,  6.4055,  4.6788,  4.6217,  4.7913,  4.2748,
        5.3102,  5.0456,  3.6078,  4.5588,  4.4797,  4.5839,  4.1501,  3.3039,  4.8895,  4.4608,  3.6975,  4.6788,  7.1927,  5.3270,  4.4300,  4.9857,
        5.1594,  5.2967,  4.3167,  5.1275,  5.0426,  5.4332,  4.4729,  3.6282,  5.0421,  5.0873,  4.0835,  4.6217,  5.3270,  8.5536,  5.3971,  6.4412,
        4.2083,  4.6162,  3.3635,  3.5897,  4.9907,  4.9513,  3.3198,  3.7115,  5.0779,  4.7124,  3.4928,  4.7913,  4.4300,  5.3971,  7.2839,  4.9262,
        4.4336,  5.1724,  3.1406,  4.7889,  5.0486,  5.6693,  4.1061,  3.2597,  5.2965,  4.6422,  4.2831,  4.2748,  4.9857,  6.4412,  4.9262,  7.7919
    };

    int n = NA;
    int lda = NA;
    float* A;

    A = (float*)malloc(lda*n*sizeof(float));
    memcpy(A, a, lda*n*sizeof(float));
    printf("\n\nA(origin)=\n\n");
    print_matrix(A, n, lda);

    test_cpu_cycle_Jac_method(A, n, lda);

    test_gpu_cycle_Jac_method_warps_batched(A, n, lda);

    cout<<"Hello aa World!"<<endl;
    return 0;
}















/* test data:
octave:4> A=a*a' + eye(16)
A =

   6.1612   4.5364   3.7169   3.9139   4.0917   4.3700   3.4825   3.2865   4.3305   4.0457   3.6477   4.1974   5.3102   5.1594   4.2083   4.4336
   4.5364   6.5672   3.2867   3.7167   4.2094   4.8452   3.2699   2.8016   4.1077   4.4042   3.4640   4.0221   5.0456   5.2967   4.6162   5.1724
   3.7169   3.2867   4.7818   3.3130   2.7890   2.4994   3.0040   2.2576   3.1962   3.3201   3.0118   3.0770   3.6078   4.3167   3.3635   3.1406
   3.9139   3.7167   3.3130   6.4944   4.1474   3.9002   4.0854   2.6071   4.3111   3.3132   3.7818   4.1045   4.5588   5.1275   3.5897   4.7889
   4.0917   4.2094   2.7890   4.1474   6.6871   4.7111   3.2987   3.3974   4.4687   4.1372   3.2490   4.2709   4.4797   5.0426   4.9907   5.0486
   4.3700   4.8452   2.4994   3.9002   4.7111   7.3204   2.9767   3.3971   4.7700   3.9394   3.9120   4.7997   4.5839   5.4332   4.9513   5.6693
   3.4825   3.2699   3.0040   4.0854   3.2987   2.9767   4.9697   2.3518   3.9516   3.0647   3.1183   3.2647   4.1501   4.4729   3.3198   4.1061
   3.2865   2.8016   2.2576   2.6071   3.3974   3.3971   2.3518   4.4785   3.5613   3.1857   2.3775   3.4221   3.3039   3.6282   3.7115   3.2597
   4.3305   4.1077   3.1962   4.3111   4.4687   4.7700   3.9516   3.5613   7.2725   4.2487   4.0750   5.0149   4.8895   5.0421   5.0779   5.2965
   4.0457   4.4042   3.3201   3.3132   4.1372   3.9394   3.0647   3.1857   4.2487   6.0102   3.2056   4.0181   4.4608   5.0873   4.7124   4.6422
   3.6477   3.4640   3.0118   3.7818   3.2490   3.9120   3.1183   2.3775   4.0750   3.2056   5.4860   3.5059   3.6975   4.0835   3.4928   4.2831
   4.1974   4.0221   3.0770   4.1045   4.2709   4.7997   3.2647   3.4221   5.0149   4.0181   3.5059   6.4055   4.6788   4.6217   4.7913   4.2748
   5.3102   5.0456   3.6078   4.5588   4.4797   4.5839   4.1501   3.3039   4.8895   4.4608   3.6975   4.6788   7.1927   5.3270   4.4300   4.9857
   5.1594   5.2967   4.3167   5.1275   5.0426   5.4332   4.4729   3.6282   5.0421   5.0873   4.0835   4.6217   5.3270   8.5536   5.3971   6.4412
   4.2083   4.6162   3.3635   3.5897   4.9907   4.9513   3.3198   3.7115   5.0779   4.7124   3.4928   4.7913   4.4300   5.3971   7.2839   4.9262
   4.4336   5.1724   3.1406   4.7889   5.0486   5.6693   4.1061   3.2597   5.2965   4.6422   4.2831   4.2748   4.9857   6.4412   4.9262   7.7919

octave:5> [u d]=eig(A)
u =

  -0.58339461  -0.20384351  -0.19395970  -0.13514756   0.00085043   0.30840120   0.20790326   0.28834776  -0.15086236  -0.17681307   0.04142857  -0.33087742  -0.14106489   0.26635488   0.14528980   0.25211355
   0.05069628   0.04847067  -0.29998523   0.18779854  -0.50224573  -0.29513987   0.08508990  -0.23006901   0.27200990   0.16843353  -0.03850948  -0.32901968  -0.43550238  -0.03358427  -0.04561696   0.25558443
   0.12092767   0.11630827   0.56334165   0.21877113  -0.27247558   0.18836382   0.08795055  -0.06448537   0.07365514  -0.27844502  -0.27221079   0.18490919  -0.15647685   0.33518784   0.34723517   0.19068755
  -0.16397812   0.10692333  -0.01009715  -0.47541023  -0.25869293  -0.16426090  -0.22401510  -0.21451259   0.15353464  -0.24489603   0.29657423   0.09613830   0.29183750  -0.22791979   0.41203551   0.24046628
   0.12295062   0.07460783  -0.06956197   0.28134214  -0.01458299   0.31554333   0.04901438   0.32789577   0.46940566  -0.14437572   0.52038801   0.21411281   0.04034155  -0.07473872  -0.23687037   0.25391211
  -0.10368896   0.50913059   0.28598785   0.02462698   0.15240708   0.00364056   0.06689561  -0.09331002  -0.19489577  -0.29267190  -0.07816047  -0.36944890   0.01740357  -0.36036744  -0.37913841   0.26657061
  -0.29035761   0.20667917  -0.03048520   0.46762399   0.33398902  -0.37153649   0.28402541  -0.02632922  -0.06867015   0.25409001   0.12414567   0.17093019   0.14737984  -0.02843934   0.37816834   0.20730518
   0.13562172  -0.02984245   0.00974536   0.02288931  -0.28951102  -0.52003812  -0.11485170   0.46681147  -0.41925302  -0.17174756   0.10690694   0.11214707   0.12216325   0.26052840  -0.21972820   0.18542218
   0.07211340   0.23512517  -0.13793747  -0.07685919  -0.33754507   0.37250456   0.13594458  -0.00384045  -0.16823925   0.49988478  -0.20894128   0.06435851   0.48280095   0.06208328  -0.08606167   0.27026714
  -0.25223105   0.27836520  -0.00101386  -0.09298904   0.21970814   0.01111624  -0.66935415   0.06361649   0.12244218   0.26667812  -0.13493066   0.19238470  -0.28229949   0.24143222  -0.11057643   0.24130877
   0.21982850  -0.08192847  -0.23828048   0.00146336   0.22129361  -0.12510401  -0.05835127   0.35973152   0.35554164  -0.19814156  -0.58787985  -0.14919936   0.23705638  -0.15209081   0.18082227   0.21258591
  -0.00196321  -0.40287487  -0.06832874   0.37098772   0.10221474   0.05206986  -0.31991718  -0.49130778  -0.05985796  -0.18508833   0.03529148  -0.15991026   0.36136108   0.23014268  -0.16546376   0.25082791
   0.49217940  -0.04130939   0.22704760  -0.25559165   0.31305035  -0.01738232   0.03455962   0.03886898  -0.01911486   0.28562378   0.31815392  -0.42855382  -0.06057050   0.22710562   0.20228170   0.27418460
   0.32474925  -0.00370486  -0.40834733  -0.02479412   0.15238190   0.20567676   0.02382913  -0.15413251  -0.44658351  -0.19701604  -0.02087666   0.34301124  -0.35329057  -0.20273052   0.14554984   0.30638332
  -0.07537896  -0.16496149   0.06469977  -0.39212601   0.18897863  -0.21734183   0.45651669  -0.23773431   0.23607483  -0.02524816  -0.12984661   0.34102006  -0.02366685   0.21652598  -0.39053381   0.26795049
  -0.12831739  -0.54039533   0.40536661   0.04173941  -0.09600476   0.01306490  -0.09187841   0.14927856  -0.08133623   0.29453002  -0.06134232   0.09003290  -0.11661387  -0.53074778  -0.03148263   0.28950727

d =

Diagonal Matrix

    1.0007         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0
         0    1.0178         0         0         0         0         0         0         0         0         0         0         0         0         0         0
         0         0    1.0290         0         0         0         0         0         0         0         0         0         0         0         0         0
         0         0         0    1.2030         0         0         0         0         0         0         0         0         0         0         0         0
         0         0         0         0    1.3384         0         0         0         0         0         0         0         0         0         0         0
         0         0         0         0         0    1.4754         0         0         0         0         0         0         0         0         0         0
         0         0         0         0         0         0    1.7647         0         0         0         0         0         0         0         0         0
         0         0         0         0         0         0         0    2.0573         0         0         0         0         0         0         0         0
         0         0         0         0         0         0         0         0    2.1815         0         0         0         0         0         0         0
         0         0         0         0         0         0         0         0         0    2.3993         0         0         0         0         0         0
         0         0         0         0         0         0         0         0         0         0    2.8318         0         0         0         0         0
         0         0         0         0         0         0         0         0         0         0         0    3.2407         0         0         0         0
         0         0         0         0         0         0         0         0         0         0         0         0    3.9635         0         0         0
         0         0         0         0         0         0         0         0         0         0         0         0         0    4.1073         0         0
         0         0         0         0         0         0         0         0         0         0         0         0         0         0    5.1441         0
         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0   68.7020

octave:6>

*/
