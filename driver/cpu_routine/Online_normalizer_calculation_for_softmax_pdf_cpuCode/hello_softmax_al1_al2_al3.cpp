#include <stdio.h>
#include <math.h>
#define LEN 5


void alg01_softmax(float* x, int len, float *softmax)
{
        float dv=0.0;

        for(int i=0; i<len; i++){
                dv += exp(x[i]);
        }

        printf("dv=%f\n", dv);

        for(int i=0; i<len; i++){
                softmax[i] = exp(x[i])/dv;
        }

}



float max_(float a, float b){
        return a>b? a : b;
}

void alg02_softmax(float* x, int len, float *softmax){

        float dv = 0.0f;
        float max=-1000000;

        for(int i=0; i<LEN; i++){
                max = max<x[i] ? x[i] : max;
        }

        for(int i=0; i<len; i++){
                dv += exp(x[i]-max);
        }

        printf("dv=%f\n", dv);

        for(int i=0; i<len; i++){
                softmax[i] = exp(x[i]-max)/dv;
        }

}

void alg03_softmax(float* x, int len, float* softmax){
        float mj_1 = -1000000.0f;
        float mj;
        float dj = 0.0f;

        for(int j=0; j<len; j++){
                mj = max_(mj_1, x[j]);
                dj = dj*exp(mj_1 - mj) + exp(x[j] - mj);
                mj_1 = mj;
        }

        printf("a3  dj=%f, mj=%f\n", dj, mj);

        for(int i=0; i<len; i++){
                softmax[i] = exp(x[i] - mj)/dj;
        }

}

void print_vector(float* A, int len){
        for(int i=0; i<len; i++){

                printf("%7.4f  |", A[i]);
        }

        printf("\nend\n");
}

int main()
{
        float x[LEN] = {        1, 2, 3, 4, 5   };

        float a1_softmax[LEN];
        float a2_softmax[LEN];
        float a3_softmax[LEN];
        float a4_softmax[LEN];

        alg01_softmax(x, LEN, a1_softmax);
        print_vector(a1_softmax, LEN);

        alg02_softmax(x, LEN, a2_softmax);
        print_vector(a2_softmax, LEN);

        alg03_softmax(x, LEN, a3_softmax);
        print_vector(a3_softmax, LEN);


        return 0;
}

