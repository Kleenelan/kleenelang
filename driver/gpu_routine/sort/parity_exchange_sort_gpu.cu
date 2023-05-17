#include <random>

using namespace std;

void init_vector(float* A, int n)
{
        random_device rd;
        mt19937 gen(rd());
        float mean = 0.0f;
        float sigma = 1.0f;

        normal_distribution<> nd(mean, sigma);

        for(int idx=0; idx<n; idx++)
        {
                A[idx] = nd(gen);
        }
}

void print_vector(float*A, int n)
{
        for(int idx=0; idx<n; idx++)
        {
                printf("%8.5f ", A[idx]);
        }
        printf("\n\n");
}

void print_int_vector(unsigned*A, int n)
{
        for(int idx=0; idx<n; idx++)
        {
                printf("%4d ", A[idx]);
        }
        printf("\n\n");
}

__device__
void sort_value_thread(float* s_a, int N)
{
        int x = threadIdx.x;
        //extern __shared__ float s_a[];
        //if(x<N)                s_a[x] = sig[x];
        __syncthreads();

        for (int i = 0; i < N; i++)
        {

                int j = i % 2;
                int idx = 2 * x + j;
                if (idx + 1 < N && s_a[idx] > s_a[idx + 1]) {
                        float tmp = s_a[idx];
                        s_a[idx] = s_a[idx + 1];
                        s_a[idx + 1] = tmp;
                }
                __syncthreads();
        }
        //if(x<N)                sig[x] = s_a[x];
        __syncthreads();
}

__device__
void sort_value_id_thread(float* s_a, int N, unsigned* s_ids)
{
        int x = threadIdx.x;

        __syncthreads();

        for (int i = 0; i < N; i++)
        {
                int j = i % 2;
                int idx = 2 * x + j;

                if (idx + 1 < N && s_a[idx] > s_a[idx + 1]) {
                        float tmp = s_a[idx];
                        s_a[idx] = s_a[idx + 1];
                        s_a[idx + 1] = tmp;

                        unsigned tmpid = s_ids[idx];
                        s_ids[idx] = s_ids[idx+1];
                        s_ids[idx+1] = tmpid;
                }
                __syncthreads();
        }

        __syncthreads();
}

__global__
void sort_value_with_id_kernel(float* sig, int N, unsigned* ids)
{
        unsigned idx = threadIdx.x;
        __shared__ float s_a[1024];
        __shared__ unsigned s_ids[1024];

        if(idx<N) s_a[idx] = sig[idx];
        if(idx<N) s_ids[idx] = idx;

        sort_value_id_thread(s_a, N, s_ids);

        if(idx<N) sig[idx] = s_a[idx];
        if(idx<N) ids[idx] = s_ids[idx];
}

void sort_value_with_id_gpu(float* sig, int N, unsigned* ids)
{
        dim3 grid_;
        dim3 block_;

        block_.x = N;
        sort_value_with_id_kernel<<<grid_, block_, 8*1024, NULL>>>(sig, N, ids);
}

__global__
void sort_value_kernel(float* sig, int N)
{
        unsigned idx = threadIdx.x;
        extern __shared__ float s_a[];

        if(idx<N)       s_a[idx] = sig[idx];

        sort_value_thread(s_a, N);
        if(idx<N)       sig[idx] = s_a[idx];
}

void sort_value_gpu(float* A_d, int N)
{
        dim3 grid_;
        dim3 block_;

        block_.x = N;

        sort_value_kernel<<<grid_, block_, 4*1024, NULL>>>(A_d, N);
}

void test_sort_value()
{
        int n = 64;
        float* A_h = nullptr;
        float* A_d = nullptr;

        A_h = (float*)malloc(n*sizeof(float));
        cudaMalloc((void**)&A_d, n*sizeof(float));
        init_vector(A_h, n);
        cudaMemcpy(A_d, A_h, n*sizeof(float), cudaMemcpyHostToDevice);

        printf("test value raw =\n");
        print_vector(A_h, n);

        sort_value_gpu(A_d, n);

        cudaMemcpy(A_h, A_d, n*sizeof(float), cudaMemcpyDeviceToHost);
        printf("test value sorted =\n");
        print_vector(A_h, n);
        printf("\n==================================================\n");

}

void test_sort_value_with_id()
{
        int n = 64;
        float* A_h = nullptr;
        float* A_d = nullptr;
        A_h = (float*)malloc(n*sizeof(float));
        cudaMalloc((void**)&A_d, n*sizeof(float));
        init_vector(A_h, n);
        cudaMemcpy(A_d, A_h, n*sizeof(float), cudaMemcpyHostToDevice);

        printf("test value raw =\n");
        print_vector(A_h, n);

        unsigned* ids_d =nullptr;

        cudaMalloc((void**)&ids_d, n*sizeof(unsigned));
        sort_value_with_id_gpu(A_d, n, ids_d);

        cudaMemcpy(A_h, A_d, n*sizeof(float), cudaMemcpyDeviceToHost);
        printf("test value sorted =\n");
        print_vector(A_h, n);

        unsigned* ids_h = nullptr;
        ids_h = (unsigned*)malloc(n*sizeof(unsigned));
        cudaMemcpy(ids_h, ids_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost);

        printf("ids_d sorted =\n");
        print_int_vector(ids_h, n);

}


int main()
{
        test_sort_value();
        test_sort_value_with_id();

        return 0;
}

