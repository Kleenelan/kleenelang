__device__
void sort_thread(float* sig, int N) {

        int x = threadIdx.x;
        extern __shared__ float s_a[];
        if(x<N)
                s_a[x] = sig[x];
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
        if(x<N)
                sig[x] = s_a[x];
        __syncthreads();
}

__device__
void sort_value_id_thread(float* sig, int N, float* s_a, unsigned* ids) {

        int x = threadIdx.x;
        //extern __shared__ float s_a[];
        if(x<N){
                s_a[x] = sig[x];
                ids[x] = x;
        }
        __syncthreads();

        for (int i = 0; i < N; i++)
        {

                int j = i % 2;
                int idx = 2 * x + j;
                if (idx + 1 < N && s_a[idx] > s_a[idx + 1]) {
                        float tmp = s_a[idx];
                        s_a[idx] = s_a[idx + 1];
                        s_a[idx + 1] = tmp;
                        unsigned tmpid = ids[idx];
                        ids[idx] = ids[idx+1];
                        ids[idx+1] = tmpid;
                }
                __syncthreads();
        }
        if(x<N)
                sig[x] = s_a[x];
        __syncthreads();
}




int main(){



}
