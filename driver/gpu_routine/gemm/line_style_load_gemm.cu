/*
// author: haibo
// date ： 2022/11/14
// sync conv test
*/
#include <stdio.h>

#include <unistd.h>
#include <cuda_runtime.h>
#include <random>
//#include ""

using namespace std;
void init_vector(float* A, int n);
template<bool ROW_MAJOR>
void print_matrix(float* A, int m, int n, int lda);
template<bool ROW_MAJOR>
void print_sub_matrix(float*A, int m, int n, int lda, int start_i, int end_i, int start_j, int end_j);
template<bool ROW_MAJOR>
void print_col_vector(float* A, int m, int n, int lda, int j);
template<bool ROW_MAJOR>
void print_row_vector(float* A, int m, int n, int lda, int i);


template<bool A_RM, bool B_RM, bool C_RM>
void cpu_gemm_proc(int M, int N, int K, float* A, int lda, float* B, int ldb, float* C, int ldc);
template<bool A_RM, bool B_RM, bool C_RM>
void cpu_gemm_test(int m, int n, int k, float* A_h, int lda, float* B_h, int ldb, float* C_h, int ldc);
void free_as_null(void* ptr);
void check_state(cudaError_t cudaS);
__global__
void gemm_kernel_one_block(int m, int n, int k, float* Ag, int lda, float* Bg, int ldb, float* Cg, int ldc);



typedef int v4i32 __attribute__((ext_vector_type(4)));

__global__ void sync_copy_kernel(const float *in,
                                 const int gvOffset,
                                 float *out)
{
    // LL:: for (int i = 0; i < 1; i++) out[i] = out[i] + 1;
 #if 0
    unsigned lane_id = threadIdx.x;
    out[lane_id] = 0.37f + lane_id;

    __shared__ float smem[64];

    v4i32 inBase;//LL:: src A
    inBase.x = (unsigned)(unsigned long long)in;
    inBase.y = (unsigned)((unsigned long long)in >> 32);//LL:: the first 64 bits, into x and y; zw=0XFFFFFFFFFFFFFFFF;
    inBase.zw = -1u;

    int svOffset = lane_id;//LL:: byte by byte
    //  int ssOffset = (unsigned)(long long)&smem[0];
    int gsOffset = 0;
    int ssOffset = reinterpret_cast<size_t>(smem);

    __builtin_bi_mllsa_async_b32x1(lane_id * 4, inBase, svOffset * 4, ssOffset, gsOffset, 0);
    //  asm volatile("sl_nop 300");
    asm volatile("sl_wait g2scnt(0)");

    // float a = smem[lane_id];
    out[lane_id] = smem[lane_id];
 #endif
}

///////////////////////////////////////
#define BM 	(128)
#define BN 	(128)
#define BK 	(128)

#define WM 	(32)
#define WN  (64)
#define WK  (BK)
///////////////////////////////////////
// neuron shape (MxN) = (4x2) (1x8) (8x1)// (2x4)
#define NEUM	(BM/WM)	// BLOCK_M/WARP_M,   neuron m, neuron(4x2)//for block C
#define NEUN   	(BN/WN)	// BLOCK_N/WAR__N,   neuron n, neuron(4x2)
//#define NEUK     (NEUN)//for load slb//     instead 1 zi changshezhen, all new conception
#define NEUS	(NEUM*NEUN)

#define TAM     (8)		// 4
#define TAK     (8)		// 16
#define TBK     (8)	
#define TBN     (8)
#define TCM     (4)		//
#define TCN     (16)

#define MAD_AMS    2	// MAD_M = MAD_AMS*TAM = MADAM		// as mad_A_ms
#define MAD_AKS    1	// MAD_K = MAD_AKS*TAK = MAD_K		// as mad_A_ks
#define MAD_BKS    1	// MAD_K = MAD_BKS*TBK = MAD_K		// as mad_B_ks
#define MAD_BNS    2	// MAD_N = MAD_BNS*TBN = MADBN		// as mad_B_ns

// mad(16 x 16 x 8) = mad(2*8 x 2*8 x 8) ;  mad(8 x 16 x 16) = mad(2*4 x 16 x 2*8)
#define MADAM   (MAD_AMS*TAM) //alternative
#define MADBN   (MAD_BNS*TBN) //alternative
#define MAD_K   (MAD_AKS*TAK) // = (MAD_BKS*TBK) // alternative

#define MAD_CMS    (MADM/TCM) // store back
#define MAD_CNS    (MADN/TCN)

//////////////////////////////////////////////////////////////////////////////////////////
// ** hyper parameters
#define WAMS   	(WM/MADAM)  	// MAS  = WARP_M / MADAM
#define WAKS   	(WK/MAD_K) 	// WAKS = WARP_K / MAD_K //  WARP_K == BLOCK_K;
#define WBNS   	(WN/MADBN) 	// NBS  = WARP_N / MADBN

#define WBKS	(WK/MAD_K) 	// BKS = La
//////////////////////////////////////////////////////////////////////////////////////////

#define STGCNT   1

typedef float (sA_calc)[STGCNT][WAKS][MAD_AKS][NEUM][WAMS][MAD_AMS][TAM*TAK];// Tile colMajor stored in sA, but
typedef float (sA_load)[STGCNT][WAKS *MAD_AKS][NEUM *WAMS *MAD_AMS][TAM*TAK];// rowMajor loaded into sA[bN][bM][64]

typedef float (sB_calc)[STGCNT][WBKS][MAD_BKS][NEUN][WBNS][MAD_BNS][TBK*TBN];// Tile rowMajor stored in sB, but
typedef float (sB_load)[STGCNT][WBKS *MAD_BKS][NEUN *WBNS *MAD_BNS][TBK*TBN];// colMajor loaded into sB[][][64]

#define SA_TILE_M (NEUM*WAMS*MAD_AMS)
#define SA_TILE_K (WAKS * MAD_AKS)
#define SA_TILE_CNT (SA_TILE_M * SA_TILE_K)
//[WAKS *MAD_AKS]*[NEUM *WAMS *MAD_AMS] tile(8x8)
#define SB_TILE_K   (WBKS*MAD_BKS)
#define SB_TILE_N   (NEUN * WBNS * MAD_BNS)
#define SB_TILE_CNT (SB_TILE_K * SB_TILE_N)

__global__ void warmup_kernel(){}

__global__
void gemm_kernel_one_block(int m, int n, int k, float* Ag, int lda, float* Bg, int ldb, float* Cg, int ldc)
{
    float* Ab = Ag;// block A start;
    unsigned warp_id;
    unsigned lane_i, lane_j, lane_id;

    warp_id = threadIdx.z*blockDim.y + threadIdx.y;//NEUM; warp_id is alternative for each warp;
    lane_id = threadIdx.x%warpSize;//     if(threadIdx.x==0&&threadIdx.z<2)printf("warp_id=%u\n", warp_id);
    lane_i = lane_id / TAK;
    lane_j = lane_id % TAK;

    // typedef float (sA_calc)[STGCNT][WAKS][MAD_AKS][NEUM][WAMS][MAD_AMS][TAM*TAK];//
    __shared__ sA_calc  sA_cl;
    sA_load* sA_ld = (sA_load*)sA_cl;

    // 1.0 load sA
    // typedef float (sA_load)[STGCNT][WAKS *MAD_AKS][NEUM *WAMS *MAD_AMS][TAM*TAK];
    for(int t_idx = warp_id; t_idx<SA_TILE_CNT; t_idx += NEUS)//A rowMajor, then load rowMajor of Tiles, that is alternative;   NEUS = blockDim.z * blockDim.y;
    {
        int i, j, idx, ti, tj;// tile_i tile_j
        float a;

        ti = t_idx / SA_TILE_K;// rowMajor load, the K direction
        tj = t_idx % SA_TILE_K;
        i = ti * TAM + lane_i;//ti*TAM*lda + tj*TAK   is s_offset
        j = tj * TAK + lane_j;//lane_id is v_offset
    
        a = Ag[i*lda + j];      //if(t_idx == 0 && threadIdx.z==0 && threadIdx.x==0 && threadIdx.y ==0){printf("\na=%7.3f\n", a);}
        //a = 173.73f;
        (*sA_ld)[0][tj][ti][lane_id] = a;
    }
#if 1

    __shared__ sB_calc  sB_cl;
    sB_load* sB_ld = (sB_load*)sB_cl;//   if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z ==0)        printf("\n\n  *sB_ld=%lu sB_cl = %lu\n\n", *sB_ld, sB_cl);
    // 2.0 load sB
    // typedef float (sB_load)[STGCNT][WBKS *MAD_BKS][NEUN *WBNS *MAD_BNS][TBK*TBN];
    for(int t_idx = warp_id; t_idx<SB_TILE_CNT; t_idx += NEUS)//NEUSB colMajor, then load colMajor of Tiles, that is alternative;
    {
        int i, j, idx, ti, tj;// tile_i tile_j
        float b;

        ti = t_idx / SB_TILE_N;
        tj = t_idx % SB_TILE_N;
        i = ti*TBK + lane_i;
        j = tj*TBN + lane_j;

        b = Bg[i + j*ldb];
        (*sB_ld)[0][ti][tj][lane_id] = b;
    }
#endif
    asm volatile("sl_nop 300");
    // 3.0 calculate warp_C
    // store sA_cl to Cg
    // typedef float (sA_calc)[STGCNT][WAKS][MAD_AKS][NEUM][WAMS][MAD_AMS][TAM*TAK];//
    //__shared__ sA_calc  sA_cl;
    //for()



    // 4.0 store warp_C
    //if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z ==0){        Cg[0] = (*sB_ld)[0][0][0][0];        Cg[1] = (*sA_ld)[0][0][0][0];        //printf("\n\nLL:: tmp_a = %7.3f,  tmp_b = %7.3f,  SA_TILE_K = %d, SB_TILE_CNT = %d, NEUS = %d\n\n", tmp_a, tmp_b, SA_TILE_K, SB_TILE_CNT, NEUS);    }
    sA_calc* C_cl = (sA_calc*)Cg;
    if(threadIdx.z==0)
    (*C_cl)[0][0][0][threadIdx.y][0][0][lane_id] = sA_cl[0][0][0][threadIdx.y][0][0][lane_id];

}


#if 0
    // ** physical parameters
    // RC == NT 
    // xM * xN = 64
    #define TAM     8
    #define TAK     8
    #define TBK     8
    #define TBN     8
    #define TCM     4 // C Tile M
    #define TCN     16// C Tile N

    #define MAMS    2// MAMS = MAD_M / TAM
    #define MAKS    1
    #define MBKS    1
    #define MBNS    2// MBNS = MAD_N / TBN
    // mad(16x16x8)  mad(8x16x16)
    #define MADM    (MAMS*TAM)
    #define MADN    (MBNS*TBN)
    #define MADK    (TAK)

    #define MCMS    (MADM/TCM)//slb load
    #define MCNS    (MADN/TCN)
    // neuron shape (MxN) = (4x2) (1x8) (8x1)// (2x4)
    #define NEUM     4   // neuron m, neuron(4x2)//for block C
    #define NEUN     2   // neuron n, neuron(4x2)
    #define NEUK     (NEUN)//for load slb
    // all neurons are busy;
    #define MIN_BLOCKM  (NEUM*MADM) //block cal  M
    #define MIN_BLOCKN  (NEUN*MADN) //block cal  N
    #define MIN_BOKCKK  (NEUN*MADK) //block load K
    //////////////////////////////////////////////////////////////////////////////////////////
    // ** hyper parameters, MS, NS, KS are 2**i, i=0, 1, ...; 
    #define MS      (2)     // MS = WARP_M / MADM
    #define NS      (2*2)   // NS = WARP_N / MADN
    #define KS      (2*2*2)     // KS = WARP_K / MADK//  WARP_K == BLOCK_K;

    #define BLOCKM (MS * MIN_BLOCKM)     //  2**i; i = 0, 1, ...
    #define BLOCKN (NS * MIN_BLOCKN)     //  2**j; j = 0, 1, ...
    #define BLOCKK (KS * MIN_BOKCKK)
    //////////////////////////////////////////////////////////////////////////////////////////
    // warp_calc_shape(i*MADM x j*MADN)
    #define WARPM   (MS*MADM)   // warp size to calculates on C(M, :) 
    #define WARPN   (NS*MADN)   // warp size to calculates on C(:, N)
    #define WARPK   (KS*MADK)   // warp size in sigma to calculates on sub_block_C

    #define STGCNT   1


    typedef float (sA_load)[STGCNT][KS][NEUN][MS*MAMS][NEUM][TAM*TAK];    //[MS*MAMS]  as MAMS is a claculate conception
    // load:	    STGCNT x KS x NEUN x (MS*MAMS) x NEUM x (TAM*TAK)
    typedef float (sA_calc)[STGCNT][KS*NEUN][NEUM][MS][MAMS][TAM*TAK];
    // calculate:   STGCNT x (KS*NEUN) x NEUM x MS x MAMS x (TAM*TAK)

    __global__ void gemm_kernel_one_block(int m, int n, int k, float* Ag, int lda, float* Bg, int ldb, float* Cg, int ldc)
    {
    float* Ab = Ag;// block A start;
    unsigned lane_id;
    unsigned lane_i, lane_j;

    lane_i = threadIdx.x / TAK;
    lane_j = threadIdx.x % TAK;

    // typedef float (sA_load)[STGCNT][KS][NEUN][MS][MAMS][NEUM][TAM][TAK];
    sA_load  sA_ld;
    // typedef float (sA_calc)[STGCNT][KS*NEUN][NEUM][MS][MAMS][TAM][TAK];
    sA_calc* sA_cl = (sA_calc*)sA_ld;// multi stage
    float a;
    for(int stage=0; stage<STGCNT; stage++)
    {
        for(int ks=0; ks<KS; ks++)
        {
            for(int msxmams=0; msxmams<MS*MAMS; msxmams++)
            {
                int i, j, idx, ti, tj;// tile_i tile_j

                ti = msxmams*NEUM + threadIdx.z;
                tj = ks*NEUN + threadIdx.y;// + stage*KS*NEUN;
                i = ti*TAM + lane_i;
                j = tj*TAK + lane_j;
                idx = i*lda + j;
                a = Ab[idx];
                sA_ld[stage][ks][threadIdx.y][msxmams][threadIdx.z][lane_id] = a;//asyn-copy
            }
        }
    }

    }
#endif

#if 0
    __global__ void block_2_line_async_copy_kernel_(float* C, int mc, int nc, int ldc, float* A, int m, int n, int lda)
    {//load tile(0,0) in global mem, into line(0) in slb;  T(0, 0) => L(0); T is 8x8, L is 1x64;
    __shared__ float sA[4][2][MAMS][64];// sA[NEUM][NEUN][MAMS][TAM*TAK]each neuron load a T(8x8)
    unsigned lane_id = threadIdx.x % 64;
    unsigned lane_i = lane_id / 8;
    unsigned lane_j = lane_id % 8;
    unsigned warp_i, warp_j;
    unsigned tb_i, tb_j;

    warp_i = threadIdx.z;
    warp_j = threadIdx.y;

    for(int mm_i=0; mm_i<MAMS; mm_i++){

        tb_i = threadIdx.z;// neuron i
        tb_j = threadIdx.y;// neuron j // rowMajor
        unsigned tb_id = blockDim.y*tb_i + tb_j;// T are rowMajor
        float a;
        int i, j;

        i = (tb_i * TAM) + lane_i;
        j = (tb_j * TAK) + lane_j;
        a = A[i*lda + j];//A[i*lda + j]

        sA[tb_i][tb_j][lane_id] = a;
    }
    // C[4x2][64]; C[(tb_i*2 + tb_j)*ldc + lane_id]
    C[(tb_i*2 + tb_j)*ldc + lane_id] = sA[tb_i][tb_j][lane_id];
    }
#endif


int main()// one CU, one BLOCK
{ 
    constexpr bool A_rowMajor = true;
    constexpr bool B_rowMajor = false;
    constexpr bool C_rowMajor = true;

    int m = 128;//128 = 1*4* 2*2*(8); = STGCNT * NEUM * MS * MAMS * TAM = 1 * 4 * 2 * 2 * 8 = 128;  //each neuron load a T;
    int n = 128;//128 = 1*2* 4*2*(8); = STGCNT * NEUN * NS * MBNS * TBN = 1 * 2 * 4 * 2 * 8 = 128;  //A RowMajor
    int k = 128;//128 =       16*(8);                                    no relationship with STAGE  * NEUN * KS * MAKS * TAK = 1 * 2 * 8 * 1 * 8 = 128;  // K

    int lda     = A_rowMajor? (k + 4)   : m+4;              // A RowMajor
    int space_A = A_rowMajor? (m * lda) : lda*k;            // A RowMajor

    int ldb     = B_rowMajor? n + 8     : (k + 8);          // B ColMajor
    int space_B = B_rowMajor? k * ldb   : (ldb * n);        // B ColMajor

    int ldc     = C_rowMajor? (n + 2)   : m + 2;            // C RowMajor
    int space_C = C_rowMajor? (m * ldc) : ldc * n;          // C RowMajor

    float *A_h = nullptr;
    float *A_d = nullptr;
    float *B_h = nullptr;
    float *B_d = nullptr;
    float *C_h = nullptr;
    float *C_d = nullptr;
    cudaError_t cudaState;

    cudaState = cudaSuccess;

    cudaState = cudaMallocHost((void**)&A_h, space_A*sizeof(float));    check_state(cudaState);
    cudaState = cudaMallocHost((void**)&B_h, space_B*sizeof(float));    check_state(cudaState);
    cudaState = cudaMallocHost((void**)&C_h, space_C*sizeof(float));    check_state(cudaState);

    cudaState = cudaMalloc(&A_d, space_A * sizeof(float));    check_state(cudaState);
    cudaState = cudaMalloc(&B_d, space_B * sizeof(float));    check_state(cudaState);
    cudaState = cudaMalloc(&C_d, space_C * sizeof(float));    check_state(cudaState);

//template<bool ROW_MAJOR>void print_sub_matrix(float*A, int m, int n, int lda, int start_i, int end_i, int start_j, int end_j)
    init_vector(A_h, space_A);if(1){    printf("A_h(:, 0:8) =\n");     print_sub_matrix<true>(A_h, m, n, lda, 0, m-1, 0, 8);}     else{   if(0){ printf("\nA_h =\n");    print_matrix<A_rowMajor>(A_h, m, k, lda);} else{    printf("\nA_h(1,:) =\n");   print_row_vector<A_rowMajor>( A_h, m, n, lda, 0);   }}
    init_vector(B_h, space_B);if(1){    printf("B_h(:, 0:8) =\n");     print_sub_matrix<true>(B_h, m, n, ldb, 0, m-1, 0, 8);}     else{   if(0){ printf("\nB_h =\n");    print_matrix<B_rowMajor>(B_h, k, n, ldb);} else{    printf("\nB_h(1,:) =\n");   print_row_vector<A_rowMajor>( B_h, m, n, lda, 0);   }}
    init_vector(C_h, space_C);if(1){    printf("C_h(:, 0:8) =\n");     print_sub_matrix<true>(C_h, m, n, ldc, 0, m-1, 0, 8);}     else{   if(0){ printf("\nC_h =\n");    print_matrix<C_rowMajor>(C_h, m, n, ldc);} else{    printf("\nC_h(1,:) =\n");   print_row_vector<A_rowMajor>( C_h, m, n, lda, 0);   }}

    cudaState = cudaMemcpy(A_d, A_h, space_A * sizeof(float), cudaMemcpyHostToDevice); sleep(1);    check_state(cudaState);
    cudaState = cudaMemcpy(B_d, B_h, space_B * sizeof(float), cudaMemcpyHostToDevice); sleep(1);    check_state(cudaState);
    cudaState = cudaMemcpy(C_d, C_h, space_C * sizeof(float), cudaMemcpyHostToDevice); sleep(1);    check_state(cudaState);

    dim3 grid_, block_;
    
    block_.x = 64;
    block_.y =  4;
    block_.z =  2;
    warmup_kernel<<<grid_, block_, 0, NULL>>>();
    //gemm_kernel_one_block<<<grid_, block_, 0, NULL>>>(m, n, k, A_d, lda, B_d, ldb, C_d, ldc);
    
    //__global__ void gemm_kernel(int m, int n, int k, float* a, int lda, float* B, int ldb, float* C, int ldc)
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0); 
    gemm_kernel_one_block<<<grid_, block_, 0, NULL>>>(m, n, k, A_d, lda, B_d, ldb, C_d, ldc);
        //block_2_line_async_copy_kernel<<<grid_, block_, 0, NULL>>>(C_d, mc, nc, ldc, A_d, m, n, lda);
cudaEventRecord(stop, 0);
cudaEventSynchronize(start);
cudaEventSynchronize(stop);
float time= 0;
cudaEventElapsedTime(&time, start, stop);
printf("GPU time: %f (ms)\n", time);



    cudaState = cudaMemcpy(C_h, C_d, space_C*sizeof(float), cudaMemcpyDeviceToHost); sleep(1);      check_state(cudaState); printf("C_h ok =\n");   print_sub_matrix<C_rowMajor>(C_h, m, n, ldc, 0, m-1, 0, 32);//    print_matrix<C_rowMajor>(C_h, m, n, ldc);

#if 0
    cpu_gemm_test<A_rowMajor, B_rowMajor, C_rowMajor>(m, n, k, A_h, lda, B_h, ldb, C_h, ldc);//C_h for verify
#endif

    if(A_h != NULL)    {   cudaState = cudaFreeHost(A_h);   check_state(cudaState);   A_h = NULL;}
    if(B_h != NULL)    {   cudaState = cudaFreeHost(B_h);   check_state(cudaState);   B_h = NULL;}
    if(C_h != NULL)    {   cudaState = cudaFreeHost(C_h);   check_state(cudaState);   C_h = NULL;}
    if(A_d != NULL)    {   cudaState = cudaFree(A_d);       check_state(cudaState);   A_d = NULL;}
    if(B_d != NULL)    {   cudaState = cudaFree(B_d);       check_state(cudaState);   B_d = NULL;}
    if(C_d != NULL)    {   cudaState = cudaFree(C_d);       check_state(cudaState);   C_d = NULL;}
printf("GPU time: %f (ms)\n", time);
    printf("\n\nend\n");

    return 1;
}







//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

template<bool ROW_MAJOR>
void print_sub_matrix(float*A, int m, int n, int lda, int start_i, int end_i, int start_j, int end_j)
{
    for(int i=(start_i>0? start_i: 0); i<((end_i+1)<m? (end_i+1) : m); i++)
    {
        for(int j=(start_j>0? start_j: 0); j<((end_j+1)<n? (end_j+1) : n); j++)
        {
            int idx;

            idx = ROW_MAJOR? i*lda+j: i + j*lda;
            printf("%7.3f ", A[idx]);
        }
        printf("\n");
    }
}

template<bool ROW_MAJOR>
void print_row_vector(float* A, int m, int n, int lda, int i)
{
    print_sub_matrix<ROW_MAJOR>(A, m, n, lda, i, i+1, 0, n);
}

template<bool ROW_MAJOR>
void print_col_vector(float* A, int m, int n, int lda, int j)
{
    print_sub_matrix<ROW_MAJOR>(A, m, n, lda, 0, m, j, j+1);
}

void free_as_null(void* ptr){
    if(ptr != NULL)
    {
        free(ptr);
        ptr = NULL;
    }
}

inline void check_state(cudaError_t cudaState)
{
#if 1
    if (cudaState != cudaSuccess)
    {
        printf("device memory allocation failed line: %d\n", __LINE__);
    }
#endif
}

template<bool A_RM, bool B_RM, bool C_RM>
void cpu_gemm_proc(int M, int N, int K, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            float sigma;
            sigma = 0.0f;

            for(int k=0; k<K; k++){
                int A_idx, B_idx;

                A_idx = A_RM? (i*lda + k) : (i + k*lda);
                B_idx = B_RM? (k*ldb + j) : (k + j*ldb);
                sigma += A[A_idx]*B[B_idx];
            }

            int C_idx;

            C_idx = C_RM? (i*ldc + j) : (i + j*ldc);
            C[C_idx] = sigma;
        }
    }

}

template<bool A_RM, bool B_RM, bool C_RM>
void cpu_gemm_test(int m, int n, int k, float* A_h, int lda, float* B_h, int ldb, float* C_h, int ldc)//C_h for verify C = A * B
{
    float* C_hh = nullptr;
    int space_C;

    space_C = C_RM? m*ldc: ldc*n;
    C_hh = (float*)malloc(space_C * sizeof(float));

    cpu_gemm_proc<A_RM, B_RM, C_RM>(m, n, k, A_h, lda, B_h, ldb, C_hh, ldc);

    printf("\nC_hh = cpu_geamm(A,B) =\n");
    print_matrix<C_RM>(C_hh, m, n, ldc);
    printf("\n\ncpu gemm end.\n\n");
    free_as_null(C_hh);
}

