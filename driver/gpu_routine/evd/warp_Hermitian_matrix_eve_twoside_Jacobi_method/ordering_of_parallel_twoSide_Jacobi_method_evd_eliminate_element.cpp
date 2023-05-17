// Ordering_twoside_Jacobi_method.cpp
// ordering_evd_Parallel_Jacobi_method.cpp
//

#include <iostream>





int all_sweep_order_symmetric_matrix(int n)//
{
    //int n = 16;
    int m = (n + 1) / 2;// 8
    int p, q;

    for (int k = 1; k <= m - 1; k++)//k-th sweep
    {
        for (int ele = 1; ele <= n - m; ele++)
        {
            p = 9999;
            q = 9999;
            q = m - k + ele;//  m-k+1, m-k+2, ... , n-k

            if (m - k + 1 <= q && q <= 2 * m - 2 * k)
            {
                p = (2 * m - 2 * k + 1) - q;
            }
            else if (2 * m - 2 * k < q && q <= 2 * m - k - 1)
            {
                p = (4 * m - 2 * k) - q;
            }
            else if (2 * m - k - 1 < q)
            {
                p = n;
            }
            printf("(%d, %d) ", p, q);
        }
        printf("\n\n");
    }
    printf("///\n");
    for (int k = m; k < 2 * m; k++)
    {
        for (int ele = 1; ele <= n - m; ele++)
        {
            p = 9999;
            q = 9999;
            q = 4 * m - n - k + ele - 1;
            if (q < 2 * m - k + 1)
            {
                p = n;
            }
            else if (2 * m - k + 1 <= q && q <= 4 * m - 2 * k - 1)
            {
                p = 4 * m - 2 * k - q;
            }
            else if (4 * m - 2 * k - 1 < q)
            {
                p = 6 * m - 2 * k - 1 - q;
            }
            printf("(%d, %d) ", p, q);
        }
        printf("\n\n");
    }

    return 0;
}

int kth_batch_of_all_sweep_order_symmetric_matrix(int n, int k)//
{
    //int n = 16;
    int m = (n + 1) / 2;// 8
    int p, q;

    //for (int k = 1; k <= m - 1; k++)//k-th sweep    {
    if (1 <= k && k <= m - 1) {
        for (int ele = 1; ele <= n - m; ele++)
        {
            p = 9999;
            q = 9999;
            q = m - k + ele;//  m-k+1, m-k+2, ... , n-k

            if (m - k + 1 <= q && q <= 2 * m - 2 * k)
            {
                p = (2 * m - 2 * k + 1) - q;
            }
            else if (2 * m - 2 * k < q && q <= 2 * m - k - 1)
            {
                p = (4 * m - 2 * k) - q;
            }
            else if (2 * m - k - 1 < q)
            {
                p = n;
            }
            printf("(%d, %d) ", p, q);
        }
        printf("\n");
    }
    else if (m <= k && k < 2 * m)//printf("///\n");    //for (int k = m; k < 2 * m; k++){
    {
        for (int ele = 1; ele <= n - m; ele++)
        {
            p = 9999;
            q = 9999;
            q = 4 * m - n - k + ele - 1;
            if (q < 2 * m - k + 1)
            {
                p = n;
            }
            else if (2 * m - k + 1 <= q && q <= 4 * m - 2 * k - 1)
            {
                p = 4 * m - 2 * k - q;
            }
            else if (4 * m - 2 * k - 1 < q)
            {
                p = 6 * m - 2 * k - 1 - q;
            }
            printf("(%d, %d) ", p, q);
        }
        printf("\n");
    }

    return 0;
}


int ith_ele_of_kth_batch_of_all_sweep_order_symmetric_matrix(int n, int k, int tid)//
{
    //int n = 16;
    int m = (n + 1) / 2;// 8
    int p, q;
    int ele = tid;
    //for (int k = 1; k <= m - 1; k++)//k-th sweep    {
    if (1 <= k && k <= m - 1) {
        //for (int ele = 1; ele <= n - m; ele++)
        if (1 <= ele && ele <= n - m)
        {
            p = 9999;
            q = 9999;
            q = m - k + ele;//  m-k+1, m-k+2, ... , n-k

            if (m - k + 1 <= q && q <= 2 * m - 2 * k)
            {
                p = (2 * m - 2 * k + 1) - q;
            }
            else if (2 * m - 2 * k < q && q <= 2 * m - k - 1)
            {
                p = (4 * m - 2 * k) - q;
            }
            else if (2 * m - k - 1 < q)
            {
                p = n;
            }
            printf("(%d, %d) ", p, q);
        }
        //printf("\n\n");
    }
    else if (m <= k && k < 2 * m)//printf("///\n");    //for (int k = m; k < 2 * m; k++){
    {
        //for (int ele = 1; ele <= n - m; ele++)
        if (1 <= ele && ele <= n - m)
        {
            p = 9999;
            q = 9999;
            q = 4 * m - n - k + ele - 1;
            if (q < 2 * m - k + 1)
            {
                p = n;
            }
            else if (2 * m - k + 1 <= q && q <= 4 * m - 2 * k - 1)
            {
                p = 4 * m - 2 * k - q;
            }
            else if (4 * m - 2 * k - 1 < q)
            {
                p = 6 * m - 2 * k - 1 - q;
            }
            printf("(%d, %d) ", p, q);
        }
        //printf("\n\n");
    }

    return 0;
}


int main()
{
    int n = 40;
    //all_sweep_order_symmetric_matrix(n);
    printf("\n\n\n");
    for (int k = 1; k <= 2 * ((n + 1) / 2) - 1; k++)
    {
        kth_batch_of_all_sweep_order_symmetric_matrix(n, k);
    }
    printf("\n\n");

    int batch = 2 * ((n + 1) / 2) - 1;//last batch
    int tid = (n + 1) / 2;
    //ith_ele_of_kth_batch_of_all_sweep_order_symmetric_matrix(n, 39, 20);

    int m = (n + 1) / 2;
    for (int batch = 1; batch <= 2 * m - 1; batch++) {
        for (int ele = 1; ele <= m; ele++) {
            ith_ele_of_kth_batch_of_all_sweep_order_symmetric_matrix(n, batch, ele);
        }
        printf("\n");

    }
    printf("\n");

    return 0;
}

