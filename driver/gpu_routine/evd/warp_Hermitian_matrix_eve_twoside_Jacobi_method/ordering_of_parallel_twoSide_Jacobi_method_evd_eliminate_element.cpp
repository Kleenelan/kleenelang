// Ordering_twoside_Jacobi_method.cpp
// ordering_evd_Parallel_Jacobi_method.cpp
//

#include <iostream>

int main()
{
    int n = 16;//A(nxn) symmetric matrix 
    int m = (n + 1) / 2;// 8 elements is eliminate in a batch, 2m-1 batch is a sweep, that eliminate all offdiagonal element once time;
    int p, q;//(p, q) pair is a element

    for (int k = 1; k <= m - 1; k++)//k-th sweep
    {
        for (int batch = 1; batch <= n - m; batch++)
        {
            int branch;
            branch = 0;
            p = 9999;
            q = 9999;
            q = m - k + batch;//  m-k+1, m-k+2, ... , n-k

            if (m - k + 1 <= q && q <= 2 * m - 2 * k)
            {
                branch = 1;
                p = (2 * m - 2 * k + 1) - q;
            }
            else if (2 * m - 2 * k < q && q <= 2 * m - k - 1)
            {
                branch = 2;
                p = (4 * m - 2 * k) - q;
            }
            else if (2 * m - k - 1 < q)
            {
                branch = 3;
                p = n;
            }
            printf("(%d, %d) ", p, q);
        }
        printf("\n\n");
    }
    printf("///////////////////////////////////////////////////////////////\n");
    for (int k = m; k < 2 * m; k++)
    {
        for (int batch = 0; batch <= n - m - 1; batch++)
        {
            p = 9999;
            q = 9999;
            q = 4*m - n - k + batch;
            if(q<2*m-k+1)
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
}


