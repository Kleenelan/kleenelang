

hello: warp_Jacobi_Hermitian_Matrix_EigenValue_sorted_eigenVector_exchanged.cu 
	nvcc $< -o $@  -L /usr/local/cuda/lib64 -lcusolver

.PHONY: clean
clean:
	rm -rf hello
