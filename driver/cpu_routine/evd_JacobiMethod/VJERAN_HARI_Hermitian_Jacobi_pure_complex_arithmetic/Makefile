all: hello_Vjeran_Hermite_Jacobi hello_Her_Jac_Vjer_stable double_hello_Her_Jac_Vjer_stable

hello_Vjeran_Hermite_Jacobi: hello_VJERAN_HARI_Hermite_twoSideJacobiMethod_evd.cpp
	g++ $^ -o $@

hello_Her_Jac_Vjer_stable: hello_Her_Jac_Vjer_stable.cpp
	g++ $^ -o $@

double_hello_Her_Jac_Vjer_stable: double_hello_Her_Jac_Vjer_stable.cpp
	g++ $^ -o $@

.PHONY: clean
clean:
	rm -f hello_Vjeran_Hermite_Jacobi hello_Her_Jac_Vjer_stable double_hello_Her_Jac_Vjer_stable
