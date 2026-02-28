NVCC = nvcc
FLAGS = -O2 -lineinfo

all: GEMM

GEMM: GEMM.cu
	$(NVCC) $(FLAGS) -o GEMM GEMM.cu

run: GEMM
	./GEMM

profile: GEMM
	ncu --set full --import-source yes --force-overwrite -o GEMM_profile ./GEMM
	ncu-ui GEMM_profile.ncu-rep &

build: clean all

clean:
	rm -f GEMM GEMM_profile.ncu-rep