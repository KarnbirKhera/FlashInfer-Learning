NVCC = nvcc
FLAGS = -O2 -lineinfo

all: reduced_sum_kernels

reduced_sum_kernels: reduced_sum_kernels.cu
	$(NVCC) $(FLAGS) -o reduced_sum_kernels reduced_sum_kernels.cu

run: reduced_sum_kernels
	./reduced_sum_kernels

profile: reduced_sum_kernels
	ncu --set full --import-source yes --force-overwrite -o reduced_sum_kernels_profile ./reduced_sum_kernels
	ncu-ui reduced_sum_kernels_profile.ncu-rep &

build: clean all

clean:
	rm -f reduced_sum_kernels reduced_sum_kernels_profile.ncu-rep