NVCC = nvcc
FLAGS = -O2 -lineinfo

all: transpose_kernels

transpose_kernels: transpose_kernels.cu
	$(NVCC) $(FLAGS) -o transpose_kernels transpose_kernels.cu

run: transpose_kernels
	./transpose_kernels

profile: transpose_kernels
	ncu --set full --import-source yes --force-overwrite -o transpose_kernels_profile ./transpose_kernels
	ncu-ui transpose_kernels_profile.ncu-rep &

build: clean all

clean:
	rm -f transpose_kernels transpose_kernels_profile.ncu-rep