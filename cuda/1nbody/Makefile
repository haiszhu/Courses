# Makefile for compiling MEX files in 1nbody

# Variables
NVCC = /usr/local/cuda-13.1/bin/nvcc
GCC = /usr/bin/g++
MATLAB_ROOT = $(shell ls -d /usr/local/MATLAB/R* 2>/dev/null | sort | tail -n1)
CUDA_ROOT = /usr/local/cuda-13.1
# Auto-detect GPU compute capabilities (removes dots and makes space-separated)
CUDA_ARCH = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sed 's/\.//g' | tr '\n' ' ')

# Common flags
NVCC_COMMON = -c --compiler-options=-D_GNU_SOURCE,-DMATLAB_MEX_FILE \
              -I"$(CUDA_ROOT)/include" \
              -I"$(MATLAB_ROOT)/extern/include" \
              -I"$(MATLAB_ROOT)/simulink/include" \
              -I"$(MATLAB_ROOT)/toolbox/parallel/gpu/extern/include/" \
              -std=c++11

LINK_COMMON = -pthread -Wl,--no-undefined -Wl,--no-as-needed -shared -O \
              -Wl,--version-script,"$(MATLAB_ROOT)/extern/lib/glnxa64/mexFunction.map" \
              -ldl \
              $(CUDA_ROOT)/targets/x86_64-linux/lib/libcusparse.so \
              $(CUDA_ROOT)/targets/x86_64-linux/lib/libcublas_static.a \
              $(CUDA_ROOT)/targets/x86_64-linux/lib/libcusparse_static.a \
              $(CUDA_ROOT)/targets/x86_64-linux/lib/libculibos.a \
              -L$(CUDA_ROOT)/lib64 -Wl,-rpath-link,$(MATLAB_ROOT)/bin/glnxa64 \
              -L"$(MATLAB_ROOT)/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -lmwgpu -lcufft -lcufftw -lcublas -lcublasLt \
              $(CUDA_ROOT)/targets/x86_64-linux/lib/libcudart.so

# Generate gencode flags from CUDA_ARCH
GENCODE_FLAGS = $(foreach arch,$(CUDA_ARCH),-gencode arch=compute_$(arch),code=sm_$(arch))

# Targets
all: mexGPUcublas.mexa64 mexGPUlapslppot.mexa64

mexGPUcublas.mexa64: mexGPUcublas.cu
	$(NVCC) $(NVCC_COMMON) \
	        $(GENCODE_FLAGS) \
	        -Xcompiler "-fexceptions -fPIC -fno-omit-frame-pointer -pthread" \
	        -O2 -DNDEBUG \
	        mexGPUcublas.cu -o mexGPUcublas.o
	$(GCC) $(LINK_COMMON) mexGPUcublas.o -o mexGPUcublas.mexa64

mexGPUlapslppot.mexa64: mexGPUlapslppot.cu
	$(NVCC) $(NVCC_COMMON) \
	        $(GENCODE_FLAGS) \
	        -Xcompiler "-fexceptions -fPIC -fno-omit-frame-pointer -pthread" \
	        -O0 -DNDEBUG \
	        mexGPUlapslppot.cu -o mexGPUlapslppot.o
	$(GCC) $(LINK_COMMON) mexGPUlapslppot.o -o mexGPUlapslppot.mexa64

clean:
	rm -f *.o *.mexa64

.PHONY: all clean