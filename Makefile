NVCC = nvcc
ARCH ?= 75
SUPPORTED_ARCHS := 35 37 50 52 60 61 62 70 72 75 80 86

ifeq (,$(findstring $(ARCH), $(SUPPORTED_ARCHS)))
 $(error Unsupported Architecture $(ARCH))
endif

NVCC_OPTIONS = --generate-code arch=compute_$(ARCH),code=sm_$(ARCH)
INCLUDES = -I$(CUDA_HOME)/include

APPS = gpu_sort radix_sort radix_pair_sort

all: $(APPS)

%.o : %.cu
	$(NVCC) -c $< -o $@ $(NVCC_OPTIONS) $(INCLUDES)

$(APPS) : % : %.o
	$(NVCC) $< -o $@ $(NVCC_OPTIONS) $(INCLUDES)

clean:
	rm -f $(APPS) *.o

.PHONY: all clean

