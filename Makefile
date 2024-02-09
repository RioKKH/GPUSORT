ARCH ?= 75
SUPPORTED_ARCHS := 35 37 50 52 60 61 62 70 72 75 80 86

ifeq (,$(findstring $(ARCH), $(SUPPORTED_ARCHS)))
 $(error Unsupported Architecture $(ARCH))
endif

NVCC_OPTIONS = --generate-code arch=compute_$(ARCH),code=sm_$(ARCH)

APPS = gpu_sort

all: $(APPS)

%.o : %.cu
	nvcc -c $< -o $@ $(NVCC_OPTIONS)

$(APPS) : % : %.o
	nvcc $< -o $@ $(NVCC_OPTIONS)

clean:
	rm -f $(APPS) *.o

.PHONY: all clean

