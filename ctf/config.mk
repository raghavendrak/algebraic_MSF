CTF_DIR     = $(HOME)/ctf_rava/ctf
CRITTER_DIR = $(HOME)/critter
CXX         = mpicxx
OPTS        = -g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512
CXXFLAGS    = $(OPTS) -Wall -Wno-format -DMPIIO -DCRITTER
INCLUDES    = -I$(CRITTER_DIR)/src -I$(CTF_DIR)/include
LIB_PATH    = -L$(CRITTER_DIR)/lib -L$(CTF_DIR)/lib -L/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64 -Wl,--no-as-needed
LIBS        = -lcritter -lctf generator/libgraph_generator_mpi.a -lmkl_scalapack_lp64 -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lmkl_def -liomp5 -lpthread -lm -ldl
DEFS        =
CUDA_ARCH   = sm_37
NVCC        = $(CXX)
NVCCFLAGS   = $(CXXFLAGS)


