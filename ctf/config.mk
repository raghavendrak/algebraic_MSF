CTFDIR    = /Users/timbaer/lpna/raghavendrak-ctf
MPI_DIR   =
CXX       = mpicxx -cxx=g++
OPTS      = -O0 -g
#CXXFLAGS  = -std=c++0x -fopenmp $(OPTS) -Wall -DPROFILE -DPMPI -DMPIIO
CXXFLAGS  = -std=c++0x $(OPTS) -Wall -Wno-format -DPMPI -DMPIIO -ferror-limit=200
INCLUDES  = -I$(CTFDIR)/include
LIBS      = -L$(CTFDIR)/lib -lctf -lblas generator/libgraph_generator_mpi.a -llapack -lblas
#LIBS      = -lctf -lblas generator/libgraph_generator_mpi.a -llapack -lblas 
DEFS      =
CUDA_ARCH = sm_37
NVCC      = $(CXX)
NVCCFLAGS = $(CXXFLAGS)
