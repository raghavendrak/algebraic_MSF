MPI_DIR   =
CXX       = mpicxx
OPTS      = -O0 -g
#CXXFLAGS  = -std=c++0x -fopenmp $(OPTS) -Wall -DPROFILE -DPMPI -DMPIIO
CXXFLAGS  = -std=c++0x $(OPTS) -Wall -Wno-format -DPMPI -DMPIIO
INCLUDES  =
LIBS      = -L /usr/local/lib -lboost_system -lboost_serialization -lboost_mpi -lboost_graph_parallel
#LIBS      = -lctf -lblas generator/libgraph_generator_mpi.a -llapack -lblas 
DEFS      =
CUDA_ARCH = sm_37
NVCC      = $(CXX)
NVCCFLAGS = $(CXXFLAGS)
