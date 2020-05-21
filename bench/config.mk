MPI_DIR         =
CXX             = mpicxx
OPTS            = -O0 -g
CXXFLAGS        = -std=c++0x $(OPTS) -Wall -Wno-format -DPMPI -DMPIIO
BOOST_INCLUDES  =
BOOST_LIBS      = -L /usr/local/lib -lboost_system -lboost_serialization -lboost_mpi -lboost_graph_parallel
STAPL_INCLUDES  = 
STAPL_LIBS      = 
DEFS            =
CUDA_ARCH       = sm_37
NVCC            = $(CXX)
NVCCFLAGS       = $(CXXFLAGS)
