include config.mk

all: test_connectivity test_mst

alg_graph.o: alg_graph.h alg_graph.cxx $(CTFDIR)
	$(NVCC) $(NVCCFLAGS) -c alg_graph.cxx  $(DEFS) $(INCLUDES) 

connectivity.o: alg_graph.o connectivity.h connectivity.cxx $(CTFDIR)
	$(NVCC) $(NVCCFLAGS) -c connectivity.cxx alg_graph.o $(DEFS) $(INCLUDES) 

mst.o: alg_graph.o mst.h mst.cxx $(CTFDIR)
	$(NVCC) $(NVCCFLAGS) -c mst.cxx alg_graph.o $(DEFS) $(INCLUDES) 

graph_io.o: graph_io.cxx graph_aux.h   $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph_io.cxx $(INCLUDES)

graph_gen.o: graph_gen.cxx graph_aux.h   $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph_gen.cxx $(INCLUDES)

test.o: alg_graph.o mst.o test.h test.cxx $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c test.cxx mst.o alg_graph.o $(INCLUDES)

test_connectivity: alg_graph.o graph_io.o graph_gen.o connectivity.o mst.o test.o test_connectivity.cxx $(CTFDIR) 
	$(CXX) $(CXXFLAGS) -o test_connectivity test_connectivity.cxx test.o connectivity.o mst.o graph_io.o graph_gen.o alg_graph.o $(INCLUDES) $(LIB_PATH) $(LIBS)

test_mst: alg_graph.o graph_io.o graph_gen.o mst.o test.o test_mst.cxx $(CTFDIR) 
	$(CXX) $(CXXFLAGS) -o test_mst test_mst.cxx test.o mst.o graph_io.o graph_gen.o alg_graph.o $(INCLUDES) $(LIB_PATH) $(LIBS)

clean:
	rm -f alg_graph.o connectivity.o mst.o graph_gen.o graph_io.o test.o test_connectivity test_mst
