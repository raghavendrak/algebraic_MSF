include config.mk

all: test_connectivity

connectivity.o: connectivity.h connectivity.cxx $(CTFDIR)
	$(NVCC) $(NVCCFLAGS) -c connectivity.cxx  $(DEFS) $(INCLUDES) 

graph_io.o: graph_io.cxx graph_aux.h   $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph_io.cxx $(INCLUDES)

graph_gen.o: graph_gen.cxx graph_aux.h   $(CTFDIR)
	$(CXX) $(CXXFLAGS) -c graph_gen.cxx $(INCLUDES)

test_connectivity: graph_io.o graph_gen.o connectivity.o test_connectivity.cxx $(CTFDIR) 
	$(CXX) $(CXXFLAGS) -o test_connectivity test_connectivity.cxx connectivity.o graph_io.o graph_gen.o $(INCLUDES) $(LIBS)

clean:
	rm -f connectivity.o graph_gen.o graph_io.o test_connectivity
