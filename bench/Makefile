include config.mk

all: boost_mst stapl_mst

boost_mst: boost_mst.cxx
	$(CXX) $(CXXFLAGS) -o boost_mst boost_mst.cxx $(BOOST_INCLUDES) $(BOOST_LIBS)

stapl_mst: stapl_mst.cxx
	$(CXX) $(CXXFLAGS) -o stapl_mst stapl_mst.cxx $(STAPL_INCLUDES) $(STAPL_LIBS)

clean:
	rm -f boost_mst stapl_mst
