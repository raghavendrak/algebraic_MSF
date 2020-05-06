// Copyright 2004 The Trustees of Indiana University.

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  Authors: Douglas Gregor
//           Andrew Lumsdaine

#include <boost/graph/use_mpi.hpp>
#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/graph/distributed/dehne_gotz_min_spanning_tree.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/distributed/vertex_list_adaptor.hpp>
#include <boost/graph/parallel/distribution.hpp>
#include <boost/test/minimal.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <iostream>
#include <cstdlib>
#include <boost/graph/metis.hpp>
#include <boost/graph/distributed/graphviz.hpp>

#ifdef BOOST_NO_EXCEPTIONS
void
boost::throw_exception(std::exception const& ex)
{
    std::cout << ex.what() << std::endl;
    abort();
}
#endif

typedef double wht;

using namespace boost;
using boost::graph::distributed::mpi_process_group;

typedef adjacency_list<listS, 
                       distributedS<mpi_process_group, vecS>,
                       undirectedS,
                       // Vertex properties
                       no_property,
                       // Edge properties
                       property<edge_weight_t, int> > Graph;

typedef graph_traits<Graph>::edge_descriptor edge_descriptor;

typedef std::pair<int, int> E;

template<typename Graph, typename WeightMap, typename InputIterator>
int
total_weight(const Graph& g, WeightMap weight_map, 
             InputIterator first, InputIterator last)
{
  typedef typename graph_traits<Graph>::vertex_descriptor vertex_descriptor;

  int total_weight = 0;
  while (first != last) {
    total_weight += get(weight_map, *first);
    if (process_id(g.process_group()) == 0) {
      vertex_descriptor u = source(*first, g);
      vertex_descriptor v = target(*first, g);
      std::cout << "(" << g.distribution().global(owner(u), local(u))
                << ", " << g.distribution().global(owner(v), local(v))
                << ")\n";
    }
    ++first;
  }

  return total_weight;
}

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int
test_distributed_dense_boruvka(const char* filename)
{
  // Open the METIS input file
  std::ifstream in(filename);
  graph::metis_reader reader(in);

  // Load the graph using the default distribution
  Graph g(reader.begin(), reader.end(), reader.weight_begin(),
          reader.num_vertices());

  if (process_id(g.process_group()) == 0)
    std::cerr << "--BOOST--\n";
  typedef property_map<Graph, edge_weight_t>::type WeightMap;
  WeightMap weight_map = get(edge_weight, g);

  std::vector<edge_descriptor> mst_edges;
  switch (2) {
    case 0  : dense_boruvka_minimum_spanning_tree(make_vertex_list_adaptor(g), 
                                                  weight_map, 
                                                  std::back_inserter(mst_edges)); break;
    case 1  : merge_local_minimum_spanning_trees(make_vertex_list_adaptor(g), 
                                                 weight_map, 
                                                 std::back_inserter(mst_edges)); break;
    case 2  : boruvka_then_merge(make_vertex_list_adaptor(g), 
                                 weight_map, 
                                 std::back_inserter(mst_edges)); break;
    default : boruvka_mixed_merge(make_vertex_list_adaptor(g), 
                                  weight_map, 
                                  std::back_inserter(mst_edges)); break;
  }

  return total_weight(g, weight_map, mst_edges.begin(), mst_edges.end());
}

int test_main(int argc, char** argv)
{
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  int const in_num = argc;
  char** input_str = argv;
  char *gfile = NULL;

  if (getCmdOption(input_str, input_str+in_num, "-f")){
    gfile = getCmdOption(input_str, input_str+in_num, "-f");
  } else gfile = NULL;
  int mst_weight = test_distributed_dense_boruvka(gfile);
  if (world.rank() == 0)
    printf("boost mst weight: %d", mst_weight);

  return 0;
}
