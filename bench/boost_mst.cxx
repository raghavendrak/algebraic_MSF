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
#include <chrono>

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

typedef typename graph_traits<Graph>::vertex_descriptor vertex_descriptor;
typedef graph_traits<Graph>::edge_descriptor edge_descriptor;
typedef property_map<Graph, edge_weight_t>::type WeightMap;

typedef std::pair<int, int> E;

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

void get_graph(Graph & g, char * gfile, int n) { // TODO: not parallel, all processes write all edges
  std::ifstream in(gfile);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    int a, b, c;
    if (!(iss >> a >> b >> c)) { break; } // error
    boost::add_edge(boost::vertex(a-1, g), boost::vertex(b-1, g), { c }, g);
  }

  // // print graph
  // WeightMap weight_map = get(edge_weight, g);
  // typename graph_traits < Graph >::edge_iterator ei, ei_end;
  // boost::tie(ei, ei_end) = edges(g);
  // while (ei != ei_end) {
  //   vertex_descriptor u = source(*ei, g);
  //   vertex_descriptor v = target(*ei, g);
  //   std::cout << "(" << g.distribution().global(owner(u), local(u)) + 1
  //             << ", " << g.distribution().global(owner(v), local(v)) + 1
  //             << ", " << get(weight_map, *ei)
  //             << ")\n";
  //   ++ei;
  // }
}

template<typename Graph, typename WeightMap, typename InputIterator>
int
total_weight(const Graph& g, WeightMap weight_map, 
             InputIterator first, InputIterator last)
{
  typedef typename graph_traits<Graph>::vertex_descriptor vertex_descriptor;

  int total_weight = 0;
  while (first != last) {
    total_weight += get(weight_map, *first);
    // // print mst
    // if (process_id(g.process_group()) == 0) {
    //   vertex_descriptor u = source(*first, g);
    //   vertex_descriptor v = target(*first, g);
    //   std::cout << "(" << g.distribution().global(owner(u), local(u))
    //             << ", " << g.distribution().global(owner(v), local(v))
    //             << ")\n";
    // }
    ++first;
  }

  return total_weight;
}

int
test_mst(Graph & g, boost::mpi::communicator &world, int mode)
{
  if (process_id(g.process_group()) == 0)
    std::cerr << "--BOOST--\n";
  typedef property_map<Graph, edge_weight_t>::type WeightMap;
  WeightMap weight_map = get(edge_weight, g);

  std::vector<edge_descriptor> mst_edges;
  MPI_Barrier(world);
  auto start = std::chrono::high_resolution_clock::now();
  switch (mode) {
    case 0  : dense_boruvka_minimum_spanning_tree(make_vertex_list_adaptor(g), 
                                                  weight_map, 
                                                  std::back_inserter(mst_edges)); break;
    case 1  : merge_local_minimum_spanning_trees(make_vertex_list_adaptor(g), 
                                                 weight_map, 
                                                 std::back_inserter(mst_edges)); break;
    case 2  : boruvka_then_merge(make_vertex_list_adaptor(g), 
                                 weight_map, 
                                 std::back_inserter(mst_edges)); break;
    case 3 :  boruvka_mixed_merge(make_vertex_list_adaptor(g), 
                                  weight_map, 
                                  std::back_inserter(mst_edges)); break;
  }
  MPI_Barrier(world);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  if (world.rank() == 0) std::cout << "Time taken to run MST: " << duration.count() << std::endl;
  return total_weight(g, weight_map, mst_edges.begin(), mst_edges.end());
}

int test_main(int argc, char** argv)
{
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  int const in_num = argc;
  char** input_str = argv;
  char *metis_file = NULL;
  char *ctf_file = NULL;
  int n = 0;
  int mode = 0;

  if (getCmdOption(input_str, input_str+in_num, "-metis")){
    metis_file = getCmdOption(input_str, input_str+in_num, "-metis");
  } else metis_file = NULL;
  if (getCmdOption(input_str, input_str+in_num, "-ctf")){
    ctf_file = getCmdOption(input_str, input_str+in_num, "-ctf");
  } else ctf_file = NULL;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
  } else n = 0;
  if (getCmdOption(input_str, input_str+in_num, "-mode")){
    mode = atoi(getCmdOption(input_str, input_str+in_num, "-mode"));
  } else mode = 0;

  assert(0 <= mode && mode <= 3);
  if (ctf_file) {
    assert(!metis_file);
    assert(n > 0);
    Graph g(n);
    get_graph(g, ctf_file, n);
    if (world.rank() == 0)
      printf("reading graph is done\n");
    int mst_weight = test_mst(g, world, mode);
    if (world.rank() == 0)
      printf("boost mst weight: %d", mst_weight);
  } else {
    assert(metis_file);
    // Open the METIS input file
    std::ifstream in(metis_file);
    graph::metis_reader reader(in);
    // Load the graph using the default distribution
    Graph g(reader.begin(), reader.end(), reader.weight_begin(),
        reader.num_vertices());
    if (world.rank() == 0)
      printf("reading graph is done\n");
    int mst_weight = test_mst(g, world, mode);
    if (world.rank() == 0)
      printf("boost mst weight: %d", mst_weight);
  }

  return 0;
}
