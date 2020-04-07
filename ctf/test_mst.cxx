#include "test.h"
#include "mst.h"
#include <ctime>

#include <boost/config.hpp>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

using namespace boost;

typedef adjacency_list < vecS, vecS, undirectedS,
property<vertex_distance_t, int>, property < edge_weight_t, int > > Boost_Graph;
typedef std::pair < int, int > E;

Boost_Graph matrix_to_graph(Matrix<EdgeExt> * A) {
  int64_t npair;
  Pair<EdgeExt> * pairs;
  A->get_all_pairs(&npair, &pairs, true); // TODO: too much overhead, work with vertices and edges from start for Boost and hook_matrix

  E edges[npair];
  int weights[npair];
  for (int64_t i = 0; i < npair; ++i) {
    edges[i] = E(pairs[i].d.src, pairs[i].d.dest); 
    weights[i] = pairs[i].d.weight;
  }

  const int num_nodes = A->nrow;

  Boost_Graph g(edges, edges + sizeof(edges) / sizeof(E), weights, num_nodes);

  delete [] pairs;

  return g;
}

// https://www.boost.org/doc/libs/1_55_0/libs/graph/example/prim-example.cpp
Vector<EdgeExt> * serial_prim(Boost_Graph g, World * world) {
  property_map<Boost_Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
  std::vector < graph_traits < Boost_Graph >::vertex_descriptor >
    p(num_vertices(g));

  property_map<Boost_Graph, vertex_distance_t>::type distance = get(vertex_distance, g);
  property_map<Boost_Graph, vertex_index_t>::type indexmap = get(vertex_index, g);
  prim_minimum_spanning_tree
    (g, *vertices(g).first, &p[0], distance, weightmap, indexmap, 
     default_dijkstra_visitor()); // TODO: occassionally segfaults or terminates

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  auto mst = new Vector<EdgeExt>(p.size(), *world, MIN_EDGE); // TODO: too much overhead? just return weight sum?

  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[p.size()];
  for (int64_t i = 0; i < p.size(); ++i) {
    pairs[i].k = i;
    pairs[i].d = EdgeExt(i, distance[i], p[i], -1);
  }
  mst->write(p.size(), pairs);

  delete [] pairs;

  return mst;
}

// requires edge weights to be distinct
// can also store mst in hashset
double compare_mst(Vector<EdgeExt> * a, Vector<EdgeExt> * b) {
  Function<EdgeExt,double> sum_weights([](EdgeExt a){ return a.src != -1 ? a.weight : 0; });

  Scalar<double> s_a;
  s_a[""] = sum_weights((*a)["i"]);

  Scalar<double> s_b;
  s_b[""] = sum_weights((*b)["i"]);

  return s_a.get_val() - s_b.get_val();
}

// requires edge weights to be distinct
// can also store mst in hashset
/*
int64_t compare_mst(Vector<EdgeExt> * a, Vector<EdgeExt> * b) {
  int64_t a_n;
  Pair<EdgeExt> * a_pairs; 
  a->get_all_pairs(&a_n, &a_pairs, true);
  std::sort(a_pairs, a_pairs + a_n, [](const Pair<EdgeExt> & lhs, const Pair<EdgeExt> & rhs) { return lhs.d.weight < rhs.d.weight; });

  int64_t b_n;
  Pair<EdgeExt> * b_pairs; 
  b->get_all_pairs(&b_n, &b_pairs, true);
  std::sort(b_pairs, b_pairs + b_n, [](const Pair<EdgeExt> & lhs, const Pair<EdgeExt> & rhs) { return lhs.d.weight < rhs.d.weight; });

  for (int64_t i = 0; i < a_n; ++i) {
    a_pairs[i].k = i;
    b_pairs[i].k = i;
  }

  a->write(a_n, a_pairs);
  b->write(b_n, b_pairs);

  // mst may store edge from src to parent or parent to src
  CTF::Scalar<int64_t> s;
  s[""] += CTF::Function<EdgeExt,EdgeExt,int64_t>([](EdgeExt a, EdgeExt b){ 
    return !(((a.src == b.src && a.dest == b.dest) || (a.src == b.dest && a.dest == b.src)) && a.weight == b.weight) ; 
  })((*a)["i"],(*b)["i"]);

  delete b_pairs;
  delete a_pairs;

  return s.get_val();
}
*/

//  0 --- 1
//  |   /     3 -- 4
//  | /
//  2
void test_trivial(World * w) {
  if (w->rank == 0) {
    printf("test_trivial\n");
  }

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
   
  int nrow = 5;
  Matrix<EdgeExt> * A = new Matrix<EdgeExt>(nrow, nrow, SP, *w, MIN_EDGE);

  int64_t npair = 2 * 4;
  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[npair];
  pairs[0] = Pair<EdgeExt>(1 * nrow + 0, EdgeExt(0, 7, 1, 0));
  pairs[1] = Pair<EdgeExt>(2 * nrow + 0, EdgeExt(0, 5, 2, 0));
  pairs[2] = Pair<EdgeExt>(2 * nrow + 1, EdgeExt(1, 6, 2, 1));
  pairs[3] = Pair<EdgeExt>(4 * nrow + 3, EdgeExt(3, 10, 4, 3));

  // perturb edge weights and produce anti symmetry
  std::srand(time(NULL));
  for (int64_t i = 0; i < npair / 2; ++i) {
    pairs[i].d.weight += (std::rand() / (double) RAND_MAX) / 1000;

    pairs[i + npair / 2].k = (pairs[i].k % nrow) * nrow + pairs[i].k / nrow;
    pairs[i + npair / 2].d = EdgeExt(pairs[i].d.dest, pairs[i].d.weight, pairs[i].d.src, pairs[i].d.dest);
  }

  A->write(npair, pairs);

  auto hm = hook_matrix(A, w);
  if (w->rank == 0) {
    printf("hook_matrix mst\n");
  }
  hm->print();

  //auto res = compare_mst(kr, hm);
  //if (w->rank == 0) {
  //  if (res) {
  //    printf("result weight of mst vectors are different by %f: FAIL\n", res);
  //  }
  //  else {
  //    printf("result weight mst vectors are same: PASS\n");
  //  }
  //}

  delete hm;

  delete [] pairs;
  delete A;
}

// graph pictured here: https://i0.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-1.png?zoom=2.625&resize=368%2C236&ssl=1
// mst pictured here: https://i1.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-12.png?zoom=2&resize=382%2C237&ssl=1
void test_simple(World * w) {
  if (w->rank == 0) {
    printf("test_simple\n");
  }

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
   
  int nrow = 7;
  Matrix<EdgeExt> * A = new Matrix<EdgeExt>(nrow, nrow, SP, *w, MIN_EDGE);

  int64_t npair = 2 * 11;
  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[npair];
  pairs[0] = Pair<EdgeExt>(1 * nrow + 0, EdgeExt(0, 7, 1, 0));
  pairs[1] = Pair<EdgeExt>(3 * nrow + 0, EdgeExt(0, 5, 3, 0));
  pairs[2] = Pair<EdgeExt>(2 * nrow + 1, EdgeExt(1, 8, 2, 1));
  pairs[3] = Pair<EdgeExt>(3 * nrow + 1, EdgeExt(1, 9, 3, 1));
  pairs[4] = Pair<EdgeExt>(4 * nrow + 1, EdgeExt(1, 7, 4, 1));
  pairs[5] = Pair<EdgeExt>(4 * nrow + 2, EdgeExt(2, 5, 4, 2));
  pairs[6] = Pair<EdgeExt>(4 * nrow + 3, EdgeExt(3, 15, 4, 3));
  pairs[7] = Pair<EdgeExt>(5 * nrow + 3, EdgeExt(3, 6, 5, 3));
  pairs[8] = Pair<EdgeExt>(5 * nrow + 4, EdgeExt(4, 8, 5, 4));
  pairs[9] = Pair<EdgeExt>(6 * nrow + 4, EdgeExt(4, 9, 6, 4));
  pairs[10] = Pair<EdgeExt>(6 * nrow + 5, EdgeExt(5, 11, 6, 5));

  // produce anti symmetry
  std::srand(std::time(NULL));
  for (int64_t i = 0; i < npair / 2; ++i) {
    pairs[i + npair / 2].k = (pairs[i].k % nrow) * nrow + pairs[i].k / nrow;
    pairs[i + npair / 2].d = EdgeExt(pairs[i].d.dest, pairs[i].d.weight, pairs[i].d.src, pairs[i].d.dest);
  }

  A->write(npair, pairs);

  auto hm = hook_matrix(A, w);
  if (w->rank == 0) {
    printf("hook_matrix mst\n");
  }
  hm->print();

  Boost_Graph g = matrix_to_graph(A);
  auto serial = serial_prim(g, w);
  if (w->rank == 0)
    printf("serial prim\n");

  auto res = compare_mst(serial, hm);
  if (w->rank == 0) {
    if (res) {
      printf("result weight of mst vectors are different by %f: FAIL\n", res);
    }
    else {
      printf("result weight of mst vectors are same: PASS\n");
    }
  }

  delete serial;
  delete hm;

  delete [] pairs;
  delete A;
}

void run_mst(Matrix<EdgeExt>* A, int64_t matSize, World *w, int batch, int shortcut, int run_serial)
{
  matSize = A->nrow; // Quick fix to avoid change in i/p matrix size after preprocessing
  double stime;
  double etime;
  //Timer_epoch thm("hook_matrix");
  //thm.begin();
  stime = MPI_Wtime();
  auto hm = hook_matrix(A, w);
  etime = MPI_Wtime();
  if (w->rank == 0) {
    printf("Time for hook_matrix(): %1.2lf\n", (etime - stime));
    printf("hook_matrix() mst:\n");
  }
  hm->print();
  printf("\n");
  //thm.end();

  if (run_serial) {
    Boost_Graph g = matrix_to_graph(A);
    auto serial = serial_prim(g, w);
    if(w->rank == 0) printf("serial prim\n");
    serial->print();
    auto res = compare_mst(serial, hm);
    if (w->rank == 0) {
      if (res) {
        printf("hook_matrix and serial mst vectors are different by %f: FAIL\n", res);
      }
      else {
        printf("hook_matrix and serial mst vectors are same: PASS\n");
      }
    }
  }
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

int main(int argc, char** argv)
{
  int rank;
  int np;
  int const in_num = argc;
  char** input_str = argv;
  uint64_t myseed;

  int64_t max_ewht;
  uint64_t edges;
  char *gfile = NULL;
  int64_t n;
  int scale;
  int ef;
  int prep;
  int batch;
  int sc2;
  int run_serial;

  int k;
  MPI_Init(&argc, &argv);
  auto w = new World(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    if (getCmdOption(input_str, input_str+in_num, "-k")) {
      k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
      if (k < 0) k = 5;
    } else k = -1;
    // K13 : 1594323 (matrix size)
    // K6 : 729; 531441 vertices
    // k5 : 243
    // k7 : 2187
    // k8 : 6561
    // k9 : 19683
    if (getCmdOption(input_str, input_str+in_num, "-f")){
      gfile = getCmdOption(input_str, input_str+in_num, "-f");
    } else gfile = NULL;
    if (getCmdOption(input_str, input_str+in_num, "-n")){
      n = atoll(getCmdOption(input_str, input_str+in_num, "-n"));
      if (n < 0) n = 27;
    } else n = 27;
    if (getCmdOption(input_str, input_str+in_num, "-S")){
      scale = atoi(getCmdOption(input_str, input_str+in_num, "-S"));
      if (scale < 0) scale=10;
    } else scale=0;
    if (getCmdOption(input_str, input_str+in_num, "-E")){
      ef = atoi(getCmdOption(input_str, input_str+in_num, "-E"));
      if (ef < 0) ef=16;
    } else ef=0;
    if (getCmdOption(input_str, input_str+in_num, "-prep")){
      prep = atoll(getCmdOption(input_str, input_str+in_num, "-prep"));
      if (prep < 0) prep = 0;
    } else prep = 0;
    if (getCmdOption(input_str, input_str+in_num, "-batch")){
      batch = atoll(getCmdOption(input_str, input_str+in_num, "-batch"));
      if (batch <= 0) batch = 1;
    } else batch = 1;
    if (getCmdOption(input_str, input_str+in_num, "-shortcut")){
      sc2 = atoi(getCmdOption(input_str, input_str+in_num, "-shortcut"));
      if (sc2 < 0) sc2 = 0;
    } else sc2 = 0;
    if (getCmdOption(input_str, input_str+in_num, "-serial")){
      run_serial = atoi(getCmdOption(input_str, input_str+in_num, "-serial"));
      if (run_serial < 0) run_serial = 0;
    } else run_serial = 0;

    if (gfile != NULL){
      //int n_nnz = 0;
      //if (w->rank == 0)
      //printf("Reading real graph n = %lld\n", n);
      //Matrix<wht> A = read_matrix(*w, n, gfile, prep, &n_nnz);
      //run_connectivity(&A, n, w, batch, sc2, run_serial);
    }
    else if (k != -1) {
      //int64_t matSize = pow(3, k);
      //auto B = generate_kronecker(w, k);

      //if (w->rank == 0) {
      //printf("Running connectivity on Kronecker graph K: %d matSize: %ld\n", k, matSize);
      //}
      //run_connectivity(B, matSize, w, batch, sc2, run_serial);
      //delete B;
    }
    else if (scale > 0 && ef > 0){
      int n_nnz = 0;
      myseed = SEED;
      if (w->rank == 0)
        printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
      Matrix<EdgeExt> A = gen_rmat_matrix<EdgeExt>(*w, scale, ef, myseed, prep, &n_nnz, max_ewht);
      int64_t matSize = A.nrow; 
      run_mst(&A, matSize, w, batch, sc2, run_serial);
    }
    else {
      if (w->rank == 0) {
        printf("Running mst on simple 7x7 graph\n");
      }
      test_simple(w);
      //test_trivial(w);
    }
  }
  MPI_Finalize();
  return 0;
}
