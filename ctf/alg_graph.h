#ifndef __ALG_GRAPH_H__
#define __ALG_GRAPH_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>
#include "graph_aux.h"

using namespace CTF;
#define SEED 23
typedef int wht;
#define MAX_WHT (INT_MAX/2)

static Semiring<wht> MAX_TIMES_SR(0,
    [](wht a, wht b) {
      return std::max(a, b);
    },
    MPI_MAX,
    1,
    [](wht a, wht b) {
      return a * b;
    });

// necessary for PTAP. TODO: refactor MAX_TIMES_SR to MIN_TIMES_SR
static Semiring<wht> MIN_TIMES_SR(MAX_WHT,
    [](wht a, wht b) {
      return std::min(a, b);
    },
    MPI_MIN,
    1,
    [](wht a, wht b) {
      return a * b;
    });

// utility //
uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges);
template <typename dtype>
int64_t are_vectors_different(CTF::Vector<dtype> & A, CTF::Vector<dtype> & B);
void init_pvector(Vector<int>* p);

// shortcut //
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);

// shortcut2 //
void shortcut2(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, int64_t sc2, World * world, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);
void roots_num(int64_t npairs, Pair<int> * loc_pairs, int64_t * loc_roots_num, int64_t * global_roots_num,  World * world);
void roots(int64_t npairs, int64_t loc_roots_num, Pair<int> * loc_pairs, int * global_roots,  World * world);
void create_nontriv_loc_indices(int64_t *& nontriv_loc_indices, int64_t * loc_nontriv_num, int64_t global_roots_num, int * global_roots, int64_t q_npairs, Pair<int> * q_loc_pairs, World * world);

// shortcut3 //
void shortcut3(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> & p_prev, MPI_Datatype & mpi_pkv, World * world);
// To collect changed parents in all processes
struct parentkv
{
  int64_t key;
  int64_t value;
};

// PTAP //
template<typename T>
Matrix<T>* PTAP(Matrix<T>* A, Vector<int>* p);

Vector<int> * star_check(Vector<int> * p, Vector<int> * gf = NULL);

// deprecated from connectivity //
class Int64Pair {
  public:
    int64_t i1;
    int64_t i2;

    Int64Pair(int64_t i1, int64_t i2);

    Int64Pair swap();
};

void mat_set(Matrix<int>* matrix, Int64Pair index, int value = 1);

class Graph {
  public:
    int numVertices;
    vector<Int64Pair>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};

#endif
