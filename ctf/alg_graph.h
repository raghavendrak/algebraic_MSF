#ifndef __ALG_GRAPH_H__
#define __ALG_GRAPH_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>
#include "graph_aux.h"

using namespace CTF;
#define SEED 23
typedef int mlt;
typedef int wht;
#define MAX_WHT (INT_MAX/2)
//typedef double wht;
//#define MAX_WHT (DBL_MAX/2)

static Semiring<int> MAX_TIMES_SR(0,
    [](int a, int b) {
    return std::max(a, b);
    },
    MPI_MAX,
    1,
    [](int a, int b) {
    return a * b;
    });

class Int64Pair {
  public:
    int64_t i1;
    int64_t i2;

    Int64Pair(int64_t i1, int64_t i2);

    Int64Pair swap();
};

struct EdgeExt {
  int64_t src, dest, parent;
  double weight;
  EdgeExt() { src = -1; weight = DBL_MAX; dest = -1; parent = -1; } // addid
  EdgeExt(int64_t src_, double weight_, int64_t dest_, int64_t parent_) { src = src_; weight = weight_; dest = dest_; parent = parent_; }

  EdgeExt(EdgeExt const & other) { src = other.src; weight = other.weight; dest = other.dest; parent = other.parent; }
};

namespace CTF {
  template <>
  inline void Set<EdgeExt>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%" PRId64 " %f " " % " PRId64 " % " PRId64 ")", ((EdgeExt*)a)[0].src, ((EdgeExt*)a)[0].weight, ((EdgeExt*)a)[0].dest, ((EdgeExt*)a)[0].parent);
  }
}

void mat_set(Matrix<int>* matrix, Int64Pair index, int value = 1);

class Graph {
  public:
    int numVertices;
    vector<Int64Pair>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};
uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges);

template <typename dtype>
int64_t are_vectors_different(CTF::Vector<dtype> & A, CTF::Vector<dtype> & B);
void init_pvector(Vector<int>* p);
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);

void shortcut2(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, int sc2, World * world, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);
void roots_num(int64_t npairs, Pair<int> * loc_pairs, int64_t * loc_roots_num, int64_t * global_roots_num,  World * world);
void roots(int64_t npairs, int64_t loc_roots_num, Pair<int> * loc_pairs, int64_t * global_roots_num, int * global_roots,  World * world);
void create_nontriv_loc_indices(int64_t *& nontriv_loc_indices, int64_t * loc_nontriv_num, int64_t * global_roots_num, int * global_roots, int64_t q_npairs, Pair<int> * q_loc_pairs, World * world);

template<typename T>
Matrix<T>* PTAP(Matrix<T>* A, Vector<int>* p);

#endif
