#ifndef __BTWN_CENTRAL_H__
#define __BTWN_CENTRAL_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>
#include "graph_aux.h"

using namespace CTF;
#define SEED 23
// From btwn_central
//typedef float mlt;
//typedef float wht;
//#define MAX_WHT (FLT_MAX/4.)
typedef int mlt;
typedef int wht;
#define MAX_WHT (INT_MAX/2)
typedef double REAL;
uint64_t gen_graph(int scale, int edgef, uint64_t seed, uint64_t **edges);

struct EdgeExt {
  int64_t key, weight, parent;
  EdgeExt() { }
  EdgeExt(int64_t key_, int64_t weight_, int64_t parent_) { key = key_; weight = weight_; parent = parent_; }

  EdgeExt(EdgeExt const & other) { key = other.key; weight = other.weight; parent = other.parent; }
};

struct Edge {
  int64_t key, weight;
  Edge() { key = 0; weight = 0; }
  Edge(int64_t key_, int64_t weight_) { key = key_; weight = weight_; }
};

static Semiring<int> MAX_TIMES_SR(0,
    [](int a, int b) {
    return std::max(a, b);
    },
    MPI_MAX,
    1,
    [](int a, int b) {
    return a * b;
    });

/*
// (key, weight, parent in p) 
// entries in A: (key, weight, -1) 
// entries in P: (key, -1, parent in p)
static Semiring<EdgeExt> MIN_TIMES_SR(
  EdgeExt(INT_MAX, INT_MAX, -1),
  [](EdgeExt a, EdgeExt b) {
    if (a.parent > a.key && b.parent > b.key)
      return EdgeExt(INT_MAX, INT_MAX, -1);
    else
      return EdgeExt(0, 0, 0);
      //return a.weight < b.weight ? a.parent : b.parent;
  },
  MPI_MAX,
  EdgeExt(-1, -1, -1),
  [](EdgeExt a, EdgeExt b) {
    return EdgeExt(a.key, a.weight, b.weight);
  });
*/

class Graph {
  public:
    int numVertices;
    vector<Edge>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};

// MST
Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world);
Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, World* world, int sc2);

// Utility functions
template <typename dtype>
int64_t are_vectors_different(CTF::Vector<dtype> & A, CTF::Vector<dtype> & B);
void init_pvector(Vector<int>* p);
Matrix<int>* pMatrix(Vector<int>* p, World* world);
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);
std::vector< Matrix<int>* > batch_subdivide(Matrix<int> & A, std::vector<float> batch_fracs);

#endif
