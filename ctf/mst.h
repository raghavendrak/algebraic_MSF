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

struct Int64Triple {
  int64_t i1, i2, i3;
  Int64Triple() { }
  Int64Triple(int64_t i1_, int64_t i2_, int64_t i3_) { i1 = i1_; i2 = i2_; i3 = i3_; }

  Int64Triple(Int64Triple const & other) { i1 = other.i1; i2 = other.i2; i3 = other.i3; }
};

struct Int64Pair {
  int64_t i1, i2;
  Int64Pair() { i1 = 0; i2 = 0; }
  Int64Pair(int64_t i1_, int64_t i2_) { i1 = i1_; i2 = i2_; }
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
static Semiring<Int64Triple> MIN_TIMES_SR(
  Int64Triple(INT_MAX, INT_MAX, -1),
  [](Int64Triple a, Int64Triple b) {
    if (a.i3 > a.i1 && b.i3 > b.i1)
      return Int64Triple(INT_MAX, INT_MAX, -1);
    else
      return Int64Triple(0, 0, 0);
      //return a.i2 < b.i2 ? a.i3 : b.i3;
  },
  MPI_MAX,
  Int64Triple(-1, -1, -1),
  [](Int64Triple a, Int64Triple b) {
    return Int64Triple(a.i1, a.i2, b.i2);
  });
*/

class Graph {
  public:
    int numVertices;
    vector<Int64Pair>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};

// MST
Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world);
Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, World* world, int sc2);

// Utility functions
template <typename dtype>
int64_t are_vectors_different(CTF::Vector<dtype> & A, CTF::Vector<dtype> & B);
void init_pvector(Vector<Int64Triple>* p);
Matrix<int>* pMatrix(Vector<int>* p, World* world);
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);
std::vector< Matrix<int>* > batch_subdivide(Matrix<int> & A, std::vector<float> batch_fracs);

#endif
