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

class Graph {
  public:
    int numVertices;
    vector<Int64Pair>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};

// Connectivity
Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world);
Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, World* world);

// Utility functions
template <typename dtype>
int64_t are_vectors_different(CTF::Vector<dtype> & A, CTF::Vector<dtype> & B);
template <typename dtype>
void max_vector(CTF::Vector<dtype> & result, CTF::Vector<dtype> & A, CTF::Vector<dtype> & B);
void init_pvector(Vector<int>* p);
Matrix<int>* pMatrix(Vector<int>* p, World* world);
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> *& leaves);
std::vector< Matrix<int>* > batch_subdivide(Matrix<int> & A, std::vector<float> batch_fracs);

// FIXME: below functions are yet to be optimized/reviewed
// ---------------------------
void mat_set(Matrix<int>* matrix, Int64Pair index, int value = 1);
int mat_get(Matrix<int>* matrix, Int64Pair index);
Matrix<int>* mat_add(Matrix<int>* A, Matrix<int>* B, World* world);
Matrix<int>* mat_I(int dim, World* world);
bool mat_eq(Matrix<int>* A, Matrix<int>* B);
Vector<int>* hook(Graph* graph, World* world);
void shortcut(Vector<int> & pi);
// ---------------------------

#endif
