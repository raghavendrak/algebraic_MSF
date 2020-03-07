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
  int64_t key, weight, parent, comp;
  EdgeExt() { }
  EdgeExt(int64_t key_, int64_t weight_, int64_t parent_, int64_t comp_) { key = key_; weight = weight_; parent = parent_; comp = comp_; }

  EdgeExt(EdgeExt const & other) { key = other.key; weight = other.weight; parent = other.parent; comp = other.comp; }
};

struct Edge {
  int64_t key, weight;
  Edge() { key = 0; weight = 0; }
  Edge(int64_t key_, int64_t weight_) { key = key_; weight = weight_; }
};

namespace CTF {
  template <>
  inline void Set<Edge>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%zu %zu)", ((Edge*)a)[0].key, ((Edge*)a)[0].weight);
  }

  template <>
  inline void Set<EdgeExt>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%zu %zu %zu %zu)", ((EdgeExt*)a)[0].key, ((EdgeExt*)a)[0].weight, ((EdgeExt*)a)[0].parent, ((EdgeExt*)a)[0].comp);
  }
}
static Semiring<int> MAX_TIMES_SR(0,
    [](int a, int b) {
    return std::max(a, b);
    },
    MPI_MAX,
    1,
    [](int a, int b) {
    return a * b;
    });

class Graph {
  public:
    int numVertices;
    vector<Edge>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};

// MST
EdgeExt EdgeExtMin(EdgeExt a, EdgeExt b);
void EdgeExt_red(EdgeExt const * a, EdgeExt * b, int n);
//Monoid<EdgeExt> get_minedge_monoid();
Semiring<EdgeExt> get_minedge_sr();
Vector<int>* hook_matrix(int n, Matrix<EdgeExt> * A, World* world);
Vector<int>* supervertex_matrix(int n, Matrix<EdgeExt>* A, Vector<int>* p, World* world, int sc2);

Matrix<EdgeExt>* PTAP(Matrix<EdgeExt>* A, Vector<EdgeExt>* p);

// Utility functions
int64_t are_vectors_different(CTF::Vector<int> & A, CTF::Vector<EdgeExt> & B);
void init_pvector(Vector<int>* p);
Matrix<int>* pMatrix(Vector<int>* p, World* world);

//template <typename T>
//void shortcut(Vector<T> & p, Vector<EdgeExt> & q, Vector<T> & rec_p, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);
#include "mst_templates.cxx"

std::vector< Matrix<int>* > batch_subdivide(Matrix<int> & A, std::vector<float> batch_fracs);

#endif
