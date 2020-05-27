#ifndef __MST_H__
#define __MST_H__

#include "alg_graph.h"

struct Edge {
  wht weight;
  int64_t parent; // stores intermediate data about which component an edge may hook onto

  Edge() { weight = MAX_WHT; parent = -1; } // addid
  Edge(wht weight_, int64_t parent_) { weight = weight_; parent = parent_; }
  Edge(Edge const & other) { weight = other.weight; parent = other.parent; }
};

Edge EdgeMin(Edge a, Edge b);
void Edge_red(Edge const * a, Edge * b, int n);
Monoid<Edge> get_minedge_monoid();

namespace CTF {
  template <>
  inline void Set<Edge>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%d" " % " PRId64 ")", ((Edge*)a)[0].weight, ((Edge*)a)[0].parent);
  }
}

Vector<Edge>* multilinear_hook(Matrix<wht> *      A, 
                                  World*          world, 
                                  int64_t         sc2, 
                                  MPI_Datatype &  mpi_pkv, 
                                  int64_t         sc3,
                                  int64_t         ptap,
                                  int64_t         star);


// utility //
// r[p[j]] = q[j] over MINWEIGHT
template <typename T>
void project(Vector<T> & r, Vector<int> & p, Vector<T> & q);

#endif
