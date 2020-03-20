#ifndef __MST_H__
#define __MST_H__

#include <ctf.hpp>
#include <float.h>
#include <math.h>
#include "graph_aux.h"

using namespace CTF;
#define SEED 23

struct EdgeExt {
  int64_t src, dest, parent;
  double weight;
  EdgeExt() { src = INT_MAX; weight = DBL_MAX; dest = INT_MAX; parent = 0; } // addid
  EdgeExt(int64_t src_, double weight_, int64_t dest_, int64_t parent_) { src = src_; weight = weight_; dest = dest_; parent = parent_; }

  EdgeExt(EdgeExt const & other) { src = other.src; weight = other.weight; dest = other.dest; parent = other.parent; }
};

namespace CTF {
  template <>
  inline void Set<EdgeExt>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%zu %f %zu %zu)", ((EdgeExt*)a)[0].src, ((EdgeExt*)a)[0].weight, ((EdgeExt*)a)[0].dest, ((EdgeExt*)a)[0].parent);
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

// MST
EdgeExt EdgeExtMin(EdgeExt a, EdgeExt b);
void EdgeExt_red(EdgeExt const * a, EdgeExt * b, int n);
Monoid<EdgeExt> get_minedge_monoid();

Vector<EdgeExt>* hook_matrix(int n, Matrix<EdgeExt> * A, World* world);

// Utility functions
int64_t are_vectors_different(CTF::Vector<int> & A, CTF::Vector<EdgeExt> & B);
void init_pvector(Vector<int>* p);
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> ** nonleaves=NULL, bool create_nonleaves=false);

#endif
