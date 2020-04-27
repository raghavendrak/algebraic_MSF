#ifndef __MST_H__
#define __MST_H__

#include "alg_graph.h"

EdgeExt EdgeExtMin(EdgeExt a, EdgeExt b);
void EdgeExt_red(EdgeExt const * a, EdgeExt * b, int n);
Monoid<EdgeExt> get_minedge_monoid();

void project(Vector<EdgeExt> & r, Vector<int> & p, Vector<EdgeExt> & q);

Vector<EdgeExt>* hook_matrix(Matrix<EdgeExt> * A, World* world);

Vector<EdgeExt>* multilinear_hook(Matrix<EdgeExt> * A, World* world);

#endif
