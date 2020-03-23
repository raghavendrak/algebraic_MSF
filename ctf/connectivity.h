#ifndef __CONNECTIVITY_H__
#define __CONNECTIVITY_H__

#include "alg_graph.h"

// Connectivity
Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world);
Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, World* world, int sc2);

// Utility functions
template <typename dtype>
void max_vector(CTF::Vector<dtype> & result, CTF::Vector<dtype> & A, CTF::Vector<dtype> & B);
Matrix<int>* pMatrix(Vector<int>* p, World* world);
std::vector< Matrix<int>* > batch_subdivide(Matrix<int> & A, std::vector<float> batch_fracs);

// FIXME: below functions are yet to be optimized/reviewed
// ---------------------------
int mat_get(Matrix<int>* matrix, Int64Pair index);
Matrix<int>* mat_add(Matrix<int>* A, Matrix<int>* B, World* world);
Matrix<int>* mat_I(int dim, World* world);
bool mat_eq(Matrix<int>* A, Matrix<int>* B);
Vector<int>* hook(Graph* graph, World* world);
void shortcut(Vector<int> & pi);
// ---------------------------

#endif

