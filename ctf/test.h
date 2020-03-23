#ifndef __TEST_H__
#define __TEST_H__

#include "alg_graph.h"
#include "mst.h"

template<typename T>
Matrix <T> preprocess_graph(int           n,
                              World &       dw,
                              Matrix<T> & A_pre,
                              bool          remove_singlets,
                              int *         n_nnz,
                              int64_t       max_ewht=1);

template<typename T>
Matrix <T> read_matrix(World  &     dw,
                         int          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int *        n_nnz,
                         int64_t      max_ewht=1);

template<typename T>
Matrix<T> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int *    n_nnz,
                             int64_t  max_ewht=1);

Matrix <wht> gen_uniform_matrix(World & dw,
                                int64_t n,
                                double  sp=.20,
                                int64_t  max_ewht=1);

Matrix<int>* generate_kronecker(World* w, int order);

#endif
