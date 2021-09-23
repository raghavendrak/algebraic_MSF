#ifndef __TEST_H__
#define __TEST_H__

#include "alg_graph.h"
#include "mst.h"
#include <random>
Matrix<wht> preprocess_graph(int64_t           n,
                              World &       dw,
                              Matrix<wht> & A_pre,
                              bool          remove_singlets,
                              int64_t *         n_nnz,
                              int64_t       max_ewht=1);

Matrix<wht> read_matrix(World  &     dw,
                         int64_t          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int64_t *        n_nnz,
                         int64_t      max_ewht=1);

Matrix<wht> read_matrix_snap(World  &     dw,
                             int64_t      n,
                             const char * fpath,
                             bool         remove_singlets,
                             int64_t *    n_nnz,
                             int64_t      max_ewht=1);

Matrix<wht> read_matrix_market(World  &     dw,
                               int64_t          n,
                               const char * fpath,
                               bool         remove_singlets,
                               int64_t *        n_nnz,
                               int          is_weight,
                               int64_t      max_ewht=1);

Matrix<wht> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int64_t *    n_nnz,
                             int64_t  max_ewht=1);

Matrix<wht> gen_uniform_matrix(World & dw,
                                int64_t n,
                                bool    remove_singlets,
                                int64_t *   n_nnz,
                                double  sp=.20,
                                int64_t max_ewht=1);

Matrix<int>* generate_kronecker(World* w, int order);

const int64_t kRandSeed = 27491095;
#endif
