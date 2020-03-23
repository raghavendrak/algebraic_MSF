#include "test.h"

bool is_sparse(int a) { return a > 0; }

bool is_sparse(EdgeExt a) { return a.weight > 0; }

template<typename T>
Scalar<T> init_addid(World &  dw);

template<>
Scalar<int> init_addid(World & dw) {
  return 0; 
}

template<>
Scalar<EdgeExt> init_addid(World & dw) {
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  Scalar<EdgeExt> s(EdgeExt(), dw, MIN_EDGE);

  return s;
}

template<typename T>
Matrix<T> init_A(int n, World * dw, char const * name);

template<>
Matrix<int> init_A(int n, World * dw, char const * name) {
  Matrix<int> A_pre(n, n, SP, *dw, MAX_TIMES_SR, name);

  return A_pre;
}

template<>
Matrix<EdgeExt> init_A(int n, World * dw, char const * name) {
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  Matrix<EdgeExt> A_pre(n, n, SP, *dw, MIN_EDGE, name);

  return A_pre;
}

template<typename T>
Matrix <T> preprocess_graph(int           n,
                              World &       dw,
                              Matrix<T> & A_pre,
                              bool          remove_singlets,
                              int *         n_nnz,
                              int64_t       max_ewht){
  Scalar<T> addid = init_addid<T>(dw);

  //printf("before zero on diagonal\n");
  //A_pre.print();
  //A_pre["ii"] = addid[""];
  /* workaround for above line.*/
  int64_t A_n;
  Pair<T> * A_loc_pairs;
  A_pre.get_local_pairs(&A_n, &A_loc_pairs);
  for (int64_t i = 0; i < n; ++i) {
    if (A_loc_pairs[i].k % A_pre.nrow == A_loc_pairs[i].k / A_pre.nrow) {
      A_loc_pairs[i].d = T();
    }
  }
  A_pre = init_A<T>(n, &dw, "A_rmat");
  A_pre.write(A_n, A_loc_pairs);
  /* workaround end. */
  //printf("after zero on diagonal\n");
  //A_pre.print();

  A_pre.sparsify([](T a){ return is_sparse(a); });

  if (dw.rank == 0)
    printf("A contains %ld nonzeros\n", A_pre.nnz_tot);

  if (remove_singlets){
    Vector<int> rc(n, dw);
    rc["i"] += ((Function<T, int>)([](T a){ return (int)(is_sparse(a)); }))(A_pre["ij"]);
    rc["i"] += ((Function<T, int>)([](T a){ return (int)(is_sparse(a)); }))(A_pre["ji"]);
    int * all_rc;
    int64_t nval;
    rc.read_all(&nval, &all_rc);
    int n_nnz_rc = 0;
    int n_single = 0;
    for (int i=0; i<nval; i++){
      if (all_rc[i] != 0){
        if (all_rc[i] == 2) n_single++;
        all_rc[i] = n_nnz_rc;
        n_nnz_rc++;
      } else {
        all_rc[i] = -1;
      }
    }
    if (dw.rank == 0) printf("n_nnz_rc = %d of %d vertices kept, %d are 0-degree, %d are 1-degree\n", n_nnz_rc, n,(n-n_nnz_rc),n_single);
    Matrix<T> A = init_A<T>(n_nnz_rc, &dw, "A");
    int * pntrs[] = {all_rc, all_rc};

    A.permute(T(), A_pre, pntrs, T()); // TODO: fix beta and alpha?
    free(all_rc);
    if (dw.rank == 0) printf("preprocessed matrix has %ld edges\n", A.nnz_tot);

    //A["ii"] = addid[""];
    /* workaround for above line.*/
    int64_t A_n;
    Pair<T> * A_loc_pairs;
    A_pre.get_local_pairs(&A_n, &A_loc_pairs);
    for (int64_t i = 0; i < n; ++i) {
      if (A_loc_pairs[i].k % A_pre.nrow == A_loc_pairs[i].k / A_pre.nrow) {
        A_loc_pairs[i].d = T();
      }
    }
    A_pre = init_A<T>(n, &dw, "A_rmat");
    A_pre.write(A_n, A_loc_pairs);
    /* workaround end. */
    *n_nnz = n_nnz_rc;
    return A;
  } else {
    *n_nnz= n;
    //A_pre["ii"] = addid[""];
    /* workaround for above line.*/
    int64_t A_n;
    Pair<T> * A_loc_pairs;
    A_pre.get_local_pairs(&A_n, &A_loc_pairs);
    for (int64_t i = 0; i < n; ++i) {
      if (A_loc_pairs[i].k % A_pre.nrow == A_loc_pairs[i].k / A_pre.nrow) {
        A_loc_pairs[i].d = T();
      }
    }
    A_pre = init_A<T>(n, &dw, "A_rmat");
    A_pre.write(A_n, A_loc_pairs);
    /* workaround end. */
    A_pre.print();
    return A_pre;
  }
}
template Matrix <wht> preprocess_graph(int           n,
                              World &       dw,
                              Matrix<wht> & A_pre,
                              bool          remove_singlets,
                              int *         n_nnz,
                              int64_t       max_ewht);
template Matrix <EdgeExt> preprocess_graph(int           n,
                              World &       dw,
                              Matrix<EdgeExt> & A_pre,
                              bool          remove_singlets,
                              int *         n_nnz,
                              int64_t       max_ewht);

void setup_A(Matrix<int> & A, uint64_t * edge, uint64_t nedges, int64_t * inds, int * vals, int64_t max_ewht) {
  int n = A.nrow;
  for (int64_t i=0; i<nedges; i++){
    inds[i] = (edge[2*i]+(edge[2*i+1])*n);
    vals[i] = 1;
  }
  A.write(nedges,inds,vals);
  A["ij"] += A["ji"];
}

void setup_A(Matrix<EdgeExt> & A, uint64_t * edge, uint64_t nedges, int64_t * inds, EdgeExt * vals, int64_t max_ewht) {
  int n = A.nrow;
  for (int64_t i=0; i<nedges; i++){
    inds[i] = (edge[2*i]+(edge[2*i+1])*n);
    vals[i] = EdgeExt(inds[i] / n, (rand()%max_ewht) + 1, inds[i] % n, inds[i] / n);

    // produce antisymmetry (i, weight, j, parent) = (j, weight, i, parent)
    inds[i + nedges] = (inds[i] % n) * n + inds[i] / n;
    vals[i + nedges] = EdgeExt(vals[i].dest, vals[i].weight, vals[i].src, vals[i].dest);
  }

  A.write(2 * nedges,inds,vals);
}

template<typename T>
Matrix <T> read_matrix(World  &     dw,
                         int          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int *        n_nnz,
                         int64_t      max_ewht){
  uint64_t *my_edges = NULL;
  uint64_t my_nedges = 0;

  //random adjacency matrix
  Matrix<T> A_pre = init_A<T>(n, &dw, "A_rmat");
 
#ifdef MPIIO
  if (dw.rank == 0) printf("Running MPI-IO graph reader n = %d... ",n);
  char **leno;
  my_nedges = read_graph_mpiio(dw.rank, dw.np, fpath, &my_edges, &leno);
  processedges(leno, my_nedges, dw.rank, &my_edges);
  free(leno[0]);
  free(leno);
#else
  if (dw.rank == 0) printf("Running graph reader n = %d... ",n);
    my_nedges = read_graph(dw.rank, dw.np, fpath, &my_edges);
#endif
  if (dw.rank == 0) printf("finished reading (%ld edges).\n", my_nedges);
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_nedges);
  T * vals = (T*)malloc(sizeof(T)*2*my_nedges);

  srand(dw.rank+1);
  setup_A(A_pre, my_edges, my_nedges, inds, vals, max_ewht);
  if (dw.rank == 0) printf("filling CTF graph\n");
  free(inds);
  free(vals);

  return preprocess_graph<T>(n,dw,A_pre,remove_singlets,n_nnz,max_ewht);
}
template Matrix <wht> read_matrix(World  &     dw,
                         int          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int *        n_nnz,
                         int64_t      max_ewht);
template Matrix <EdgeExt> read_matrix(World  &     dw,
                         int          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int *        n_nnz,
                         int64_t      max_ewht);


template<typename T>
Matrix<T> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int *    n_nnz,
                             int64_t  max_ewht){
  uint64_t *edge=NULL;
  uint64_t nedges = 0;
  //random adjacency matrix
  int n = pow(2,scale);
  Matrix<T> A_pre = init_A<T>(n, &dw, "A_rmat");
  if (dw.rank == 0) printf("Running graph generator n = %d... ",n);
  nedges = gen_graph(scale, ef, gseed, &edge);
  if (dw.rank == 0) printf("done.\n");
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*2*nedges);
  T * vals = (T*)malloc(sizeof(T)*2*nedges);

  srand(dw.rank+1);
  setup_A(A_pre, edge, nedges, inds, vals, max_ewht);
  if (dw.rank == 0) printf("filling CTF graph\n");
  free(inds);
  free(vals);

  return preprocess_graph<T>(n,dw,A_pre,remove_singlets,n_nnz,max_ewht);
}
template Matrix<wht> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int *    n_nnz,
                             int64_t  max_ewht);
template Matrix<EdgeExt> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int *    n_nnz,
                             int64_t  max_ewht);

Matrix <wht> gen_uniform_matrix(World & dw,
                                int64_t n,
                                double  sp,
                                int64_t  max_ewht){
  Semiring<wht> s(MAX_WHT,
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });

  //random adjacency matrix
  Matrix<wht> A(n, n, SP, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
//  A.fill_random(1, std::min(n*n,100));
  int nmy = ((int)std::max((int)(n*sp),(int)1))*((int)((n+dw.np-1)/dw.np));
  int64_t inds[nmy];
  wht vals[nmy];
  int i=0;
  for (int64_t row=dw.rank*n/dw.np; row<(int)(dw.rank+1)*n/dw.np; row++){
    int64_t cols[std::max((int)(n*sp),1)];
    for (int64_t col=0; col<std::max((int)(n*sp),1); col++){
      bool is_rep;
      do {
        cols[col] = rand()%n;
        is_rep = 0;
        for (int c=0; c<col; c++){
          if (cols[c] == cols[col]) is_rep = 1;
        }
      } while (is_rep);
      inds[i] = cols[col]*n+row;
      vals[i] = (rand()%max_ewht)+1;
      i++;
    }
  }
  A.write(i,inds,vals);

  A["ii"] = 0;

  //keep only values smaller than 20 (about 20% sparsity)
  //A.sparsify([=](int a){ return a<sp*100; });
   return A;
}

Matrix<int>* generate_kronecker(World* w, int order)
{
  auto g = new Graph();
  g->numVertices = 3;
  g->edges->emplace_back(0, 0);
  g->edges->emplace_back(0, 1);
  g->edges->emplace_back(1, 1);
  g->edges->emplace_back(1, 2);
  g->edges->emplace_back(2, 2);
  auto kinitiator = g->adjacencyMatrix(w);
  auto B = g->adjacencyMatrix(w);

  int64_t len = 1;
  int64_t matSize = 3;
  for (int i = 2; i <= order; i++) {
    len *= 3;
    int64_t lens[] = {3, len, 3, len};
    /**
    int * lens = new int[4];
    lens[0] = 3;
    lens[1] = len;
    lens[2] = 3;
    lens[3] = len;
    **/
    auto D = Tensor<int>(4, B->is_sparse, lens);
    D["ijkl"] = (*kinitiator)["ik"] * (*B)["jl"];

    matSize *= 3;
    auto B2 = new Matrix<int>(matSize, matSize, B->is_sparse * SP, *w, *B->sr);
    delete B;
    B2->reshape(D);
    B = B2;
    // B->print_matrix();
    // hook on B
  }
  delete kinitiator;
  return B;
}
