#include "connectivity.h"

template <typename dtype>
void max_vector(CTF::Vector<dtype> & result, CTF::Vector<dtype> & A, CTF::Vector<dtype> & B)
{
  result["i"] = CTF::Function<dtype,dtype,dtype>([](dtype a, dtype b){return ((a > b) ? a : b);})(A["i"], B["i"]);
}

Matrix<int>* pMatrix(Vector<int>* p, World* world)
{
  /*
  auto n = p->len;
  auto A = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);

  (*A)["ij"] = CTF::Function<int,int,int>([](int a, int b){ return a==b; })((*p)["i"],(*pg)["j"]);
  return A;
  */
  // FIXME: below is not benchmarked, nor sure if this is the right way of doing it
  int64_t n = p->len;
  auto A = new Matrix<int>(n, n, SP|SY, *world, MAX_TIMES_SR);
  int64_t npairs;
  Pair<int> * loc_pairs;
  p->read_local(&npairs, &loc_pairs);
  int64_t *gIndex = new int64_t[npairs];
  int *gData = new int[npairs];
  for (int64_t i = 0; i < npairs; i++){
    gIndex[i] = loc_pairs[i].k + loc_pairs[i].d * n;
    gData[i] = 1;
  }
  A->write(npairs, gIndex, gData);
  delete [] gIndex;
  delete [] gData;
  delete [] loc_pairs;
  return A;
}

// return B where B[i,j] = A[p[i],p[j]], or if P is P[i,j] = p[i], compute B = P^T A P
Matrix<int>* PTAP(Matrix<int>* A, Vector<int>* p){
  Timer t_ptap("CONNECTIVITY_PTAP");
  t_ptap.start();
  int np = p->wrld->np;
  int64_t n = p->len;
  Pair<int> * pprs;
  int64_t npprs;
  //get local part of p
  p->get_local_pairs(&npprs, &pprs);
  assert((npprs <= (n+np-1)/np) && (npprs >= (n/np)));
  assert(A->ncol == n);
  assert(A->nrow == n);
  Pair<int> * A_prs;
  int64_t nprs;
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the row of A (A1)
    Matrix<int> A1(n, n, "ij", Partition(1,&np)["i"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A1["ij"] = A->operator[]("ij");
    A1.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and rows of A are distributed cyclically, to compute P^T * A
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k/n)*n + pprs[(A_prs[i].k%n)/np].d;
    }
  }
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the column of A (A1)
    Matrix<int> A2(n, n, "ij", Partition(1,&np)["j"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    //write in P^T A into A2
    A2.write(nprs, A_prs);
    delete [] A_prs;
    A2.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and cols of A are distributed cyclically, to compute P^T A * P
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k%n) + pprs[(A_prs[i].k/n)/np].d*n;
    }
  }
  Matrix<int> * PTAP = new Matrix<int>(n, n, SP*(A->is_sparse), *A->wrld, *A->sr);
  PTAP->write(nprs, A_prs);
  delete [] A_prs;
  t_ptap.stop();
  return PTAP;
}


//recursive projection based algorithm
Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, World* world, int sc2)
{
  Timer t_relax("CONNECTIVITY_Relaxation");
  t_relax.start();
  //relax all edges
  auto q = new Vector<int>(n, SP*p->is_sparse, *world, MAX_TIMES_SR);
  (*q)["i"] = (*p)["i"];
  (*q)["i"] += (*A)["ij"] * (*p)["j"];
  t_relax.stop();
  Vector<int> * nonleaves;
  //check for convergence
  int64_t diff = are_vectors_different(*q, *p);
  if (p->wrld->rank == 0)
    printf("Diff is %ld\n",diff);
  if (!diff){
    return p;
  } else {
    //compute shortcutting q[i] = q[q[i]], obtain nonleaves or roots (FIXME: can we also remove roots that are by themselves?)
    //shortcut2(*q, *q, *q, sc2, world, &nonleaves, true);
    shortcut(*q, *q, *q, &nonleaves, true);
    if (p->wrld->rank == 0)
      printf("Number of nonleaves or roots is %ld\n",nonleaves->nnz_tot);
    //project to reduced graph with all vertices
    auto rec_A = PTAP(A, q);
    //recurse only on nonleaves
    auto rec_p = supervertex_matrix(n, rec_A, nonleaves, world, sc2);
    delete rec_A;
    //perform one step of shortcutting to update components of leaves
    shortcut(*p, *q, *rec_p);
    //shortcut2(*p, *q, *rec_p, 0, world);
    delete q;
    delete rec_p;
    return p;
  }
}

Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world)
{
  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);
  auto prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  while (are_vectors_different(*p, *prev)) {
    (*prev)["i"] = (*p)["i"];
    auto q = new Vector<int>(n, *world, MAX_TIMES_SR);
    Timer t_relax("CONNECTIVITY_Relaxation");
    t_relax.start();
    (*q)["i"] = (*A)["ij"] * (*p)["j"];
    t_relax.stop();
    auto r = new Vector<int>(n, *world, MAX_TIMES_SR);
    max_vector(*r, *p, *q);
    //auto P = pMatrix(p, world);
    auto s = new Vector<int>(n, *world, MAX_TIMES_SR);
    //(*s)["i"] = (*P)["ji"] * (*r)["j"];
    //shortcut(*s, *r, *p);
    shortcut2(*s, *r, *p, 1, world);
    max_vector(*p, *p, *s);
    Vector<int> * pi = new Vector<int>(*p);
    //shortcut(*p, *p, *p);
    shortcut2(*p, *p, *p, 1, world);

    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut2(*p, *p, *p, 1, world);
    }
    delete pi;

    delete q;
    delete r;
    delete s;
  }
  return p;
}


std::vector< Matrix<int>* > batch_subdivide(Matrix<int> & A, std::vector<float> batch_fracs){
  Matrix<float> B(A.nrow, A.ncol, SP*A.is_sparse, *A.wrld, MAX_TIMES_SR);
  Pair<int> * prs;
  int64_t nprs;
  A.get_local_pairs(&nprs, &prs, true);
  srand48(A.wrld->rank*4+10);
  Pair<float> * rprs = new Pair<float>[nprs];
  for (int64_t i=0; i<nprs; i++){
    rprs[i].k = prs[i].k;
    rprs[i].d = drand48();
  }
  std::sort(rprs, rprs+nprs, [](const Pair<float> & a, const Pair<float> & b){ return a.d<b.d; });
  float prefix = 0.;
  int64_t iprefix = 0;
  std::vector< Matrix<int>* > vp;
  for (int i=0; i<batch_fracs.size(); i++){
    prefix += batch_fracs[i];
    int64_t old_iprefix = iprefix;
    while (iprefix < nprs && rprs[iprefix].d < prefix){ iprefix++; }
    Matrix<int> * P = new Matrix<int>(A.nrow, A.ncol, SP*A.is_sparse, *A.wrld, MAX_TIMES_SR);
    Pair<int> * part_pairs = new Pair<int>[iprefix-old_iprefix];
    for (int64_t j=0; j<iprefix-old_iprefix; j++){
      part_pairs[j].k = rprs[old_iprefix+j].k;
      part_pairs[j].d = 1;
    }
    P->write(iprefix-old_iprefix, part_pairs);
    delete [] part_pairs;
    vp.push_back(P);
  }
  return vp;
}


// FIXME: remove these functions or document at least
// ---------------------------
Vector<int>* hook(Graph* graph, World* world) {
  auto n = graph->numVertices;
  auto A = graph->adjacencyMatrix(world);
  return hook_matrix(n, A, world);
}

Matrix<int>* mat_add(Matrix<int>* A, Matrix<int>* B, World* world) {
  int n = A->nrow;
  int m = A->ncol;
  auto C = new Matrix<int>(n, m, *world, MAX_TIMES_SR);
  for (auto row = 0; row < n; row++) {
    for (auto col = 0; col < m; col++) {
      auto idx = Int64Pair(row, col);
      auto aVal = mat_get(A, idx);
      auto bVal = mat_get(B, idx);
      mat_set(C, idx, min(1, aVal + bVal));
    }
  }
  return C;
}

bool mat_eq(Matrix<int>* A, Matrix<int>* B) {
  for (int r = 0; r < A->nrow; r++) {
    for (int c = 0; c < A->ncol; c++) {
      if (mat_get(A, Int64Pair(r, c)) != mat_get(B, Int64Pair(r, c))) {
        return false;
      }
    }
  }
  return true;
}


Matrix<int>* mat_I(int dim, World* world) {
  auto I = new Matrix<int>(dim, dim, *world, MAX_TIMES_SR);
  for (auto i = 0; i < dim; i++) {
    mat_set(I, Int64Pair(i, i), 1);
  }
  return I;
}

int mat_get(Matrix<int>* matrix, Int64Pair index) {
  auto data = new int[matrix->nrow * matrix->ncol];
  matrix->read_all(data);
  int value = data[index.i2 * matrix->nrow + index.i1];
  delete [] data;
  return value;
}
// ---------------------------


