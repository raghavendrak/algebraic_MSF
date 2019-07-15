#include "connectivity.h"

// NOTE: can't use bool as return
template <typename dtype>
int64_t are_vectors_different(CTF::Vector<dtype> & A, CTF::Vector<dtype> & B)
{
  CTF::Scalar<int64_t> s;
  s[""] = CTF::Function<dtype,dtype,int64_t>([](dtype a, dtype b){ return a!=b; })(A["i"],B["i"]);
  return s.get_val();
}

template <typename dtype>
void max_vector(CTF::Vector<dtype> & result, CTF::Vector<dtype> & A, CTF::Vector<dtype> & B)
{
  result["i"] = CTF::Function<dtype,dtype,dtype>([](dtype a, dtype b){return ((a > b) ? a : b);})(A["i"], B["i"]);
}

void init_pvector(Vector<int>* p)
{
  int64_t npairs;
  Pair<int> * loc_pairs;
  p->read_local(&npairs, &loc_pairs);
  for (int64_t i = 0; i < npairs; i++){
    loc_pairs[i].d = loc_pairs[i].k;
  }
  p->write(npairs, loc_pairs);
  delete [] loc_pairs;
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
  auto A = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);
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

// p[i] = rec_p[q[i]]
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p)
{
  int64_t npairs;
  Pair<int> * loc_pairs;
  q.read_local(&npairs, &loc_pairs);
  Pair<int> * remote_pairs = new Pair<int>[npairs];
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d;
  }
  rec_p.read(npairs, remote_pairs);
  for (int64_t i=0; i<npairs; i++){
    loc_pairs[i].d = remote_pairs[i].d;
  }
  delete [] remote_pairs;
  p.write(npairs, loc_pairs);
  delete [] loc_pairs;
}

Matrix<int>* PTAP(Matrix<int>* A, Vector<int>* p){
  int np = p->wrld->np;
  int64_t n = p->len;
  Pair<int> * pprs;
  int64_t npprs;
  p->get_local_pairs(&npprs, &pprs);
  assert(npprs == (n+np-1)/np);
  assert(A->ncol == n);
  assert(A->nrow == n);
  Pair<int> * A_prs;
  int64_t nprs;
  {
    Matrix<int> A1(n, n, "ij", Partition(1,&np)["i"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A1["ij"] = A->operator[]("ij");
    A1.get_local_pairs(&nprs, &A_prs);
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k/n)*n + pprs[(A_prs[i].k%n)/np].d;
    }
  }
  {
    Matrix<int> A2(n, n, "ij", Partition(1,&np)["j"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A2.write(nprs, A_prs);
    delete [] A_prs;
    A2.get_local_pairs(&nprs, &A_prs);
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k%n) + pprs[(A_prs[i].k/n)/np].d*n;
    }
  }
  Matrix<int> * PTAP = new Matrix<int>(n, n, SP*(A->is_sparse), *A->wrld, *A->sr);
  PTAP->write(nprs, A_prs);
  delete [] A_prs;
  return PTAP;
}

Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, World* world)
{
  auto q = new Vector<int>(n, *world, MAX_TIMES_SR);
  (*q)["i"] = (*p)["i"] + (*A)["ij"] * (*p)["j"];
  if (!are_vectors_different(*p, *q)) {
    return q;
  }
  else {
    auto rec_A = PTAP(A, q);
    //auto P = pMatrix(q, world);
    //auto rec_A = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);
    //auto inter = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);
    //(*inter)["ik"] = (*P)["ji"] * (*A)["jk"];
    //(*rec_A)["ik"] = (*inter)["ij"] * (*P)["jk"];
    //// (*rec_A)["il"] = (*P)["ji"] * (*A)["jk"] * (*P)["kl"];
    //delete inter;
    auto rec_p = supervertex_matrix(n, rec_A, q, world);
    delete rec_A;
    // p[i] = rec_p[q[i]]
    shortcut(*p, *q, *rec_p);
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
    (*q)["i"] = (*A)["ij"] * (*p)["j"];
    auto r = new Vector<int>(n, *world, MAX_TIMES_SR);
    max_vector(*r, *p, *q);
    auto P = pMatrix(p, world);
    auto s = new Vector<int>(n, *world, MAX_TIMES_SR);
    (*s)["i"] = (*P)["ji"] * (*r)["j"];
    max_vector(*p, *p, *s);
    Vector<int> * pi = new Vector<int>(*p);
    shortcut(*p, *p, *p);

    while (are_vectors_different(*pi, *p)){
      free(pi);
      pi = new Vector<int>(*p);
      shortcut(*p, *p, *p);
    }
    free(pi);

    free(q);
    free(r);
    free(P);
    free(s);
  }
  return p;
}


// FIXME: below functions are yet to be optimized/reviewed
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

Int64Pair::Int64Pair(int64_t i1, int64_t i2) {
  this->i1 = i1;
  this->i2 = i2;
}

Int64Pair Int64Pair::swap() {
  return {this->i2, this->i1};
}

Matrix<int>* mat_I(int dim, World* world) {
  auto I = new Matrix<int>(dim, dim, *world, MAX_TIMES_SR);
  for (auto i = 0; i < dim; i++) {
    mat_set(I, Int64Pair(i, i), 1);
  }
  return I;
}

Graph::Graph() {
  this->numVertices = 0;
  this->edges = new vector<Int64Pair>();
}

Matrix<int>* Graph::adjacencyMatrix(World* world, bool sparse) {
  auto attr = 0;
  if (sparse) {
    attr = SP;
  }
  auto A = new Matrix<int>(numVertices, numVertices,
      attr, *world, MAX_TIMES_SR);
  for (auto edge : *edges) {
    mat_set(A, edge);
    mat_set(A, edge.swap());
  }
  return A;
}

void mat_set(Matrix<int>* matrix, Int64Pair index, int value) {
  int64_t idx[1];
  idx[0] = index.i2 * matrix->nrow + index.i1;
  int fill[1];
  fill[0] = value;
  matrix->write(1, idx, fill);
}

int mat_get(Matrix<int>* matrix, Int64Pair index) {
  auto data = new int[matrix->nrow * matrix->ncol];
  matrix->read_all(data);
  int value = data[index.i2 * matrix->nrow + index.i1];
  free(data);
  return value;
}
// ---------------------------
