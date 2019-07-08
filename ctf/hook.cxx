#include <ctf.hpp>

#define PLACE_VERTEX (1)

using namespace CTF;

static Semiring<int> MAX_TIMES_SR(0,
    [](int a, int b) {
    return std::max(a, b);
    },
    MPI_MAX,
    1,
    [](int a, int b) {
    return a * b;
    });

class Int64Pair {
  public:
    int64_t i1;
    int64_t i2;

    Int64Pair(int64_t i1, int64_t i2);

    Int64Pair swap();
};

class Graph {
  public:
    int numVertices;
    vector<Int64Pair>* edges;

    Graph();

    Matrix<int>* adjacencyMatrix(World* world, bool sparse = false);
};

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

Matrix<int>* pMatrix(Vector<int> *p, Vector<int> *pg, World* world);

// FIXME: below functions are yet to be optimized/reviewed
// ---------------------------
void mat_set(Matrix<int>* matrix, Int64Pair index, int value = PLACE_VERTEX);
int mat_get(Matrix<int>* matrix, Int64Pair index);
Matrix<int>* mat_add(Matrix<int>* A, Matrix<int>* B, World* world);
Matrix<int>* mat_I(int dim, World* world);
bool mat_eq(Matrix<int>* A, Matrix<int>* B);
Vector<int>* hook(Graph* graph, World* world);
void shortcut(Vector<int> & pi);
// ---------------------------

Vector<int>* supervertex_matrix(int n, Matrix<int> *A, Vector<int> *p, Vector<int> *pg,  World *world);
Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world);

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

void test_simple(World* w){
  printf("TEST1: SIMPLE GRAPH 6*6\n");
  auto g = new Graph();
  g->numVertices = 6;
  g->edges->emplace_back(0, 1);
  g->edges->emplace_back(2, 4);
  g->edges->emplace_back(4, 3);
  g->edges->emplace_back(3, 5);
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}

void test_disconnected(World *w){
  printf("TEST2: DISCONNECTED 6*6\n");
  auto g = new Graph();
  g->numVertices = 6;
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}

void test_fully_connected(World *w){
  printf("TEST3: FULLY CONNECTED 6*6\n");
  auto g = new Graph();
  g->numVertices = 6;
  for(int i = 0; i < 5; i++){
    for(int j = i + 1; j < 6; j++){
      g->edges->emplace_back(i, j);
    }
  }
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}


void test_random1(World *w){
  printf("TEST4-1: RANDOM 1 6*6\n");
  Matrix<int> * B = new Matrix<int>(6,6,SP|SH,*w,MAX_TIMES_SR);
  B->fill_sp_random(1.0,1.0,0.1);
  B->print_matrix();
  hook_matrix(6, B, w)->print();
}

void test_random2(World *w){
  printf("TEST4-2: RANDOM 2 10*10\n");
  Matrix<int> * B = new Matrix<int>(10,10,SP|SH,*w,MAX_TIMES_SR);
  B->fill_sp_random(1.0,1.0,0.1);
  B->print_matrix();
  hook_matrix(10, B, w)->print();
}

void test_6Blocks_simply_connected(World *w){
  printf("TEST5: 6 Blocks of 6*6 simply connected graph\n");
  auto g = new Graph();
  g->numVertices = 36;
  for(int b = 0; b < 6; b++){
    for(int i = 0; i < 5; i++){
      g->edges->emplace_back(b*6+i, b*6+i+1);
    }
  }
  auto A = g->adjacencyMatrix(w);
  hook_matrix(36, A, w)->print();
  int matSize = 36;
  auto p = new Vector<int>(matSize, *w, MAX_TIMES_SR);
  init_pvector(p);
  Vector<int> *pg = new Vector<int>(*p);
  supervertex_matrix(matSize, A, p, pg, w)->print();
}

void test_6Blocks_fully_connected(World *w){
  printf("TEST6: 6 Blocks of 6*6 fully connected graph\n");
  auto g = new Graph();
  g->numVertices = 36;
  for(int b = 0; b < 6; b++){
    for(int i = 0; i < 5; i++){
      for(int j = i+1; j < 6; j++)
        g->edges->emplace_back(b*6+i, b*6+j);
    }
  }
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}

Matrix<int>* generate_kronecker(World* w, int order){
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

void driver(World *w) {
  // K13 : 1594323 (matrix size)
  // K6 : 729; 531441 vertices
  // k5 : 243
  // k7 : 2187
  // k8 : 6561
  // k9 : 19683
  int64_t matSize = 243;
  auto B = generate_kronecker(w, 5);

  Timer_epoch thm("hook_matrix");
  thm.begin();
  auto hm = hook_matrix(matSize, B, w);
  thm.end();

  auto p = new Vector<int>(matSize, *w, MAX_TIMES_SR);
  init_pvector(p);
  Timer_epoch tsv("super_vertex");
  tsv.begin();
  auto pg = new Vector<int>(matSize, *w, MAX_TIMES_SR);
  init_pvector(pg);
  auto sv = supervertex_matrix(matSize, B, p, pg, w);
  tsv.end();

  int64_t result = are_vectors_different(*hm, *sv);
  if (w->rank == 0) {
    if (result) {
      printf("result vectors are different: FAIL\n");
    }
    else {
      printf("result vectors are same: PASS\n");
    }
  }
  delete B;
  test_6Blocks_simply_connected(w);
}

int main(int argc, char** argv) {
  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  auto w = new World(argc, argv);
  driver(w);
  return 0;
}


Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world) {
  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);
  Vector<int> *pg = new Vector<int>(*p);
  auto prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  while (are_vectors_different(*p, *prev)) {
    (*prev)["i"] = (*p)["i"];
    auto q = new Vector<int>(n, *world, MAX_TIMES_SR);
    (*q)["i"] = (*A)["ij"] * (*p)["j"];
    auto r = new Vector<int>(n, *world, MAX_TIMES_SR);
    max_vector(*r, *p, *q);
    auto P = pMatrix(p, pg, world);
    auto s = new Vector<int>(n, *world, MAX_TIMES_SR);
    (*s)["i"] = (*P)["ji"] * (*r)["i"];
    max_vector(*p, *p, *s);
    Vector<int> * pi = new Vector<int>(*p);
    shortcut(*p);

    while (are_vectors_different(*pi, *p)){
      free(pi);
      pi = new Vector<int>(*p);
      shortcut(*p);
    }
    free(pi);

    free(q);
    free(r);
    free(P);
    free(s);		
  }
  free(pg);
  return p;
}

Matrix<int>* pMatrix(Vector<int>* p, Vector<int> *pg, World* world) {
  /*
  auto n = p->len;
  auto A = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);

  (*A)["ij"] = CTF::Function<int,int,int>([](int a, int b){ return a==b; })((*p)["i"],(*pg)["j"]);
  return A;
  */
  // FIXME: below is not benchmarked, nor sure if this is the right way of doing it
  auto n = p->len;
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
  return A;
}

void shortcut(Vector<int> & pi){
  int64_t npairs;
  Pair<int> * loc_pairs;
  // obtain all values of pi on this process
  pi.read_local(&npairs, &loc_pairs);
  Pair<int> * remote_pairs = new Pair<int>[npairs];
  // set keys to value of pi, so remote_pairs[i].k = pi[loc_pairs[i].k]
  for (int64_t i=0; i<npairs; i++){
    //cout << "k: " << remote_pairs[i].k << " d: " << loc_pairs[i].d << endl;
    remote_pairs[i].k = loc_pairs[i].d;
  }
  // obtains values at each pi[i] by remote read, so remote_pairs[i].d = pi[loc_pairs[i].k]
  pi.read(npairs, remote_pairs);
  // set loc_pairs[i].d = remote_pairs[d] and write back to local data
  for (int64_t i=0; i<npairs; i++){
    loc_pairs[i].d = remote_pairs[i].d;
  }
  delete [] remote_pairs;
  pi.write(npairs, loc_pairs);
  delete [] loc_pairs;
}

Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, Vector<int> *pg, World* world) {
  auto q = new Vector<int>(n, *world, MAX_TIMES_SR);
  (*q)["i"] = (*p)["i"] + (*A)["ij"] * (*p)["j"];
  if (!are_vectors_different(*p, *q)) {
    return q;
  }
  else {
    auto P = pMatrix(q, pg, world);
    auto rec_A = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);
    (*rec_A)["il"] = (*P)["ji"] * (*A)["jk"] * (*P)["kl"];
    auto rec_p = supervertex_matrix(n, rec_A, q, pg, world);
    delete rec_A;
    // p[i] = rec_p[q[i]]
    int64_t npairs;
    Pair<int>* loc_pairs;
    q->read_local(&npairs, &loc_pairs);
    Pair<int> * remote_pairs = new Pair<int>[npairs];
    for (int64_t i = 0; i < npairs; i++) {
      remote_pairs[i].k = loc_pairs[i].d;
    }
    rec_p->read(npairs, remote_pairs);
    for (int64_t i = 0; i < npairs; i++) {
      loc_pairs[i].d = remote_pairs[i].d;
    }
    // FIXME: assumption: p & q use the same distribution across processes
    p->write(npairs, loc_pairs);
    delete [] remote_pairs;
    delete [] loc_pairs;
    return p;
  }
  free(pg);
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
