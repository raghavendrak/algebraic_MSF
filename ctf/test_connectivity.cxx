#include "connectivity.h"

void test_6Blocks_simply_connected(World *w)
{
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

void driver(World *w)
{
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

int main(int argc, char** argv)
{
  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  auto w = new World(argc, argv);
  driver(w);
  return 0;
}


// ------------------------------------
// FIXME: below tests are yet to be reviewed with the driver

void test_simple(World* w)
{
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

void test_disconnected(World *w)
{
  printf("TEST2: DISCONNECTED 6*6\n");
  auto g = new Graph();
  g->numVertices = 6;
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}

void test_fully_connected(World *w)
{
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


void test_random1(World *w)
{
  printf("TEST4-1: RANDOM 1 6*6\n");
  Matrix<int> * B = new Matrix<int>(6,6,SP|SH,*w,MAX_TIMES_SR);
  B->fill_sp_random(1.0,1.0,0.1);
  B->print_matrix();
  hook_matrix(6, B, w)->print();
}

void test_random2(World *w)
{
  printf("TEST4-2: RANDOM 2 10*10\n");
  Matrix<int> * B = new Matrix<int>(10,10,SP|SH,*w,MAX_TIMES_SR);
  B->fill_sp_random(1.0,1.0,0.1);
  B->print_matrix();
  hook_matrix(10, B, w)->print();
}

void test_6Blocks_fully_connected(World *w)
{
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
//-----------------------------------------------------------
