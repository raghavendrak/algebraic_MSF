#include <ctf.hpp>
using namespace CTF;


Tensor<int> connectivity(int n, World & dw){
  Semiring<int> s(n*n,
                [](int a, int b){ return std::max(a,b); },
                MPI_MAX,
                0,
                [](int a, int b){ return a+b; });
  Matrix<int> A(n, n, dw, s);
  srand48(dw.rank);
  double sp_frac = 0.1;
  A.fill_sp_random(1.0,1.0,sp_frac);
  printf("matrix is: \n");
  A.print_matrix();
  Vector<int> W(n, dw, s);

  W["i"] += A["ij"] * W["j"];
  return W;
}


int main(int argc, char ** argv){
  int rank, np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  World dw(argc, argv);
  Tensor<int> ret = connectivity(7, dw);
  //generate graph with blocks
  /**
  for i = 1 to n/b:
    A[bi:b(i+1), bi:b(i+1)] = B[i]
  **/
    //block sparse

  printf("return w: \n");
  ret.print();
}
