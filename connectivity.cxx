#include <ctf.hpp>
using namespace CTF;

Tensor<int> connectivity(Matrix<int> & A){
  int n = A.nrow;
  Vector<int> v(n);
  Vector<int> w(n);

  v.get_local_data
  //change the dqta

  w["i"] = v["i"];

  for(int i = 1; i <= n; i++){
    //semering
    w["j"] += A["jk"]*w["k"];

  }
  return w;
}

int main(int argc, char ** argv){
  int rank, np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  float sp_frac = 0.1;
  int n = 8;
  Matrix<int> A(n,n,SH|SP);
  srand48(A.wrld->rank);
  A.fill_sp_random(1,1,sp_frac);
  //generate graph with blocks
  for i = 1 to n/b:
    A[bi:b(i+1), bi:b(i+1)] = B[i]
    //block sparse
  printf("matrix before: \n");
  A.print_matrix();
  printf("return w: \n");
  Vector<int> w = connectivity(A);
  w.print();
}
