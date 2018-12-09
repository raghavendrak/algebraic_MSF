#include <ctf.hpp>
using namespace CTF;


Tensor<int> connectivity(int n, World & dw){
  Semiring<int> s(0,
                [](int a, int b){ return std::max(a,b); },
                MPI_MAX,
                0,
                [](int a, int b){ return a+b; });
  Matrix<int> A(n, n, get_universe(), s);
  //srand48(dw.rank);
  //double sp_frac = 0.1;

  //A.fill_sp_random(1.0,1.0,sp_frac);
  A["ij"] = 0;
  int64_t * idx = new int64_t[12]{1, 2, 7, 9, 10, 14, 15, 18, 19, 22, 30, 37};
  int * data = new int[12]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  A.write(12, idx, data);
  printf("matrix is: \n");
  A.print_matrix();
  Vector<int> W(n, dw, s);

  for(int j = 0; i < n; j++)
    W["j"] += A["jk"] * W["k"];
  free(idx);
  free(data);
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
