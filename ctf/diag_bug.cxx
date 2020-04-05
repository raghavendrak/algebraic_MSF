/* 
 * Demonstrates bug when trying to "zero" diagonal of A
 * ./diag_bug
 * <int> case correctly "zero" diagonal but <EdgeExt> case makes the matrix itself diagonal
 */
#include <ctf.hpp>
#include "alg_graph.h"
#include "mst.h"

using namespace CTF;

int main(int argc, char **argv)
{
  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  auto w = new World(argc, argv);

  int n = 2;

  Matrix<int> * A1 = new Matrix<int>(n, n, SP, *w, MAX_TIMES_SR);
  int64_t npair = 4;
  Pair<int> * pairs1 = new Pair<int>[npair];
  pairs1[0] = Pair<int>(0 * n + 0, 10);
  pairs1[1] = Pair<int>(0 * n + 1, 10);
  pairs1[2] = Pair<int>(1 * n + 0, 10);
  pairs1[3] = Pair<int>(1 * n + 1, 10);
  A1->write(npair, pairs1);

  Scalar<int> addid1(5, *w, MAX_TIMES_SR);
  (*A1)["ii"] = addid1[""];
  A1->print();

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  Matrix<EdgeExt> * A2 = new Matrix<EdgeExt>(n, n, SP, *w, MIN_EDGE);
  Pair<EdgeExt> * pairs2 = new Pair<EdgeExt>[npair];
  pairs2[0] = Pair<EdgeExt>(0 * n + 0, EdgeExt(10, 10, 10, 10));
  pairs2[1] = Pair<EdgeExt>(0 * n + 1, EdgeExt(10, 10, 10, 10));
  pairs2[2] = Pair<EdgeExt>(1 * n + 0, EdgeExt(10, 10, 10, 10));
  pairs2[3] = Pair<EdgeExt>(1 * n + 1, EdgeExt(10, 10, 10, 10));
  A2->write(npair, pairs2);

  Scalar<EdgeExt> addid2(EdgeExt(5, 5.0, 5, 5), *w, MIN_EDGE);
  (*A2)["ii"] = addid2[""];
  A2->print();

  delete A2;
  delete A1;
  
  return 0;
}
