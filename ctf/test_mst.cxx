#include "mst.h"

#include <unordered_set>

// graph pictured here: https://i0.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-1.png?zoom=2.625&resize=368%2C236&ssl=1
// mst pictured here: https://i1.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-12.png?zoom=2&resize=382%2C237&ssl=1
void test_simple(World * w) {
  printf("test_simple\n");

  class HashFunction {
    public:
      size_t operator() (const Pair<int>& p) const {
        return hash<int>()(p.k) + hash<int>()(p.d);
      }
  };

  int nrow = 7;
  int ncol = 7;

  // unordered of edges: (key, weight) where key encodes (row, col)
  std::unordered_set<Pair<int>, HashFunction> expected; // TODO: not symmetric
  expected.insert(Pair<int>(0 * nrow + 1, 7)); // NOTE: overloaded == for Pair in tensor.h to allow for this
  expected.insert(Pair<int>(0 * nrow + 3, 5));
  expected.insert(Pair<int>(1 * nrow + 4, 7));
  expected.insert(Pair<int>(2 * nrow + 4, 5));
  expected.insert(Pair<int>(3 * nrow + 5, 6));
  expected.insert(Pair<int>(4 * nrow + 6, 9));
 
  Matrix<int> A(nrow, ncol, SP|SY, *w, MAX_TIMES_SR); // TODO: use (min, times) semiring

  int64_t npair = 11;
  Pair<int> pairs[npair];
  pairs[0] = Pair<int>(0 * nrow + 1, 7);
  pairs[1] = Pair<int>(0 * nrow + 3, 5);
  pairs[2] = Pair<int>(1 * nrow + 2, 8);
  pairs[3] = Pair<int>(1 * nrow + 3, 9);
  pairs[4] = Pair<int>(1 * nrow + 4, 7);
  pairs[5] = Pair<int>(2 * nrow + 4, 5);
  pairs[6] = Pair<int>(3 * nrow + 4, 15);
  pairs[7] = Pair<int>(3 * nrow + 5, 6);
  pairs[8] = Pair<int>(4 * nrow + 5, 8);
  pairs[9] = Pair<int>(4 * nrow + 6, 9);
  pairs[10] = Pair<int>(5 * nrow + 6, 11);

  A.write(npair, pairs);

  A.print_matrix();

  std::unordered_set<Pair<int>, HashFunction> mst;
  
  if (expected == mst)
    printf("CORRECT MST FOUND");
  else
    printf("INCORRECT MST FOUND");
}

int main(int argc, char** argv) {
  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  auto w = new World(argc, argv);
  test_simple(w);
}
