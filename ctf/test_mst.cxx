#include "mst.h"

namespace CTF {
  template <>
  inline void Set<Int64Triple>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%zu %zu %zu)", ((Int64Triple*)a)[0].i1, ((Int64Triple*)a)[0].i2, ((Int64Triple*)a)[0].i3);
  }
}

// graph pictured here: https://i0.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-1.png?zoom=2.625&resize=368%2C236&ssl=1
// mst pictured here: https://i1.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-12.png?zoom=2&resize=382%2C237&ssl=1
void test_simple(World * w) {
  // (key, weight, parent in p) 
  // entries in A: (key, weight, -1) 
  // entries in P: (key, -1, parent in p)
  static Semiring<Int64Triple> MIN_TIMES_SR( // TODO: mpi runtime error when in .h file
    Int64Triple(INT_MAX, INT_MAX, -1),
    [](Int64Triple a, Int64Triple b) {
      if (a.i3 > a.i1 && b.i3 > b.i1)
        return Int64Triple(INT_MAX, INT_MAX, -1);
      else
        return Int64Triple(0, 0, 0);
        //return a.i2 < b.i2 ? a.i3 : b.i3;
    },
    MPI_MAX,
    Int64Triple(-1, -1, -1),
    [](Int64Triple a, Int64Triple b) {
      return Int64Triple(a.i1, a.i2, b.i2);
    });

  printf("test_simple\n");
  
  int nrow = 7;
  int ncol = 7;
  Matrix<Int64Triple> * A = new Matrix<Int64Triple>(nrow, ncol, SP|SY, *w, MIN_TIMES_SR);

  int64_t npair = 11;
  Pair<Int64Triple> * pairs = new Pair<Int64Triple>[npair];
  pairs[0] = Pair<Int64Triple>(0 * nrow + 1, Int64Triple(0, 7, -1));
  pairs[1] = Pair<Int64Triple>(0 * nrow + 3, Int64Triple(0, 5, -1));
  pairs[2] = Pair<Int64Triple>(1 * nrow + 2, Int64Triple(1, 8, -1));
  pairs[3] = Pair<Int64Triple>(1 * nrow + 3, Int64Triple(1, 9, -1));
  pairs[4] = Pair<Int64Triple>(1 * nrow + 4, Int64Triple(1, 7, -1));
  pairs[5] = Pair<Int64Triple>(2 * nrow + 4, Int64Triple(2, 5, -1));
  pairs[6] = Pair<Int64Triple>(3 * nrow + 4, Int64Triple(3, 15, -1));
  pairs[7] = Pair<Int64Triple>(3 * nrow + 5, Int64Triple(3, 6, -1));
  pairs[8] = Pair<Int64Triple>(4 * nrow + 5, Int64Triple(4, 8, -1));
  pairs[9] = Pair<Int64Triple>(4 * nrow + 6, Int64Triple(4, 9, -1));
  pairs[10] = Pair<Int64Triple>(5 * nrow + 6, Int64Triple(5, 11, -1));

  A->write(npair, pairs);

  A->print_matrix();

  // DELETE
  int64_t temp_npairs;
  Pair<Int64Triple> * temp_loc_pairs;
  A->get_local_pairs(&temp_npairs, &temp_loc_pairs, true);

  for (int i=0; i<temp_npairs; i++) {
    printf("(%zu %zu %zu)\n", temp_loc_pairs[i].d.i1, temp_loc_pairs[i].d.i2, temp_loc_pairs[i].d.i3);
  }
  // END DELETE

  auto p = new Vector<Int64Triple>(nrow, *w, MIN_TIMES_SR);
  init_pvector(p);

  printf("p:\n");
  p->print();

  // relax all edges //
  auto q = new Vector<Int64Triple>(nrow, SP*p->is_sparse, *w, MIN_TIMES_SR);
  (*q)["i"] = (*p)["i"];
  (*q)["i"] += (*A)["ij"] * (*p)["j"];

  printf("q:\n");
  q->print();

  delete q;
  delete p;
  delete [] pairs;
  delete A;
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
