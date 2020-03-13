#include "mst.h"

/*
void test_are_vectors_different(Vector<int> * p, Vector<EdgeExt> * q) {
  printf("test are_vectors_different\n");
  printf("p:\n");
  p->print();

  printf("q:\n");
  q->print();

  int64_t diff = are_vectors_different(*p, *q);
  if (p->wrld->rank == 0)
    printf("Diff is %ld\n",diff);
}
*/

/*
void test_shortcut1(Vector<int> * p, Vector<EdgeExt> * q, Vector<int> * nonleaves) {
  printf("test shortcut1\n");
  shortcut<EdgeExt>(*q, *q, *q, &nonleaves, true);
  if (p->wrld->rank == 0)
    printf("Number of nonleaves or roots is %ld\n",nonleaves->nnz_tot);
}
*/

/*
Matrix<EdgeExt> * test_PTAP(Matrix<EdgeExt> * A, Vector<EdgeExt> * q) {
  printf("test_PTAP\n");
  auto rec_A = PTAP(A, q);
  return rec_A;
}
*/


//void test_shortcut2(int n, Matrix<Edge> * A, Vector<int> * nonleaves, World * w, int sc2) {
//  auto rec_p = supervertex_matrix(n, rec_A, nonleaves, world, sc2);
//  shortcut<int>(*p, *q, *rec_p);
//}


// does not use path parentression
unsigned int find(unsigned int p[], unsigned int i) {
  while (p[i] != i) {
    i = p[i];
  }

  return i;
}

// not a smart union
void union1(unsigned int p[], unsigned int a, unsigned int b) {
  unsigned int a_dest = find(p, a);
  unsigned int b_dest = find(p, b);

  p[a_dest] = b_dest;
}

// Kruskal
Vector<int> * serial_mst(Matrix<EdgeExt> * A) {
  int64_t npair;
  Pair<EdgeExt> * pairs;
  A->get_all_pairs(&npair, &pairs, true);

  EdgeExt edges[npair];
  for (unsigned int i = 0; i < npair; ++i) {
    edges[i] = EdgeExt(pairs[i].k / A->nrow, pairs[i].d.weight, pairs[i].d.dest, pairs[i].d.parent);
  }

  std::sort(edges, edges + npair, [](const EdgeExt & lhs, const EdgeExt & rhs) { return lhs.weight < rhs.weight; });

  unsigned int p[A->nrow];
  for (unsigned int i = 0; i < A->nrow; ++i) {
    p[i] = i;
  }

  for (unsigned int i = 0; i < npair; ++i) {
    if (find(p, edges[i].src) != find(p, edges[i].dest)) {
      union1(p, edges[i].src, edges[i].dest);
      printf("src, weight, dest: %d, %d, %d\n", edges[i].src, edges[i].weight, edges[i].dest); // TODO: store in hashset
    }
  }

  Vector<int> * mst = new Vector<int>();
  return mst;
}

// graph pictured here: https://i0.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-1.png?zoom=2.625&resize=368%2C236&ssl=1
// mst pictured here: https://i1.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-12.png?zoom=2&resize=382%2C237&ssl=1
void test_simple(World * w) {
  printf("test_simple\n");

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
   
  int nrow = 7;
  Matrix<EdgeExt> * A = new Matrix<EdgeExt>(nrow, nrow, SP, *w, MIN_EDGE);

  int64_t npair = 11;
  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[npair];
  pairs[0] = Pair<EdgeExt>(0 * nrow + 1, EdgeExt(0, 7, 1, 0));
  pairs[1] = Pair<EdgeExt>(0 * nrow + 3, EdgeExt(0, 5, 3, 0));
  pairs[2] = Pair<EdgeExt>(1 * nrow + 2, EdgeExt(1, 8, 2, 1));
  pairs[3] = Pair<EdgeExt>(1 * nrow + 3, EdgeExt(1, 9, 3, 1));
  pairs[4] = Pair<EdgeExt>(1 * nrow + 4, EdgeExt(1, 7, 4, 1));
  pairs[5] = Pair<EdgeExt>(2 * nrow + 4, EdgeExt(2, 5, 4, 2));
  pairs[6] = Pair<EdgeExt>(3 * nrow + 4, EdgeExt(3, 15, 4, 3));
  pairs[7] = Pair<EdgeExt>(3 * nrow + 5, EdgeExt(3, 6, 5, 3));
  pairs[8] = Pair<EdgeExt>(4 * nrow + 5, EdgeExt(4, 8, 5, 4));
  pairs[9] = Pair<EdgeExt>(4 * nrow + 6, EdgeExt(4, 9, 6, 5));
  pairs[10] = Pair<EdgeExt>(5 * nrow + 6, EdgeExt(5, 11, 6, 5));

  pairs[11] = Pair<EdgeExt>(1 * nrow + 0, EdgeExt(1, 7, 0, 1));
  pairs[12] = Pair<EdgeExt>(3 * nrow + 0, EdgeExt(3, 5, 0, 3));
  pairs[13] = Pair<EdgeExt>(2 * nrow + 1, EdgeExt(2, 8, 1, 2));
  pairs[14] = Pair<EdgeExt>(3 * nrow + 1, EdgeExt(3, 9, 1, 3));
  pairs[15] = Pair<EdgeExt>(4 * nrow + 1, EdgeExt(4, 7, 1, 4));
  pairs[16] = Pair<EdgeExt>(4 * nrow + 2, EdgeExt(4, 5, 2, 4));
  pairs[17] = Pair<EdgeExt>(4 * nrow + 3, EdgeExt(4, 15, 3, 4));
  pairs[18] = Pair<EdgeExt>(5 * nrow + 3, EdgeExt(5, 6, 3, 5));
  pairs[19] = Pair<EdgeExt>(5 * nrow + 4, EdgeExt(5, 8, 4, 5));
  pairs[20] = Pair<EdgeExt>(6 * nrow + 4, EdgeExt(6, 9, 4, 6));
  pairs[21] = Pair<EdgeExt>(6 * nrow + 5, EdgeExt(6, 11, 5, 6));

  /*
  int64_t npair = 22;
  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[npair];
  pairs[0] = Pair<EdgeExt>(0 * nrow + 1, EdgeExt(0, 7, 0, 1));
  pairs[1] = Pair<EdgeExt>(0 * nrow + 3, EdgeExt(0, 5, 0, 3));
  pairs[2] = Pair<EdgeExt>(1 * nrow + 2, EdgeExt(1, 8, 1, 2));
  pairs[3] = Pair<EdgeExt>(1 * nrow + 3, EdgeExt(1, 9, 1, 3));
  pairs[4] = Pair<EdgeExt>(1 * nrow + 4, EdgeExt(1, 7, 1, 4));
  pairs[5] = Pair<EdgeExt>(2 * nrow + 4, EdgeExt(2, 5, 2, 4));
  pairs[6] = Pair<EdgeExt>(3 * nrow + 4, EdgeExt(3, 15, 3, 4));
  pairs[7] = Pair<EdgeExt>(3 * nrow + 5, EdgeExt(3, 6, 3, 5));
  pairs[8] = Pair<EdgeExt>(4 * nrow + 5, EdgeExt(4, 8, 4, 5));
  pairs[9] = Pair<EdgeExt>(4 * nrow + 6, EdgeExt(4, 9, 4, 6));
  pairs[10] = Pair<EdgeExt>(5 * nrow + 6, EdgeExt(5, 11, 5, 6));
  
  pairs[11] = Pair<EdgeExt>(1 * nrow + 0, EdgeExt(0, 7, 0, 0));
  pairs[12] = Pair<EdgeExt>(3 * nrow + 0, EdgeExt(0, 5, 0, 0));
  pairs[13] = Pair<EdgeExt>(2 * nrow + 1, EdgeExt(1, 8, 1, 1));
  pairs[14] = Pair<EdgeExt>(3 * nrow + 1, EdgeExt(1, 9, 1, 1));
  pairs[15] = Pair<EdgeExt>(4 * nrow + 1, EdgeExt(1, 7, 1, 1));
  pairs[16] = Pair<EdgeExt>(4 * nrow + 2, EdgeExt(2, 5, 2, 2));
  pairs[17] = Pair<EdgeExt>(4 * nrow + 3, EdgeExt(3, 15, 3, 3));
  pairs[18] = Pair<EdgeExt>(5 * nrow + 3, EdgeExt(3, 6, 3, 3));
  pairs[19] = Pair<EdgeExt>(5 * nrow + 4, EdgeExt(4, 8, 4, 4));
  pairs[20] = Pair<EdgeExt>(6 * nrow + 4, EdgeExt(4, 9, 4, 4));
  pairs[21] = Pair<EdgeExt>(6 * nrow + 5, EdgeExt(5, 11, 5, 5));
  */

  /*
  int nrow = 3;
  Matrix<EdgeExt> * A = new Matrix<EdgeExt>(nrow, nrow, SP|SY, *w, MIN_EDGE);

  int64_t npair = 2;
  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[npair];
  //pairs[0] = Pair<EdgeExt>(0 * nrow + 1, EdgeExt(0, 30, 1));
  //pairs[1] = Pair<EdgeExt>(0 * nrow + 3, EdgeExt(0, 10, 2));
  pairs[0] = Pair<EdgeExt>(1 * nrow + 0, EdgeExt(0, 30, 0));
  pairs[1] = Pair<EdgeExt>(2 * nrow + 0, EdgeExt(0, 10, 0));
  // pairs[2] = Pair<EdgeExt>(2 * nrow + 1, EdgeExt(1, 5, 0));
  */
  A->write(npair, pairs);

  printf("hook_matrix\n");
  auto hm = hook_matrix(A->nrow, A, w);
  hm->print();
  delete hm;

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
