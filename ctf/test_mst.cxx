#include "mst.h"
#include "mst_serial.cxx"

// requires edge weights to be distinct
int64_t compare_mst(Vector<EdgeExt> * a, Vector<EdgeExt> * b) {
  int64_t a_n;
  Pair<EdgeExt> * a_pairs; 
  a->get_all_pairs(&a_n, &a_pairs, true);
  std::sort(a_pairs, a_pairs + a_n, [](const Pair<EdgeExt> & lhs, const Pair<EdgeExt> & rhs) { return lhs.d.weight < rhs.d.weight; });

  int64_t b_n;
  Pair<EdgeExt> * b_pairs; 
  b->get_all_pairs(&b_n, &b_pairs, true);
  std::sort(b_pairs, b_pairs + b_n, [](const Pair<EdgeExt> & lhs, const Pair<EdgeExt> & rhs) { return lhs.d.weight < rhs.d.weight; });

  for (int64_t i = 0; i < a_n; ++i) {
    a_pairs[i].k = i;
    b_pairs[i].k = i;
  }

  a->write(a_n, a_pairs);
  b->write(b_n, b_pairs);

  // mst may store edge from src to parent or parent to src
  CTF::Scalar<int64_t> s;
  s[""] += CTF::Function<EdgeExt,EdgeExt,int64_t>([](EdgeExt a, EdgeExt b){ 
    return !(((a.src == b.src && a.dest == b.dest) || (a.src == b.dest && a.dest == b.src)) && a.weight == b.weight) ; 
  })((*a)["i"],(*b)["i"]);

  delete b_pairs;
  delete a_pairs;

  return s.get_val();
}

// graph pictured here: https://i0.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-1.png?zoom=2.625&resize=368%2C236&ssl=1
// mst pictured here: https://i1.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-12.png?zoom=2&resize=382%2C237&ssl=1
void test_simple(World * w) {
  if (w->rank == 0) {
    printf("test_simple\n");
  }

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
   
  int nrow = 7;
  Matrix<EdgeExt> * A = new Matrix<EdgeExt>(nrow, nrow, SP, *w, MIN_EDGE);

  int64_t npair = 2 * 11;
  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[npair];
  pairs[0] = Pair<EdgeExt>(1 * nrow + 0, EdgeExt(0, 7, 1, 0));
  pairs[1] = Pair<EdgeExt>(3 * nrow + 0, EdgeExt(0, 5, 3, 0));
  pairs[2] = Pair<EdgeExt>(2 * nrow + 1, EdgeExt(1, 8, 2, 1));
  pairs[3] = Pair<EdgeExt>(3 * nrow + 1, EdgeExt(1, 9, 3, 1));
  pairs[4] = Pair<EdgeExt>(4 * nrow + 1, EdgeExt(1, 7, 4, 1));
  pairs[5] = Pair<EdgeExt>(4 * nrow + 2, EdgeExt(2, 5, 4, 2));
  pairs[6] = Pair<EdgeExt>(4 * nrow + 3, EdgeExt(3, 15, 4, 3));
  pairs[7] = Pair<EdgeExt>(5 * nrow + 3, EdgeExt(3, 6, 5, 3));
  pairs[8] = Pair<EdgeExt>(5 * nrow + 4, EdgeExt(4, 8, 5, 4));
  pairs[9] = Pair<EdgeExt>(6 * nrow + 4, EdgeExt(4, 9, 6, 4));
  pairs[10] = Pair<EdgeExt>(6 * nrow + 5, EdgeExt(5, 11, 6, 5));

  // perturb edge weights and produce anti symmetry
  std::srand(std::time(NULL));
  for (int64_t i = 0; i < npair / 2; ++i) {
    pairs[i].d.weight += std::rand() / (double) RAND_MAX;

    pairs[i + npair / 2].k = (pairs[i].k % nrow) * nrow + pairs[i].k / nrow;
    pairs[i + npair / 2].d = EdgeExt(pairs[i].d.dest, pairs[i].d.weight, pairs[i].d.src, pairs[i].d.dest);
  }

  A->write(npair, pairs);

  auto kr = serial_mst(A, w);
  if (w->rank == 0) {
    printf("serial mst\n");
  }
  kr->print();

  auto hm = hook_matrix(A->nrow, A, w);
  if (w->rank == 0) {
    printf("hook_matrix mst\n");
  }
  hm->print();

  int64_t res = compare_mst(kr, hm);
  if (w->rank == 0) {
    if (res) {
      printf("result mst vectors are different by %zu: FAIL\n", res);
    }
    else {
      printf("result mst vectors are same: PASS\n");
    }
  }

  delete kr;
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
