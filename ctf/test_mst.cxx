#include "mst.h"

// graph pictured here: https://i0.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-1.png?zoom=2.625&resize=368%2C236&ssl=1
// mst pictured here: https://i1.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-12.png?zoom=2&resize=382%2C237&ssl=1
void test_simple(World * w) {
  // (key, weight, parent in p) 
  // entries in A: (key, weight, -1) 
  // entries in P: (key, -1, parent in p)

  printf("test_simple\n");

  //static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  
  int nrow = 7; 
  Matrix<Edge> * A = new Matrix<Edge>(nrow, nrow, SP, *w, Set<Edge>());

  int64_t npair = 11;
  Pair<Edge> * pairs = new Pair<Edge>[npair];
  pairs[0] = Pair<Edge>(0 * nrow + 1, Edge(0, 7));
  pairs[1] = Pair<Edge>(0 * nrow + 3, Edge(0, 5));
  pairs[2] = Pair<Edge>(1 * nrow + 2, Edge(1, 8));
  pairs[3] = Pair<Edge>(1 * nrow + 3, Edge(1, 9));
  pairs[4] = Pair<Edge>(1 * nrow + 4, Edge(1, 7));
  pairs[5] = Pair<Edge>(2 * nrow + 4, Edge(2, 5));
  pairs[6] = Pair<Edge>(3 * nrow + 4, Edge(3, 15));
  pairs[7] = Pair<Edge>(3 * nrow + 5, Edge(3, 6));
  pairs[8] = Pair<Edge>(4 * nrow + 5, Edge(4, 8));
  pairs[9] = Pair<Edge>(4 * nrow + 6, Edge(4, 9));
  pairs[10] = Pair<Edge>(5 * nrow + 6, Edge(5, 11));

  A->write(npair, pairs);

  //printf("A:\n");
  //A->print_matrix();

  auto p = new Vector<int>(nrow, SP, *w);
  init_pvector(p);

  //printf("p:\n");
  //p->print();

  //int sc2 = 0;
  //supervertex_matrix(nrow, A, p, w, sc2);
  
  // tests setup
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid(); // TODO: correct usage?
  auto q = new Vector<EdgeExt>(nrow, p->is_sparse, *w, MIN_EDGE);
  (*q)["i"] = Function<int,EdgeExt>([](int p){ return EdgeExt(INT_MAX, INT_MAX, p); })((*p)["i"]);
  Bivar_Function<Edge,int,EdgeExt> fmv([](Edge e, int p){ return EdgeExt(e.key, e.weight, p); });
  fmv.intersect_only=true;
  (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);
  (*p)["i"] = Function<EdgeExt,int>([](EdgeExt e){ return e.parent; })((*q)["i"]);
  // tests setup end

  // test are_vectors_different // TODO: convergence when p["i"] == q["i"].key? or .parent?
  printf("test are_vectors_different\n");
  printf("p:\n");
  p->print();

  printf("q:\n");
  q->print();

  //int64_t diff = are_vectors_different(*p, *q);
  //if (p->wrld->rank == 0)
  //  printf("Diff is %ld\n",diff);
  // end test are_vectors_different //

  // test shortcut1 //
  printf("test shortcut1\n");
  Vector<int> * nonleaves;
  //shortcut<EdgeExt>(*q, *q, *q, &nonleaves, true);
  if (p->wrld->rank == 0)
    printf("Number of nonleaves or roots is %ld\n",nonleaves->nnz_tot);
  // end test shortcut //
  
  // test PTAP //
  //auto rec_A = PTAP(A, q);
  // end test PTAP //
  
  // test shortcut2 //
  //auto rec_p = supervertex_matrix(n, rec_A, nonleaves, world, sc2);
  //shortcut(*p, *q, *rec_p);
  // end test shortcut2 //

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
