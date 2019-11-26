#include "mst.h"

namespace CTF {
  template <>
  inline void Set<Edge>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%zu %zu)", ((Edge*)a)[0].key, ((Edge*)a)[0].weight);
  }

  template <>
  inline void Set<EdgeExt>::print(char const * a, FILE * fp) const {
    fprintf(fp, "(%zu %zu %zu)", ((EdgeExt*)a)[0].key, ((EdgeExt*)a)[0].weight, ((EdgeExt*)a)[0].parent);
  }
}

EdgeExt EdgeExtMin(EdgeExt a, EdgeExt b){
  if (a.parent < b.parent)
    return a.weight < b.weight ? a : b;
  else
    return a;
}

void EdgeExt_red(EdgeExt const * a,
                 EdgeExt * b,
                 int n){
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i=0; i<n; i++){
    b[i] = EdgeExtMin(a[i], b[i]);
  }
}

// graph pictured here: https://i0.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-1.png?zoom=2.625&resize=368%2C236&ssl=1
// mst pictured here: https://i1.wp.com/www.techiedelight.com/wp-content/uploads/2016/11/Kruskal-12.png?zoom=2&resize=382%2C237&ssl=1
void test_simple(World * w) {
  // (key, weight, parent in p) 
  // entries in A: (key, weight, -1) 
  // entries in P: (key, -1, parent in p)
  MPI_Op omee;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        EdgeExt_red((EdgeExt*)a, (EdgeExt*)b, *n);
      },
      1, &omee);

  static Monoid<EdgeExt> MIN_EDGE(EdgeExt(INT_MAX, INT_MAX, INT_MAX), EdgeExtMin, omee);

  printf("test_simple\n");
  
  int nrow = 7; 
  int ncol = 7;
  Matrix<Edge> * A = new Matrix<Edge>(nrow, ncol, SP, *w, Set<Edge>());

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

  printf("A:\n");
  A->print_matrix();

  auto p = new Vector<int>(nrow, *w);
  init_pvector(p);

  printf("p:\n");
  p->print();

  // relax all edges //
  auto q = new Vector<EdgeExt>(nrow, SP, *w, MIN_EDGE); // TODO: SP*p->is_sparse ruins q
  (*q)["i"] = Function<int,EdgeExt>([](int p){ return EdgeExt(INT_MAX, INT_MAX, p); })((*p)["i"]);
  Bivar_Function<Edge,int,EdgeExt> fmv([](Edge e, int p){ return EdgeExt(e.key, e.weight, p); });
  fmv.intersect_only=true;
  (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);  // TODO: errors
  (*p)["i"] = Function<EdgeExt,int>([](EdgeExt e){ return e.parent; })((*q)["i"]);

  printf("q:\n");
  q->print();
  
  printf("p:\n");
  p->print();

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
