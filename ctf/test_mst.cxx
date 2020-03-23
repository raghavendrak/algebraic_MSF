#include "test.h"
#include "mst.h"

// does not use path compression
int64_t find(int64_t p[], int64_t i) {
  while (p[i] != i) {
    i = p[i];
  }

  return i;
}

// not a smart union
void union1(int64_t p[], int64_t a, int64_t b) {
  int64_t a_dest = find(p, a);
  int64_t b_dest = find(p, b);

  p[a_dest] = b_dest;
}

// Kruskal
Vector<EdgeExt> * serial_mst(Matrix<EdgeExt> * A, World * world) {
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();

  int64_t npair;
  Pair<EdgeExt> * pairs;
  A->get_all_pairs(&npair, &pairs, true);

  EdgeExt edges[npair];
  for (int64_t i = 0; i < npair; ++i) {
    edges[i] = EdgeExt(pairs[i].d.src, pairs[i].d.weight, pairs[i].d.dest, pairs[i].d.parent);
  }

  std::sort(edges, edges + npair, [](const EdgeExt & lhs, const EdgeExt & rhs) { return lhs.weight < rhs.weight; });

  int64_t p[A->nrow];
  for (int64_t i = 0; i < A->nrow; ++i) {
    p[i] = i;
  }

  int64_t mst_npair = A->nrow - 1;
  Pair<EdgeExt> * mst_pairs = new Pair<EdgeExt>[mst_npair];
  int64_t j = 0;
  for (int64_t i = 0; i < npair; ++i) {
  find(p, edges[i].src) != find(p, edges[i].dest);
    if (find(p, edges[i].src) != find(p, edges[i].dest)) {
      mst_pairs[j].k = j;
      mst_pairs[j].d = edges[i];
      ++j;
      union1(p, edges[i].src, edges[i].dest);
    }
  }

  Vector<EdgeExt> * mst = new Vector<EdgeExt>(A->nrow, *world, MIN_EDGE);
  mst->write(mst_npair, mst_pairs);

  delete [] mst_pairs;
  delete [] pairs;

  return mst;
}

/*
// modification of Awerbuch and Shiloach
Vector<EdgeExt> * as(Matrix<EdgeExt> * A, World * world) {
  int n = A->nrow;

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  auto mst = new Vector<EdgeExt>(n, *world, MIN_EDGE);
        
  while(are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];

    int64_t A_n;
    Pair<EdgeExt> * A_loc_pairs;
    A.get_local_pairs(&A_n, &A_loc_pairs);

    // unconditional star hooking
    int64_t edges_n;
    Pair<EdgeExt> * edges_loc_pairs;

    int64_t j = 0;
    for (int64_t i = 0; i < A_n; ++i) {
      int64_t src = A_loc_pairs.d.src;
      int64_t dest = A_loc_pairs.d.dest;
      if (p(p(src)) == p(src) && p(src) == p(dest)) {
        p(p(i)) = p(dest);

        edges[j] = edges_loc_pairs[i];
        ++j;
      }
      mst->write(edges_n, edges_loc_pairs); // accumulate over MINWEIGHT
    }
  
    // tie breaking
    for (int64_t i = 0; i < A_n; ++i) {
      if (i < p(i) && i = p(p(i))) {
        p(i) = i;
      }
    }

    // shortcutting
    shortcut(*p, *p, *p, NULL, false);
  }
}
*/

// requires edge weights to be distinct
// can also store mst in hashset
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

//  0 --- 1
//  |   /     3 -- 4
//  | /
//  2
void test_trivial(World * w) {
  if (w->rank == 0) {
    printf("test_simple\n");
  }

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
   
  int nrow = 5;
  Matrix<EdgeExt> * A = new Matrix<EdgeExt>(nrow, nrow, SP, *w, MIN_EDGE);

  int64_t npair = 2 * 4;
  Pair<EdgeExt> * pairs = new Pair<EdgeExt>[npair];
  pairs[0] = Pair<EdgeExt>(1 * nrow + 0, EdgeExt(0, 7, 1, 0));
  pairs[1] = Pair<EdgeExt>(2 * nrow + 0, EdgeExt(0, 5, 2, 0));
  pairs[2] = Pair<EdgeExt>(2 * nrow + 1, EdgeExt(1, 6, 2, 1));
  pairs[3] = Pair<EdgeExt>(4 * nrow + 3, EdgeExt(3, 10, 4, 3));

  // perturb edge weights and produce anti symmetry
  std::srand(std::time(NULL));
  for (int64_t i = 0; i < npair / 2; ++i) {
    pairs[i].d.weight += (std::rand() / (double) RAND_MAX) / 1000;

    pairs[i + npair / 2].k = (pairs[i].k % nrow) * nrow + pairs[i].k / nrow;
    pairs[i + npair / 2].d = EdgeExt(pairs[i].d.dest, pairs[i].d.weight, pairs[i].d.src, pairs[i].d.dest);
  }

  A->write(npair, pairs);

  auto kr = serial_mst(A, w);
  if (w->rank == 0) {
    printf("serial mst\n");
  }
  kr->print();

  auto hm = hook_matrix(A, w);
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
    pairs[i].d.weight += (std::rand() / (double) RAND_MAX) / 1000;

    pairs[i + npair / 2].k = (pairs[i].k % nrow) * nrow + pairs[i].k / nrow;
    pairs[i + npair / 2].d = EdgeExt(pairs[i].d.dest, pairs[i].d.weight, pairs[i].d.src, pairs[i].d.dest);
  }

  A->write(npair, pairs);

  auto kr = serial_mst(A, w);
  if (w->rank == 0) {
    printf("serial mst\n");
  }
  kr->print();

  auto hm = hook_matrix(A, w);
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

void run_mst(Matrix<EdgeExt>* A, int64_t matSize, World *w, int batch, int shortcut, int run_serial)
{
  matSize = A->nrow; // Quick fix to avoid change in i/p matrix size after preprocessing
  double stime;
  double etime;
  Timer_epoch thm("hook_matrix");
  thm.begin();
  stime = MPI_Wtime();
  auto hm = hook_matrix(A, w);
  etime = MPI_Wtime();
  if (w->rank == 0) {
    printf("Time for hook_matrix(): %1.2lf\n", (etime - stime));
    printf("hook_matrix() mst:\n");
  }
  hm->print();
  thm.end();

  if (run_serial) {
    auto serial= serial_mst(A, w);
    serial->print();
    int64_t res = compare_mst(serial, hm);
    if (w->rank == 0) {
      if (res) {
        printf("result mst vectors are different by %zu: FAIL\n", res);
      }
      else {
        printf("result mst vectors are same: PASS\n");
      }
    }
  }
}

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char** argv)
{
  int rank;
  int np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  auto w = new World(argc, argv);
  int const in_num = argc;
  char** input_str = argv;
  uint64_t myseed;

  int64_t max_ewht;
  uint64_t edges;
  char *gfile = NULL;
  int64_t n;
  int scale;
  int ef;
  int prep;
  int batch;
  int sc2;
  int run_serial;

  int k;
  if (getCmdOption(input_str, input_str+in_num, "-k")) {
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 5;
  } else k = -1;
  // K13 : 1594323 (matrix size)
  // K6 : 729; 531441 vertices
  // k5 : 243
  // k7 : 2187
  // k8 : 6561
  // k9 : 19683
  if (getCmdOption(input_str, input_str+in_num, "-f")){
    gfile = getCmdOption(input_str, input_str+in_num, "-f");
  } else gfile = NULL;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoll(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 27;
  } else n = 27;
  if (getCmdOption(input_str, input_str+in_num, "-S")){
    scale = atoi(getCmdOption(input_str, input_str+in_num, "-S"));
    if (scale < 0) scale=10;
  } else scale=0;
  if (getCmdOption(input_str, input_str+in_num, "-E")){
    ef = atoi(getCmdOption(input_str, input_str+in_num, "-E"));
    if (ef < 0) ef=16;
  } else ef=0;
  if (getCmdOption(input_str, input_str+in_num, "-prep")){
    prep = atoll(getCmdOption(input_str, input_str+in_num, "-prep"));
    if (prep < 0) prep = 0;
  } else prep = 0;
  if (getCmdOption(input_str, input_str+in_num, "-batch")){
    batch = atoll(getCmdOption(input_str, input_str+in_num, "-batch"));
    if (batch <= 0) batch = 1;
  } else batch = 1;
  if (getCmdOption(input_str, input_str+in_num, "-shortcut")){
    sc2 = atoi(getCmdOption(input_str, input_str+in_num, "-shortcut"));
    if (sc2 < 0) sc2 = 0;
  } else sc2 = 0;
  if (getCmdOption(input_str, input_str+in_num, "-serial")){
    run_serial = atoi(getCmdOption(input_str, input_str+in_num, "-serial"));
    if (run_serial < 0) run_serial = 0;
  } else run_serial = 0;

  if (gfile != NULL){
    //int n_nnz = 0;
    //if (w->rank == 0)
      //printf("Reading real graph n = %lld\n", n);
    //Matrix<wht> A = read_matrix(*w, n, gfile, prep, &n_nnz);
    //run_connectivity(&A, n, w, batch, sc2, run_serial);
  }
  else if (k != -1) {
    //int64_t matSize = pow(3, k);
    //auto B = generate_kronecker(w, k);

    //if (w->rank == 0) {
      //printf("Running connectivity on Kronecker graph K: %d matSize: %ld\n", k, matSize);
    //}
    //run_connectivity(B, matSize, w, batch, sc2, run_serial);
    //delete B;
  }
  else if (scale > 0 && ef > 0){
    int n_nnz = 0;
    myseed = SEED;
    if (w->rank == 0)
      printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
    Matrix<EdgeExt> A = gen_rmat_matrix<EdgeExt>(*w, scale, ef, myseed, prep, &n_nnz, max_ewht);
    int64_t matSize = A.nrow; 
    run_mst(&A, matSize, w, batch, sc2, run_serial);
  }
  else {
    if (w->rank == 0) {
      printf("Running mst on simple 7x7 graph\n");
    }
    test_simple(w);
  }
  return 0;
}
