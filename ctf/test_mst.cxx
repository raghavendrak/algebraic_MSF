#include "test.h"
#include "mst.h"
#include <ctime>

Matrix<EdgeExt> to_EdgeExt_mat(Matrix<wht> * A_pre) {
  //(*A_pre)["ii"] = INT_MAX;
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  // START FIXME
  Matrix<EdgeExt> A(A_pre->nrow, A_pre->nrow, A_pre->is_sparse, *(A_pre->wrld), MIN_EDGE, A_pre->name); // hook_matrix() fails
  //Matrix<EdgeExt> A(A_pre->nrow, A_pre->nrow, 0, *(A_pre->wrld), MIN_EDGE, A_pre->name); // multilinear_hook() fails
  // END FIXME
  int64_t npairs;
  Pair<wht> * pre_loc_pairs;
  A_pre->get_local_pairs(&npairs, &pre_loc_pairs, A_pre->is_sparse);
  Pair<EdgeExt> * write_pairs = new Pair<EdgeExt>[2 * npairs];
  for (int64_t i = 0; i < npairs; ++i) {
    int64_t row = pre_loc_pairs[i].k / A_pre->nrow;
    int64_t col = pre_loc_pairs[i].k % A_pre->nrow;
    write_pairs[i].k = row + col * A_pre->nrow;
    write_pairs[i].d = EdgeExt(row, pre_loc_pairs[i].d, col, row); 

    // produce symmetry
    write_pairs[i + npairs].k = col + row * A_pre->nrow;
    write_pairs[i + npairs].d = EdgeExt(col, pre_loc_pairs[i].d, row, col); 
  }
  A.write(2 * npairs, write_pairs);

  return A;
}

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

static Monoid<bool> OR_STAR(
    true,
    [](bool a, bool b) { return a || b; },
    MPI_LOR);

Vector<bool> * star_check(Vector<int> * p) {
  Vector<bool> * star = new Vector<bool>(p->len, *(p->wrld), OR_STAR);

  int64_t p_npairs;
  Pair<int> * p_loc_pairs;
  p->get_local_pairs(&p_npairs, &p_loc_pairs);

  // If F(i) =/= GF(i) then ST(i) <- FALSE and ST(GF(i)) <- FALSE
  // excludes vertices that have nontrivial grandparent or grandchild
  Pair<int> * p_parents = new Pair<int>[p_npairs];
  for (int64_t i = 0; i < p_npairs; ++i) {
    p_parents[i].k = p_loc_pairs[i].d;
  } 
  p->read(p_npairs, p_parents);

  Pair<bool> * nontriv_grandX = new Pair<bool>[p_npairs];
  int64_t grandX_npairs = 0;
  for (int64_t i = 0; i < p_npairs; ++i) {
    if (p_loc_pairs[i].d != p_parents[i].d) {
      nontriv_grandX[i].k = p_loc_pairs[i].k;
      nontriv_grandX[i].d = false;

      nontriv_grandX[i].k = p_parents[i].d;
      nontriv_grandX[i].d = false;

      ++grandX_npairs;
    }
  }
  star->write(grandX_npairs, nontriv_grandX);

  // ST(i) <- ST(F(i))
  // excludes vertices that have nontrivial nephews
  Pair<bool> * nontriv_nephews = new Pair<bool>[p_npairs];
  for (int64_t i = 0; i < p_npairs; ++i) {
    nontriv_nephews[i].k = p_loc_pairs[i].d;
  }
  star->read(p_npairs, nontriv_nephews);

  Pair<bool> * updated_nephews = new Pair<bool>[p_npairs];
  for (int64_t i = 0; i < p_npairs; ++i) {
    updated_nephews[i].k = p_loc_pairs[i].k;
    updated_nephews[i].d = nontriv_nephews[i].d;
  }
  star->write(p_npairs, updated_nephews);

  delete [] updated_nephews;
  delete [] nontriv_nephews;
  delete [] nontriv_grandX;
  delete [] p_parents;
  delete [] p_loc_pairs;

  return star;
}


Vector<EdgeExt> * hooking(int64_t A_npairs, Pair<EdgeExt> * A_loc_pairs, Vector<int> * p, Vector<bool> * star) {
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid(); // TODO: pass by reference

  auto r = new Vector<EdgeExt>(p->len, p->is_sparse, *(p->wrld), MIN_EDGE);

  Pair<int> * src_loc_pairs = new Pair<int>[A_npairs];
  for (int64_t i = 0; i < A_npairs; ++i) {
    src_loc_pairs[i].k = A_loc_pairs[i].d.src;
  }
  p->read(A_npairs, src_loc_pairs);

  /*
  Pair<int> * src_parents = new Pair<int>[A_npairs];
  for (int64_t i = 0; i < A_npairs; ++i) {
    src_parents[i].k = src_loc_pairs[i].d;
  } 
  p->read(A_npairs, src_parents);
  */

  Pair<int> * dest_loc_pairs = new Pair<int>[A_npairs];
  for (int64_t i = 0; i < A_npairs; ++i) {
    dest_loc_pairs[i].k = A_loc_pairs[i].d.dest;
  }
  p->read(A_npairs, dest_loc_pairs);

  Pair<bool> * star_loc_pairs = new Pair<bool>[A_npairs];
  for (int64_t i = 0; i < A_npairs; ++i) {
    star_loc_pairs[i].k = src_loc_pairs[i].k;
  }
  star->read(A_npairs, star_loc_pairs);

  //Pair<int> * updated_p_pairs = new Pair<int>[A_npairs];
  Pair<EdgeExt> * updated_r_pairs = new Pair<EdgeExt>[A_npairs];
  int64_t updated_npairs = 0;
  for (int64_t i = 0; i < A_npairs; ++i) {
    if (star_loc_pairs[i].d && src_loc_pairs[i].d != dest_loc_pairs[i].d) {
      updated_r_pairs[updated_npairs].k = src_loc_pairs[i].d;
      updated_r_pairs[updated_npairs].d = A_loc_pairs[i].d;
      updated_r_pairs[updated_npairs].d.parent = dest_loc_pairs[i].d;

      ++updated_npairs; 
    }
  }
  r->write(updated_npairs, updated_r_pairs); // accumulates over MINWEIGHT

  delete [] src_loc_pairs;
  //delete [] src_parents;
  delete [] dest_loc_pairs;
  delete [] star_loc_pairs;
  //delete [] updated_p_pairs;
  delete [] updated_r_pairs;
  delete star;

  return r;
}

// Awerbuch and Shiloach with modified tie breaking scheme
Vector<EdgeExt> * as(Matrix<EdgeExt> * A, World * world) {
  int n = A->nrow;

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  auto mst = new Vector<EdgeExt>(n, *world, MIN_EDGE);

  int64_t A_npairs;
  Pair<EdgeExt> * A_loc_pairs;
  A->get_local_pairs(&A_npairs, &A_loc_pairs, true);

  while(are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];

    // unconditional star hooking
    Vector<bool> * star = star_check(p);

    Vector<EdgeExt> * r = hooking(A_npairs, A_loc_pairs, p, star);

    // tie breaking
    // hook only onto larger stars and update p
    (*p)["i"] += Function<EdgeExt, int>([](EdgeExt e){ return e.parent; })((*r)["i"]);

    // hook only onto larger stars and update mst
    (*mst)["i"] += Bivar_Function<EdgeExt, int, EdgeExt>([](EdgeExt e, int a){ return e.parent >= a ? e : EdgeExt(); })((*r)["i"], (*p)["i"]);

    delete r;

    // shortcutting
    int sc2 = 1000;
    Vector<int> * pi = new Vector<int>(*p);
    shortcut2(*p, *p, *p, sc2, world, NULL, false);
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
    }
    delete pi;
  }
  delete [] A_loc_pairs;
  delete p_prev;
  delete p;

  return mst;
}

// requires edge weights to be distinct
// can also store mst in hashset
double compare_mst(Vector<EdgeExt> * a, Vector<EdgeExt> * b) {
  //Function<EdgeExt,double> sum_weights([](EdgeExt a){ return a.src != -1 ? a.weight : 0; });
  a->sparsify(); // TODO: workaround
  b->sparsify(); // TODO: workaround
  Function<EdgeExt,double> sum_weights([](EdgeExt a){ return a.weight; });

  Scalar<double> s_a;
  s_a[""] = sum_weights((*a)["i"]);

  Scalar<double> s_b;
  s_b[""] = sum_weights((*b)["i"]);

  return s_a.get_val() - s_b.get_val();
}

// requires edge weights to be distinct
// can also store mst in hashset
/*
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
*/

//  0 --- 1
//  |   /     3 -- 4
//  | /
//  2
void test_trivial(World * w) {
  if (w->rank == 0) {
    printf("test_trivial\n");
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
  std::srand(time(NULL));
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

  auto as_mst = as(A, w);
  if (w->rank == 0) {
    printf("as mst\n");
  }
  as_mst->print();

  auto res = compare_mst(kr, hm);
  if (w->rank == 0) {
    if (res) {
      printf("result weight of mst vectors are different by %f: FAIL\n", res);
    }
    else {
      printf("result weight mst vectors are same: PASS\n");
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

  auto as_mst = as(A, w);
  if (w->rank == 0) {
    printf("as mst\n");
  }
  as_mst->print();

  auto mult_mst = multilinear_hook(A, w);
  if (w->rank == 0) {
    printf("multilinear mst\n");
  }
  mult_mst->print();

  /*
  auto res = compare_mst(kr, hm);
  if (w->rank == 0) {
    if (res) {
      printf("result weight of mst vectors are different by %f: FAIL\n", res);
    }
    else {
      printf("result weight of mst vectors are same: PASS\n");
    }
  }
  */

  delete as_mst;
  delete kr;
  delete hm;

  delete [] pairs;
  delete A;
}

void run_mst(Matrix<EdgeExt>* A, int64_t matSize, World *w, int batch, int shortcut, int run_serial, int run_multilinear)
{
  double stime;
  double etime;
  matSize = A->nrow; // Quick fix to avoid change in i/p matrix size after preprocessing
  Vector<EdgeExt>* mult_mst;
  if (run_multilinear) {
    Timer_epoch tmh("multilinear_hook");
    tmh.begin();
    stime = MPI_Wtime();
    mult_mst = multilinear_hook(A, w);
    etime = MPI_Wtime();
    tmh.end();
    if (w->rank == 0) {
      printf("multilinear mst done in %1.2lf\n", (etime - stime));
    }
    mult_mst->print();
    return;
  }
  Vector<EdgeExt> * hm;
  int run_hook = 0;
  if (run_hook) {
    Timer_epoch thm("hook_matrix");
    thm.begin();
    stime = MPI_Wtime();
    hm = hook_matrix(A, w);
    etime = MPI_Wtime();
    if (w->rank == 0) {
      printf("Time for hook_matrix(): %1.2lf\n", (etime - stime));
      printf("hook_matrix() mst:\n");
    }
    hm->print();
    printf("\n");
    //thm.end();
  }
  if (run_multilinear && run_hook) {
    auto res = compare_mst(hm, mult_mst);
    if (w->rank == 0) {
      if (res) {
        printf("multilinear and hook mst vectors are different by %f: FAIL\n", res);
      }
      else {
        printf("multilinear and hook mst vectors are same: PASS\n");
      }
    }
  }
  if (run_serial) {
    auto serial = serial_mst(A, w);
    if(w->rank == 0) printf("serial\n");
    serial->print();
    auto serial_res = compare_mst(serial, mult_mst);
    if (w->rank == 0) {
      if (serial_res) {
        printf("multilinear and serial mst vectors are different by %f: FAIL\n", serial_res);
      }
      else {
        printf("multilinear and serial mst vectors are same: PASS\n");
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
  MPI_Init(&argc, &argv);
  auto w = new World(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
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
      int n_nnz = 0;
      if (w->rank == 0)
      printf("Reading real graph n = %lld\n", n);
      Matrix<wht> A_pre = read_matrix(*w, n, gfile, prep, &n_nnz);
      Matrix<EdgeExt> A = to_EdgeExt_mat(&A_pre);
      int64_t matSize = A.nrow; 
      run_mst(&A, matSize, w, batch, sc2, run_serial, 1);
    }
    else if (k != -1) {
      //int64_t matSize = pow(3, k);
      //auto B = generate_kronecker(w, k);

      //if (w->rank == 0) {
      //printf("Running connectivity on Kronecker graph K: %d matSize: %ld\n", k, matSize);
      //}
      //Matrix<EdgeExt> A = to_EdgeExt_mat(&B);
      //run_mst(&A, matSize, w, batch, sc2, run_serial);
      //delete B;
    }
    else if (scale > 0 && ef > 0){
      int n_nnz = 0;
      myseed = SEED;
      if (w->rank == 0)
        printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
      Matrix<wht> A_pre = gen_rmat_matrix(*w, scale, ef, myseed, prep, &n_nnz, max_ewht);
      Matrix<EdgeExt> A = to_EdgeExt_mat(&A_pre);
      int64_t matSize = A.nrow; 
      run_mst(&A, matSize, w, batch, sc2, run_serial, 1);
    }
    else {
      if (w->rank == 0) {
        printf("Running mst on simple 7x7 graph\n");
      }
      test_simple(w);
      //test_trivial(w);
    }
  }
  MPI_Finalize();
  return 0;
}
