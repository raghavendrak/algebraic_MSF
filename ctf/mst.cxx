#include "mst.h"

/* TODO: add shortcut2 (originally omitted for readbilitity) */

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

/*
Monoid<EdgeExt> get_minedge_monoid(){
    MPI_Op omee;
    MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        EdgeExt_red((EdgeExt*)a, (EdgeExt*)b, *n);
      },
    1, 
    &omee);

    Monoid<EdgeExt> MIN_EDGE(
      EdgeExt(INT_MAX, INT_MAX, INT_MAX), 
      [](EdgeExt a, EdgeExt b){ return EdgeExtMin(a, b); }, 
      omee);

  return MIN_EDGE; 
}
*/

Semiring<EdgeExt> get_minedge_sr(){
    MPI_Op omee;
    MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        EdgeExt_red((EdgeExt*)a, (EdgeExt*)b, *n);
      },
    1, 
    &omee);

   Semiring<EdgeExt> MIN_EDGE(
      EdgeExt(INT_MAX, INT_MAX, INT_MAX), 
      [](EdgeExt a, EdgeExt b){ return EdgeExtMin(a, b); }, 
      omee,
      EdgeExt(INT_MAX, INT_MAX, INT_MAX), // mult needed for p.write in shortcut
      [](EdgeExt a, EdgeExt b) { return a; } );

  return MIN_EDGE; 
}

// NOTE: can't use bool as return
int64_t are_vectors_different(CTF::Vector<int> & A, CTF::Vector<EdgeExt> & B)
{
  CTF::Scalar<int64_t> s;
  if (!A.is_sparse && !B.is_sparse){
    s[""] += CTF::Function<int,EdgeExt,int64_t>([](int a, EdgeExt b){ return a!=b.parent; })(A["i"],B["i"]);
  } else {
    auto C = Vector<int>(A.len, SP*A.is_sparse, *A.wrld);
    C["i"] += A["i"];
    auto B_keys = Vector<int>(B.len, SP*B.is_sparse, *B.wrld);
    B_keys["i"] = CTF::Function<EdgeExt,int64_t>([](EdgeExt b){ return b.parent; })(B["i"]);
    ((int64_t)-1)*C["i"] += B_keys["i"];
    s[""] += CTF::Function<int,int64_t>([](int a){ return (int64_t)(a!=0); })(C["i"]);

  }
  return s.get_val();
}

void init_pvector(Vector<int>* p)
{
  int64_t npairs;
  Pair<int> * loc_pairs;
  p->read_local(&npairs, &loc_pairs);
  for (int64_t i = 0; i < npairs; i++){
    loc_pairs[i].d = loc_pairs[i].k;
  }
  p->write(npairs, loc_pairs);
  delete [] loc_pairs;
}

Matrix<int>* pMatrix(Vector<int>* p, World* world)
{
  /*
  auto n = p->len;
  auto A = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);

  (*A)["ij"] = CTF::Function<int,int,int>([](int a, int b){ return a==b; })((*p)["i"],(*pg)["j"]);
  return A;
  */
  // FIXME: below is not benchmarked, nor sure if this is the right way of doing it
  int64_t n = p->len;
  auto A = new Matrix<int>(n, n, SP|SY, *world, MAX_TIMES_SR);
  int64_t npairs;
  Pair<int> * loc_pairs;
  p->read_local(&npairs, &loc_pairs);
  int64_t *gIndex = new int64_t[npairs];
  int *gData = new int[npairs];
  for (int64_t i = 0; i < npairs; i++){
    gIndex[i] = loc_pairs[i].k + loc_pairs[i].d * n;
    gData[i] = 1;
  }
  A->write(npairs, gIndex, gData);
  delete [] gIndex;
  delete [] gData;
  delete [] loc_pairs;
  return A;
}


// return B where B[i,j] = A[p[i],p[j]], or if P is P[i,j] = p[i], compute B = P^T A P
Matrix<EdgeExt>* PTAP(Matrix<EdgeExt>* A, Vector<EdgeExt>* p){
  Timer t_ptap("CONNECTIVITY_PTAP");
  t_ptap.start();
  int np = p->wrld->np;
  int64_t n = p->len;
  Pair<EdgeExt> * pprs;
  int64_t npprs;
  //get local part of p
  p->get_local_pairs(&npprs, &pprs);
  assert((npprs <= (n+np-1)/np) && (npprs >= (n/np)));
  assert(A->ncol == n);
  assert(A->nrow == n);
  Pair<EdgeExt> * A_prs;
  int64_t nprs;
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the row of A (A1)
    Matrix<EdgeExt> A1(n, n, "ij", Partition(1,&np)["i"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A1["ij"] = A->operator[]("ij");
    A1.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and rows of A are distributed cyclically, to compute P^T * A
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k/n)*n + pprs[(A_prs[i].k%n)/np].d.parent;
    }
  }
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the column of A (A1)
    Matrix<EdgeExt> A2(n, n, "ij", Partition(1,&np)["j"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    //write in P^T A into A2
    A2.write(nprs, A_prs);
    delete [] A_prs;
    A2.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and cols of A are distributed cyclically, to compute P^T A * P
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k%n) + pprs[(A_prs[i].k/n)/np].d.parent*n;
    }
  }
  Matrix<EdgeExt> * PTAP = new Matrix<EdgeExt>(n, n, SP*(A->is_sparse), *A->wrld, *A->sr);
  PTAP->write(nprs, A_prs);
  delete [] A_prs;
  t_ptap.stop();
  return PTAP;
}


//recursive projection based algorithm
Vector<int>* supervertex_matrix(int n, Matrix<EdgeExt>* A, Vector<int>* p, World* world, int sc2)
{
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_sr(); // TODO: correct usage?

  //relax all edges
  Timer t_relax("CONNECTIVITY_Relaxation");
  t_relax.start();
  auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
  (*q)["i"] = Function<int,EdgeExt>([](int p){ return EdgeExt(INT_MAX, INT_MAX, p); })((*p)["i"]);
  Bivar_Function<EdgeExt,int,EdgeExt> fmv([](EdgeExt e, int p){ return EdgeExt(e.key, e.weight, p); });
  fmv.intersect_only=true;
  (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);
  (*p)["i"] = Function<EdgeExt,int>([](EdgeExt e){ return e.parent; })((*q)["i"]);
  t_relax.stop();

  Vector<int> * nonleaves;
  //check for convergence
  int64_t diff = are_vectors_different(*p, *q);
  if (p->wrld->rank == 0)
    printf("Diff is %ld\n",diff);
  if (!diff){
    return p;
  } else {
    //compute shortcutting q[i] = q[q[i]], obtain nonleaves or roots (FIXME: can we also remove roots that are by themselves?)
    shortcut_EdgeExt(*q, *q, *q, &nonleaves, true);
    if (p->wrld->rank == 0)
      printf("Number of nonleaves or roots is %ld\n",nonleaves->nnz_tot);
    //project to reduced graph with all vertices
    auto rec_A = PTAP(A, q);
    //recurse only on nonleaves
    auto rec_p = supervertex_matrix(n, rec_A, nonleaves, world, sc2);
    delete rec_A;
    //perform one step of shortcutting to update components of leaves
    shortcut_int(*p, *q, *rec_p);
    delete q;
    delete rec_p;
    return p;
  }
}

void shortcut_EdgeExt(Vector<EdgeExt> & p, Vector<EdgeExt> & q, Vector<EdgeExt> & rec_p, Vector<int> ** nonleaves, bool create_nonleaves)
{
  Timer t_shortcut("CONNECTIVITY_Shortcut");
  t_shortcut.start();
  int64_t npairs;
  Pair<EdgeExt> * loc_pairs;
  if (q.is_sparse){
    //if we have updated only a subset of the vertices
    q.get_local_pairs(&npairs, &loc_pairs, true);
  } else {
    //if we have potentially updated all the vertices
    q.get_local_pairs(&npairs, &loc_pairs);
  }
  Pair<EdgeExt> * remote_pairs = new Pair<EdgeExt>[npairs];
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d.parent;
  }
  Timer t_shortcut_read("CONNECTIVITY_Shortcut_read");
  t_shortcut_read.start();
  rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
  t_shortcut_read.stop();
  
  Pair<EdgeExt> * updated_loc_pairs = new Pair<EdgeExt>[npairs];
  for (int64_t i=0; i<npairs; i++){
      updated_loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
  }
  delete [] remote_pairs;
  //p.write(npairs, updated_loc_pairs); //enter data into p[i] // TODO: no multiplication operation for this algebraic structure

  //prune out leaves
  if (create_nonleaves){
    *nonleaves = new Vector<int>(p.len, *p.wrld, MAX_TIMES_SR);
    //set nonleaves[i] = max_j p[j], i.e. set nonleaves[i] = 1 if i has child, i.e. is nonleaf
    Pair<int> * updated_nonleaves = new Pair<int>[npairs];
    for (int64_t i=0; i<npairs; i++){
      updated_nonleaves[i].k = updated_loc_pairs[i].d.parent;
      updated_nonleaves[i].d = 1;
    }
    //FIXME: here and above potential optimization is to avoid duplicate queries to parent
    (*nonleaves)->write(npairs, updated_nonleaves);

    auto p_parents = Vector<int>(p.len, SP*p.is_sparse, *p.wrld);
    p_parents["i"] = CTF::Function<EdgeExt,int64_t>([](EdgeExt p){ return p.parent; })(p["i"]);

    (*nonleaves)->operator[]("i") = (*nonleaves)->operator[]("i")*p_parents["i"];
    (*nonleaves)->sparsify();
  }
  
  delete [] loc_pairs;
  delete [] updated_loc_pairs;
  t_shortcut.stop();
}

void shortcut_int(Vector<int> & p, Vector<EdgeExt> & q, Vector<int> & rec_p, Vector<int> ** nonleaves, bool create_nonleaves)
{
  Timer t_shortcut("CONNECTIVITY_Shortcut");
  t_shortcut.start();
  int64_t npairs;
  Pair<EdgeExt> * loc_pairs;
  if (q.is_sparse){
    //if we have updated only a subset of the vertices
    q.get_local_pairs(&npairs, &loc_pairs, true);
  } else {
    //if we have potentially updated all the vertices
    q.get_local_pairs(&npairs, &loc_pairs);
  }
  Pair<int> * remote_pairs = new Pair<int>[npairs];
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d.parent;
  }
  Timer t_shortcut_read("CONNECTIVITY_Shortcut_read");
  t_shortcut_read.start();
  rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
  t_shortcut_read.stop();
  
  Pair<int> * updated_loc_pairs = new Pair<int>[npairs];
  for (int64_t i=0; i<npairs; i++){
      updated_loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
  }
  delete [] remote_pairs;
  //p.write(npairs, updated_loc_pairs); //enter data into p[i] // TODO: no multiplication operation for this algebraic structure

  //prune out leaves
  if (create_nonleaves){
    *nonleaves = new Vector<int>(p.len, *p.wrld, MAX_TIMES_SR);
    //set nonleaves[i] = max_j p[j], i.e. set nonleaves[i] = 1 if i has child, i.e. is nonleaf
    Pair<int> * updated_nonleaves = new Pair<int>[npairs];
    for (int64_t i=0; i<npairs; i++){
      updated_nonleaves[i].k = updated_loc_pairs[i].d;
      updated_nonleaves[i].d = 1;
    }
    //FIXME: here and above potential optimization is to avoid duplicate queries to parent
    (*nonleaves)->write(npairs, updated_nonleaves);

    auto p_parents = Vector<int>(p.len, SP*p.is_sparse, *p.wrld);
    p_parents["i"] = CTF::Function<EdgeExt,int64_t>([](EdgeExt p){ return p.parent; })(p["i"]);

    (*nonleaves)->operator[]("i") = (*nonleaves)->operator[]("i")*p_parents["i"];
    (*nonleaves)->sparsify();
  }
  
  delete [] loc_pairs;
  delete [] updated_loc_pairs;
  t_shortcut.stop();
}
