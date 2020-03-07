#include "mst.h"

/* TODO: add shortcut2 (originally omitted for readbilitity) */
EdgeExt EdgeExtMin(EdgeExt a, EdgeExt b){
  /*
  return (a.parent > b.parent || 
          ((a.parent == b.parent) && 
           (a.weight < b.weight))) ? a : b;
  */
  // Choose the minimum weight
  // Parent is chosen through MAX_TIMES_SR
  if (a.key != INT_MAX && b.key != INT_MAX && a.parent == a.comp) {
    return b;
  }
  else if (b.key != INT_MAX && a.key != INT_MAX && b.parent == b.comp) {
    return a;
  }
  else { 
    return ((a.weight <= b.weight) ? a : b);
  }
}

/*
EdgeExt EdgeExtMin(EdgeExt a, EdgeExt b){
  if ((a.weight < b.weight) && (a.parent > b.parent)) return a;
  else return b;
}
*/
EdgeExt ChangeComp(EdgeExt a, EdgeExt b){
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

// MIN_EDGE semiring: type EdgeExt is 3-tuple (key, weight, parent), additive operator performs a op b = a.weight < b.weight ? a : b, multiplicative operator performs FIXME become Monoid
Semiring<EdgeExt> get_minedge_sr(){
    MPI_Op omee;
    MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        EdgeExt_red((EdgeExt*)a, (EdgeExt*)b, *n);
      },
    1, 
    &omee);

   Semiring<EdgeExt> MIN_EDGE(
      EdgeExt(INT_MAX, INT_MAX, -1, INT_MAX),
      [](EdgeExt a, EdgeExt b){ return EdgeExtMin(a, b); }, 
      omee,
      EdgeExt(INT_MAX, INT_MAX, -1, INT_MAX), // mult needed for A.write in hook_matrix
      //[](EdgeExt a, EdgeExt b) { return a; } );
      [](EdgeExt a, EdgeExt b){ return EdgeExtMin(a, b); } );

  return MIN_EDGE; 
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
  (*q)["i"] = Function<int,EdgeExt>([](int p){ return EdgeExt(INT_MAX, INT_MAX, p, INT_MAX); })((*p)["i"]);
  Bivar_Function<EdgeExt,int,EdgeExt> fmv([](EdgeExt e, int p){ return EdgeExt(e.key, e.weight, p, e.comp); });
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
    shortcut<EdgeExt, EdgeExt>(*q, *q, *q, &nonleaves, true);
    if (p->wrld->rank == 0)
      printf("Number of nonleaves or roots is %ld\n",nonleaves->nnz_tot);
    //project to reduced graph with all vertices
    auto rec_A = PTAP(A, q);
    //recurse only on nonleaves
    auto rec_p = supervertex_matrix(n, rec_A, nonleaves, world, sc2);
    delete rec_A;
    //perform one step of shortcutting to update components of leaves
    shortcut<int, EdgeExt>(*p, *q, *rec_p, NULL, false);
    delete q;
    delete rec_p;
    return p;
  }
}

/*
Vector<int>* hook_matrix(int n, Matrix<EdgeExt> * A, World* world)
{
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_sr(); // TODO: correct usage?

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);
  auto prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  while (are_vectors_different(*p, *prev)) {
    (*prev)["i"] = (*p)["i"];
    //auto q = new Vector<int>(n, *world, MAX_TIMES_SR);
    Timer t_relax("CONNECTIVITY_Relaxation");
    t_relax.start();
    auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    (*q)["i"] = Function<int,EdgeExt>([](int p){ return EdgeExt(INT_MAX, INT_MAX, p); })((*p)["i"]);
    Bivar_Function<EdgeExt,int,EdgeExt> fmv([](EdgeExt e, int p){ return EdgeExt(e.key, e.weight, p); });
    fmv.intersect_only=true;
    (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);
    (*p)["i"] = Function<EdgeExt,int>([](EdgeExt e){ return e.parent; })((*q)["i"]);
    //(*q)["i"] = (*A)["ij"] * (*p)["j"];
    t_relax.stop();
    auto r = new Vector<int>(n, *world, MAX_TIMES_SR);
    max_vector(*r, *p, *q);
    //auto P = pMatrix(p, world);
    auto s = new Vector<int>(n, *world, MAX_TIMES_SR);
    //(*s)["i"] = (*P)["ji"] * (*r)["j"];
    //shortcut(*s, *r, *p);
    shortcut<int, int>(*s, *r, *p, NULL, false);
    max_vector(*p, *p, *s);
    Vector<int> * pi = new Vector<int>(*p);
    //shortcut(*p, *p, *p);
    shortcut<int, int>(*p, *p, *p, NULL, false);

    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut<int, int>(*p, *p, *p, NULL, false);
    }
    delete pi;

    delete q;
    delete r;
    delete s;
  }
  return p;
}
*/

Vector<int>* hook_matrix(int n, Matrix<EdgeExt> * A, World* world)
{
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_sr(); // TODO: correct usage?
  //const static Semiring<EdgeExt> MIN_EDGE = get_minedge_sr(); // TODO: correct usage?

  A->print_matrix();
  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);
  /*
  auto mst = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(mst);
  */
  auto ref = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(ref);
  auto prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  auto mst = new Vector<EdgeExt>(n, *world, MIN_EDGE);
  (*mst)["i"] = Function<int, EdgeExt>([](int p) {return EdgeExt(p, p, p, p); })((*ref)["i"]);
  mst->print();


  while (are_vectors_different(*p, *prev)) {
    (*prev)["i"] = (*p)["i"];
    Timer t_relax("CONNECTIVITY_Relaxation");
    t_relax.start();
    auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    // q_i = (inf, inf, p_i)
    (*q)["i"] = Function<int,EdgeExt>([](int p){ return EdgeExt(INT_MAX, INT_MAX, p, INT_MAX); })((*p)["i"]);
    // fmv(e, p) = (e.key, e.w, p)
    Bivar_Function<EdgeExt,int,EdgeExt> fmv([](EdgeExt e, int p){ return EdgeExt(e.key, e.weight, p, e.comp); });
    // fmv should only be applied to nonzeros
    fmv.intersect_only=true;
    // q_i = minweight_{i} fmv(a_{ij},p_j)}
    //printf("HERE0\n");
    /*
    printf("q before:\n");
    q->print();
    printf("p before:\n");
    p->print();
    */
    A->print_matrix();
    (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);
    printf("q after:\n");
    q->print();
    //printf("HERE\n");
    

    auto oldp = new Vector<int>(n, *world, MAX_TIMES_SR);
    (*oldp)["i"] = (*p)["i"];
    // Replace this function to check if the parent has an edge emanating from it which is not the same edge this node is using to hook onto
    (*p)["i"] += Function<EdgeExt,int>([](EdgeExt e){ return e.parent; })((*q)["i"]);
    /*
    Bivar_Function<EdgeExt,int,int> updateP([](EdgeExt e, int p){
        if (e.parent != p) return p;
        else return (int)e.parent;
        });
    updateP.intersect_only = true;
    (*p)["i"] = updateP((*q)["i"], (*oldp)["i"]);
    */
    printf("p after:\n");
    p->print();
    // Aggressive shortcut
    // Shortcut seems to have a bug
    /*
    Vector<int> * pi = new Vector<int>(*p);
    shortcut<int, int>(*p, *p, *p, NULL, false);
    printf("p after shortcut:\n");
    p->print();
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut<int, int>(*p, *p, *p, NULL, false);
    }
    delete pi;
    printf("p after shortcut:\n");
    p->print();
    */
    
    
    Bivar_Function<int, int, int> ch([](int n, int o) {
          if (n == o) return 0;
          else return n;
        });
    auto updated = new Vector<int>(n, *world, MAX_TIMES_SR);
    (*updated)["i"] = ch((*p)["i"], (*oldp)["i"]); 
    
    printf("updated:\n");
    updated->print();
    
    int64_t u_n;
    Pair<int> * u_loc_pairs;
    updated->get_local_pairs(&u_n, &u_loc_pairs, true);
    for (int64_t i = 0; i < u_n; ++i) {
      //printf("u_loc_pairs.k: %lld u_loc_pairs.d: %d\n", u_loc_pairs[i].k, u_loc_pairs[i].d);
      int rowno = u_loc_pairs[i].k; // row no changed
      int compno = u_loc_pairs[i].d; // rowno has a new component
      int nrow_read = A->nrow;
      Pair<EdgeExt> *row_read = new Pair<EdgeExt>[nrow_read];
      for(int j = 0; j < nrow_read; j++) {
        row_read[j].k = rowno + j * nrow_read; // get the whole row data
      }
      A->read(nrow_read, row_read);
      for(int j = 0; j < nrow_read; j++) {
        EdgeExt e = row_read[j].d;
        if (e.parent == -1) continue;
        //printf("row_read[j].k: %d row_read[j].d: %ld %lld %lld %lld new_compno: %d\n", row_read[j].k, e.key, e.weight, e.parent, e.comp, compno);
        row_read[j].d = EdgeExt(e.key, e.weight, e.parent, compno); // update with new component number
      }
      A->write(nrow_read, row_read);
      // A->print();
    }


    /* 
    Bivar_Function<EdgeExt, int, EdgeExt> up([](EdgeExt e, int r){
        return EdgeExt(r, e.weight, e.parent);
        });

    (*A)["ij"] = up((*A)["ii"], (*p)["i"]);
    A->print();
    */

    /*
    (*mst)["i"] = Function<EdgeExt, int, int, int, int>([](EdgeExt e, int p, int ref, int m) {
        // m == ref: my first hook
        // p != ref: only if the parent has changed 
        if (m == ref && p != ref) return e.key;
        else return m;
        })((*q)["i"], (*p)["i"], (*ref)["i"], (*mst)["i"]);
        */


    // mst: <key/mst, ref, parent>
    Bivar_Function<EdgeExt,EdgeExt,EdgeExt> mstf([](EdgeExt e, EdgeExt r){ 
        if (r.key == r.weight && e.parent > r.weight && e.key != r.weight) return EdgeExt(e.key, r.weight, r.parent, e.comp);
        else if (r.key == r.weight && e.parent > r.weight && e.key == r.weight) return EdgeExt(e.parent, r.weight, r.parent, e.comp);
        else return EdgeExt(r.key, r.weight, r.parent, r.comp); 
        });
    // mstf.intersect_only=true;
    // can use Bivar_Transform
    // (*mst)["i"] = mstf((*q)["i"], (*mst)["i"]);
    
    auto mstt = new Vector<EdgeExt>(n, *world, MIN_EDGE);
    (*mstt)["i"] = Function<EdgeExt, EdgeExt>([](EdgeExt e) {return EdgeExt(e.key, e.weight, e.parent, e.comp); })((*mst)["i"]);
    (*mst)["i"] = mstf((*q)["i"], (*mstt)["i"]);

    printf("mst:\n");
    mst->print();
    t_relax.stop();

    /* 
    // zero out edges taken in A
    int64_t p_n;
    Pair<int> * p_loc_pairs;
    p->read_local(&p_n, &p_loc_pairs);

    CTF::Pair<EdgeExt> * updated_loc_pairs = new CTF::Pair<EdgeExt>[n];
    for (int64_t i = 0; i < p_n; ++i) {
      updated_loc_pairs[i].k = p_loc_pairs[i].k + p_loc_pairs[i].d * p_n;
      updated_loc_pairs[i].d = EdgeExt(INT_MAX, INT_MAX, INT_MAX);
    }
    A->write(n, updated_loc_pairs);
    */
    


    delete q;
  }
  return p;
}
