#include "mst.h"

EdgeExt EdgeExtMin(EdgeExt a, EdgeExt b){
  return a.weight < b.weight ? a : b;
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

Monoid<EdgeExt> get_minedge_monoid(){
    MPI_Op omee;
    MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        EdgeExt_red((EdgeExt*)a, (EdgeExt*)b, *n);
      },
    1, 
    &omee);

    Monoid<EdgeExt> MIN_EDGE(
      EdgeExt(-1, INT_MAX, -1, -1), 
      [](EdgeExt a, EdgeExt b){ return EdgeExtMin(a, b); }, 
      omee);

  return MIN_EDGE; 
}

// r[p[j]] = q[j] over MINWEIGHT
void project(Vector<EdgeExt> & r, Vector<int> & p, Vector<EdgeExt> & q)
{ 
  TAU_FSTART(CONNECTIVITY_Project);
  
  int64_t q_npairs;
  Pair<EdgeExt> * q_loc_pairs;
  if (q.is_sparse){
    //if we have updated only a subset of the vertices
    q.get_local_pairs(&q_npairs, &q_loc_pairs, true);
  } else {
    //if we have potentially updated all the vertices
    q.get_local_pairs(&q_npairs, &q_loc_pairs);
  }

  Pair<int> * p_read_pairs = new Pair<int>[q_npairs];
  for(int64_t i = 0; i < q_npairs; ++i) {
    p_read_pairs[i].k = q_loc_pairs[i].k;
  }
  p.read_local(&q_npairs, &p_read_pairs);
  //p.read(q_npairs, p_read_pairs); // if q->sparsify() enabled

  Pair<EdgeExt> * r_loc_pairs = new Pair<EdgeExt>[q_npairs];
  for (int64_t i = 0; i < q_npairs; ++i){
    r_loc_pairs[i].k = p_read_pairs[i].d;
    r_loc_pairs[i].d = q_loc_pairs[i].d;
  }
  r.write(q_npairs, r_loc_pairs); // enter data into r[i], accumulates over MINWEIGHT
  
  delete [] r_loc_pairs;
  delete [] q_loc_pairs;
  delete [] p_read_pairs;
  TAU_FSTOP(CONNECTIVITY_Project);
}

Vector<EdgeExt>* hook_matrix(Matrix<EdgeExt> * A, World* world) {
  int64_t n = A->nrow;

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  auto mst = new Vector<EdgeExt>(n, *world, MIN_EDGE);

  int np = p->wrld->np;

  while (are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];

    // q_i = MINWEIGHT {fmv(a_{ij},p_j) : j in [n]}
    TAU_FSTART(Compute q);
    auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    Bivar_Function<EdgeExt, int, EdgeExt> fmv([](EdgeExt e, int p) {
      return e.parent != p ? EdgeExt(e.src, e.weight, e.dest, p) : EdgeExt();
    });
    fmv.intersect_only = true;
    (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);
    //q->sparsify(); // optional optimization: q grows sparse as nodes have no more edges to new components
    TAU_FSTOP(Compute q);

    // r[p[j]] = q[j] over MINWEIGHT
    TAU_FSTART(Project);
    auto r = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    project(*r, *p, *q);
    TAU_FSTOP(Project);

    // hook only onto larger stars and update p
    TAU_FSTART(Update p);
    (*p)["i"] += Function<EdgeExt, int>([](EdgeExt e){ return e.parent; })((*r)["i"]);
    TAU_FSTOP(Update p);

    // hook only onto larger stars and update mst
    TAU_FSTART(Update mst);
    (*mst)["i"] += Bivar_Function<EdgeExt, int, EdgeExt>([](EdgeExt e, int a){ return e.parent >= a ? e : EdgeExt(); })((*r)["i"], (*p)["i"]);
    TAU_FSTOP(Update mst);

    delete r;
    delete q;

    // aggressive shortcutting
    int sc2 = 1000;
    Vector<int> * pi = new Vector<int>(*p);
    shortcut2(*p, *p, *p, sc2, world, NULL, false);
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
    }
    delete pi;

    //A = PTAP<EdgeExt>(A, p); // optimal optimization

    // update edges parent in A[ij]
    TAU_FSTART(Update A);
    Transform<int, EdgeExt>([](int p, EdgeExt & e){ e.parent = p; })((*p)["i"], (*A)["ij"]);
    TAU_FSTOP(Update A);
  }

  delete p;
  delete p_prev;

  return mst;
}

Vector<EdgeExt>* multilinear_hook(Matrix<wht> * A, World* world, int sc2) {
  int64_t n = A->nrow;

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  auto mst = new Vector<EdgeExt>(n, *world, MIN_EDGE);

  std::function<EdgeExt(int, int, int)> f = [](int x, int a, int y){ // TODO: fix templating for wht
    if (x != y) {
      return EdgeExt(-1, a, -1, x);
    } else {
      return EdgeExt();
    }
  };

  while (are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];

    // q_i = MINWEIGHT {fmv(a_{ij},p_j) : j in [n]}
    auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    Tensor<int> * vec_list[2] = {p, p};
    TAU_FSTART(Update A);
    Multilinear1<int, EdgeExt>(A, vec_list, q, f); // in Raghavendra fork of CTF on multilinear branch
    TAU_FSTOP(Update A);
    //q->sparsify(); // optional optimization: q grows sparse as nodes have no more edges to new components

    // r[p[j]] = q[j] over MINWEIGHT
    TAU_FSTART(Project);
    auto r = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    project(*r, *p, *q);
    TAU_FSTOP(Project);

    // hook only onto larger stars and update p
    TAU_FSTART(Update p);
    (*p)["i"] += Function<EdgeExt, int>([](EdgeExt e){ return e.parent; })((*r)["i"]);
    TAU_FSTOP(Update p);

    // hook only onto larger stars and update mst
    TAU_FSTART(Update mst);
    (*mst)["i"] += Bivar_Function<EdgeExt, int, EdgeExt>([](EdgeExt e, int a){ return e.parent >= a ? e : EdgeExt(); })((*r)["i"], (*p)["i"]);
    TAU_FSTOP(Update mst);

    delete r;
    delete q;

    // aggressive shortcutting
    //int sc2 = 1000;
    TAU_FSTART(aggressive shortcut);
    Vector<int> * pi = new Vector<int>(*p);
    shortcut2(*p, *p, *p, sc2, world, NULL, false);
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
    }
    delete pi;
    TAU_FSTOP(aggressive shortcut);

    //A = PTAP<wht>(A, p); // optimal optimization
  }

  delete p;
  delete p_prev;

  return mst;
}
