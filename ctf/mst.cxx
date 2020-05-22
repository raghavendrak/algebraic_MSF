#include "mst.h"

Edge EdgeMin(Edge a, Edge b){
  return a.weight < b.weight ? a : b;
}

void Edge_red(Edge const * a,
              Edge * b,
              int n){
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i=0; i<n; i++){
    b[i] = EdgeMin(a[i], b[i]);
  } 
}

Monoid<Edge> get_minedge_monoid(){
    MPI_Op omee;
    MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        Edge_red((Edge*)a, (Edge*)b, *n);
      },
    1, 
    &omee);

    Monoid<Edge> MIN_EDGE(
      Edge(MAX_WHT, -1), 
      [](Edge a, Edge b){ return EdgeMin(a, b); }, 
      omee);

  return MIN_EDGE; 
}

// r[p[j]] = q[j] over MINWEIGHT
template <typename T>
void project(Vector<T> & r, Vector<int> & p, Vector<T> & q)
{ 
  TAU_FSTART(CONNECTIVITY_Project);
  
  int64_t q_npairs;
  Pair<T> * q_loc_pairs;
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

  Pair<T> * r_loc_pairs = new Pair<T>[q_npairs];
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
template void project<Edge>(Vector<Edge> & r, Vector<int> & p, Vector<Edge> & q);

Vector<Edge>* multilinear_hook(Matrix<wht> *      A, 
                                  World*          world, 
                                  int64_t         sc2, 
                                  MPI_Datatype &  mpi_pkv, 
                                  int64_t         sc3,
                                  int64_t         ptap) {
  assert(!(sc2 > 0 && sc3 > 0)); // TODO: cannot run both shortcut2 and shortcut3
  //A->sr = &MIN_TIMES_SR; // TODO: refactor p to use MIN_TIMES_SR for simplicity
  int64_t n = A->nrow;

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  const static Monoid<Edge> MIN_EDGE = get_minedge_monoid();
  auto mst = new Vector<Edge>(n, *world, MIN_EDGE);

  std::function<Edge(int, int, int)> f = [](int x, int a, int y){ // TODO: fix templating for wht
    if (x != y) {
      return Edge(a, x);
    } else {
      return Edge();
    }
  };

  while (are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];

    if (world->rank == 0) printf("p\n");
    p->print();

    if (world->rank == 0) printf("star_check\n");
    auto temp = star_check(p);
    temp->print();
    delete temp;

    // q_i = MINWEIGHT {fmv(a_{ij},p_j) : j in [n]}
    auto q = new Vector<Edge>(n, p->is_sparse, *world, MIN_EDGE);
    Tensor<int> * vec_list[2] = {p, p};
    TAU_FSTART(Update A);
    Multilinear1<int, Edge>(A, vec_list, q, f); // in Raghavendra fork of CTF on multilinear branch
    TAU_FSTOP(Update A);
    //q->sparsify(); // optional optimization: q grows sparse as nodes have no more edges to new components

    // r[p[j]] = q[j] over MINWEIGHT
    TAU_FSTART(Project);
    auto r = new Vector<Edge>(n, p->is_sparse, *world, MIN_EDGE);
    project(*r, *p, *q);
    TAU_FSTOP(Project);

    // hook only onto larger stars and update p
    TAU_FSTART(Update p);
    (*p)["i"] += Function<Edge, int>([](Edge e){ return e.parent; })((*r)["i"]);
    TAU_FSTOP(Update p);

    // hook only onto larger stars and update mst
    TAU_FSTART(Update mst);
    (*mst)["i"] += Bivar_Function<Edge, int, Edge>([](Edge e, int a){ return e.parent >= a ? e : Edge(); })((*r)["i"], (*p)["i"]);
    TAU_FSTOP(Update mst);

    delete r;
    delete q;

    TAU_FSTART(aggressive shortcut);
    // 256kB: 32768
    if (sc3 > 0) {
      int64_t diff = are_vectors_different(*p, *p_prev);
      if (diff < sc3) {
        shortcut3(*p, *p, *p, *p_prev, mpi_pkv, world);
        continue;
      }
    }

    // aggressive shortcutting
    /*
    Vector<int> * pi = new Vector<int>(*p);
    shortcut2(*p, *p, *p, sc2, world, NULL, false);
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
    }
    delete pi;
    */
    TAU_FSTOP(aggressive shortcut);

    if (ptap > 0) {
      int64_t npairs;
      Pair<int> * loc_pairs;
      p->get_local_pairs(&npairs, &loc_pairs);
      int64_t nloc_roots, nglobal_roots;
      roots_num(npairs, loc_pairs, &nloc_roots, &nglobal_roots, world);
      if (nglobal_roots < ptap) {
        A = PTAP<wht>(A, p);
      }
    }
  }

  delete p;
  delete p_prev;
  mst->print();
  return mst;
}

/*
Vector<EdgeExt>* hook_matrix(Matrix<EdgeExt> * A, World* world) {
  int64_t n = A->nrow;
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();
  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);
  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);
  auto mst = new Vector<EdgeExt>(n, *world, MIN_EDGE);

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
    int64_t sc2 = 1000;
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
*/
