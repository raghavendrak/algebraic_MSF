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
      EdgeExt(INT_MAX, INT_MAX, INT_MAX, 0), 
      [](EdgeExt a, EdgeExt b){ return EdgeExtMin(a, b); }, 
      omee);

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

// NOTE: can't use bool as return
template <typename dtype>
int64_t are_vectors_different(CTF::Vector<dtype> & A, CTF::Vector<dtype> & B)
{
  CTF::Scalar<int64_t> s;
  if (!A.is_sparse && !B.is_sparse){
    s[""] += CTF::Function<dtype,dtype,int64_t>([](dtype a, dtype b){ return a!=b; })(A["i"],B["i"]);
  } else {
    auto C = Vector<dtype>(A.len, SP*A.is_sparse, *A.wrld);
    C["i"] += A["i"];
    ((int64_t)-1)*C["i"] += B["i"];
    s[""] += CTF::Function<dtype,int64_t>([](dtype a){ return (int64_t)(a!=0); })(C["i"]);

  }
  return s.get_val();
}

// r[p[j]] = q[j] over MINWEIGHT
void project(Vector<EdgeExt> & r, Vector<int> & p, Vector<EdgeExt> & q)
{
  Timer t_project("CONNECTIVITY_Project");
  t_project.start();

  int64_t p_npairs;
  Pair<int> * p_loc_pairs;
  if (p.is_sparse){
    //if we have updated only a subset of the vertices
    p.get_local_pairs(&p_npairs, &p_loc_pairs, true);
  } else {
    //if we have potentially updated all the vertices
    p.get_local_pairs(&p_npairs, &p_loc_pairs);
  }

  int64_t q_npairs;
  Pair<EdgeExt> * q_loc_pairs;
  if (q.is_sparse){
    //if we have updated only a subset of the vertices
    q.get_local_pairs(&q_npairs, &q_loc_pairs, true);
  } else {
    //if we have potentially updated all the vertices
    q.get_local_pairs(&q_npairs, &q_loc_pairs);
  }

  Pair<EdgeExt> * remote_pairs = new Pair<EdgeExt>[q_npairs];
  for (int64_t i = 0; i < q_npairs; ++i){
    remote_pairs[i].k = p_loc_pairs[i].d;
    remote_pairs[i].d = q_loc_pairs[i].d;
  }
  r.write(q_npairs, remote_pairs); // enter data into r[i], accumulates over MINWEIGHT
  
  delete [] remote_pairs;
  delete [] q_loc_pairs;
  delete [] p_loc_pairs;
  t_project.stop();
}

// p[i] = rec_p[q[i]]
// if create_nonleaves=true, computing non-leaf vertices in parent forest
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> ** nonleaves, bool create_nonleaves)
{
  Timer t_shortcut("CONNECTIVITY_Shortcut");
  t_shortcut.start();
  int64_t npairs;
  Pair<int> * loc_pairs;
  if (q.is_sparse){
    //if we have updated only a subset of the vertices
    q.get_local_pairs(&npairs, &loc_pairs, true);
  } else {
    //if we have potentially updated all the vertices
    q.get_local_pairs(&npairs, &loc_pairs);
  }
  Pair<int> * remote_pairs = new Pair<int>[npairs];
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d;
  }
  Timer t_shortcut_read("CONNECTIVITY_Shortcut_read");
  //t_shortcut_read.start();
  rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
  t_shortcut_read.stop();
  for (int64_t i=0; i<npairs; i++){
    loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
  }
  delete [] remote_pairs;
  p.write(npairs, loc_pairs); //enter data into p[i]
  
  //prune out leaves
  if (create_nonleaves){
    *nonleaves = new Vector<int>(p.len, *p.wrld, *p.sr);
    //set nonleaves[i] = max_j p[j], i.e. set nonleaves[i] = 1 if i has child, i.e. is nonleaf
    for (int64_t i=0; i<npairs; i++){
      loc_pairs[i].k = loc_pairs[i].d;
      loc_pairs[i].d = 1;
    }
    //FIXME: here and above potential optimization is to avoid duplicate queries to parent
    (*nonleaves)->write(npairs, loc_pairs);
    (*nonleaves)->operator[]("i") = (*nonleaves)->operator[]("i")*p["i"];
    (*nonleaves)->sparsify();
  }
  
  delete [] loc_pairs;
  t_shortcut.stop();
}

Vector<EdgeExt>* hook_matrix(int n, Matrix<EdgeExt> * A, World* world) {
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  auto mst = new Vector<EdgeExt>(n, *world, MIN_EDGE);

  while (are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];

    // q_i = MINWEIGHT {fmv(a_{ij},p_j) : j \in [n]}
    auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    Bivar_Function<EdgeExt,int,EdgeExt> fmv([](EdgeExt e, int p){ 
      return EdgeExt(e.src, e.weight, e.dest, p);
    });
    fmv.intersect_only=true; // fmv should only be applied to nonzeros
    (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);

    // r[p[j]] = q[j] over MINWEIGHT
    auto r = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    project(*r, *p, *q);

    // hook only onto larger stars and update p
    (*p)["i"] = Bivar_Function<EdgeExt, int, int>([](EdgeExt e, int a){ return e.parent > a ? e.parent : a; })((*r)["i"], (*p)["i"]);

    // hook only onto larger stars and update mst
    (*mst)["i"] = Bivar_Function<EdgeExt, int, EdgeExt>([](EdgeExt e, int a){ return e.parent > a ? e : EdgeExt(); })((*r)["i"], (*p)["i"]);

    // Aggressive shortcutting
    Vector<int> * pi = new Vector<int>(*p);
    shortcut(*p, *p, *p, NULL, false);
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut(*p, *p, *p, NULL, false);
    }
    delete pi;
  }
}

/*
Vector<EdgeExt>* hook_matrix(int n, Matrix<EdgeExt> * A, World* world)
{
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto mst = new Vector<int>(n, *world, MAX_TIMES_SR);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  while (are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];
    Timer t_relax("CONNECTIVITY_Relaxation");
    t_relax.start();
    auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    // q_i = (inf, inf, p_i)
    (*q)["i"] = Function<int,EdgeExt>([](int p){ return EdgeExt(INT_MAX, INT_MAX, p, INT_MAX); })((*p)["i"]);
    // fmv(e, p) = (e.src, e.w, p)
    Bivar_Function<EdgeExt,int,EdgeExt> fmv([](EdgeExt e, int p){ return EdgeExt(e.src, e.weight, p, e.parent); });
    // fmv should only be applied to nonzeros
    fmv.intersect_only=true;
    // q_i = minweight_{i} fmv(a_{ij},p_j)}
    (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);

    auto oldp = new Vector<int>(n, *world, MAX_TIMES_SR);
    (*oldp)["i"] = (*p)["i"];
    // Replace this function to check if the dest has an edge emanating from it which is not the same edge this node is using to hook onto
    (*p)["i"] += Function<EdgeExt,int>([](EdgeExt e){ return e.dest; })((*q)["i"]);

    // Aggressive shortcut
    Vector<int> * pi = new Vector<int>(*p);
    shortcut<int, int>(*p, *p, *p, NULL, false);
    p->print();
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut<int, int>(*p, *p, *p, NULL, false);
    }
    delete pi;
    
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
      int rowno = u_loc_pairs[i].k; // row no changed
      int parentno = u_loc_pairs[i].d; // rowno has a new parentonent
      int nrow_read = A->nrow;
      Pair<EdgeExt> *row_read = new Pair<EdgeExt>[nrow_read];
      for(int j = 0; j < nrow_read; j++) {
        row_read[j].k = rowno + j * nrow_read; // get the whole row data
      }
      A->read(nrow_read, row_read);
      for(int j = 0; j < nrow_read; j++) {
        EdgeExt e = row_read[j].d;
        if (e.dest == -1) continue;
        row_read[j].d = EdgeExt(e.src, e.weight, e.dest, parentno); // update with new parentonent number
      }
      A->write(nrow_read, row_read);
    }

    // mst: <src/mst, ref, dest>
    Bivar_Function<EdgeExt,EdgeExt,EdgeExt> mstf([](EdgeExt e, EdgeExt r){ 
        if (r.src == r.weight && e.dest > r.weight && e.src != r.weight) return EdgeExt(e.src, r.weight, r.dest, e.parent);
        else if (r.src == r.weight && e.dest > r.weight && e.src == r.weight) return EdgeExt(e.dest, r.weight, r.dest, e.parent);
        else return EdgeExt(r.src, r.weight, r.dest, r.parent); 
        });
    
    auto mstt = new Vector<EdgeExt>(n, *world, MIN_EDGE);
    (*mstt)["i"] = Function<EdgeExt, EdgeExt>([](EdgeExt e) {return EdgeExt(e.src, e.weight, e.dest, e.parent); })((*mst)["i"]);
    (*mst)["i"] = mstf((*q)["i"], (*mstt)["i"]);

    printf("mst:\n");
    mst->print();
    t_relax.stop();

    delete q;
  }
  return mst;
}
*/
