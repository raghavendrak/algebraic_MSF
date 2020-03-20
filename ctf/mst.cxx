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
  t_shortcut_read.start();
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

    // q_i = MINWEIGHT {fmv(a_{ij},p_j) : j in [n]}
    auto q = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    (*q)["i"] = Bivar_Function<EdgeExt, int, EdgeExt>([](EdgeExt e, int p) {
      return e.parent != p ? EdgeExt(e.src, e.weight, e.dest, p) : EdgeExt();                
    })((*A)["ij"], (*p)["j"]);

    // r[p[j]] = q[j] over MINWEIGHT
    auto r = new Vector<EdgeExt>(n, p->is_sparse, *world, MIN_EDGE);
    project(*r, *p, *q);

    // hook only onto larger stars and update p
    (*p)["i"] += Function<EdgeExt, int>([](EdgeExt e){ return e.parent; })((*r)["i"]);

    // hook only onto larger stars and update mst
    (*mst)["i"] += Bivar_Function<EdgeExt, int, EdgeExt>([](EdgeExt e, int a){ return e.parent >= a ? e : EdgeExt(); })((*r)["i"], (*p)["i"]);

    delete r;
    delete q;

    // aggressive shortcutting
    Vector<int> * pi = new Vector<int>(*p);
    shortcut(*p, *p, *p, NULL, false);
    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut(*p, *p, *p, NULL, false);
    }
    delete pi;
    
    // update edges parent in A[ij]
    Transform<int, EdgeExt>([](int p, EdgeExt & e){ e.parent = p; })((*p)["i"], (*A)["ij"]);
  }

  delete p;
  delete p_prev;

  return mst;
}
