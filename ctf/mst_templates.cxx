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

// NOTE: can't use bool as return
inline int64_t are_vectors_different(CTF::Vector<int> & A, CTF::Vector<EdgeExt> & B)
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

/*for (int64_t i=0; i<npairs; i++){
  remote_pairs[i].k = loc_pairs[i].d.parent;
}*/
template <typename T>
void remote_pairs_k(CTF::Pair<T> * remote_pairs, CTF::Pair<EdgeExt> * loc_pairs, int64_t npairs) {
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d.parent;
  }
}

inline void remote_pairs_k(CTF::Pair<int> * remote_pairs, CTF::Pair<int> * loc_pairs, int64_t npairs) {
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d;
  }
}


template <typename T>
CTF::Vector<int> * get_nonleaves(CTF::Vector<T> & p, int64_t npairs, CTF::Pair<T> * updated_loc_pairs){
  CTF::Vector<int> * nonleaves = new CTF::Vector<int>(p.len, *p.wrld, MAX_TIMES_SR);
  //set nonleaves[i] = max_j p[j], i.e. set nonleaves[i] = 1 if i has child, i.e. is nonleaf
  CTF::Pair<int> * updated_nonleaves = new CTF::Pair<int>[npairs];
  for (int64_t i=0; i<npairs; i++){
    updated_nonleaves[i].k = updated_loc_pairs[i].d.parent;
    updated_nonleaves[i].d = 1;
  }
  //FIXME: here and above potential optimization is to avoid duplicate queries to parent
  nonleaves->write(npairs, updated_nonleaves);

  auto p_parents = CTF::Vector<int>(p.len, SP*p.is_sparse, *p.wrld);
  p_parents["i"] = CTF::Function<EdgeExt,int64_t>([](EdgeExt p){ return p.parent; })(p["i"]);

  nonleaves->operator[]("i") = nonleaves->operator[]("i")*p_parents["i"];
  nonleaves->sparsify();

  return nonleaves;
}

template <>
inline CTF::Vector<int> * get_nonleaves<int>(CTF::Vector<int> & p, int64_t npairs, CTF::Pair<int> * updated_loc_pairs){
  return NULL;
}

// p[i] = rec_p[q[i]]
// if create_nonleaves=true, computing non-leaf vertices in parent forest
// NOTE: only works for T = int or EdgeExt
template <typename T, typename S>
void shortcut(CTF::Vector<T> & p, CTF::Vector<S> & q, CTF::Vector<T> & rec_p, CTF::Vector<int> ** nonleaves, bool create_nonleaves)
{
  CTF::Timer t_shortcut("CONNECTIVITY_Shortcut");
  t_shortcut.start();
  int64_t npairs;
  CTF::Pair<S> * loc_pairs;
  if (q.is_sparse){
    //if we have updated only a subset of the vertices
    //q.get_local_pairs(&npairs, &loc_pairs, true);
    q.get_local_pairs(&npairs, &loc_pairs);
  } else {
    //if we have potentially updated all the vertices
    q.get_local_pairs(&npairs, &loc_pairs);
  }
  CTF::Pair<T> * remote_pairs = new CTF::Pair<T>[npairs];
  /*for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d.parent;
  }*/
  remote_pairs_k(remote_pairs, loc_pairs, npairs); // remote_pairs[i].k = loc_pairs[i].d, overloaded for dtypes
  
  CTF::Timer t_shortcut_read("CONNECTIVITY_Shortcut_read");
  t_shortcut_read.start();
  rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
  t_shortcut_read.stop();
 
  rec_p.print(); 
  CTF::Pair<T> * updated_loc_pairs = new CTF::Pair<T>[npairs];
  for (int64_t i=0; i<npairs; i++){
      updated_loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
  }

  delete [] remote_pairs;
  p.write(npairs, updated_loc_pairs); //enter data into p[i] // TODO: causing a seg fault on multiple processes

  //prune out leaves
  if (create_nonleaves && std::is_same<T, EdgeExt>::value){
    *nonleaves = get_nonleaves<T>(p, npairs, updated_loc_pairs);
  }

  
  delete [] loc_pairs;
  delete [] updated_loc_pairs;
  t_shortcut.stop();
}

template <typename dtype>
void max_vector(CTF::Vector<dtype> & result, CTF::Vector<dtype> & A, CTF::Vector<dtype> & B) {
  result["i"] = CTF::Function<dtype,dtype,dtype>([](dtype a, dtype b){return ((a > b) ? a : b);})(A["i"], B["i"]);
}

inline void max_vector(CTF::Vector<int> & result, CTF::Vector<int> & A, CTF::Vector<EdgeExt> & B) {
  result["i"] = CTF::Function<int,EdgeExt,int>([](int a, EdgeExt b){return ((a > b.parent) ? a : b.parent);})(A["i"], B["i"]);
}
