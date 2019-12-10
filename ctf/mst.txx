// p[i] = rec_p[q[i]]
// if create_nonleaves=true, computing non-leaf vertices in parent forest
template <typename T>
void shortcut(Vector<T> & p, Vector<EdgeExt> & q, Vector<T> & rec_p, Vector<int> ** nonleaves, bool create_nonleaves)
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
  Pair<T> * remote_pairs = new Pair<T>[npairs];
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d.parent;
  }
  Timer t_shortcut_read("CONNECTIVITY_Shortcut_read");
  t_shortcut_read.start();
  rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
  t_shortcut_read.stop();
  
  Pair<T> * updated_loc_pairs = new Pair<T>[npairs];
  for (int64_t i=0; i<npairs; i++){
      updated_loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
  }
  delete [] remote_pairs;
  //p.write(npairs, updated_loc_pairs); //enter data into p[i] // TODO: no multiplication operation for this algebraic structure

  //prune out leaves
  if (create_nonleaves && std::is_same<T, EdgeExt>::value){
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
