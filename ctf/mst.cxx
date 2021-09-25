#include "mst.h"

Edge EdgeMin(Edge a, Edge b){
  return a.weight < b.weight || (a.weight == b.weight && a.parent > b.parent) ? a : b;
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
  //Timer t_cp("Connectivity_Project");
  //t_cp.start();
  
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

  // Build a map for p[j] to avoid duplicate updates to r[p[j]]
  // can sort: nlogn
  std::unordered_map<int, T> m_pq;
  for (int64_t i = 0; i < q_npairs; i++) {
    typename std::unordered_map<int, T>::iterator it;
    it = m_pq.find(p_read_pairs[i].d);
    if (it != m_pq.end()) {
      // update with minimum weight
      if (q_loc_pairs[i].d.weight < it->second.weight
          || (q_loc_pairs[i].d.weight == it->second.weight && q_loc_pairs[i].d.parent > it->second.parent)) {
        it->second = q_loc_pairs[i].d;
      }
    }
    else {
      if (q_loc_pairs[i].d.weight != MAX_WHT) {
        m_pq.insert({p_read_pairs[i].d, q_loc_pairs[i].d});
      }
    }
  }


  /*
  Pair<T> * r_loc_pairs = new Pair<T>[q_npairs];
  for (int64_t i = 0; i < q_npairs; ++i){
    r_loc_pairs[i].k = p_read_pairs[i].d;
    r_loc_pairs[i].d = q_loc_pairs[i].d;
  }
  r.write(q_npairs, r_loc_pairs); // enter data into r[i], accumulates over MINWEIGHT
  */

  Pair<T> * r_loc_pairs = new Pair<T>[m_pq.size()];
  int64_t ir = 0;
  for (const auto& pq : m_pq) {
    r_loc_pairs[ir].k = pq.first;
    r_loc_pairs[ir++].d = pq.second;
  }
  //Timer t_wr("Write_r_in_project");
  //t_wr.start();
#ifdef TIME_ITERATION
  double stime;
  double etime;
  MPI_Barrier(MPI_COMM_WORLD);
  stime = MPI_Wtime();
#endif
  
  TAU_FSTART(Write_r_in_project);
  r.write(ir, r_loc_pairs);

#ifdef TIME_ITERATION
  MPI_Barrier(MPI_COMM_WORLD);
  etime = MPI_Wtime();
  if (q.wrld->rank == 0) {
    printf("r.write in %1.2lf  ", (etime - stime));
  }
#endif
            
  //t_wr.stop();
  TAU_FSTOP(Write_r_in_project);
 
  delete [] r_loc_pairs;
  delete [] q_loc_pairs;
  delete [] p_read_pairs;
  TAU_FSTOP(CONNECTIVITY_Project);
  //t_cp.stop();
}
template void project<Edge>(Vector<Edge> & r, Vector<int> & p, Vector<Edge> & q);

Vector<Edge>* as_hook(Matrix<Edge> *  A, 
                      World*          world,
                      int64_t         sc2,
                      MPI_Datatype &  mpi_pkv,
                      int64_t         sc3,
                      int64_t         star) {
  assert(!(sc2 > 0 && sc3 > 0)); // TODO: cannot run both shortcut2 and shortcut3

  int64_t n = A->nrow;

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  const static Monoid<Edge> MIN_EDGE = get_minedge_monoid();
  auto mst = new Vector<Edge>(n, *world, MIN_EDGE);

  int niter = 0;
  while (are_vectors_different(*p, *p_prev)) {
    ++niter;
    (*p_prev)["i"] = (*p)["i"];

    // if (world->rank == 0)
    //   printf("p:\n");
    // p->print();

    // q_i = MINWEIGHT {f(a_{ij},p_j) : i belongs to a star}
    TAU_FSTART(Compute q);
    auto q = new Vector<Edge>(n, p->is_sparse, *world, MIN_EDGE);
    Bivar_Function<Edge, int, Edge> fmv([](Edge e, int p) {
      return e.parent != p ? Edge(e.weight, p) : Edge();
    });
    fmv.intersect_only = true;
    (*q)["i"] = fmv((*A)["ij"], (*p)["j"]);
    Vector<int> * star_mask = star_check(p);
    // if (world->rank == 0)
    //   printf("star_mask:\n");
    // star_mask->print();
    Transform<int, Edge>([](int s, Edge & e){ if (!s) e = Edge(); })((*star_mask)["i"], (*q)["i"]); // output filter for vertices belonging to a star
    delete star_mask;
    TAU_FSTOP(Compute q);

    // if (world->rank == 0)
    //   printf("q:\n");
    // q->print();

    // r[p[j]] = q[j] over MINWEIGHT
    TAU_FSTART(Project);
    //Timer t_p("Project");
    //t_p.start();
    auto r = new Vector<Edge>(n, p->is_sparse, *world, MIN_EDGE);
    project(*r, *p, *q);
    TAU_FSTOP(Project);
    //t_p.stop();

    // hook only onto larger stars and update p
    TAU_FSTART(Update p);
    //Timer t_up("Update p");
    //t_up.start();
    (*p)["i"] += Function<Edge, int>([](Edge e){ return e.parent; })((*r)["i"]);
    TAU_FSTOP(Update p);
    //t_up.stop();

    // hook only onto larger stars and update mst
    TAU_FSTART(Update mst);
    //Timer t_mst("Update mst");
    //t_mst.start();
    (*mst)["i"] += Bivar_Function<Edge, int, Edge>([](Edge e, int a){ return e.parent >= a ? e : Edge(); })((*r)["i"], (*p)["i"]);
    TAU_FSTOP(Update mst);
    //t_mst.stop();
 
    delete r;
    delete q;

    // if (sc3 > 0) {
    //   shortcut3(*p, *p, *p, *p_prev, mpi_pkv, world);
    // } else {
    //   shortcut2(*p, *p, *p, 0, world, NULL, false);
    // }
    
    // 256kB: 32768
    bool is_shortcutted = false;
    int64_t st2 = 0;
#ifdef TIME_ITERATION
    double stimes;
    double etimes;
    MPI_Barrier(MPI_COMM_WORLD);
    stimes = MPI_Wtime();
#endif
    if (sc3 > 0) {
      TAU_FSTART(sc3 aggressive shortcut);
      //Timer t_sc3("sc3_aggressive_shortcut");
      //t_sc3.start();
      int64_t diff = are_vectors_different(*p, *p_prev);
#ifdef TIME_ST_ITERATION
      if (world->rank == 0) {
        printf("diff p p_prev: %lld  ", diff);
      }
#endif
      if (diff < sc3) {
        shortcut3(*p, *p, *p, *p_prev, mpi_pkv, world);
        is_shortcutted = true;
      }
      TAU_FSTOP(sc3 aggressive shortcut);
      //t_sc3.stop();
    } 
    if (star && !is_shortcutted) { // shortcut once
      TAU_FSTART(single shortcut);
      //Timer t_ss("single shortcut2");
      //t_ss.start();
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
      st2++;
      is_shortcutted = true;
      //t_ss.stop();
      TAU_FSTART(single shortcut);
    }
    if (!is_shortcutted) {
      TAU_FSTART(sc2 aggressive shortcut);
      //Timer t_sc2("sc2_aggressive_shortcut");
      //t_sc2.start();
      Vector<int> * pi = new Vector<int>(*p);
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
      st2++;
      while (are_vectors_different(*pi, *p)){
        delete pi;
        pi = new Vector<int>(*p);
        shortcut2(*p, *p, *p, sc2, world, NULL, false);
        st2++;
      }
      delete pi;
      is_shortcutted = true;
      //t_sc2.stop();
      TAU_FSTOP(sc2 aggressive shortcut);
    }
    assert(is_shortcutted);

    // update edges parent in A
    TAU_FSTART(Update A);
    Transform<int, Edge>([](int p, Edge & e){ e.parent = p; })((*p)["i"], (*A)["ij"]);
    TAU_FSTOP(Update A);
  }
  if (world->rank == 0) printf("number of iterations: %d\n", niter);

  delete p;
  delete p_prev;
  return mst;
}

Vector<Edge>* multilinear_hook(Matrix<wht> *      A, 
                                  World*          world, 
                                  int64_t         sc2, 
                                  MPI_Datatype &  mpi_pkv, 
                                  int64_t         sc3,
                                  int64_t         ptap,
                                  int64_t         star,
                                  int64_t         convgf) {
  //assert(!(sc2 > 0 && sc3 > 0)); // TODO: cannot run both shortcut2 and shortcut3

  int64_t n = A->nrow;

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  const static Monoid<Edge> MIN_EDGE = get_minedge_monoid();

  Pair<Edge> mst_prs[n];
  int64_t mst_nprs = 0;

  std::function<Edge(int, wht, int)> f = [](int x, wht a, int y){
    if (x != y) {
      return Edge(a, x);
    } else {
      return Edge();
    }
  };

  bool first_ptap = true; // only perform PTAP once

  Vector<int> * gf;
  Vector<int> * gf_prev;
  if (convgf) {
    gf = new Vector<int>(n, *world, MAX_TIMES_SR);
    init_pvector(gf);
    gf_prev = new Vector<int>(n, *world, MAX_TIMES_SR);
  }

  int niter = 0;
#ifdef TIME_ITERATION
  double stime;
  double etime;
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  while (convgf ? are_vectors_different(*gf, *gf_prev) : are_vectors_different(*p, *p_prev)) { // for most real world graphs, gf converges one iteration before p (see FastSV)

#ifdef TIME_ITERATION
    stime = MPI_Wtime();
#endif

    ++niter;
    if (convgf) {
      (*gf_prev)["i"] = (*gf)["i"];
    }
    if (!convgf || sc3 > 0) {
      (*p_prev)["i"] = (*p)["i"];
    }

    // if (world->rank == 0)
    //   printf("p at beginning of iteration:\n");
    // p->print();

    // q_i = MINWEIGHT {f(p_i,a_{ij},p_j) : j in [n]}
    auto q = new Vector<Edge>(n, p->is_sparse, *world, MIN_EDGE);
    Tensor<int> * vec_list[2] = {p, p};
    Vector<int> * p_star;
    if (star) { // optional optimization: relaxes star requirement, allows stars to hook onto trees
      Vector<int> * star_mask = convgf ? star_check(p, gf) : star_check(p);
      // if (world->rank == 0)
      //   printf("star_mask:\n");
      // star_mask->print();
      p_star = new Vector<int>(n, p->is_sparse, *world); 
      (*p_star)["i"] = Function<int, int, int>([](int parent, int s){ return s ? parent : 0; })((*p)["i"], (*star_mask)["i"]); // 0 is addid for p 
      vec_list[0] = p;
      vec_list[1] = p_star;
      // vec_list[0] = p_star;
      // vec_list[1] = p;
      delete star_mask;
    }
#ifdef TIME_ITERATION
    double stimea;
    double etimea;
    MPI_Barrier(MPI_COMM_WORLD);
    stimea = MPI_Wtime();
#endif
    TAU_FSTART(Update A);
    //Timer t_ua("Update A");
    //t_ua.start();
    Multilinear<wht, int, Edge>(A, vec_list, q, f); // in Tim fork of CTF on multilinear branch
    // Multilinear<int, Edge>(A, vec_list, q, f); // in Raghavendra fork of CTF on multilinear branch
    
    // if (world->rank == 0)
    //   printf("q:\n");
    // q->print();

#ifdef TIME_ITERATION
    MPI_Barrier(MPI_COMM_WORLD);
    etimea = MPI_Wtime();
    if (world->rank == 0) {
      printf("multilinear in %1.2lf  ", (etimea - stimea));
    }
#endif

    if (star) delete p_star;
    TAU_FSTOP(Update A);
    //t_ua.stop();
    //q->sparsify(); // optional optimization: q grows sparse as nodes have no more edges to new components
#ifdef TIME_ITERATION
    double stimep;
    double etimep;
    MPI_Barrier(MPI_COMM_WORLD);
    stimep = MPI_Wtime();
#endif
    // r[p[j]] = q[j] over MINWEIGHT
    TAU_FSTART(Project);
    //Timer t_p("Project");
    //t_p.start();
    auto r = new Vector<Edge>(n, p->is_sparse, *world, MIN_EDGE);
    project(*r, *p, *q);

#ifdef TIME_ITERATION
    MPI_Barrier(MPI_COMM_WORLD);
    etimep = MPI_Wtime();
    if (world->rank == 0) {
      printf("project in %1.2lf  ", (etimep - stimep));
    }
#endif
    TAU_FSTOP(Project);
    //t_p.stop();

    // (*p)["i"] = Function<Edge, int>([](Edge e){ return e.parent; })((*r)["i"]);
    Transform<Edge,int>([](Edge e, int & y){ if (e.parent != -1) y = e.parent; })((*r)["i"], (*p)["i"]);
    
    Vector<int> gf2(n, *world);
    // start reading grandparent
  //Timer t_us("Unoptimized_shortcut");
  //t_us.start();
  int64_t npairs;
  Pair<int> * loc_pairs;
  if (p->is_sparse){
    //if we have updated only a subset of the vertices
    p->get_local_pairs(&npairs, &loc_pairs, true);
  } else {
    //if we have potentially updated all the vertices
    p->get_local_pairs(&npairs, &loc_pairs);
  }

  // read might be from the local node
  // duplicate requests - can squash
  // <q.d, q.k>
  std::unordered_map<int64_t, int64_t> q_data;
  for(int i = 0; i < npairs; i++) {
    std::unordered_map<int64_t, int64_t>::iterator it;
    
    // is data available in my local node?
    it = q_data.find(loc_pairs[i].k);
    if (it != q_data.end()) {
      it->second = loc_pairs[i].d;
    }
    
    // has this been queried already?
    it = q_data.find(loc_pairs[i].d);
    if (it == q_data.end()) {
      q_data.insert({loc_pairs[i].d, -1});
    }
  }

  /*
  Pair<int> * remote_pairs = new Pair<int>[npairs];
  for (int64_t i=0; i<npairs; i++){
    remote_pairs[i].k = loc_pairs[i].d;
  }
  */
  
  // the size will be lesser than q_data.size()
  Pair<int> * remote_pairs = new Pair<int>[q_data.size()];
  int64_t tot_recp_reqs = 0;
  for (const auto& qd : q_data) {
    if (qd.second == -1) {
      remote_pairs[tot_recp_reqs++].k = qd.first;
    }
  }

  TAU_FSTART(Unoptimized_shortcut_rread);
  //Timer t_usr("Unoptimized_shortcut_rread");
  //t_usr.start();
  //rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
  p->read(tot_recp_reqs, remote_pairs); //obtains rec_p[q[i]]

  int64_t ir = 0;
  for (auto& qd : q_data) {
    if (qd.second == -1) {
      qd.second = remote_pairs[ir++].d;
    }
  }

  for (int64_t i = 0; i < npairs; i++){
    // loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
    std::unordered_map<int64_t, int64_t>::iterator it;
    
    it = q_data.find(loc_pairs[i].d);
    if (it != q_data.end()) {
      assert(it->second != -1);
      loc_pairs[i].d = it->second;
    }
    else {
      assert(0);
    }

  }
  delete [] remote_pairs;
  gf2.write(npairs, loc_pairs); //enter data into p[i]
  delete [] loc_pairs;
    
    // end reading granparent

    Vector<int> * tie_break = new Vector<int>(n, p->is_sparse, *world);
    Vector<int> * id = new Vector<int>(n, p->is_sparse, *world);
    init_pvector(id);
    (*tie_break)["i"] += Function<int, int, int>([](int x, int id){ return x < id; })((*p)["i"], (*id)["i"]);
    (*tie_break)["i"] += Function<int, int, int>([](int x, int id){ return x == id; })(gf2["i"], (*id)["i"]);

    // if (world->rank == 0) printf("r\n");
    // r->print();

    // if (world->rank == 0) printf("tie break\n");
    // tie_break->print();

    Transform<int,int,int>([](int tie, int id, int & x){ if (tie == 2) x = id; })((*tie_break)["i"], (*id)["i"], (*p)["i"]);
    // Transform<int,Edge,Edge>([](int tie, Edge x, Edge & y){ if (tie != 2) y = x; })((*tie_break)["i"], (*r)["i"], (*mst)["i"]);

    int64_t r_nprs;
    Pair<Edge> * r_prs;
    r->get_local_pairs(&r_nprs, &r_prs);
    int64_t tie_nprs;
    Pair<int> * tie_prs;
    tie_break->get_local_pairs(&tie_nprs, &tie_prs);
    assert(r_nprs == tie_nprs);
    for(int64_t i = 0; i < r_nprs; ++i) {
      assert(r_prs[i].k == tie_prs[i].k); // r_prs and tie_nprs should be lined up correctly
      if (tie_prs[i].d != 2 && r_prs[i].d.parent != -1) {
        mst_prs[mst_nprs].k = -1;
        mst_prs[mst_nprs].d = r_prs[i].d;
        ++mst_nprs;
      }
    }


    // (*p)["i"] = Function<Edge, int, int>([](int x, int tie){ return tie == 2 ? -1 : x; })((*p)["i"], (*tie_break)["i"]);
    // (*p)["i"] = Function<int, int, int>([](int id, int x){ return x == -1 ? id : x; })((*id)["i"], (*p)["i"]);

    // (*p)["i"] = Function<Edge, int, int>([](Edge e, int tie){ return tie == 2 ? -1 : e.parent; })((*r)["i"], (*tie_break)["i"]);
    // (*p)["i"] = Function<int, int, int>([](int id, int x){ return x == -1 ? id : x; })((*id)["i"], (*p)["i"]);
    // (*mst)["i"] += Bivar_Function<Edge, int, Edge>([](Edge e, int a){ return e.parent >= a ? e : Edge(); })((*r)["i"], (*p)["i"]);
    // (*mst)["i"] += Function<Edge, int, Edge>([](Edge e, int tie){ return tie == 2 ? Edge() : e; })((*r)["i"], (*tie_break)["i"]);

    // if (world->rank == 0) printf("p after\n");
    // p->print();


    // // hook only onto larger stars and update p
    // TAU_FSTART(Update p);
    // //Timer t_up("Update p");
    // //t_up.start();
    // (*p)["i"] += Function<Edge, int>([](Edge e){ return e.parent; })((*r)["i"]);
    // TAU_FSTOP(Update p);
    // //t_up.stop();

    // // hook only onto larger stars and update mst
    // TAU_FSTART(Update mst);
    // //Timer t_mst("Update mst");
    // //t_mst.start();
    // (*mst)["i"] += Bivar_Function<Edge, int, Edge>([](Edge e, int a){ return e.parent >= a ? e : Edge(); })((*r)["i"], (*p)["i"]);
    // TAU_FSTOP(Update mst);
    // //t_mst.stop();

    delete r;
    delete q;

    // 256kB: 32768
    bool is_shortcutted = false;
    int64_t st2 = 0;
#ifdef TIME_ITERATION
    double stimes;
    double etimes;
    MPI_Barrier(MPI_COMM_WORLD);
    stimes = MPI_Wtime();
#endif
    if (sc3 > 0) {
      TAU_FSTART(sc3 aggressive shortcut);
      //Timer t_sc3("sc3_aggressive_shortcut");
      //t_sc3.start();
      int64_t diff = are_vectors_different(*p, *p_prev);
#ifdef TIME_ST_ITERATION
      if (world->rank == 0) {
        printf("diff p p_prev: %lld  ", diff);
      }
#endif
      if (diff < sc3) {
        shortcut3(*p, *p, *p, *p_prev, mpi_pkv, world);
        is_shortcutted = true;
      }
      TAU_FSTOP(sc3 aggressive shortcut);
      //t_sc3.stop();
    } 
    if (star && !is_shortcutted) { // shortcut once
      TAU_FSTART(single shortcut);
      //Timer t_ss("single shortcut2");
      //t_ss.start();
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
      st2++;
      is_shortcutted = true;
      //t_ss.stop();
      TAU_FSTART(single shortcut);
    }
    if (!is_shortcutted) {
      TAU_FSTART(sc2 aggressive shortcut);
      //Timer t_sc2("sc2_aggressive_shortcut");
      //t_sc2.start();
      Vector<int> * pi = new Vector<int>(*p);
      shortcut2(*p, *p, *p, sc2, world, NULL, false);
      st2++;
      while (are_vectors_different(*pi, *p)){
        // printf("stuck\n");
        delete pi;
        pi = new Vector<int>(*p);
        shortcut2(*p, *p, *p, sc2, world, NULL, false);
        st2++;
      }
      delete pi;
      is_shortcutted = true;
      //t_sc2.stop();
      TAU_FSTOP(sc2 aggressive shortcut);
    }
    assert(is_shortcutted);

    if (ptap > 0 && first_ptap) {
      // first_ptap = false;
      int64_t npairs;
      Pair<int> * loc_pairs;
      p->get_local_pairs(&npairs, &loc_pairs);
      int64_t nloc_roots, nglobal_roots;
      roots_num(npairs, loc_pairs, &nloc_roots, &nglobal_roots, world);
      if (nglobal_roots < ptap) {
        A = PTAP<wht>(A, p);
      }
    }

    if (convgf) {
      shortcut2(*gf, *p, *p, sc2, world, NULL, false); // gf = p[p[i]]
      st2++;
    }

#ifdef TIME_ITERATION
    MPI_Barrier(MPI_COMM_WORLD);
    etimes = MPI_Wtime();
    if (world->rank == 0) {
      printf("shortcut in %1.2lf  st2: %lld  ", (etimes - stimes), st2);
    }
    etime = MPI_Wtime();
    if (world->rank == 0) {
      printf("iteration %d in %1.2lf\n\n\n", niter, (etime - stime));
    }
#endif
  }
  if (world->rank == 0) printf("number of iterations: %d\n", niter);

  auto mst = new Vector<Edge>(n, *world, MIN_EDGE);
  int64_t off;
  MPI_Exscan(&mst_nprs, &off, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  if (world->rank == 0)
    off = 0;
  for (int64_t i = 0; i < mst_nprs; ++i) {
    mst_prs[i].k = i + off;
  }
  mst->write(mst_nprs, mst_prs);

  // if (world->rank == 0) printf("mst:\n");
  // mst->print();

  if (convgf) {
    delete gf;
    delete gf_prev;
  }

  delete p;
  delete p_prev;
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

