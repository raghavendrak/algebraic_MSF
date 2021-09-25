#include "test.h"
#include "mst.h"
#include <ctime>

Matrix<Edge> * wht_to_edge(Matrix<wht> * A) {
  int n = A->nrow;
  const static Monoid<Edge> MIN_EDGE = get_minedge_monoid();
  Matrix<Edge> * B = new Matrix<Edge>(n, n, A->symm|(A->is_sparse*SP), *(A->wrld), MIN_EDGE);
  int64_t nprs;
  Pair<wht> * prs;
  A->get_local_pairs(&nprs, &prs, true);
  Pair<Edge> B_prs[nprs];
  for (int64_t i = 0; i < nprs; ++i) {
    B_prs[i].k = prs[i].k;
    B_prs[i].d = Edge(prs[i].d, prs[i].k % n);
  }
  B->write(nprs, B_prs);
  return B;
}

void run_mst(Matrix<wht>* A, int64_t matSize, World *w, int batch, int64_t sc2, int run_serial, int run_as, int run_multilinear, int64_t sc3, int64_t ptap, int64_t star, int64_t convgf)
{
  TAU_FSTART(run_mst);
  MPI_Datatype mpi_pkv;
  struct parentkv pkv;
  MPI_Datatype type[2] = {MPI_LONG_LONG, MPI_LONG_LONG};
  MPI_Aint disp[2];
  int blocklen[2] = {1, 1};
  disp[0] = (size_t)&(pkv.key) - (size_t)&pkv;
  disp[1] = (size_t)&(pkv.value) - (size_t)&pkv;
  MPI_Type_create_struct(2, blocklen, disp, type, &mpi_pkv);
  MPI_Type_commit(&mpi_pkv);

  Function<Edge,wht> sum_weights([](Edge a){ return a.weight != MAX_WHT ? a.weight : 0; }); // workaround, sometimes it returns wrong result without checking if != INT_MAX

  double stime;
  double etime;
  matSize = A->nrow; // Quick fix to avoid change in i/p matrix size after preprocessing
  if (run_as) {
    Matrix<Edge> * B = wht_to_edge(A);
    TAU_FSTART(as_hook);
    //Timer_epoch tmh("multilinear_hook");
    //tmh.begin();
    stime = MPI_Wtime();
    Vector<Edge> * as_mst = as_hook(B, w, sc2, mpi_pkv, sc3, star);
    etime = MPI_Wtime();
    TAU_FSTOP(as_hook);
    delete B;
    //tmh.end();
    if (w->rank == 0) {
      printf("as mst done in %1.2lf\n", (etime - stime));
    }

    Scalar<wht> s(*w);
    s[""] = sum_weights((*as_mst)["i"]);
    int64_t sweight = s.get_val();
    if (w->rank == 0)
    	printf("weight of Awerbuch-Shiloach mst: %ld\n", sweight);
    delete as_mst;
  }
  else if (run_multilinear) {
    TAU_FSTART(multilinear_hook);
    //Timer_epoch tmh("multilinear_hook");
    //tmh.begin();
    stime = MPI_Wtime();
    Vector<Edge> * mult_mst = multilinear_hook(A, w, sc2, mpi_pkv, sc3, ptap, star, convgf);
    etime = MPI_Wtime();
    TAU_FSTOP(multilinear_hook);
    //tmh.end();
    if (w->rank == 0) {
      printf("multilinear mst done in %1.2lf\n", (etime - stime));
    }

    Scalar<wht> s(*w);
    s[""] = sum_weights((*mult_mst)["i"]);
    int64_t sweight = s.get_val();
    if (w->rank == 0)
    	printf("weight of multilinear mst: %ld\n", sweight);
    delete mult_mst;
  }
  /*
  if (run_serial) {
    Vector<Edge> * serial = serial_mst(A, w);

    Scalar<wht> s(*w);
    s[""] = sum_weights((*serial)["i"]);
    int64_t sweight = s.get_val();
    if (w->rank == 0)
    	printf("weight of serial mst: %ld\n", sweight);
    delete mult_mst;
  }
  */
  TAU_FSTOP(run_mst);
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
  char *gfile = NULL;
  int64_t n;
  int scale;
  int ef;
  int prep;
  int batch;
  int64_t sc2;
  int64_t sc3;
  int run_serial;
  int run_as;
  int critter_mode=0;
  int ptap;
  int star;
  int convgf;
  double sp;
  char *write = NULL;
  int snap_dataset = 0;
  int mm = 0, mm_w = 0;

  int k;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World w(argc, argv);
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
    if (getCmdOption(input_str, input_str+in_num, "-sp")){
      sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
      if (sp < 0.) sp = 0.2;
    } else sp = 0.;
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
    if (getCmdOption(input_str, input_str+in_num, "-shortcut2")){
      sc2 = atoll(getCmdOption(input_str, input_str+in_num, "-shortcut2"));
      if (sc2 < 0) sc2 = 0;
    } else sc2 = 0;
    if (getCmdOption(input_str, input_str+in_num, "-serial")){
      run_serial = atoi(getCmdOption(input_str, input_str+in_num, "-serial"));
      if (run_serial < 0) run_serial = 0;
    } else run_serial = 0;
    if (getCmdOption(input_str, input_str+in_num, "-as")){
      run_as = atoi(getCmdOption(input_str, input_str+in_num, "-as"));
      if (run_as < 0) run_as = 0;
    } else run_as = 0;
    if (getCmdOption(input_str, input_str+in_num, "-shortcut3")){
      sc3 = atoll(getCmdOption(input_str, input_str+in_num, "-shortcut3"));
      if (sc3 < 0) sc3 = 0;
    } else sc3 = 0;
    if (getCmdOption(input_str, input_str+in_num, "-critter_mode")){
      critter_mode = atoi(getCmdOption(input_str, input_str+in_num, "-critter_mode"));
      if (critter_mode < 0) critter_mode = 0;
    } else critter_mode = 0;
    if (getCmdOption(input_str, input_str+in_num, "-ptap")){
      ptap = atoll(getCmdOption(input_str, input_str+in_num, "-ptap"));
      if (ptap < 0) ptap = 0;
    } else ptap = 0;
    if (getCmdOption(input_str, input_str+in_num, "-star")){
      star = atoll(getCmdOption(input_str, input_str+in_num, "-star"));
      if (star < 0) star = 0;
    } else star = 0;
    if (getCmdOption(input_str, input_str+in_num, "-convgf")){
      convgf = atoll(getCmdOption(input_str, input_str+in_num, "-convgf"));
      if (convgf < 0) convgf = 0;
    } else convgf = 0;
    if (getCmdOption(input_str, input_str+in_num, "-wf")){
      write = getCmdOption(input_str, input_str+in_num, "-wf");
    } else write = NULL;
    if (getCmdOption(input_str, input_str+in_num, "-snap")){
      snap_dataset = atoll(getCmdOption(input_str, input_str+in_num, "-snap"));
      if (snap_dataset < 0) snap_dataset = 0;
    } else snap_dataset = 0;
    if (getCmdOption(input_str, input_str+in_num, "-mm")){
      mm = atoll(getCmdOption(input_str, input_str+in_num, "-mm"));
      if (mm < 0) mm = 0;
    } else mm = 0;
    if (getCmdOption(input_str, input_str+in_num, "-mm_w")){
      mm_w = atoll(getCmdOption(input_str, input_str+in_num, "-mm_w"));
      if (mm_w < 0) mm_w = 0;
    } else mm_w = 0;

    if (gfile != NULL){
      int64_t n_nnz = 0;
      if (w.rank == 0)
      printf("Reading real graph n = %lld\n", n);
      Matrix<wht> A;
      if (snap_dataset) {
        A = read_matrix_snap(w, n, gfile, prep, &n_nnz);
      }
      else if (mm) {
        A = read_matrix_market(w, n, gfile, prep, &n_nnz, mm_w);
      }
      else {
        A = read_matrix(w, n, gfile, prep, &n_nnz);
      }
      if (write)
        A.write_sparse_to_file(write, true);
      int64_t matSize = A.nrow; 
#ifdef CRITTER
      critter::start(critter_mode);
#endif
      run_mst(&A, matSize, &w, batch, sc2, run_serial, run_as, 1, sc3, ptap, star, convgf);
#ifdef CRITTER
      critter::stop(critter_mode);
#endif
    }
    else if (k != -1) {
      int64_t matSize = pow(3, k);
      auto B = generate_kronecker(&w, k);
      if (write)
        B->write_sparse_to_file(write, true);

      if (w.rank == 0) {
        printf("Running connectivity on Kronecker graph K: %d matSize: %ld\n", k, matSize);
      }
      run_mst(B, matSize, &w, batch, sc2, run_serial, run_as, 1, sc3, ptap, star, convgf);
      delete B;
    }
    else if (scale > 0 && ef > 0){
      int64_t n_nnz = 0;
      myseed = SEED;
      if (w.rank == 0)
        printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
      Matrix<wht> A = gen_rmat_matrix(w, scale, ef, myseed, prep, &n_nnz, max_ewht);
      if (write)
        A.write_sparse_to_file(write, true);
      int64_t matSize = A.nrow; 
      run_mst(&A, matSize, &w, batch, sc2, run_serial, run_as, 1, sc3, ptap, star, convgf);
    }
    else if (sp != 0.) {
      int64_t n_nnz = 0;
      if (w.rank == 0)
        printf("uniform matrix n: %lld sparsity: %lf\n", n, sp);
      Matrix<wht> A = gen_uniform_matrix(w, n, prep, &n_nnz, sp, max_ewht);
      if (write)
        A.write_sparse_to_file(write, true);
      int64_t matSize = A.nrow; 
      run_mst(&A, matSize, &w, batch, sc2, run_serial, run_as, 1, sc3, ptap, star, convgf);
    }
    else {
      if (w.rank == 0) {
        printf("No graph specified\n");
      }
    }
#ifdef CRITTER
    std::vector<std::string> symbols = {
      "init_tensor","set_zero_tsr","slice","write_pairs","sparsify","sparsify_dense","read_local_pairs",
      "read_all_pairs","redistribute_for_sum","map_tensor_pair","map_sum_indices","check_sum_mapping","sum_postprocessing","post_sum_func_barrier","sum_func","activate_topo",
      "pre_sum_func_barrier","zero_sum_padding","map_fold","sum_tensors_map","sum_tensors","sum_preprocessing","redistribute_for_sum_home","spA_spB_seq_sum_pre","spA_spB_seq_sum",
      "spA_dnB_seq_sum","sym_seq_sum_cust","sym_seq_sum_inr","sym_seq_sum_ref","compute_syoffs","sum_virt","spsum_pin","spsum_permute",
      "Local_shortcut3","Local_shortcut3_write","tspsum_map","spsum_virt",
      "is_last_col_zero","CTF","sp_scal_diag","scal_diag",
      "zero_padding","zero_padding_nonsym","depad_tsr_move","depad_tsr_cnt","depad_tsr","pad_key","unpack_virt_buf","pack_virt_buf",
      "calc_cnt_displs","cyclic_reshuffle","order_globally","cyclic_pup_move","cyclic_pup_bucket","barrier_after_dgtog_reshuffle","redist_debucket","redist_fence","COMM_RESHUFFLE",
      "redist_bucket","dgtog_reshuffle","nosym_transpose_thr","nosym_transpose","check_key_ranges","wr_pairs_layout","readwrite","bucket_by_virt_sort","bucket_by_virt_move",
      "bucket_by_virt_assemble_offsets","bucket_by_virt_omp_cnt","bucket_by_virt","bucket_by_pe_move","bucket_by_pe_count","spsfy_tsr","assign_keys",
      "permute_keys","push_slice","precompute_offsets","calc_drv_displs","block_reshuffle","compute_bucket_offsets","padded_reshuffle","unpack_virt_buf",
      "pack_virt_buf","calc_cnt_displs","cyclic_reshuffle","cyclic_pup_move","cyclic_pup_bucket","sp_scl","redistribute_for_scale_home","redistribute_for_scale","scaling",
      "sym_seq_sum_cust","sym_seq_sum_ref","strp_tsr","scl_virt","symmetrize","desymmetrize","spctr_pin_keys","spctr_virt",
      "spA_dnB_dnC_seq","spctr_2d_general","spctr_2d_general_barrier","redistribute_for_ctr_home","unfold_contraction_output",
      "post_ctr_func_barrier","ctr_func","pre_ctr_func_barrier","map_fold","pre_fold_barrier","pre_map_barrier","prescale_operands","contract","construct_contraction",
      "redistribute_for_contraction","ctr_sig_map_insert","get_best_exh_map","get_best_sel_map","ctr_sig_map_find","all_select_ctr_map","evaluate_mappings","detail_estimate_mem_and_time","get_num_map_vars",
      "offload_axpy","spctr_offload","offload_scale","spctr_offload","offload_scale","ctr_offload","spctr_replicate","spA_dnB_dnC_seq_ctr","ctr_virt",
      "gemm","sym_seq_ctr_inner","sym_seq_ctr_cust","sym_seq_ctr_ref","compute_syoffs","ctr_2d_general",
      "sparse_transpose","zero_out_sparse_diagonal","scale_diagonals","zero_out_padding","gen_graph","CONNECTIVITY_PTAP","Unoptimized_shortcut_Opruneleaves","Unoptimized_shortcut_Owrite","Unoptimized_shortcut_Orread",
      "Optimized_shortcut_write","Optimized_shortcut_read","Optimized_shortcut","Unoptimized_shortcut_pruneleaves","Unoptimized_shortcut_write","Unoptimized_shortcut_rread","Unoptimized_shortcut","CONNECTIVITY_Shortcut_read",
      "CONNECTIVITY_Shortcut","aggressive shortcut","Update mst","Update p","Project","Update A","Update mst","Compute q","CONNECTIVITY_Project",
      "hook_matrix","multilinear_hook","run_mst","super_vertex","hook_matrix","CONNECTIVITY_Relaxation","CONNECTIVITY_Relaxation"
    };
    critter::init(symbols);

#endif 
  }
  MPI_Finalize();
  return 0;
}

// deprecated //
// does not use path compression
/*
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
Vector<Edge> * serial_mst(Matrix<Edge> * A, World * world) {
  const static Monoid<Edge> MIN_EDGE = get_minedge_monoid();

  int64_t npair;
  Pair<Edge> * pairs;
  A->get_all_pairs(&npair, &pairs, true);

  Edge edges[npair];
  for (int64_t i = 0; i < npair; ++i) {
    edges[i] = Edge(pairs[i].d.src, pairs[i].d.weight, pairs[i].d.dest, pairs[i].d.parent);
  }

  std::sort(edges, edges + npair, [](const Edge & lhs, const Edge & rhs) { return lhs.weight < rhs.weight; });

  int64_t p[A->nrow];
  for (int64_t i = 0; i < A->nrow; ++i) {
    p[i] = i;
  }

  int64_t mst_npair = A->nrow - 1;
  Pair<Edge> * mst_pairs = new Pair<Edge>[mst_npair];
  int64_t j = 0;
  for (int64_t i = 0; i < npair; ++i) {
    // find(p, edges[i].src) != find(p, edges[i].dest);
    if (find(p, edges[i].src) != find(p, edges[i].dest)) {
      mst_pairs[j].k = j;
      mst_pairs[j].d = edges[i];
      ++j;
      union1(p, edges[i].src, edges[i].dest);
    }
  }

  Vector<Edge> * mst = new Vector<Edge>(A->nrow, *world, MIN_EDGE);
  mst->write(mst_npair, mst_pairs);

  delete [] mst_pairs;
  delete [] pairs;

  return mst;
}
*/

/*
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

Vector<Edge> * hooking(int64_t A_npairs, Pair<Edge> * A_loc_pairs, Vector<int> * p, Vector<bool> * star) {
  const static Monoid<Edge> MIN_EDGE = get_minedge_monoid();

  auto r = new Vector<Edge>(p->len, p->is_sparse, *(p->wrld), MIN_EDGE);

  Pair<int> * src_loc_pairs = new Pair<int>[A_npairs];
  for (int64_t i = 0; i < A_npairs; ++i) {
    src_loc_pairs[i].k = A_loc_pairs[i].d.src;
  }
  p->read(A_npairs, src_loc_pairs);

  //Pair<int> * src_parents = new Pair<int>[A_npairs];
  //for (int64_t i = 0; i < A_npairs; ++i) {
  //  src_parents[i].k = src_loc_pairs[i].d;
  //} 
  //p->read(A_npairs, src_parents);

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
  Pair<Edge> * updated_r_pairs = new Pair<Edge>[A_npairs];
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
Vector<Edge> * as(Matrix<Edge> * A, World * world) {
  int n = A->nrow;

  const static Monoid<Edge> MIN_EDGE = get_minedge_monoid();

  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);

  auto p_prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  auto mst = new Vector<Edge>(n, *world, MIN_EDGE);

  int64_t A_npairs;
  Pair<Edge> * A_loc_pairs;
  A->get_local_pairs(&A_npairs, &A_loc_pairs, true);

  while(are_vectors_different(*p, *p_prev)) {
    (*p_prev)["i"] = (*p)["i"];

    // unconditional star hooking
    Vector<bool> * star = star_check(p);

    Vector<Edge> * r = hooking(A_npairs, A_loc_pairs, p, star);

    // tie breaking
    // hook only onto larger stars and update p
    (*p)["i"] += Function<Edge, int>([](Edge e){ return e.parent; })((*r)["i"]);

    // hook only onto larger stars and update mst
    (*mst)["i"] += Bivar_Function<Edge, int, Edge>([](EdgeExt e, int a){ return e.parent >= a ? e : EdgeExt(); })((*r)["i"], (*p)["i"]);

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
*/

