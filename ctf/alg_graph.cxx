#include "alg_graph.h"

Int64Pair::Int64Pair(int64_t i1, int64_t i2) {
  this->i1 = i1;
  this->i2 = i2;
}

Int64Pair Int64Pair::swap() {
  return {this->i2, this->i1};
}

void mat_set(Matrix<int>* matrix, Int64Pair index, int value) {
  int64_t idx[1];
  idx[0] = index.i2 * matrix->nrow + index.i1;
  int fill[1];
  fill[0] = value;
  matrix->write(1, idx, fill);
}

Graph::Graph() {
  this->numVertices = 0;
  this->edges = new vector<Int64Pair>();
}

Matrix<int>* Graph::adjacencyMatrix(World* world, bool sparse) {
  auto attr = 0;
  if (sparse) {
    attr = SP;
  }
  auto A = new Matrix<int>(numVertices, numVertices,
      attr, *world, MAX_TIMES_SR);
  for (auto edge : *edges) {
    mat_set(A, edge);
    mat_set(A, edge.swap());
  }
  return A;
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
template int64_t are_vectors_different<int>(CTF::Vector<int> & A, CTF::Vector<int> & B);

// p[i] = rec_p[q[i]]
// if create_nonleaves=true, computing non-leaf vertices in parent forest
void shortcut(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, Vector<int> ** nonleaves, bool create_nonleaves)
{
  TAU_START(Unoptimized_shortcut);
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
  TAU_START(Unoptimized_shortcut_rread);
  rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
  TAU_STOP(Unoptimized_shortcut_rread);
  for (int64_t i=0; i<npairs; i++){
    loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
  }
  delete [] remote_pairs;
  TAU_START(Unoptimized_shortcut_write);
  p.write(npairs, loc_pairs); //enter data into p[i]
  TAU_STOP(Unoptimized_shortcut_write);
  
  //prune out leaves
  if (create_nonleaves){
    TAU_START(Unoptimized_shortcut_pruneleaves);
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
    TAU_STOP(Unoptimized_shortcut_pruneleaves);
  }
   
  delete [] loc_pairs;
  TAU_STOP(Unoptimized_shortcut);
}

// p[i] = rec_p[q[i]]
// if create_nonleaves=true, computing non-leaf vertices in parent forest
void shortcut2(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, int sc2, World * world, Vector<int> ** nonleaves, bool create_nonleaves)
{
  if (sc2 <= 0) { // run unoptimized shortcut
    shortcut(p, q, rec_p, nonleaves, create_nonleaves);
    return;
  }

  TAU_START(Optimized_shortcut);
  
  int64_t rec_p_npairs;
  Pair<int> * rec_p_loc_pairs;
  if (rec_p.is_sparse) {
    rec_p.get_local_pairs(&rec_p_npairs, &rec_p_loc_pairs, true);
  } else {
    rec_p.get_local_pairs(&rec_p_npairs, &rec_p_loc_pairs);
  }
  
  int64_t q_npairs;
  Pair<int> * q_loc_pairs;
  bool delete_p = true;
  if (&q == &rec_p) {
    q_npairs = rec_p_npairs;
    q_loc_pairs = rec_p_loc_pairs; // TODO: optimize with reference?
    delete_p = false;
  } else {
    p["i"] += q["i"];
    if (q.is_sparse){
      //if we have updated only a subset of the vertices
      q.get_local_pairs(&q_npairs, &q_loc_pairs, true);
    } else {
      //if we have potentially updated all the vertices
      q.get_local_pairs(&q_npairs, &q_loc_pairs);
    }
  }
  
  int64_t * global_roots_num = new int64_t;
  int64_t * loc_roots_num = new int64_t;
  roots_num(rec_p_npairs, rec_p_loc_pairs, loc_roots_num, global_roots_num, world);
  
  if (*global_roots_num < sc2) {
    int * global_roots = new int[*global_roots_num];
    roots(rec_p_npairs, *loc_roots_num, rec_p_loc_pairs, global_roots_num, global_roots, world);
    
    int64_t * nontriv_loc_indices;
    int64_t * loc_nontriv_num = new int64_t;
    create_nontriv_loc_indices(nontriv_loc_indices, loc_nontriv_num, global_roots_num, global_roots, q_npairs, q_loc_pairs, world);

    Pair<int> * nontriv_loc_pairs = new Pair<int>[*loc_nontriv_num];
    Pair<int> * remote_pairs = new Pair<int>[*loc_nontriv_num];
    for (int64_t i = 0; i < *loc_nontriv_num; i++) {
      int64_t nontriv_index = nontriv_loc_indices[i];
      nontriv_loc_pairs[i] = q_loc_pairs[nontriv_index];
      remote_pairs[i].k = q_loc_pairs[nontriv_index].d;
    }
  
    TAU_START(Optimized_shortcut_read);
    rec_p.read(*loc_nontriv_num, remote_pairs); //obtains rec_p[q[i]]
    TAU_STOP(Optimized_shortcut_read);
    for(int64_t i = 0; i < *loc_nontriv_num; i++) {
      nontriv_loc_pairs[i].d = remote_pairs[i].d;
    }
    
    for (int64_t i = 0; i < *loc_nontriv_num; i++) { // update loc_pairs for create_nonleaves step
      int64_t nontriv_index = nontriv_loc_indices[i];
      q_loc_pairs[nontriv_index].d = remote_pairs[i].d; // p[i] = rec_p[q[i]]
    }
 
    TAU_START(Optimized_shortcut_write); 
    p.write(*loc_nontriv_num, nontriv_loc_pairs); //enter data into p[i]
    TAU_STOP(Optimized_shortcut_write); 
    
    delete [] remote_pairs;
    delete [] global_roots;
    delete [] nontriv_loc_pairs;
    delete [] nontriv_loc_indices;
    delete loc_nontriv_num;
  } else { // original shortcut
    Pair<int> * remote_pairs = new Pair<int>[q_npairs];
    for (int64_t i=0; i<q_npairs; i++) {
      remote_pairs[i].k = q_loc_pairs[i].d;
    }
    TAU_START(Unoptimized_shortcut_Orread);
    rec_p.read(q_npairs, remote_pairs); //obtains rec_p[q[i]]
    TAU_STOP(Unoptimized_shortcut_Orread);
    for (int64_t i=0; i<q_npairs; i++){
      q_loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
    }
    TAU_START(Unoptimized_shortcut_Owrite);
    p.write(q_npairs, q_loc_pairs); //enter data into p[i]
    TAU_STOP(Unoptimized_shortcut_Owrite);

    delete [] remote_pairs;
  }
  
  //prune out leaves
  if (create_nonleaves) {
    TAU_START(Unoptimized_shortcut_Opruneleaves);
    *nonleaves = new Vector<int>(p.len, *p.wrld, *p.sr); //set nonleaves[i] = max_j p[j], i.e. set nonleaves[i] = 1 if i has child, i.e. is nonleaf
    for (int64_t i=0; i<q_npairs; i++){
      q_loc_pairs[i].k = q_loc_pairs[i].d;
      q_loc_pairs[i].d = 1;
    }
    //FIXME: here and above potential optimization is to avoid duplicate queries to parent
    (*nonleaves)->write(q_npairs, q_loc_pairs);
    (*nonleaves)->operator[]("i") = (*nonleaves)->operator[]("i")*p["i"];
    (*nonleaves)->sparsify();
    TAU_STOP(Unoptimized_shortcut_Opruneleaves);
  }
  TAU_STOP(Optimized_shortcut);

  if (delete_p)
    delete [] q_loc_pairs;
  delete [] rec_p_loc_pairs;
  delete global_roots_num;
  delete loc_roots_num;
}

void roots_num(int64_t npairs, Pair<int> * loc_pairs, int64_t * loc_roots_num, int64_t * global_roots_num,  World * world) {
  *loc_roots_num = 0;
  for (int64_t i = 0; i < npairs; i++) {
    Pair<int> loc_pair = loc_pairs[i];
    if (loc_pair.d == loc_pair.k) {
      (*loc_roots_num)++;
    }
  }
    
  MPI_Allreduce(loc_roots_num, global_roots_num, 1, MPI_LONG_LONG, MPI_SUM, world->comm);
}

void roots(int64_t npairs, int64_t loc_roots_num, Pair<int> * loc_pairs, int64_t * global_roots_num, int * global_roots,  World * world) {
  int world_size;
  MPI_Comm_size(world->comm, &world_size);
 
  int loc_roots [loc_roots_num];
  int64_t j = 0;
  for (int64_t i=0; i<npairs; i++) {
    // same loop as roots_num but would introduce overhead
    Pair<int> loc_pair = loc_pairs[i];
    if (loc_pair.d == loc_pair.k) {
      loc_roots[j] = loc_pair.k;
      j++;
    }
  }

  int global_roots_nums [world_size];
  MPI_Allgather(&loc_roots_num, 1, MPI_INT, global_roots_nums, 1, MPI_INT, world->comm); // [3, 1, 2, 0, 4]

  // prefix sum
  int displs_roots [world_size];
  int64_t sum_roots = 0;
  for (int64_t i=0; i<world_size; i++) {
    displs_roots[i] = sum_roots;
    sum_roots += global_roots_nums[i];
  }

  MPI_Allgatherv(loc_roots, loc_roots_num, MPI_INT, global_roots, global_roots_nums, displs_roots, MPI_INT, world->comm); // [., ., ., ., ., ., ., ., ., ., .]?
}

void create_nontriv_loc_indices(int64_t *& nontriv_loc_indices, int64_t * loc_nontriv_num, int64_t * global_roots_num, int * global_roots, int64_t q_npairs, Pair<int> * q_loc_pairs, World * world) {
  std::sort(global_roots, global_roots + *global_roots_num);

  nontriv_loc_indices = new int64_t[q_npairs]; // wastes a bit of memory
  int64_t nontriv_index = 0;
  int64_t q_index = 0;
  bool end_roots = false;
  for (int64_t root_index = 0; root_index < *global_roots_num && q_index < q_npairs; q_index++) { // construct nontrivial local indices O((n+m)log(n+m))
    if (q_loc_pairs[q_index].d < global_roots[root_index]) { // if a node's parent is not a root
      nontriv_loc_indices[nontriv_index] = q_index;
      nontriv_index++;
    }
    while (q_loc_pairs[q_index].d > global_roots[root_index]) {
      root_index++;
      if (root_index >= *global_roots_num) { end_roots = true; break; }
       
      if (q_loc_pairs[q_index].d < global_roots[root_index]) { // if this node's parent is greater than previous root but less than current root
        nontriv_loc_indices[nontriv_index] = q_index;
        nontriv_index++;
      }
    }
    if (end_roots) { break; }
  }

  for (; q_index < q_npairs; q_index++) { // add nodes with parent greater than max root
    nontriv_loc_indices[nontriv_index] = q_index;
    nontriv_index++;
  }

  *loc_nontriv_num = nontriv_index;
}

// return B where B[i,j] = A[p[i],p[j]], or if P is P[i,j] = p[i], compute B = P^T A P
template<typename T>
Matrix<T>* PTAP(Matrix<T>* A, Vector<int>* p){
  TAU_START(CONNECTIVITY_PTAP);
  int np = p->wrld->np;
  int64_t n = p->len;
  Pair<int> * pprs;
  int64_t npprs;
  //get local part of p
  p->get_local_pairs(&npprs, &pprs);
  assert((npprs <= (n+np-1)/np) && (npprs >= (n/np)));
  assert(A->ncol == n);
  assert(A->nrow == n);
  Pair<T> * A_prs;
  int64_t nprs;
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the row of A (A1)
    Matrix<T> A1(n, n, "ij", Partition(1,&np)["i"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A1["ij"] = A->operator[]("ij");
    A1.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and rows of A are distributed cyclically, to compute P^T * A
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k/n)*n + pprs[(A_prs[i].k%n)/np].d;
    }
  }
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the column of A (A1)
    Matrix<T> A2(n, n, "ij", Partition(1,&np)["j"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    //write in P^T A into A2
    A2.write(nprs, A_prs);
    delete [] A_prs;
    A2.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and cols of A are distributed cyclically, to compute P^T A * P
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k%n) + pprs[(A_prs[i].k/n)/np].d*n;
    }
  }
  Matrix<T> * PTAP = new Matrix<T>(n, n, SP*(A->is_sparse), *A->wrld, *A->sr);
  PTAP->write(nprs, A_prs);
  delete [] A_prs;
  TAU_STOP(CONNECTIVITY_PTAP);
  return PTAP;
}
template Matrix<int>* PTAP<int>(Matrix<int>* A, Vector<int>* p);
template Matrix<EdgeExt>* PTAP<EdgeExt>(Matrix<EdgeExt>* A, Vector<int>* p);
