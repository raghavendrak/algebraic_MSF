#include "connectivity.h"

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

template <typename dtype>
void max_vector(CTF::Vector<dtype> & result, CTF::Vector<dtype> & A, CTF::Vector<dtype> & B)
{
  result["i"] = CTF::Function<dtype,dtype,dtype>([](dtype a, dtype b){return ((a > b) ? a : b);})(A["i"], B["i"]);
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
  auto A = new Matrix<int>(n, n, SP, *world, MAX_TIMES_SR);
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

// p[i] = rec_p[q[i]]
// if create_nonleaves=true, computing non-leaf vertices in parent forest
void shortcut2(Vector<int> & p, Vector<int> & q, Vector<int> & rec_p, World * world, Vector<int> ** nonleaves, bool create_nonleaves)
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
  
  int * triv_num = new int;
  int * loc_triv_num = new int;
  roots_num(npairs, loc_pairs, loc_triv_num, triv_num, world);
  
  if (*triv_num < 1000) {
    int * global_triv = new int[*triv_num];
    triv(npairs, *loc_triv_num, loc_pairs, triv_num, global_triv, world);
    
    int loc_nontriv_num = npairs - (*loc_triv_num);

    Pair<int> * nontriv_loc_pairs = new Pair<int>[loc_nontriv_num];
    Pair<int> * remote_pairs = new Pair<int>[loc_nontriv_num];
	  
	  int nontriv_loc_indices[loc_nontriv_num];
	  bool trivial = false;
	  int k = 0;
	  for (int i = 0; i < npairs; i++) { // construct nontrivial local indices
	    for (int j = 0; j < *triv_num; j++) {
	      if (loc_pairs[i].k == global_triv[j]) {
			    trivial = true;
			    break;
		  }
	  } if (!trivial) {
		    nontriv_loc_indices[k] = i;
		    k++;
	    }
	    trivial = false;
	  }
	
	  for (int i = 0; i < loc_nontriv_num; i++) { // construct nontrivial local pairs
	    int nontriv_index = nontriv_loc_indices[i];
	    nontriv_loc_pairs[i] = loc_pairs[nontriv_index];
	    remote_pairs[i].k = loc_pairs[nontriv_index].d;
	  }
	
    Timer t_shortcut2_read("CONNECTIVITY_Shortcut2_read");
    t_shortcut2_read.start();
    rec_p.read(loc_nontriv_num, remote_pairs); //obtains rec_p[q[i]]
    t_shortcut2_read.stop();
    for(int64_t i = 0; i < loc_nontriv_num; i++) {
      nontriv_loc_pairs[i].d = remote_pairs[i].d; 
    }
    
    for (int64_t i = 0; i < loc_nontriv_num; i++) { // update loc_pairs for create_nonleaves step
      int nontriv_index = nontriv_loc_indices[i];
	    loc_pairs[nontriv_index].d = remote_pairs[i].d; // p[i] = rec_p[q[i]]
	  }
    
    p.write(loc_nontriv_num, nontriv_loc_pairs); //enter data into p[i]
    
    delete [] remote_pairs;
    delete [] global_triv;
    delete [] nontriv_loc_pairs;
    t_shortcut.stop();
  }
  
  else { // original shortcut
    Pair<int> * remote_pairs = new Pair<int>[npairs];
    for (int64_t i=0; i<npairs; i++) {
		  remote_pairs[i].k = loc_pairs[i].d;
    }
    Timer t_shortcut_read("CONNECTIVITY_Shortcut_read");
    t_shortcut_read.start();
    rec_p.read(npairs, remote_pairs); //obtains rec_p[q[i]]
    t_shortcut_read.stop();
    for (int64_t i=0; i<npairs; i++){
      loc_pairs[i].d = remote_pairs[i].d; //p[i] = rec_p[q[i]]
    }
    p.write(npairs, loc_pairs); //enter data into p[i]

    delete [] remote_pairs;
  }
  
  //prune out leaves
  if (create_nonleaves) {
	  *nonleaves = new Vector<int>(p.len, *p.wrld, *p.sr); //set nonleaves[i] = max_j p[j], i.e. set nonleaves[i] = 1 if i has child, i.e. is nonleaf
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
  delete triv_num;
  delete loc_triv_num;
}

void roots_num(int64_t npairs, Pair<int> * loc_pairs, int * loc_roots_num, int * global_roots_num,  World * world) {
  for (int i=0; i<npairs; i++) {
    Pair<int> loc_pair = loc_pairs[i];
    if (loc_pair.d == loc_pair.k) {
      (*loc_roots_num)++;
    }
  }
    
  MPI_Allreduce(loc_roots_num, global_roots_num, 1, MPI_INT, MPI_SUM, world->comm);
}

void triv(int64_t npairs, int loc_roots_num, Pair<int> * loc_pairs, int * global_roots_num, int * global_roots,  World * world) {
  int world_size;
  MPI_Comm_size(world->comm, &world_size);
 
  int loc_roots [loc_roots_num];
  int j = 0;
  for (int i=0; i<npairs; i++) {
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
  int sum_roots = 0;
  for (int i=0; i<world_size; i++) {
    displs_roots[i] = sum_roots;
    sum_roots += global_roots_nums[i];
  }

  MPI_Allgatherv(loc_roots, loc_roots_num, MPI_INT, global_roots, global_roots_nums, displs_roots, MPI_INT, world->comm); // [., ., ., ., ., ., ., ., ., ., .]?
}

// return B where B[i,j] = A[p[i],p[j]], or if P is P[i,j] = p[i], compute B = P^T A P
Matrix<int>* PTAP(Matrix<int>* A, Vector<int>* p){
  Timer t_ptap("CONNECTIVITY_PTAP");
  t_ptap.start();
  int np = p->wrld->np;
  int64_t n = p->len;
  Pair<int> * pprs;
  int64_t npprs;
  //get local part of p
  p->get_local_pairs(&npprs, &pprs);
  assert((npprs <= (n+np-1)/np) && (npprs >= (n/np)));
  assert(A->ncol == n);
  assert(A->nrow == n);
  Pair<int> * A_prs;
  int64_t nprs;
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the row of A (A1)
    Matrix<int> A1(n, n, "ij", Partition(1,&np)["i"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    A1["ij"] = A->operator[]("ij");
    A1.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and rows of A are distributed cyclically, to compute P^T * A
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k/n)*n + pprs[(A_prs[i].k%n)/np].d;
    }
  }
  {
    //map matrix so rows are distributed as elements of p, ensures for each element of p, this process also owns the column of A (A1)
    Matrix<int> A2(n, n, "ij", Partition(1,&np)["j"], Idx_Partition(), SP*(A->is_sparse), *A->wrld, *A->sr);
    //write in P^T A into A2
    A2.write(nprs, A_prs);
    delete [] A_prs;
    A2.get_local_pairs(&nprs, &A_prs, true);
    //use fact p and cols of A are distributed cyclically, to compute P^T A * P
    for (int64_t i=0; i<nprs; i++){
      A_prs[i].k = (A_prs[i].k%n) + pprs[(A_prs[i].k/n)/np].d*n;
    }
  }
  Matrix<int> * PTAP = new Matrix<int>(n, n, SP*(A->is_sparse), *A->wrld, *A->sr);
  PTAP->write(nprs, A_prs);
  delete [] A_prs;
  t_ptap.stop();
  return PTAP;
}


//recursive projection based algorithm
Vector<int>* supervertex_matrix(int n, Matrix<int>* A, Vector<int>* p, World* world)
{
  Timer t_relax("CONNECTIVITY_Relaxation");
  t_relax.start();
  //relax all edges
  auto q = new Vector<int>(n, SP*p->is_sparse, *world, MAX_TIMES_SR);
  (*q)["i"] = (*p)["i"];
  (*q)["i"] += (*A)["ij"] * (*p)["j"];
  t_relax.stop();
  Vector<int> * nonleaves;
  //check for convergence
  int64_t diff = are_vectors_different(*q, *p);
  if (p->wrld->rank == 0)
    printf("Diff is %ld\n",diff);
  if (!diff){
    return p;
  } else {
    //compute shortcutting q[i] = q[q[i]], obtain nonleaves or roots (FIXME: can we also remove roots that are by themselves?)
    shortcut2(*q, *q, *q, world, &nonleaves, true);
    if (p->wrld->rank == 0)
      printf("Number of nonleaves or roots is %ld\n",nonleaves->nnz_tot);
    //project to reduced graph with all vertices
    auto rec_A = PTAP(A, q);
    //recurse only on nonleaves
    auto rec_p = supervertex_matrix(n, rec_A, nonleaves, world);
    delete rec_A;
    //perform one step of shortcutting to update components of leaves
    (*p)["i"] += (*rec_p)["i"];
    shortcut2(*p, *q, *rec_p, world);
    delete q;
    delete rec_p;
    return p;
  }
}

Vector<int>* hook_matrix(int n, Matrix<int> * A, World* world)
{
  auto p = new Vector<int>(n, *world, MAX_TIMES_SR);
  init_pvector(p);
  auto prev = new Vector<int>(n, *world, MAX_TIMES_SR);

  while (are_vectors_different(*p, *prev)) {
    (*prev)["i"] = (*p)["i"];
    auto q = new Vector<int>(n, *world, MAX_TIMES_SR);
    Timer t_relax("CONNECTIVITY_Relaxation");
    t_relax.start();
    (*q)["i"] = (*A)["ij"] * (*p)["j"];
    t_relax.stop();
    auto r = new Vector<int>(n, *world, MAX_TIMES_SR);
    max_vector(*r, *p, *q);
    //auto P = pMatrix(p, world);
    auto s = new Vector<int>(n, *world, MAX_TIMES_SR);
    //(*s)["i"] = (*P)["ji"] * (*r)["j"];
    shortcut(*s, *r, *p);
    max_vector(*p, *p, *s);
    Vector<int> * pi = new Vector<int>(*p);
    shortcut(*p, *p, *p);

    while (are_vectors_different(*pi, *p)){
      delete pi;
      pi = new Vector<int>(*p);
      shortcut(*p, *p, *p);
    }
    delete pi;

    delete q;
    delete r;
    delete s;
  }
  return p;
}


std::vector< Matrix<int>* > batch_subdivide(Matrix<int> & A, std::vector<float> batch_fracs){
  Matrix<float> B(A.nrow, A.ncol, SP*A.is_sparse, *A.wrld, MAX_TIMES_SR);
  Pair<int> * prs;
  int64_t nprs;
  A.get_local_pairs(&nprs, &prs, true);
  srand48(A.wrld->rank*4+10);
  Pair<float> * rprs = new Pair<float>[nprs];
  for (int64_t i=0; i<nprs; i++){
    rprs[i].k = prs[i].k;
    rprs[i].d = drand48();
  }
  std::sort(rprs, rprs+nprs, [](const Pair<float> & a, const Pair<float> & b){ return a.d<b.d; });
  float prefix = 0.;
  int64_t iprefix = 0;
  std::vector< Matrix<int>* > vp;
  for (int i=0; i<batch_fracs.size(); i++){
    prefix += batch_fracs[i];
    int64_t old_iprefix = iprefix;
    while (iprefix < nprs && rprs[iprefix].d < prefix){ iprefix++; }
    Matrix<int> * P = new Matrix<int>(A.nrow, A.ncol, SP*A.is_sparse, *A.wrld, MAX_TIMES_SR);
    Pair<int> * part_pairs = new Pair<int>[iprefix-old_iprefix];
    for (int64_t j=0; j<iprefix-old_iprefix; j++){
      part_pairs[j].k = rprs[old_iprefix+j].k;
      part_pairs[j].d = 1;
    }
    P->write(iprefix-old_iprefix, part_pairs);
    delete [] part_pairs;
    vp.push_back(P);
  }
  return vp;
}


// FIXME: remove these functions or document at least
// ---------------------------
Vector<int>* hook(Graph* graph, World* world) {
  auto n = graph->numVertices;
  auto A = graph->adjacencyMatrix(world);
  return hook_matrix(n, A, world);
}

Matrix<int>* mat_add(Matrix<int>* A, Matrix<int>* B, World* world) {
  int n = A->nrow;
  int m = A->ncol;
  auto C = new Matrix<int>(n, m, *world, MAX_TIMES_SR);
  for (auto row = 0; row < n; row++) {
    for (auto col = 0; col < m; col++) {
      auto idx = Int64Pair(row, col);
      auto aVal = mat_get(A, idx);
      auto bVal = mat_get(B, idx);
      mat_set(C, idx, min(1, aVal + bVal));
    }
  }
  return C;
}

bool mat_eq(Matrix<int>* A, Matrix<int>* B) {
  for (int r = 0; r < A->nrow; r++) {
    for (int c = 0; c < A->ncol; c++) {
      if (mat_get(A, Int64Pair(r, c)) != mat_get(B, Int64Pair(r, c))) {
        return false;
      }
    }
  }
  return true;
}

Int64Pair::Int64Pair(int64_t i1, int64_t i2) {
  this->i1 = i1;
  this->i2 = i2;
}

Int64Pair Int64Pair::swap() {
  return {this->i2, this->i1};
}

Matrix<int>* mat_I(int dim, World* world) {
  auto I = new Matrix<int>(dim, dim, *world, MAX_TIMES_SR);
  for (auto i = 0; i < dim; i++) {
    mat_set(I, Int64Pair(i, i), 1);
  }
  return I;
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

void mat_set(Matrix<int>* matrix, Int64Pair index, int value) {
  int64_t idx[1];
  idx[0] = index.i2 * matrix->nrow + index.i1;
  int fill[1];
  fill[0] = value;
  matrix->write(1, idx, fill);
}

int mat_get(Matrix<int>* matrix, Int64Pair index) {
  auto data = new int[matrix->nrow * matrix->ncol];
  matrix->read_all(data);
  int value = data[index.i2 * matrix->nrow + index.i1];
  delete [] data;
  return value;
}
// ---------------------------

