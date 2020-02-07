#include "connectivity.h"

Matrix <wht> preprocess_graph(int           n,
                              World &       dw,
                              Matrix<wht> & A_pre,
                              bool          remove_singlets,
                              int *         n_nnz,
                              int64_t       max_ewht=1){
  Semiring<wht> s(MAX_WHT,
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });

  A_pre["ii"] = 0;

  A_pre.sparsify([](int a){ return a>0; });

  if (dw.rank == 0)
    printf("A contains %ld nonzeros\n", A_pre.nnz_tot);

  if (remove_singlets){
    Vector<int> rc(n, dw);
    rc["i"] += ((Function<wht>)([](wht a){ return (int)(a>0); }))(A_pre["ij"]);
    rc["i"] += ((Function<wht>)([](wht a){ return (int)(a>0); }))(A_pre["ji"]);
    int * all_rc; // = (int*)malloc(sizeof(int)*n);
    int64_t nval;
    rc.read_all(&nval, &all_rc);
    int n_nnz_rc = 0;
    int n_single = 0;
    for (int i=0; i<nval; i++){
      if (all_rc[i] != 0){
        if (all_rc[i] == 2) n_single++;
        all_rc[i] = n_nnz_rc;
        n_nnz_rc++;
      } else {
        all_rc[i] = -1;
      }
    }
    if (dw.rank == 0) printf("n_nnz_rc = %d of %d vertices kept, %d are 0-degree, %d are 1-degree\n", n_nnz_rc, n,(n-n_nnz_rc),n_single);
    Matrix<wht> A(n_nnz_rc, n_nnz_rc, SP, dw, MAX_TIMES_SR, "A");
    int * pntrs[] = {all_rc, all_rc};

    A.permute(0, A_pre, pntrs, 1);
    free(all_rc);
    if (dw.rank == 0) printf("preprocessed matrix has %ld edges\n", A.nnz_tot);

    A["ii"] = 0;
    *n_nnz = n_nnz_rc;
    return A;
  } else {
    *n_nnz= n;
    A_pre["ii"] = 0;
    //A_pre.print();
    return A_pre;
  }
//  return n_nnz_rc;

}

Matrix <wht> read_matrix(World  &     dw,
                         int          n,
                         const char * fpath,
                         bool         remove_singlets,
                         int *        n_nnz,
                         int64_t      max_ewht=1){
  uint64_t *my_edges = NULL;
  uint64_t my_nedges = 0;
  Semiring<wht> s(MAX_WHT,
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });
  //random adjacency matrix
  Matrix<wht> A_pre(n, n, SP, dw, MAX_TIMES_SR, "A_rmat");
#ifdef MPIIO
  if (dw.rank == 0) printf("Running MPI-IO graph reader n = %d... ",n);
  char **leno;
  my_nedges = read_graph_mpiio(dw.rank, dw.np, fpath, &my_edges, &leno);
  processedges(leno, my_nedges, dw.rank, &my_edges);
  free(leno[0]);
  free(leno);
#else
  if (dw.rank == 0) printf("Running graph reader n = %d... ",n);
    my_nedges = read_graph(dw.rank, dw.np, fpath, &my_edges);
#endif
  if (dw.rank == 0) printf("finished reading (%ld edges).\n", my_nedges);
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_nedges);
  wht * vals = (wht*)malloc(sizeof(wht)*my_nedges);

  srand(dw.rank+1);
  for (int64_t i=0; i<my_nedges; i++){
    inds[i] = my_edges[2*i]+my_edges[2*i+1]*n;
    //vals[i] = (rand()%max_ewht) + 1;
    vals[i] = 1;
  }
  if (dw.rank == 0) printf("filling CTF graph\n");
  A_pre.write(my_nedges,inds,vals);
  //A_pre["ij"] += A_pre["ji"];
  A_pre["ij"] += A_pre["ji"];
  free(inds);
  free(vals);

  Matrix<wht> newA =  preprocess_graph(n,dw,A_pre,remove_singlets,n_nnz,max_ewht);
  /*int64_t nprs;
  newA.read_local_nnz(&nprs,&inds,&vals);

  for (int64_t i=0; i<nprs; i++){
    printf("%d %d\n",inds[i]/newA.nrow,inds[i]%newA.nrow);
  }*/
  return newA;
}

Matrix <wht> gen_rmat_matrix(World  & dw,
                             int      scale,
                             int      ef,
                             uint64_t gseed,
                             bool     remove_singlets,
                             int *    n_nnz,
                             int64_t  max_ewht=1){
  uint64_t *edge=NULL;
  uint64_t nedges = 0;
  Semiring<wht> s(MAX_WHT,
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });
  //random adjacency matrix
  int n = pow(2,scale);
  Matrix<wht> A_pre(n, n, SP, dw, MAX_TIMES_SR, "A_rmat");
  if (dw.rank == 0) printf("Running graph generator n = %d... ",n);
  nedges = gen_graph(scale, ef, gseed, &edge);
  if (dw.rank == 0) printf("done.\n");
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*nedges);
  wht * vals = (wht*)malloc(sizeof(wht)*nedges);

  srand(dw.rank+1);
  for (int64_t i=0; i<nedges; i++){
    inds[i] = (edge[2*i]+(edge[2*i+1])*n);
    // vals[i] = (rand()%max_ewht) + 1;
    vals[i] = 1;
  }
  if (dw.rank == 0) printf("filling CTF graph\n");
  A_pre.write(nedges,inds,vals);
  A_pre["ij"] += A_pre["ji"];
  free(inds);
  free(vals);

  return preprocess_graph(n,dw,A_pre,remove_singlets,n_nnz,max_ewht);

}
Matrix <wht> gen_uniform_matrix(World & dw,
                                int64_t n,
                                double  sp=.20,
                                int64_t  max_ewht=1){
  Semiring<wht> s(MAX_WHT,
                  [](wht a, wht b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](wht a, wht b){ return a+b; });

  //random adjacency matrix
  Matrix<wht> A(n, n, SP, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
//  A.fill_random(1, std::min(n*n,100));
  int nmy = ((int)std::max((int)(n*sp),(int)1))*((int)((n+dw.np-1)/dw.np));
  int64_t inds[nmy];
  wht vals[nmy];
  int i=0;
  for (int64_t row=dw.rank*n/dw.np; row<(int)(dw.rank+1)*n/dw.np; row++){
    int64_t cols[std::max((int)(n*sp),1)];
    for (int64_t col=0; col<std::max((int)(n*sp),1); col++){
      bool is_rep;
      do {
        cols[col] = rand()%n;
        is_rep = 0;
        for (int c=0; c<col; c++){
          if (cols[c] == cols[col]) is_rep = 1;
        }
      } while (is_rep);
      inds[i] = cols[col]*n+row;
      vals[i] = (rand()%max_ewht)+1;
      i++;
    }
  }
  A.write(i,inds,vals);

  A["ii"] = 0;

  //keep only values smaller than 20 (about 20% sparsity)
  //A.sparsify([=](int a){ return a<sp*100; });
   return A;
}


void test_6Blocks_simply_connected(World *w)
{
  auto g = new Graph();
  g->numVertices = 36;
  for(int b = 0; b < 6; b++){
    for(int i = 0; i < 5; i++){
      g->edges->emplace_back(b*6+i, b*6+i+1);
    }
  }
  auto A = g->adjacencyMatrix(w);
  hook_matrix(36, A, w)->print();
  int matSize = 36;
  auto p = new Vector<int>(matSize, *w, MAX_TIMES_SR);
  init_pvector(p);

  int sc2 = 0;
  supervertex_matrix(matSize, A, p, w, sc2)->print();
}

void test_batch_subdivide(World *w)
{
  /**
  auto g = new Graph();
  g->numVertices = 36;
  for(int b = 0; b < 6; b++){
    for(int i = 0; i < 5; i++){
      g->edges->emplace_back(b*6+i, b*6+i+1);
    }
  }
  auto A = g->adjacencyMatrix(w);
  **/
  auto A = new Matrix<int>(10,10,SP|SH,*w,MAX_TIMES_SR);
  A->fill_sp_random(1.0,1.0,0.4);
  A->print_matrix();
  std::vector<float> fracs = {0.1, 0.1, 0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1};
  std::vector<Matrix<int>*> batches = batch_subdivide(*A, fracs);
  int count = 1;
  for(Matrix<int>* mat: batches) {
    cout << "Batch " << count << endl;
    mat->print_matrix();
    count++;
  }
  /**
  hook_matrix(36, A, w)->print();
  int matSize = 36;
  auto p = new Vector<int>(matSize, *w, MAX_TIMES_SR);
  init_pvector(p);
  supervertex_matrix(matSize, A, p, w)->print();
  **/
}

void test_shortcut2(World *w) {
  int scale = 10;
  int ef = 8;
  uint64_t myseed;
  int prep = 0;
  int n_nnz = 0;
  myseed = SEED;
  int max_ewht;
  //int batch = 0;
  int sc2 = 0;

  if (w->rank == 0)
    printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
  Matrix<wht> A = gen_rmat_matrix(*w, scale, ef, myseed, prep, &n_nnz, max_ewht);
  int64_t matSize = A.nrow;
  if (w->rank == 0)
    printf("matSize = %ld\n",matSize);

  auto p = new Vector<int>(100, *w, MAX_TIMES_SR);
  init_pvector(p);
  (*p)["i"] = (A)["ij"] * (*p)["j"];

  auto p2 = new Vector<int>(100, *w, MAX_TIMES_SR);
  init_pvector(p2);
  (*p2)["i"] = (A)["ij"] * (*p2)["j"];

  if (w->rank == 0) { printf("P"); }
  p->print();
  
  Vector<int> * nonleaves;
  Vector<int> * nonleaves2;
  shortcut(*p, *p, *p, &nonleaves, true);
  shortcut2(*p2, *p2, *p2, sc2, w, &nonleaves2, true);

  int64_t result = are_vectors_different(*p, *p2);
  if (w->rank == 0) {
    if (result) {
      printf("result p vectors are different by %ld: FAIL\n", result);
    }
    else {
      printf("result p vectors are same: PASS\n");
    }
  }
  
  int64_t result_nonleaves = are_vectors_different(*nonleaves, *nonleaves2);
  if (w->rank == 0) {
    if (result_nonleaves) {
      printf("result nonleaves vectors are different by %ld: FAIL\n", result_nonleaves);
    }
    else {
      printf("result nonleaves vectors are same: PASS\n");
    }
  }
  
  delete p;
  delete p2;
}

Matrix<int>* generate_kronecker(World* w, int order)
{
  auto g = new Graph();
  g->numVertices = 3;
  g->edges->emplace_back(0, 0);
  g->edges->emplace_back(0, 1);
  g->edges->emplace_back(1, 1);
  g->edges->emplace_back(1, 2);
  g->edges->emplace_back(2, 2);
  auto kinitiator = g->adjacencyMatrix(w);
  auto B = g->adjacencyMatrix(w);

  int64_t len = 1;
  int64_t matSize = 3;
  for (int i = 2; i <= order; i++) {
    len *= 3;
    int64_t lens[] = {3, len, 3, len};
    /**
    int * lens = new int[4];
    lens[0] = 3;
    lens[1] = len;
    lens[2] = 3;
    lens[3] = len;
    **/
    auto D = Tensor<int>(4, B->is_sparse, lens);
    D["ijkl"] = (*kinitiator)["ik"] * (*B)["jl"];

    matSize *= 3;
    auto B2 = new Matrix<int>(matSize, matSize, B->is_sparse * SP, *w, *B->sr);
    delete B;
    B2->reshape(D);
    B = B2;
    // B->print_matrix();
    // hook on B
  }
  delete kinitiator;
  return B;
}

void serial_connectivity_dfs(int64_t v, std::vector<std::vector<int64_t> > &adj, bool *visited)
{
  visited[v] = true;
  for (int64_t i = 0; i < adj[v].size(); i++) {
    if (visited[adj[v][i]] == false) {
      serial_connectivity_dfs(adj[v][i], adj, visited);
    }
  }
}

int64_t serial_connectivity(Matrix<int>* A)
{
  // A->print_matrix();
  int64_t numpair = 0;
  Pair<int> *vpairs = nullptr;
  A->get_all_pairs(&numpair, &vpairs, true);
  std::vector<std::vector<int64_t> > adj (A->nrow);
  for (int64_t i = 0; i < numpair; i++) {
    int64_t rowNo = vpairs[i].k % A->nrow;
    int64_t colNo = vpairs[i].k / A->nrow;
    if (rowNo < colNo) {
      adj[rowNo].push_back(colNo);
    }
  }
  bool visited[A->nrow];
  for (int64_t i = 0; i < A->nrow; i++) {
    visited[i] = false;
  }
  int64_t connected_components = 0;
  for(int64_t i = 0; i < adj.size(); i++) {
    if (visited[i] == false) {
      serial_connectivity_dfs(i, adj, visited);
      connected_components++;
    }
  }

  /*
  for(int64_t i = 0; i < adj.size(); i++) {
    for (int64_t j = 0; j < adj[i].size(); j++) {
      std::cout << "row: " << i << " col: " << adj[i][j] << endl;
    }
  }
  */
  return connected_components;
}

void run_connectivity(Matrix<int>* A, int64_t matSize, World *w, int batch, int shortcut, int run_serial)
{
  matSize = A->nrow; // Quick fix to avoid change in i/p matrix size after preprocessing
  double stime;
  double etime;
  auto pg = new Vector<int>(matSize, *w, MAX_TIMES_SR);
  init_pvector(pg);
  Scalar<int64_t> count(*w);
  Timer_epoch thm("hook_matrix");
  thm.begin();
  stime = MPI_Wtime();
  auto hm = hook_matrix(matSize, A, w);
  etime = MPI_Wtime();
  if (w->rank == 0) {
    printf("Time for hook_matrix(): %1.2lf\n", (etime - stime));
  }
  thm.end();
  count[""] = Function<int,int,int64_t>([](int a, int b){ return (int64_t)(a==b); })((*pg)["i"], hm->operator[]("i"));
  int64_t cnt = count.get_val();
  if (w->rank == 0) {
    printf("Found %ld components with hook_matrix, pg is of length %d, hm of length %d, matSize is %ld.\n",cnt,pg->len,hm->len,matSize);
  }

  auto p = new Vector<int>(matSize, *w, MAX_TIMES_SR);
  init_pvector(p);
  Timer_epoch tsv("super_vertex");
  tsv.begin();
  Vector<int>* sv;
  stime = MPI_Wtime();
  if (batch == 1) {
    sv = supervertex_matrix(matSize, A, p, w, shortcut);
  }
  else {
    std::vector<float> fracs;
    float frac = 1.0f / batch;
    for (int i=0; i<batch; i++) {
      fracs.push_back(frac);
    }
    std::vector<Matrix<int>*> batches = batch_subdivide(*A, fracs);
    sv = p;
    bool st = true;
    for(Matrix<int>* mat: batches) {
      if (!st)
        mat->operator[]("ij") += pMatrix(sv, sv->wrld)->operator[]("ij");
      st = false;
      sv = supervertex_matrix(matSize, mat, sv, w, shortcut);
    }
  }
  etime = MPI_Wtime();
  if (w->rank == 0) {
    printf("Time for supervertex_matrix(): %1.2lf\n", (etime - stime));
  }
  tsv.end();
  count[""] = Function<int,int,int64_t>([](int a, int b){ return (int64_t)(a==b); })((*pg)["i"], sv->operator[]("i")); // TODO: returning incorrect result on multiple processes
  cnt = count.get_val();

  if (w->rank == 0) {
    printf("SV:\n");
  }
  sv->print();
  if (w->rank == 0) {
    printf("\n");
  printf("PG\n");
  }
  pg->print();
  if (w->rank == 0) {
    printf("\n");

    printf("count\n");
  }
  count.print();

  if (w->rank == 0) {
    printf("Found %ld components with supervertex_matrix.\n",cnt);
  }

  int64_t result = are_vectors_different(*hm, *sv);
  if (w->rank == 0) {
    if (result) {
      printf("result vectors are different by %ld: FAIL\n", result);
    }
    else {
      printf("result vectors are same: PASS\n");
    }
  }
  if (run_serial) {
    int64_t serial_cnt;
    serial_cnt = serial_connectivity(A);
    if (w->rank == 0) {
      printf("Found %d components with serial_connectivity\n", serial_cnt);
      if (cnt == serial_cnt) {
        printf("Number of components between supervertex_matrix() and serial_connectivity() are same: PASS\n");
      }
      else {
        printf("Number of components between supervertex_matrix() and serial_connectivity() are different: FAIL\n");
      }
    }
  }
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
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  auto w = new World(argc, argv);
  int const in_num = argc;
  char** input_str = argv;
  uint64_t myseed;

  int64_t max_ewht;
  uint64_t edges;
  char *gfile = NULL;
  int64_t n;
  int scale;
  int ef;
  int prep;
  int batch;
  int sc2;
  int run_serial;

  int k;
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
  if (getCmdOption(input_str, input_str+in_num, "-shortcut")){
    sc2 = atoi(getCmdOption(input_str, input_str+in_num, "-shortcut"));
    if (sc2 < 0) sc2 = 0;
  } else sc2 = 0;
  if (getCmdOption(input_str, input_str+in_num, "-serial")){
    run_serial = atoi(getCmdOption(input_str, input_str+in_num, "-serial"));
    if (run_serial < 0) run_serial = 0;
  } else run_serial = 0;

  if (gfile != NULL){
    int n_nnz = 0;
    if (w->rank == 0)
      printf("Reading real graph n = %lld\n", n);
    Matrix<wht> A = read_matrix(*w, n, gfile, prep, &n_nnz);
    // A.print_matrix();
    run_connectivity(&A, n, w, batch, sc2, run_serial);
  }
  else if (k != -1) {
    int64_t matSize = pow(3, k);
    auto B = generate_kronecker(w, k);

    if (w->rank == 0) {
      printf("Running connectivity on Kronecker graph K: %d matSize: %ld\n", k, matSize);
    }
    run_connectivity(B, matSize, w, batch, sc2, run_serial);
    delete B;
  }
  else if (scale > 0 && ef > 0){
    int n_nnz = 0;
    myseed = SEED;
    if (w->rank == 0)
      printf("R-MAT scale = %d ef = %d seed = %lu\n", scale, ef, myseed);
    Matrix<wht> A = gen_rmat_matrix(*w, scale, ef, myseed, prep, &n_nnz, max_ewht);
    int64_t matSize = A.nrow; 
    run_connectivity(&A, matSize, w, batch, sc2, run_serial);
  }
  else {
    if (w->rank == 0) {
      printf("Running connectivity on 6X6 graph\n");
    }
    //test_6Blocks_simply_connected(w);
    //test_batch_subdivide(w);
    test_shortcut2(w);
  }
  return 0;
}


// ------------------------------------
// FIXME: below tests are yet to be reviewed with the driver

void test_simple(World* w)
{
  printf("TEST1: SIMPLE GRAPH 6*6\n");
  auto g = new Graph();
  g->numVertices = 6;
  g->edges->emplace_back(0, 1);
  g->edges->emplace_back(2, 4);
  g->edges->emplace_back(4, 3);
  g->edges->emplace_back(3, 5);
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}

void test_disconnected(World *w)
{
  printf("TEST2: DISCONNECTED 6*6\n");
  auto g = new Graph();
  g->numVertices = 6;
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}

void test_fully_connected(World *w)
{
  printf("TEST3: FULLY CONNECTED 6*6\n");
  auto g = new Graph();
  g->numVertices = 6;
  for(int i = 0; i < 5; i++){
    for(int j = i + 1; j < 6; j++){
      g->edges->emplace_back(i, j);
    }
  }
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}


void test_random1(World *w)
{
  printf("TEST4-1: RANDOM 1 6*6\n");
  Matrix<int> * B = new Matrix<int>(6,6,SP|SH,*w,MAX_TIMES_SR);
  B->fill_sp_random(1.0,1.0,0.1);
  B->print_matrix();
  hook_matrix(6, B, w)->print();
}

void test_random2(World *w)
{
  printf("TEST4-2: RANDOM 2 10*10\n");
  Matrix<int> * B = new Matrix<int>(10,10,SP|SH,*w,MAX_TIMES_SR);
  B->fill_sp_random(1.0,1.0,0.1);
  B->print_matrix();
  hook_matrix(10, B, w)->print();
}

void test_6Blocks_fully_connected(World *w)
{
  printf("TEST6: 6 Blocks of 6*6 fully connected graph\n");
  auto g = new Graph();
  g->numVertices = 36;
  for(int b = 0; b < 6; b++){
    for(int i = 0; i < 5; i++){
      for(int j = i+1; j < 6; j++)
        g->edges->emplace_back(b*6+i, b*6+j);
    }
  }
  auto A = g->adjacencyMatrix(w);
  A->print_matrix();
  hook(g, w)->print();
}

//-----------------------------------------------------------
