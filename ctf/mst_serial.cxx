// does not use path compression
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
Vector<EdgeExt> * serial_mst(Matrix<EdgeExt> * A, World * world) {
  const static Monoid<EdgeExt> MIN_EDGE = get_minedge_monoid();

  int64_t npair;
  Pair<EdgeExt> * pairs;
  A->get_all_pairs(&npair, &pairs, true);

  EdgeExt edges[npair];
  for (int64_t i = 0; i < npair; ++i) {
    edges[i] = EdgeExt(pairs[i].d.src, pairs[i].d.weight, pairs[i].d.dest, pairs[i].d.parent);
  }

  std::sort(edges, edges + npair, [](const EdgeExt & lhs, const EdgeExt & rhs) { return lhs.weight < rhs.weight; });

  int64_t p[A->nrow];
  for (int64_t i = 0; i < A->nrow; ++i) {
    p[i] = i;
  }

  int64_t mst_npair = A->nrow - 1;
  Pair<EdgeExt> * mst_pairs = new Pair<EdgeExt>[mst_npair];
  int64_t j = 0;
  for (int64_t i = 0; i < npair; ++i) {
    if (find(p, edges[i].src) != find(p, edges[i].dest)) {
      mst_pairs[j].k = j;
      mst_pairs[j].d = edges[i];
      ++j;
      union1(p, edges[i].src, edges[i].dest);
    }
  }

  Vector<EdgeExt> * mst = new Vector<EdgeExt>(A->nrow, *world, MIN_EDGE);
  mst->write(mst_npair, mst_pairs);

  delete [] mst_pairs;
  delete [] pairs;

  return mst;
}
