#include <ctf.hpp>

#define PLACE_VERTEX (1);

using namespace CTF;

class IntPair {
public:
	int64_t i1;
	int64_t i2;
	
	IntPair(int64_t i1, int64_t i2);
	
	IntPair(const IntPair &p);
};

class Graph {
public:
	int n;
	vector<IntPair> edges;
	
	Graph(int n, vector<IntPair> edges);
	
	Matrix<int>* adj_mat(World* w);
};

Vector<int>* connectivity(Matrix<int>* A);

int main(int argc, char** argv) {
	int rank;
	int np;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	World w(argc, argv);
	
	auto edges = vector<IntPair>();
	edges.emplace_back(1, 2);
	edges.emplace_back(3, 4);
	edges.emplace_back(3, 5);
	edges.emplace_back(4, 5);
	auto g = Graph(6, edges);
	auto B = g.adj_mat(&w);
	B->print_matrix();
	connectivity(B)->print();
	
	int n = 7;
	auto A = new Matrix<int>(n, n, w);
	(*A)["ij"] = 0; // init to all 0
	int v = 12;
	auto idx = new int64_t[v]{1, 2, 7, 9, 10, 14, 15, 18, 19, 22, 30, 37};
	auto data = new int[v];
	for (int i = 0; i < v; i++) {
		data[i] = PLACE_VERTEX;
	}
	A->write(v, idx, data);
	printf("matrix before is: \n");
	A->print_matrix();
	free(idx);
	free(data);
	
	auto ret = connectivity(A);
	free(A);
	printf("return w: \n");
	ret->print();
	free(ret);
}

IntPair::IntPair(int64_t i1, int64_t i2) {
	this->i1 = i1;
	this->i2 = i2;
}

IntPair::IntPair(const IntPair &p) {
	this->i1 = p.i1;
	this->i2 = p.i2;
}

Graph::Graph(int n, vector<IntPair> edges) {
	this->n = n;
	this->edges = edges;
}

Matrix<int>* Graph::adj_mat(World* w) {
	auto A = new Matrix<int>(n, n, *w);
	(*A)["ij"] = 0;
	
	auto n64 = (int64_t) n;
	auto m = (int64_t) edges.size();
	auto m2 = m * 2;
	auto idx = new int64_t[m2];
	for (int i = 0; i < m; i++) {
		auto edge = edges[i];
		auto v1 = edge.i1;
		auto v2 = edge.i2;
		idx[i] = v1 * n64 + v2;
		idx[i + m] = v2 * n64 + v1;
	}
	auto fill = new int[m2];
	for (int i = 0; i < m2; i++) {
		fill[i] = PLACE_VERTEX;
	}
	A->write(m2, idx, fill);
	
	return A;
}

Vector<int>* connectivity(Matrix<int>* A) {
	assert(A->nrow == A->ncol);
	
	// tropical semiring
	Semiring<int> tsr(0,
	                  [](int a, int b) { return std::max(a, b); },
	                  MPI_MAX,
	                  0,
	                  [](int a, int b) { return a + b; });
	int n = A->nrow;
	auto w = new Vector<int>(n, *A->wrld, tsr);
	auto tmp = A->sr;
	// equip A w/ tsr
	A->sr = &tsr;
	
	// update adj mat
	for (int j = 0; j < n; j++) {
		(*w)["j"] += (*A)["jk"] * (*w)["k"];
	}
	
	A->sr = tmp;
	return w;
}

