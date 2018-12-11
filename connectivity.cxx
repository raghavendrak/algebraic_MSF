#include <ctf.hpp>
#include <utility>
#include <cfloat>

#define PLACE_VERTEX (1.0);

using namespace CTF;

class IntPair {
public:
	int64_t i1;
	int64_t i2;
	
	IntPair(int64_t i1, int64_t i2);
};

class Graph {
public:
	int n;
	vector<IntPair> edges;
	
	Graph(int n, vector<IntPair> edges);
	
	Matrix<float>* adj_mat(World* w);
};

Vector<float>* connectivity(Matrix<float> A);

int main(int argc, char** argv) {
	int rank;
	int np;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	World w(argc, argv);
	
	cout << "Example 1:" << endl;
	auto edges = vector<IntPair>();
	edges.emplace_back(1, 2);
	edges.emplace_back(3, 4);
	edges.emplace_back(3, 5);
	edges.emplace_back(4, 5);
	auto g = Graph(6, edges);
	auto B = g.adj_mat(&w);
	B->print_matrix();
	connectivity(*B)->print();
	free(B);
	
	cout << endl;
	
	cout << "Example 2:" << endl;
	int n = 7;
	Semiring<float> tsr(0.0,
	                    [](float a, float b) { return std::max(a, b); },
	                    MPI_MAX,
	                    1.0,
	                    [](float a, float b) { return a * b; });
	auto A = Matrix<float>(n, n, w, tsr);
	int v = 12;
	auto idx = new int64_t[v]{1, 2, 7, 9, 10, 14, 15, 18, 19, 22, 30, 37};
	auto data = new float[v];
	for (int i = 0; i < v; i++) {
		data[i] = PLACE_VERTEX;
	}
	A.write(v, idx, data);
	A.print_matrix();
	free(idx);
	free(data);
	
	auto ret = connectivity(A);
	ret->print();
	free(ret);
}

IntPair::IntPair(int64_t i1, int64_t i2) {
	this->i1 = i1;
	this->i2 = i2;
}

Graph::Graph(int n, vector<IntPair> edges) {
	this->n = n;
	this->edges = std::move(edges);
}

Matrix<float>* Graph::adj_mat(World* w) {
	// tropical semiring (tsr)
	Semiring<float> tsr(0.0,
	                    [](float a, float b) { return std::max(a, b); },
	                    MPI_MAX,
	                    1.0,
	                    [](float a, float b) { return a * b; });
	auto A = new Matrix<float>(n, n, *w, tsr);
	
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
	auto fill = new float[m2];
	for (int i = 0; i < m2; i++) {
		fill[i] = PLACE_VERTEX;
	}
	A->write(m2, idx, fill);
	
	free(idx);
	free(fill);
	return A;
}

Vector<float>* connectivity(Matrix<float> A) {
	assert(A.nrow == A.ncol);
	int n = A.nrow;
	
	Semiring<float> tsr(0.0,
	                    [](float a, float b) { return std::max(a, b); },
	                    MPI_MAX,
	                    1.0,
	                    [](float a, float b) { return a * b; });
	auto w = new Vector<float>(n, *A.wrld, tsr);
	auto idx = new int64_t[n];
	auto fill = new float[n];
	for (int i = 0; i < n; i++) {
		idx[i] = i;
		fill[i] = i;
	}
	w->write(n, idx, fill);
	cout << "w setup: " << endl;
	w->print();
	
	// update adj mat
	for (int i = 0; i < n; i++) {
		(*w)["j"] += A["jk"] * (*w)["k"];
//		w->print();
	}
	
	return w;
}
