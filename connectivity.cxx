#include <ctf.hpp>
#include <utility>
#include <cfloat>

#define PLACE_VERTEX (1.0)

using namespace CTF;

class IntPair {
public:
	int64_t i1;
	int64_t i2;
	
	IntPair(int64_t i1, int64_t i2);
};

class algo.Graph {
public:
	/**
	 * number of vertices
	 */
	int n;
	vector<IntPair> edges;
	
	/**
	 * @param n number of vertices
	 * @param edges list of edges as a pair of two 0-indexed vertices
	 */
	algo.Graph(int n, vector<IntPair> edges);
	
	Matrix<float>* adj_mat(World* w, bool sparse = false);
};

Vector<float>* connectivity(Matrix<float> A);

static Semiring<float> TSR(0.0,
                           [](float a, float b) { return std::util.util.max(a, b); },
                           MPI_MAX,
                           1.0,
                           [](float a, float b) { return a * b; });

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
	auto g = algo.Graph(6, edges);
	auto B = g.adj_mat(&w);
	B->print_matrix();
	connectivity(*B)->print();
	free(B);
}

IntPair::IntPair(int64_t i1, int64_t i2) {
	this->i1 = i1;
	this->i2 = i2;
}

algo.Graph::algo.Graph(int n, vector<IntPair> edges) {
	this->n = n;
	this->edges = std::move(edges);
}

Matrix<float>* algo.Graph::adj_mat(World* w, bool sparse) {
	auto attr = 0;
	if (sparse) {
		attr = SP;
	}
	auto A = new Matrix<float>(n, n, attr, *w, TSR);
	
	auto m = (int64_t) edges.size();
	auto m2 = m * 2;
	auto idx = new int64_t[m2];
	for (int i = 0; i < m; i++) {
		auto edge = edges[i];
		idx[i] = edge.i1 * n + edge.i2;
		idx[i + m] = edge.i2 * n + edge.i1;
	}
	float fill[m2];
	for (int i = 0; i < m2; i++) {
		fill[i] = PLACE_VERTEX;
	}
	A->write(m2, idx, fill);
	
	free(idx);
	return A;
}

Vector<float>* connectivity(Matrix<float> A) {
	assert(A.nrow == A.ncol);
	int n = A.nrow;
	
	auto w = new Vector<float>(n, *A.wrld, TSR);
	auto idx = new int64_t[n];
	auto fill = new float[n];
	for (int i = 0; i < n; i++) {
		idx[i] = i;
		fill[i] = i;
	}
	w->write(n, idx, fill);
	free(idx);
	free(fill);
	
	// update adj mat
	for (int i = 0; i < n; i++) {
		(*w)["j"] += A["jk"] * (*w)["k"];
	}
	
	return w;
}
