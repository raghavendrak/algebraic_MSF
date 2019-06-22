package graph

import algebra.INT_MAX_TIMES_SEMIRING
import algebra.Vector

fun main() {
	val G = Graph(11, listOf(
			5 to 4,
			4 to 1,
			4 to 3,
			3 to 2,
			11 to 9,
			10 to 9,
			9 to 6,
			9 to 8,
			6 to 8)
	)
	G.connectedComponents().prettyPrintln()
}

fun Graph.connectedComponents(): Vector<Vertex> {
	val A = adjacencyMatrix(INT_MAX_TIMES_SEMIRING)

	// goal: w[i] = the largest label in the same component with vertex i
	// initialized s.t. each vertex is in its own component, i.e. w[i] = i
	var w = verticesVector(INT_MAX_TIMES_SEMIRING)

	var w_prev: Vector<Vertex>? = null
	while (w != w_prev) {
		w_prev = w
		w += A * w
	}

	return w
}
