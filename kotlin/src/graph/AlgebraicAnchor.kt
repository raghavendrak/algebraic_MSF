package graph

import algebra.INT_MAX_TIMES_SEMIRING
import algebra.Matrix
import algebra.Vector
import algebra.identityIntMatrix

fun main() {
	val G = Graph(6, listOf(
			1 to 2,
			3 to 5,
			5 to 4,
			4 to 6)
	)
	G.algebraicAnchor().prettyPrintln(true)
}

// more algebraic operations, fewer loops
fun Graph.algebraicAnchor(): Vector<Vertex> {
	val v = verticesVector(INT_MAX_TIMES_SEMIRING)
	val A = adjacencyMatrix(INT_MAX_TIMES_SEMIRING) // symmetric
	val I = identityIntMatrix(numVertices, INT_MAX_TIMES_SEMIRING)

	var P = ((A + I) * v).toParentMatrix()
	var Prev: Matrix<Int>? = null
	while (P != Prev) {
		Prev = P
		P = ((A + I) * v).toParentMatrix()
		val B = P.transpose() * A * P
		P *= B.upperTriangular()
		P *= P
	}

	return (P + I) * v
}
