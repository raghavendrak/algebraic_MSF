package graph

import algebra.*
import util.max

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

	G.anchor().prettyPrintln(true)
}

fun Graph.anchor(): Vector<Vertex> {
	val A = adjacencyMatrix(INT_MAX_TIMES_SEMIRING).upperTriangular()

	var prev_pi: Vector<Int>? = null
	var (pi, P) = A.getParent()

	while (prev_pi != pi) {
		val B = P.transpose() * A

		(1..numVertices).forEach { i ->
			(i + 1..numVertices).forEach { j ->
				A[i, j] = max(A[i, j], B[i, j], B[j, i])
			}
		}

		prev_pi = pi
		pi = A.getParentVector()
		P = pi.toParentMatrix()
		P *= P
	}

	return pi.shortcut().shortcut()
}

fun Matrix<Vertex>.getParentVector() = intVector(numRows, semiring) { i ->
	max(i, (1..numCols).map { j -> this[i, j] * j }.max()!!)
}

fun Vector<Vertex>.toParentMatrix() =
		intMatrix(length by length, semiring) { i, j ->
			if (this[i] == j) 1 else 0
		}

fun Matrix<Vertex>.getParent() =
		getParentVector().run { this to toParentMatrix() }