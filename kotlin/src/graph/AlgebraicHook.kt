package graph

import algebra.INT_MAX_TIMES_SEMIRING
import algebra.Vector
import algebra.intVector
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

	G.algebraicHook().prettyPrintln()
}

fun Graph.algebraicHook(): Vector<Vertex> {
	val SR = INT_MAX_TIMES_SEMIRING
	val A = adjacencyMatrix(SR)

	var p = verticesVector(SR)
	var prev: Vector<Vertex>? = null
	while (p != prev) {
		prev = p
		val q = A * p
		val r = intVector(numVertices, SR) { i -> max(p[i], q[i]) }
		val P = p.toParentMatrix()
		val s = P.transpose() * r
		p = intVector(numVertices, SR) { i -> max(p[i], s.toVector()[i]) }
				.shortcut()
	}
	return p
}
