package graph

import algebra.INT_MAX_TIMES_SEMIRING
import algebra.Matrix
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
	G.recursiveAnchor().prettyPrintln(true)
}

fun Graph.recursiveAnchor(batchFactor: Int = max(2.0, getAvgDeg()).toInt())
		: Vector<Vertex> {
	val A = adjacencyMatrix(INT_MAX_TIMES_SEMIRING).upperTriangular()

	val P = A.getParentMatrix()
	val B = P.transpose() * A
	(1..numVertices).forEach { i ->
		(i + 1..numVertices).forEach { j ->
			A[i, j] = max(A[i, j], B[i, j], B[j, i])
		}
	}
	val pi = A.getParentVector().shortcut()

	// <supervertex, corresponding children as set>
	val superPairs = vertices
			.map { u -> u to pi.asSequence().filter { pi[it] == u }.toSet() }
			.filter { (_, children) -> children.size >= batchFactor }
			.toMap()
	if (superPairs.isEmpty()) {
		return pi
	} else {
		val isSupervertex = { v: Vertex -> superPairs.keys.contains(v) }
		val parentIsSupervertex = { v: Vertex -> isSupervertex(pi[v]) }

		val compressed = Graph(numVertices, edges
				.map { (v1, v2) ->
					val u = if (parentIsSupervertex(v1)) pi[v1] else v1
					val v = if (parentIsSupervertex(v2)) pi[v2] else v2
					u to v
				}
				.filterNot { (u, v) -> u == v } // ignore self-loops
				.toList()
		)
		val superPi = compressed.recursiveAnchor(batchFactor)

		return intVector(numVertices) { v ->
			// restore parent for non-supervertices
			if (isSupervertex(v)) superPi[v] else superPi[pi[v]]
		}.shortcut()
	}
}

fun Graph.getAvgDeg() = (2.0 * numEdges) / numVertices

fun Matrix<Vertex>.getParentMatrix() = getParentVector().toParentMatrix()

fun Vector<Vertex>.shortcut() = intVector(length, semiring) { this[this[it]] }
