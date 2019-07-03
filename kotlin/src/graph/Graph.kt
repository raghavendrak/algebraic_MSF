package graph

import algebra.*

typealias Vertex = Int
typealias Edge = Pair<Vertex, Vertex>

infix fun Int.with(v: Vertex) = this to v

// 1-indexed vertices
data class Graph(val numVertices: Int = 0, val edges: List<Edge> = emptyList()) {
	val numEdges = edges.size
	val vertices = 1..numVertices

	fun adjacencyMatrix(semiring: Semiring<Vertex> = INT_DEFAULT_SEMIRING,
	                    noEdge: Vertex = semiring.addId,
	                    placeEdge: Vertex = semiring.multId): Matrix<Vertex> {
		val A = Matrix(numVertices by numVertices, semiring) { _, _ -> noEdge }
		edges.forEach { (v1, v2) ->
			A[v1, v2] = placeEdge
			A[v2, v1] = placeEdge
		}
		return A
	}

	fun verticesVector(semiring: Semiring<Vertex> = INT_DEFAULT_SEMIRING) =
			intVector(numVertices, semiring) { it }
}
