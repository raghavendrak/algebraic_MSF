package graph

import algebra.INT_MIN_PLUS_SEMIRING
import algebra.Vector
import algebra.intVector
import util.INF
import util.min

fun main(args: Array<String>) {
	val G = Graph(4, listOf(1 with 2, 1 with 3, 1 with 4, 3 with 4))

	G.bfs(2).prettyPrintln()
}

fun Graph.bfs(startVertex: Vertex): Vector<Vertex> {
	// custom adj mat equipped w/ tropical semiring
	val A = adjacencyMatrix(INT_MIN_PLUS_SEMIRING, placeEdge = 1)

	val frontier = intVector(numVertices, INT_MIN_PLUS_SEMIRING)
	frontier[startVertex] = 0

	val dist = frontier.copy()

	// # of iterations, irrelevant to actual vertices
	(1 until numVertices).forEach {
		val tmp = A * frontier // mat-vec mult defined on tropical semiring
		vertices.forEach { v ->
			frontier[v] = if (dist[v] != INF) INF else tmp[v]
			dist[v] = min(dist[v], frontier[v])
		}
	}

	return dist
}
