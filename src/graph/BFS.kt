package graph

import linalg.*
import util.INF

fun main(args: Array<String>) {
	val G = Graph(4, listOf(1 to 2, 1 to 3, 1 to 4, 3 to 4))

	G.bfs(2).prettyPrint(true)
}

fun Graph.bfs(s: Int): Vector<Int> {
	val A = intMatrix(numVertices by numVertices, INT_TROPICAL_SEMIRING_MIN, INF)
	edges.forEach { (v1, v2) ->
		A[v1, v2] = 1
		A[v2, v1] = 1
	}
	A.prettyPrint(true)
	println()

	val f = intVector(numVertices, INT_TROPICAL_SEMIRING_MIN, INF)
	f[s] = 0
	var x: Vector<Int>

	(1..numVertices).forEach { _ ->
		x = A * f
		vertices.forEach { f[it] = if (f[it] != INF) INF else x[it] }
		x.prettyPrint(true)
		f.prettyPrint(true)
	}

	TODO()
}
