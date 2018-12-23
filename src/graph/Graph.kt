package graph

import linalg.by
import linalg.intMatrix

// 1-indexed vertices
class Graph(val numVertices: Int, val edges: List<Pair<Int, Int>>) {
	val vertices = 1..numVertices
	val adjMat = intMatrix(numVertices by numVertices)

	init {
		edges.forEach { (r, c) ->
			adjMat[r, c] = 1
			adjMat[c, r] = 1
		}
	}
}
