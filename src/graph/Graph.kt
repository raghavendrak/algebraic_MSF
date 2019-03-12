package graph

// 1-indexed vertices
class Graph(val numVertices: Int, val edges: List<Pair<Int, Int>>) {
	val vertices = 1..numVertices
}
