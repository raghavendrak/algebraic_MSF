package graph

import algebra.Vector
import algebra.intVector
import util.max

fun main(args: Array<String>) {
	val G = Graph(18, listOf(
			1 with 2,
			2 with 3,
			3 with 4,
			3 with 11,
			4 with 11,
			16 with 17,
			2 with 12,
			2 with 5,
			5 with 6,
			14 with 15,
			6 with 13,
			13 with 14,
			10 with 9,
			8 with 9,
			7 with 8))

	G.hook().prettyPrintln(true)
}

fun Graph.hook(): Vector<Vertex> {
	// P-arent vector
	// each vertex is initially hooked to the largest among neighbors and itself
	val P = intVector(numVertices) { it }
	edges.forEach { (v1, v2) ->
		if (v1 > v2) P[v2] = max(P[v2], v1)
		else P[v1] = max(P[v1], v2)
	}

	val isRoot = { v: Vertex -> P[v] == v }

	val rootOf = { v: Vertex ->
		// union-find like
		// collapse pathToRoot on the fly
		val pathToRoot = hashSetOf(v)
		var current = v
		while (!isRoot(current)) {
			pathToRoot.add(current)
			current = P[current]
		}
		pathToRoot.forEach { P[it] = current }
		current
	}

	val collapse = { v: Vertex, root: Vertex ->
		val pathToRoot = hashSetOf(v)
		var current = v
		while (!isRoot(current)) {
			pathToRoot.add(current)
			current = P[current]
		}
		pathToRoot.forEach { P[it] = root }
	}

	val A = adjacencyMatrix()
	val isEdge = { v1: Vertex, v2: Vertex -> A[v1, v2] == 1 }

	vertices.forEach { v1 ->
		vertices.forEach { v2 ->
			val v1Root = rootOf(v1)
			val v2Root = rootOf(v2)

			if (isEdge(v1, v2)) {
				if (v1Root > v2Root) collapse(v2, v1Root)
				if (v1Root < v2Root) collapse(v1, v2Root)
			}
		}
	}

	return P
}
