package graph


fun main() {
	// an isolated vertex [1] UNION an edge [2, 3] UNION a triangle [4, 5, 6]
	val G = Graph(6, listOf(2 with 3, 4 with 5, 4 with 6, 5 with 6))


	println(G.starHook())
}

fun Graph.starHook(): Map<Vertex, List<Vertex>> {
	// vertex -> parent
	val F = ParentMap()

	// init. s.t. all vertices have a parent
	// isolated vertices and root vertices have parents to themselves
	vertices.forEach { F[it] = it }
	// vertex w/ deg >= 2 has the last vertex in the edge list as its parent
	edges.forEach { (v1, v2) -> F[v1] = v2 }

	// keep looping if any vertex is NOT in a star
	// algo will conv in O(log N) steps
	while (vertices.any { !F.inStar(it) }) {
		// and each step will cost O(N + numEdges)
		// cond star hook
		edges
				.filter { (i, j) -> F.inStar(i) && F[i] > F[j] }
				.forEach { (i, j) -> F[F[i]] = F[j] }

		// uncond star hook
		edges
				.filter { (i, j) -> F.inStar(i) && F[i] != F[j] }
				.forEach { (i, j) -> F[F[i]] = F[j] }

		// shortcut
		vertices
				.filter { !F.inStar(it) }
				.forEach { F[it] = F[F[it]] }
	}

	// parent -> list of vertices with the parent
	return F.entries
			.groupBy { it.value }
			.mapValues { pair -> pair.value.map { it.key } }
}

class ParentMap : HashMap<Vertex, Vertex>() {

	override operator fun get(key: Vertex) = super.get(key)!!

	// in star iff. grandparent == parent
	fun inStar(v: Vertex) = this[this[v]] == this[v]
}