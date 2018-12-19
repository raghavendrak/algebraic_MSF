fun main(args: Array<String>) {
	// an isolated vertex [1] UNION an edge [2, 3] UNION a triangle [4, 5, 6]
	val G = Graph(6, listOf(2 to 3, 4 to 5, 4 to 6, 5 to 6))

	val components = G.connectedComponents()
	println(components)
}

fun Graph.connectedComponents(): Map<Int, List<Int>> {
	// vertex -> parent
	val F = ParentMap()

	// init. s.t. all vertices have a parent
	// isolated vertices and root vertices have parents to themselves
	vertices.forEach { F[it] = it }
	// vertex w/ deg >= 2 has the last vertex in the edge list as its parent
	edges.forEach { (v1, v2) -> F[v1] = v2 }

	// keep looping if any vertex is NOT in a star
	while (vertices.any { !inStar(F, it) }) {
		// cond star hook
		edges
				.filter { (i, j) -> inStar(F, i) && F[i] > F[j] }
				.forEach { (i, j) -> F[F[i]] = F[j] }

		// uncond star hook
		edges
				.filter { (i, j) -> inStar(F, i) && F[i] != F[j] }
				.forEach { (i, j) -> F[F[i]] = F[j] }

		// shortcut
		vertices
				.filter { !inStar(F, it) }
				.forEach { F[it] = F[F[it]] }
	}

	// parent -> list of vertices with the parent
	return F.entries
			.groupBy { it.value }
			.mapValues { pair -> pair.value.map { it.key } }
}

class Graph(val numVertices: Int, val edges: List<Pair<Int, Int>>) {
	val vertices = IntRange(1, numVertices)

	// in star iff. grandparent == parent
	fun inStar(F: ParentMap, v: Int) = F[F[v]] == F[v]
}

class ParentMap : HashMap<Int, Int>() {

	override operator fun get(key: Int) = super.get(key)!!
}