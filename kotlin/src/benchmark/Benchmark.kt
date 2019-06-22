package benchmark

import graph.*


fun main() {
	testCompleteGraph()
}

fun testCompleteGraph() {
	val N = 100
	val G = Graph(N, (1..N).flatMap { u ->
		(u + 1..N).map { v -> u to v }
	})
	benchmark(G)
}

fun testIsolatedGraph() {
	val N = 200
	val G = Graph(N)
	benchmark(G)
}

fun testSparseGraph() {
	val N = 200
	val G = Graph(N)
	benchmark(G)
}

fun benchmark(G: Graph) {
	val durationAlgebraicAnchor = { G.algebraicAnchor() }.duration()
	println("AlgebraicAnchor: $durationAlgebraicAnchor")

	val durationRecursiveAnchor = { G.recursiveAnchor() }.duration()
	println("RecursiveAnchor: $durationRecursiveAnchor")

	val durationNaive = { G.connectedComponents() }.duration()
	println("Naive: $durationNaive")

	val durationSuperVertex = { G.superVertex() }.duration()
	println("SuperVertex: $durationSuperVertex")
}

/**
 * @return execution time the current method takes in milliseconds
 */
fun <T> (() -> (T)).duration(): Long {
	val start = System.currentTimeMillis()
	this()
	val end = System.currentTimeMillis()
	return end - start
}