package graph

import algebra.*
import util.length

private val SR = INT_MAX_TIMES_SEMIRING

fun main() {
	val G = Graph(6, listOf(
			1 to 2,
			3 to 4,
			4 to 5,
			4 to 6)
	)

//	G.edgeBatchRec().prettyPrintln()
	G.edgeBatchRand()
}

fun Graph.edgeBatchRec(A: Matrix<Vertex> = adjacencyMatrix(),
                       rowRange: IntRange = 1..numVertices,
                       colRange: IntRange = 1..numVertices): Vector<Vertex> {
	val (rowStart, rowEnd) = rowRange.start to rowRange.endInclusive
	val (colStart, colEnd) = colRange.start to colRange.endInclusive

	if (rowRange.length == 1 && colRange.length == 1) {
		return Vector(numVertices, SR) {
			if (it == rowStart && A[rowStart, colStart] == SR.multId) colStart
			else it
		}
	} else {
		val rowMid = rowStart + (rowEnd - rowStart) / 2
		val colMid = colStart + (colEnd - colStart) / 2

		// do in parallel {
		val topLeftRowRange = rowStart..rowMid
		val topLeftColRange = colStart..colMid
		val topLeft = edgeBatchRec(A[topLeftRowRange, topLeftColRange,
				false], // pass in `false` to avoid matrix reshaping
				topLeftRowRange, topLeftColRange)
				.shortcut() // `shortcut` => p[i] = p[p[i]]

		val btmRightRowRange = rowMid + 1..rowEnd
		val btmRightColRange = colMid + 1..colEnd
		val btmRight = edgeBatchRec(A[btmRightRowRange, btmRightColRange, false],
				btmRightRowRange, btmRightColRange).shortcut()

		val topRightRowRange = rowStart..rowMid
		val topRightColRange = colMid + 1..colEnd
		val topRight = edgeBatchRec(A[topRightRowRange, topRightColRange, false],
				topRightRowRange, topRightColRange).shortcut()
		// } until all finished

		return (topLeft + topRight + btmRight).shortcut() // max operation
	}
}

fun Graph.edgeBatchRand(): Vector<Vertex> {
	val EDGE_EACH_BATCH = 1
	// shuffle edges if we truly want randomness
	val batches = edges.indices.groupBy { it / EDGE_EACH_BATCH }
			.map { it.value.map { edges[it] } }
	var A = intMatrix(numVertices by numVertices, SR)
	var p = verticesVector(SR) // p[i] = i
	for (batch in batches) {
		// add edges in this batch to current A
		for ((u, v) in batch) {
			A[u, v] = 1
			A[v, u] = 1
		}
		// run connectivity for current batch
		var prev: Vector<Vertex>? = null
		while (p != prev) {
			prev = p
			p += A * p
		}
		// shortcut convergence
		prev = null
		while (p != prev) {
			prev = p
			p = p.shortcut()
		}
		p.prettyPrintln()
		// update supervertex for current batch
		val P = p.toParentMatrix()
		A = P.transpose() * A * P
	}
	return p
}
