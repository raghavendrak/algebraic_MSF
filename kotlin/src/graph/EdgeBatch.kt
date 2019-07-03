package graph

import algebra.INT_MAX_TIMES_SEMIRING
import algebra.Matrix
import algebra.Vector
import util.length

private val SR = INT_MAX_TIMES_SEMIRING

fun main() {
	val G = Graph(8, listOf(1 to 2, 3 to 4, 4 to 5, 5 to 6, 6 to 7, 7 to 8))

	G.edgeBatch().prettyPrintln()
}

fun Graph.edgeBatch(A: Matrix<Vertex>
                    = adjacencyMatrix(), // assuming `A` is of size 2^n
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
		val topLeft = edgeBatch(A[topLeftRowRange, topLeftColRange,
				false], // pass in `false` to avoid matrix reshaping
				topLeftRowRange, topLeftColRange)
				.shortcut() // `shortcut` => p[i] = p[p[i]]

		val btmRightRowRange = rowMid + 1..rowEnd
		val btmRightColRange = colMid + 1..colEnd
		val btmRight = edgeBatch(A[btmRightRowRange, btmRightColRange, false],
				btmRightRowRange, btmRightColRange).shortcut()

		val topRightRowRange = rowStart..rowMid
		val topRightColRange = colMid + 1..colEnd
		val topRight = edgeBatch(A[topRightRowRange, topRightColRange, false],
				topRightRowRange, topRightColRange).shortcut()
		// } until all finished

		return (topLeft + topRight + btmRight).shortcut() // max operation
	}
}

