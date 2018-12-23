package linalg

class Shape(val numRows: Int, val numCols: Int) {

	operator fun component1() = numRows

	operator fun component2() = numCols

	override fun toString(): String {
		return "($numRows by $numCols)"
	}

	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (other !is Shape) return false

		if (numRows != other.numRows) return false
		if (numCols != other.numCols) return false

		return true
	}

	override fun hashCode(): Int {
		var result = numRows
		result = 31 * result + numCols
		return result
	}
}

infix fun Int.by(numCols: Int) = Shape(this, numCols)
