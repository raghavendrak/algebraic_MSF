package algebra

import util.*
import java.util.*
import kotlin.math.max

typealias Shape = Pair<Int, Int>

infix fun Int.by(numCols: Int) = this to numCols

/**
 * 1-indexed
 */
open class Matrix<T>(val shape: Shape,
                     var semiring: Semiring<T>,
                     init: (Int, Int) -> T) {
	val numRows = shape.first
	val numCols = shape.second
	val rowIndices = 1..numRows
	val colIndices = 1..numCols

	/**
	 * COLUMN-dominant 2d arrays are used internally, i.e. arrays[col][row]
	 * however public getter/setter interfaces are still ROW-dominant,
	 * i.e. M[row, col]. This way achieves better performance for Vector as
	 * it represents a matrix w/ a single column and many rows (space locality)
	 */
	protected val arrays =
			Array(numCols) { arrayOfNulls<Any?>(numRows) } as Array<Array<T>>

	init {
		rowIndices.forEach { r ->
			colIndices.forEach { c ->
				this[r, c] = init(r, c)
			}
		}
	}

	constructor(array: Array<Array<T>>,
	            semiring: Semiring<T>) : this(
			array.size by array.first().size,
			semiring, { r, c -> array[r - 1, c - 1] })

	operator fun get(rowIdx: Int, colRange: IntRange) =
			this[rowIdx..rowIdx, colRange].toVector()

	operator fun get(rowRange: IntRange, colIdx: Int) =
			this[rowRange, colIdx..colIdx].toVector()

	operator fun get(rowRange: IntRange,
	                 colRange: IntRange,
	                 reshape: Boolean = true): Matrix<T> {
		return if (reshape) {
			val rowStart = rowRange.start
			val colStart = colRange.start

			Matrix(rowRange.length by colRange.length, semiring) { r, c ->
				this[rowStart + r - 1, colStart + c - 1]
			}
		} else {
			// "zero" out other entries not in range
			Matrix(numRows by numCols, semiring) { r, c ->
				if (r in rowRange && c in colRange) {
					this[r, c]
				} else {
					semiring.addId
				}
			}
		}
	}

	operator fun get(r: Int, c: Int) = arrays[c - 1, r - 1]

	operator fun set(rowIdx: Int, colRange: IntRange, vals: Vector<T>) {
		this[rowIdx..rowIdx, colRange] = vals.transpose()
	}

	operator fun set(rowRange: IntRange, colIdx: Int, vals: Vector<T>) {
		this[rowRange, colIdx..colIdx] = vals
	}

	operator fun set(rowRange: IntRange, colRange: IntRange, vals: Matrix<T>) {
		val rowStart = rowRange.start
		val colStart = colRange.start

		if (rowRange.length by colRange.length != vals.shape) {
			throw IllegalArgumentException(
					"Inconsistent shape between range and values."
			)
		}

		rowRange.forEach { r ->
			colRange.forEach { c ->
				this[r, c] = vals[r - rowStart + 1, c - colStart + 1]
			}
		}
	}

	operator fun set(r: Int, c: Int, v: T) {
		arrays[c - 1, r - 1] = v
	}

	open fun prettyPrintln(printIndex: Boolean = true) {
		if (numCols == 0 || numRows == 0) {
			println("[]")
			return
		}

		var maxLenEle = numCols.toString().length
		rowIndices.forEach { r ->
			colIndices.forEach { c ->
				maxLenEle = max(maxLenEle, this[r, c].toString().length)
			}
		}

		if (printIndex) {
			print(" " * (numRows.toString().length + 2))
			colIndices.forEach { c ->
				print(c)
				print(" " * (maxLenEle - c.toString().length + 1))
			}
			println()
		}

		rowIndices.forEach { r ->
			if (printIndex) {
				print(" " * (numRows.toString().length - r.toString().length))
				print("$r ")
			}
			print("[")
			colIndices.forEach { c ->
				print(this[r, c])
				print(" " * (maxLenEle - this[r, c].toString().length))
				if (c == numCols) {
					println("]")
				} else {
					print(" ")
				}
			}
		}
	}

	override fun equals(other: Any?): Boolean {
		if (this === other) {
			return true
		}
		if (other !is Matrix<*>) {
			return false
		}

		if (shape != other.shape) {
			return false
		}
		if (semiring != other.semiring) {
			return false
		}
		if (!Arrays.deepEquals(arrays, other.arrays)) {
			return false
		}

		return true
	}

	override fun hashCode(): Int {
		var result = shape.hashCode()
		result = 31 * result + semiring.hashCode()
		result = 31 * result + Arrays.deepHashCode(arrays)
		return result
	}

	operator fun plus(m: Matrix<T>) = when {
		shape != m.shape ->
			throw IllegalArgumentException("Inconsistent shape.")
		semiring != m.semiring ->
			throw IllegalArgumentException("Inconsistent semiring.")
		else -> Matrix(shape, semiring) { r, c -> this[r, c] + m[r, c] }
	}

	operator fun times(v: Vector<T>) = when {
		numCols != v.length ->
			throw IllegalArgumentException("Inconsistent shape.")
		semiring != v.semiring ->
			throw IllegalArgumentException("Inconsistent semiring.")
		else -> Vector(v.length, semiring) { r ->
			var acc = semiring.addId
			colIndices.forEach { acc += this[r, it] * v[it] }
			acc
		}
	}

	operator fun times(m: Matrix<T>) = when {
		numCols != m.numRows ->
			throw IllegalArgumentException("Inconsistent shape.")
		semiring != m.semiring ->
			throw IllegalArgumentException("Inconsistent semiring.")
		else -> Matrix(numRows by m.numCols, semiring) { r, c ->
			var acc = semiring.addId
			colIndices.forEach { acc += this[r, it] * m[it, c] }
			acc
		}
	}

	operator fun T.plus(t: T) = semiring.addOp(this, t)

	operator fun T.times(t: T) = semiring.multOp(this, t)

	fun toVector() = when {
		numRows == 1 -> Vector(numCols, semiring) { this[1, it] }
		numCols == 1 -> Vector(numRows, semiring) { this[it, 1] }
		else -> throw IllegalArgumentException("Not a vector.")
	}

	fun transpose() = Matrix(numCols by numRows, semiring) { r, c ->
		this[c, r]
	}

	open fun copy() = Matrix(numRows by numCols, semiring) { r, c ->
		this[r, c]
	}

	fun asSequence() = arrays.asSequence().flatMap { it.asSequence() }

	fun upperTriangular(empty: T = semiring.addId) =
			Matrix(numRows by numCols, semiring) { r, c ->
				if (r <= c) this[r, c] else empty
			}

	fun lowerTriangular(empty: T = semiring.addId) =
			Matrix(numRows by numCols, semiring) { r, c ->
				if (r >= c) this[r, c] else empty
			}

	fun diagonalVector() =
			Vector(min(numRows, numCols), semiring) { this[it, it] }

	fun diagonalMatrix(empty: T = semiring.addId) =
			Matrix(numRows by numCols, semiring) { r, c ->
				if (r == c) this[r, c] else empty
			}

	override fun toString(): String = Arrays.deepToString(arrays)

}

fun intMatrix(shape: Shape,
              semiring: Semiring<Int> = INT_DEFAULT_SEMIRING,
              init: (Int, Int) -> Int = { _, _ -> semiring.addId }) =
		Matrix(shape, semiring, init)

fun intMatrix(shape: Shape,
              semiring: Semiring<Int> = INT_DEFAULT_SEMIRING,
              initValue: Int) = intMatrix(shape, semiring) { _, _ -> initValue }

fun <T> identityMatrix(size: Int,
                       semiring: Semiring<T>,
                       diagonal: T = semiring.multId,
                       nondiagonal: T = semiring.addId) =
		Matrix(size by size, semiring) { r, c ->
			if (r == c) diagonal else nondiagonal
		}

fun identityIntMatrix(size: Int,
                      semiring: Semiring<Int> = INT_DEFAULT_SEMIRING,
                      diagonal: Int = semiring.multId,
                      nondiagonal: Int = semiring.addId) =
		identityMatrix(size, semiring, diagonal, nondiagonal)

operator fun Matrix<Int>.times(scalar: Int) =
		Matrix(shape, semiring) { r, c -> scalar * this[r, c] }

operator fun Matrix<Int>.minus(other: Matrix<Int>) =
		Matrix(shape, semiring) { r, c ->
			this[r, c] - other[r, c]
		}

operator fun Int.times(m: Matrix<Int>) = m * this

fun <T> Array<Array<T>>.toMatrix(semiring: Semiring<T>) = Matrix(this, semiring)

fun Array<Array<Int>>.toMatrix(semiring: Semiring<Int> = INT_DEFAULT_SEMIRING) =
		Matrix(this, semiring)
