package linalg

import util.get
import util.set
import util.times
import java.util.*
import kotlin.math.max

typealias Shape = Pair<Int, Int>

/**
 * 1-indexed
 */
open class Matrix<T>(val shape: Shape,
                     val semiring: Semiring<T>,
                     init: (Int, Int) -> T) {
	val numRows = shape.first
	val numCols = shape.second
	val rows = 1..numRows
	val cols = 1..numCols

	/**
	 * COLUMN-dominant 2d arrays are used internally, i.e. array[col][row]
	 * however public getter/setter interfaces are still ROW-dominant,
	 * i.e. M[row, col]. This mechanism achieve better performance for Vector as
	 * it represents a (length by 1) matrix, i.e. a single column with many rows
	 */
	protected val array =
			Array(numCols) { arrayOfNulls<Any?>(numRows) } as Array<Array<T>>

	init {
		rows.forEach { r ->
			cols.forEach { c ->
				this[r, c] = init(r, c)
			}
		}
	}

	constructor(array: Array<Array<T>>,
	            semiring: Semiring<T>) : this(
			array.size by array[0].size,
			semiring, { r, c -> array[r - 1, c - 1] })

	operator fun get(rowRange: IntRange, colRange: IntRange): Matrix<T> {
		val rowStart = rowRange.start
		val colStart = colRange.start
		val rows = rowRange.last - rowStart + 1
		val cols = colRange.last - colStart + 1
		return Matrix(rows by cols, semiring) { r, c ->
			this[rowStart + r - 1, colStart + c - 1]
		}
	}

	operator fun get(r: Int, c: Int) = array[c - 1, r - 1]

	operator fun set(rowRange: IntRange, colRange: IntRange, vals: Matrix<T>) {
		val rowStart = rowRange.start
		val colStart = colRange.start
		rowRange.forEach { r ->
			colRange.forEach { c ->
				this[r, c] = vals[r - rowStart + 1, c - colStart + 1]
			}
		}
	}

	operator fun set(r: Int, c: Int, v: T) {
		array[c - 1, r - 1] = v
	}

	fun prettyPrint(printIndex: Boolean = false) {
		var maxLenEle = numCols.toString().length
		rows.forEach { r ->
			cols.forEach { c ->
				maxLenEle = max(maxLenEle, this[r, c].toString().length)
			}
		}

		if (printIndex) {
			print(" " * (numRows.toString().length + 2))
			cols.forEach { c ->
				print(c)
				print(" " * (maxLenEle - c.toString().length + 1))
			}
			println()
		}

		rows.forEach { r ->
			if (printIndex) {
				print(" " * (numRows.toString().length - r.toString().length))
				print(r)
				print(" ")
			}
			print("[")
			cols.forEach { c ->
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
		if (this === other) return true
		if (other !is Matrix<*>) return false

		if (shape != other.shape) return false
		if (semiring != other.semiring) return false
		if (!Arrays.deepEquals(array, other.array)) return false

		return true
	}

	override fun hashCode(): Int {
		var result = shape.hashCode()
		result = 31 * result + semiring.hashCode()
		result = 31 * result + Arrays.deepHashCode(array)
		return result
	}

	operator fun plus(m: Matrix<T>): Matrix<T> {
		if (shape != m.shape) {
			throw IllegalArgumentException("inconsistent shape")
		}

		if (semiring != m.semiring) {
			throw IllegalArgumentException("inconsistent semiring")
		}

		return Matrix(shape, semiring) { r, c ->
			semiring.addOp(this[r, c], m[r, c])
		}
	}

	operator fun times(v: Vector<T>): Vector<T> {
		if (numCols != v.length) {
			throw IllegalArgumentException("inconsistent length")
		}

		if (semiring != v.semiring) {
			throw IllegalArgumentException("inconsistent semiring")
		}

		return Vector(v.length, semiring) { r ->
			var acc = semiring.addIdentity
			cols.forEach { c ->
				acc = semiring.addOp(acc, semiring.multOp(this[r, c], v[r]))
			}
			acc
		}
	}
}

infix fun Int.by(numCols: Int) = this to numCols

fun intMatrix(shape: Shape,
              semiring: Semiring<Int> = INT_DEFAULT_SEMIRING,
              init: (Int, Int) -> Int = { _, _ -> 0 }) =
		Matrix(shape, semiring, init)

fun intMatrixIdentity(size: Int,
                      semiring: Semiring<Int> = INT_DEFAULT_SEMIRING) =
		Matrix(size by size, semiring) { r, c -> if (r == c) 1 else 0 }

operator fun Matrix<Int>.times(scalar: Int) = Matrix(shape, semiring) { r, c ->
	semiring.multOp(scalar, this[r, c])
}

operator fun Int.times(m: Matrix<Int>) = m * this

fun <T> Array<Array<T>>.toMatrix(semiring: Semiring<T>) = Matrix(this, semiring)

fun Array<Array<Int>>.toMatrix(semiring: Semiring<Int> = INT_DEFAULT_SEMIRING) =
		Matrix(this, semiring)

