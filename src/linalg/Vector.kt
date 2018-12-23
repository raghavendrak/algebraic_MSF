package linalg

import util.times
import java.util.*

/**
 * 1-indexed
 */
open class Vector<T>(val length: Int,
                     semiring: Semiring<T>,
                     init: (Int) -> T)
	: Matrix<T>(length by 1, semiring, { r, _ -> init(r) }) {
	val indices = rows

	operator fun get(indexRange: IntRange) =
			super.get(indexRange, cols) as Vector<T>

	operator fun get(index: Int) = super.get(index, 1)

	operator fun set(index: Int, value: T) = super.set(index, numCols, value)

	operator fun set(indexRange: IntRange, values: Vector<T>) =
			super.set(indexRange, cols, values)

	fun prettyPrintVector(printIndex: Boolean = false) {
		if (!printIndex) {
			println(Arrays.toString(array[0]))
			return
		}

		print(" ")
		indices.forEach {
			print(it)
			print(" ")
			val lenIdx = it.toString().length
			val lenEle = this[it].toString().length
			if (lenEle > lenIdx) {
				print(" " * (lenEle - lenIdx))
			}
		}
		println()
		print("[")
		indices.forEach {
			print(this[it])
			val lenIdx = it.toString().length
			val lenEle = this[it].toString().length
			if (lenIdx > lenEle) {
				print(" " * (lenIdx - lenEle))
			}
			if (it == length) {
				println("]")
			} else {
				print(" ")
			}
		}
	}
}

fun intVector(length: Int,
              semiring: Semiring<Int> = INT_SEMIRING_DEFAULT,
              init: (Int) -> Int = { 0 }) = Vector(length, semiring, init)

operator fun Vector<Int>.times(scalar: Int) = Vector(length, semiring) {
	semiring.multOp(scalar, this[it])
}

operator fun Int.times(v: Vector<Int>) = v * this
