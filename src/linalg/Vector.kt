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
	val indices = rowIndices

	constructor(array: Array<out T>, semiring: Semiring<T>) :
			this(array.size, semiring, { array[it - 1] })

	operator fun get(indexRange: IntRange) =
			super.get(indexRange, colIndices) as Vector<T>

	operator fun get(index: Int) = super.get(index, 1)

	operator fun set(index: Int, value: T) = super.set(index, numCols, value)

	operator fun set(indexRange: IntRange, values: Vector<T>) =
			super.set(indexRange, colIndices, values)

	override fun prettyPrint(printIndex: Boolean) {
		if (!printIndex) {
			println(Arrays.toString(arrays.first()))
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

	operator fun plus(v: Vector<T>): Vector<T> {
		if (length != v.length) {
			throw IllegalArgumentException("inconsistent length")
		}

		if (semiring != v.semiring) {
			throw IllegalArgumentException("inconsistent semiring")
		}

		return Vector(length, semiring) { this[it] + v[it] }
	}

	infix fun inner(v: Vector<T>): T {
		if (length != v.length) {
			throw IllegalArgumentException("inconsistent length")
		}

		if (semiring != v.semiring) {
			throw IllegalArgumentException("inconsistent semiring")
		}

		var acc = semiring.addIdentity
		indices.forEach { acc += this[it] * v[it] }

		return acc
	}

	infix fun outer(v: Vector<T>): Matrix<T> {
		if (length != v.length) {
			throw IllegalArgumentException("inconsistent length")
		}

		if (semiring != v.semiring) {
			throw IllegalArgumentException("inconsistent semiring")
		}

		return Matrix(length by length, semiring) { r, c -> this[r] * v[c] }
	}
}

fun intVector(length: Int,
              semiring: Semiring<Int> = INT_DEFAULT_SEMIRING,
              init: (Int) -> Int = { INT_DEFAULT_SEMIRING.addIdentity }) =
		Vector(length, semiring, init)

operator fun Vector<Int>.times(scalar: Int) =
		Vector(length, semiring) { scalar * this[it] }

operator fun Int.times(v: Vector<Int>) = v * this

fun <T> Array<T>.toVector(semiring: Semiring<T>) = Vector(this, semiring)

fun Array<Int>.toVector(semiring: Semiring<Int> = INT_DEFAULT_SEMIRING) =
		Vector(this, semiring)
