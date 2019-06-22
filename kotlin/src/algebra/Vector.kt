package algebra

import util.times
import java.util.*

/**
 * Vector = Matrix with `length` rows and exactly 1 column
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

	operator fun set(index: Int, value: T) = super.set(index, 1, value)

	operator fun set(indexRange: IntRange, values: Vector<T>) =
			super.set(indexRange, colIndices, values)

	override fun prettyPrintln(printIndex: Boolean) {
		if (length == 0) {
			println("[]")
			return
		}

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

	operator fun plus(v: Vector<T>) = when {
		length != v.length ->
			throw IllegalArgumentException("Inconsistent length.")
		semiring != v.semiring ->
			throw IllegalArgumentException("Inconsistent semiring.")
		else -> Vector(length, semiring) { this[it] + v[it] }
	}


	infix fun inner(v: Vector<T>) = when {
		length != v.length ->
			throw IllegalArgumentException("Inconsistent length.")
		semiring != v.semiring ->
			throw IllegalArgumentException("Inconsistent semiring.")
		else -> {
			var acc = semiring.additiveIdentity
			indices.forEach { acc += this[it] * v[it] }
			acc
		}
	}

	infix fun outer(v: Vector<T>) = when {
		length != v.length ->
			throw IllegalArgumentException("Inconsistent length.")
		semiring != v.semiring ->
			throw IllegalArgumentException("Inconsistent semiring.")
		else -> Matrix(length by length, semiring) { r, c -> this[r] * v[c] }
	}

	override fun copy() = Vector(length, semiring) { this[it] }

	fun asDiagonal(empty: T = semiring.additiveIdentity) =
			Matrix(length by length, semiring) { r, c ->
				if (r == c) this[r] else empty
			}
}

fun intVector(length: Int,
              semiring: Semiring<Int> = INT_DEFAULT_SEMIRING,
              init: (Int) -> Int = { semiring.additiveIdentity }) =
		Vector(length, semiring, init)

fun intVector(length: Int,
              semiring: Semiring<Int> = INT_DEFAULT_SEMIRING,
              initValue: Int) = intVector(length, semiring) { initValue }

operator fun Vector<Int>.times(scalar: Int) =
		Vector(length, semiring) { scalar * this[it] }

operator fun Int.times(v: Vector<Int>) = v * this

fun <T> Array<T>.toVector(semiring: Semiring<T>) = Vector(this, semiring)

fun Array<Int>.toVector(semiring: Semiring<Int> = INT_DEFAULT_SEMIRING) =
		Vector(this, semiring)

