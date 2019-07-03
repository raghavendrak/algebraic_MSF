package util

fun <T : Comparable<T>> max(vararg ts: T) = ts.max()
		?: throw NoSuchElementException("no max value")

fun <T : Comparable<T>> min(vararg ts: T) = ts.min()
		?: throw NoSuchElementException("no min value")

const val INF = Int.MAX_VALUE / 2

val IntRange.length: Int
	get() = last - first + 1

