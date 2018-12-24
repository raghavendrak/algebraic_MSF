package util

fun <T : Comparable<T>> max(vararg ts: T) = ts.max()
		?: throw NullPointerException("no max value")

fun <T : Comparable<T>> min(vararg ts: T) = ts.min()
		?: throw NullPointerException("no min value")

const val INF = Int.MAX_VALUE / 2
const val N_INF = Int.MIN_VALUE / 2
