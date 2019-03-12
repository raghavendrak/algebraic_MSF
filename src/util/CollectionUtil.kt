package util

operator fun <T> Array<Array<T>>.get(r: Int, c: Int) = this[r][c]

operator fun <T> Array<Array<T>>.set(r: Int, c: Int, v: T) {
	this[r][c] = v
}

operator fun Array<IntArray>.get(r: Int, c: Int) = this[r][c]

operator fun Array<IntArray>.set(r: Int, c: Int, v: Int) {
	this[r][c] = v
}
