package linalg

import util.INF
import util.N_INF
import util.max
import util.min

class Semiring<T>(val addIdentity: T,
                  val multIdentity: T,
                  val addOp: (T, T) -> T,
                  val multOp: (T, T) -> T)

val INT_DEFAULT_SEMIRING = Semiring(0, 1,
		{ i1, i2 -> min(INF, i1 + i2) }, { i1, i2 -> min(INF, i1 * i2) })
val INT_TROPICAL_SEMIRING_MIN = Semiring(INF, 0,
		{ i1, i2 -> min(i1, i2) }, { i1, i2 -> i1 + i2 })
val INT_TROPICAL_SEMIRING_MAX = Semiring(N_INF, 0,
		{ i1, i2 -> max(i1, i2) }, { i1, i2 -> i1 + i2 })
