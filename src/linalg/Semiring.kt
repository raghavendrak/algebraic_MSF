package linalg

import util.INF
import util.max
import util.min

class Semiring<T>(val infinity: T,
                  val addIdentity: T,
                  val multIdentity: T,
                  val addOp: (T, T) -> T,
                  val multOp: (T, T) -> T)

val INT_DEFAULT_SEMIRING =
		Semiring(INF, 0, 1, { i1, i2 -> i1 + i2 }, { i1, i2 -> i1 * i2 })
val INT_TROPICAL_SEMIRING_MIN =
		Semiring(INF, 0, 1, { i1, i2 -> min(i1, i2) }, { i1, i2 -> i1 + i2 })
val INT_TROPICAL_SEMIRING_MAX =
		Semiring(INF, 0, 1, { i1, i2 -> max(i1, i2) }, { i1, i2 -> i1 + i2 })
