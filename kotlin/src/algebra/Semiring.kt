package algebra

import util.INF
import util.max
import util.min

class Semiring<T>(val addId: T,
                  val multId: T,
                  val addOp: (T, T) -> T,
                  val multOp: (T, T) -> T)

val INT_DEFAULT_SEMIRING = Semiring(0, 1,
		{ i1, i2 -> min(INF, i1 + i2) }, { i1, i2 -> min(INF, i1 * i2) })
val INT_MIN_PLUS_SEMIRING = Semiring(INF, 0,
		{ i1, i2 -> min(i1, i2) }, { i1, i2 -> min(INF, i1 + i2) })
val INT_MAX_TIMES_SEMIRING = Semiring(0, 1,
		{ i1, i2 -> max(i1, i2) }, { i1, i2 -> min(INF, i1 * i2) })
