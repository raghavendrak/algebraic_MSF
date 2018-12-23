package linalg

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import kotlin.test.assertEquals

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MatrixTest {

	@Test
	fun testMatrixSliceGet() {
		val m = intIdentityMatrix(5)
		m.prettyPrint(true)
		println()
		m[1..3, 2..3].prettyPrint(true)
	}

	@Test
	fun testMatrixSliceSet() {
		val m = intIdentityMatrix(3)
		m.prettyPrint(true)
		println()
		m[1..2, 2..3] = intIdentityMatrix(2)
		m.prettyPrint(true)
	}

	@Test
	fun testMatrixAdd() {
		val m1 = intIdentityMatrix(3)
		val m2 = intIdentityMatrix(3)
		assertEquals(m1 * 2, m1 + m2)
	}

	@Test
	fun testMatrixVectorMult() {
		val m = 2 * intIdentityMatrix(4)
		val v = intVector(4) { it } * 100
		(m * v).prettyPrint(true)
	}

	@Test
	fun testMatrixMatrixMult() {
		val m1 = arrayOf(arrayOf(1, 2), arrayOf(3, 4)).toMatrix()
		(m1 * m1).prettyPrint(true)
	}

	@Test
	fun testMatrixVectorMultTropicalSemring() {
		val v = arrayOf(1, 2).toVector(INT_TROPICAL_SEMIRING_MAX)
		val m = arrayOf(
				arrayOf(1, 2),
				arrayOf(5, 1)).toMatrix(INT_TROPICAL_SEMIRING_MAX)
		(m * v).prettyPrint(true)
	}

	@Test
	fun testMatrixTranspose() {
		val m = arrayOf(arrayOf(3, 4), arrayOf(2, 5), arrayOf(2, 2)).toMatrix()
		m.transpose().prettyPrint(true)
	}
}
