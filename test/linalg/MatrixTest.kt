package linalg

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import kotlin.test.assertEquals

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MatrixTest {

	@Test
	fun testMatrixSlice() {
		val m = intMatrixIdentity(5)
		m.prettyPrint(true)
		println()
		m[1..3, 2..3].prettyPrint(true)
	}

	@Test
	fun testMatrixAdd() {
		val m1 = intMatrixIdentity(3)
		val m2 = intMatrixIdentity(3)
		assertEquals(m1 * 2, m1 + m2)
	}

	@Test
	fun testMatrixVectorMult() {
		val m = 2 * intMatrixIdentity(4)
		val v = intVector(4) { it } * 100
		(m * v).prettyPrintVector(true)
	}
}
