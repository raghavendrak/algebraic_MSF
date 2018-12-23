package linalg

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import kotlin.test.assertEquals

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VectorTest {

	@Test
	fun testInnerProduct() {
		val v1 = arrayOf(1, 2, 3).toVector()
		val v2 = arrayOf(0, 1, 2).toVector()
		assertEquals(8, v1 inner v2)
	}

	@Test
	fun testOuterProduct() {
		val v1 = arrayOf(1, 2, 3).toVector()
		val v2 = arrayOf(0, 1, 2).toVector()
		(v1 outer v2).prettyPrint(true)
	}
}

