package com.gojungparkjo.routetracker

import android.util.Log
import java.math.BigDecimal
import java.math.RoundingMode


/**
 * The Ramer–Douglas–Peucker algorithm (RDP) is an algorithm for reducing the number of points in a
 * curve that is approximated by a series of points.
 *
 *
 * @see [Ramer–Douglas–Peucker Algorithm
 * @author Justin Wetherell <phishman3579></phishman3579>@gmail.com>
](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) */
object RamerDouglasPeucker {
    private fun sqr(x: BigDecimal): BigDecimal {
        return x * x
    }

    private fun distanceBetweenPoints(vx: BigDecimal, vy: BigDecimal, wx: BigDecimal, wy: BigDecimal): BigDecimal {
        return sqr(vx - wx) + sqr(vy - wy)
    }

    private fun distanceToSegmentSquared(
        px: BigDecimal,
        py: BigDecimal,
        vx: BigDecimal,
        vy: BigDecimal,
        wx: BigDecimal,
        wy: BigDecimal
    ): BigDecimal {
        val l2 = distanceBetweenPoints(vx, vy, wx, wy)
        Log.d("TAG", "distanceToSegmentSquared: ${l2.toString()}")
        if (l2 == BigDecimal.ZERO) return distanceBetweenPoints(px, py, vx, vy)
        val t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2
        if (t < BigDecimal.ZERO) return distanceBetweenPoints(px, py, vx, vy)
        return if (t > BigDecimal.ONE) distanceBetweenPoints(px, py, wx, wy) else distanceBetweenPoints(
            px, py,
            vx + t * (wx - vx),
            vy + t * (wy - vy)
        )
    }

    private fun perpendicularDistance(
        px: BigDecimal,
        py: BigDecimal,
        vx: BigDecimal,
        vy: BigDecimal,
        wx: BigDecimal,
        wy: BigDecimal
    ): BigDecimal {
        return bigSqrt(distanceToSegmentSquared(px, py, vx, vy, wx, wy))
    }

    private fun douglasPeucker(
        list: List<Array<BigDecimal>>,
        s: Int,
        e: Int,
        epsilon: BigDecimal,
        resultList: MutableList<Array<BigDecimal>>
    ) {
        // Find the point with the maximum distance
        var dmax = BigDecimal.ZERO
        var index = 0
        val end = e - 1
        for (i in s + 1 until end) {
            // Point
            val px = list[i][0]
            val py = list[i][1]
            // Start
            val vx = list[s][0]
            val vy = list[s][1]
            // End
            val wx = list[end][0]
            val wy = list[end][1]
            val d = perpendicularDistance(px, py, vx, vy, wx, wy)
            if (d > dmax) {
                index = i
                dmax = d
            }
        }
        // If max distance is greater than epsilon, recursively simplify
        if (dmax > epsilon) {
            // Recursive call
            douglasPeucker(list, s, index, epsilon, resultList)
            douglasPeucker(list, index, e, epsilon, resultList)
        } else {
            if (end - s > 0) {
                resultList.add(list[s])
                resultList.add(list[end])
            } else {
                resultList.add(list[s])
            }
        }
    }

    /**
     * Given a curve composed of line segments find a similar curve with fewer points.
     *
     * @param list List of _root_ide_package_.java.math.BigDecimal[] points (x,y)
     * @param epsilon Distance dimension
     * @return Similar curve with fewer points
     */
    fun douglasPeucker(list: List<Array<BigDecimal>>, epsilon: BigDecimal): List<Array<BigDecimal>> {
        val resultList: MutableList<Array<BigDecimal>> = ArrayList()
        douglasPeucker(list, 0, list.size, epsilon, resultList)
        return resultList
    }
}

private val SQRT_DIG = BigDecimal(1000)
private val SQRT_PRE = BigDecimal(10).pow(SQRT_DIG.toInt())

/**
 * Private utility method used to compute the square root of a BigDecimal.
 *
 * @author Luciano Culacciatti
 * @url http://www.codeproject.com/Tips/257031/Implementing-SqrtRoot-in-BigDecimal
 */
private fun sqrtNewtonRaphson(c: BigDecimal, xn: BigDecimal, precision: BigDecimal): BigDecimal {
    val fx = xn.pow(2).add(c.negate())
    val fpx = xn.multiply(BigDecimal(2))
    var xn1 = fx.divide(fpx, 2 * SQRT_DIG.toInt(), RoundingMode.HALF_DOWN)
    xn1 = xn.add(xn1.negate())
    val currentSquare = xn1.pow(2)
    var currentPrecision = currentSquare.subtract(c)
    currentPrecision = currentPrecision.abs()
    return if (currentPrecision.compareTo(precision) <= -1) {
        xn1
    } else sqrtNewtonRaphson(c, xn1, precision)
}

/**
 * Uses Newton Raphson to compute the square root of a BigDecimal.
 *
 * @author Luciano Culacciatti
 * @url http://www.codeproject.com/Tips/257031/Implementing-SqrtRoot-in-BigDecimal
 */
fun bigSqrt(c: BigDecimal): BigDecimal {
    return sqrtNewtonRaphson(c, BigDecimal.ONE, BigDecimal.ONE.divide(SQRT_PRE))
}