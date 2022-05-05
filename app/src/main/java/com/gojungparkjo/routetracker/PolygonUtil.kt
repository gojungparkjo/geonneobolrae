package com.gojungparkjo.routetracker

import android.util.Log
import com.gojungparkjo.routetracker.ProjUtil.toEPSG5186
import com.gojungparkjo.routetracker.ProjUtil.toLatLng
import com.naver.maps.geometry.LatLng
import com.naver.maps.map.overlay.PolygonOverlay
import org.locationtech.proj4j.ProjCoordinate
import java.math.BigDecimal
import java.util.*

fun PolygonOverlay.getSimplified(ep:Double) = PolygonOverlay(RamerDouglasPeucker.douglasPeucker(
    this.coords.toList().map { arrayOf(it.latitude.toBigDecimal(), it.longitude.toBigDecimal()) }, ep.toBigDecimal()
).map { LatLng(it[0].toDouble(), it[1].toDouble()) })

typealias Point = Pair<BigDecimal, BigDecimal>

fun PolygonOverlay.getConvexHull(): PolygonOverlay {

    val arr = this.coords.map { Point(it.latitude.toBigDecimal(),it.longitude.toBigDecimal()) }.toMutableList()
    var bottomPoint = Point(1000000.0.toBigDecimal(), 1000000.0.toBigDecimal())
    for (point in arr){
        if(point.first<bottomPoint.first){
            bottomPoint = point
        }else if(point.first==bottomPoint.first){
            if(point.second<bottomPoint.second){
                bottomPoint = point
            }
        }
    }
    arr.sortWith(kotlin.Comparator { o1, o2 ->
        val result = ccw(bottomPoint,o1,o2)
        if(result>0) return@Comparator -1
        else if(result<0) return@Comparator 1
        else{
            if(dist(bottomPoint,o1)>dist(bottomPoint,o2)){
                return@Comparator 1
            }
        }
        return@Comparator -1
    })

    val stack = Stack<Point>()
    stack.add(bottomPoint)
    for (i in 1 until arr.size){
        while(stack.size>1&&ccw(stack[stack.size-2], stack[stack.size-1],arr[i])<=0){
            stack.pop()
        }
        stack.add(arr[i])
    }
    if(stack.size!=4) {
        Log.d("TAG", "getConvexHull: stack size is ${stack.size}")
        for(datum in stack){
            Log.d("TAG", "getConvexHull: $datum")
        }
    }
    return PolygonOverlay(stack.toList().map { LatLng(it.first.toDouble(),it.second.toDouble()) })
}

fun ccw(p1: Point, p2: Point, p3: Point): Int {
    /*
    x1 x2 x3 x1
    y1 y2 y3 y1
     */
    val result =
        (p1.first * p2.second + p2.first * p3.second + p3.first * p1.second) - (p2.first * p1.second + p3.first * p2.second + p1.first * p3.second)
    if (result > 0.toBigDecimal()) return 1
    if (result < 0.toBigDecimal()) return -1
    return 0
}

fun dist(p1: Point, p2: Point): BigDecimal {
    return (p2.first - p1.first) * (p2.first - p1.first) + (p2.second - p1.second) * (p2.second - p1.second)
}