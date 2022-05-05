package com.gojungparkjo.routetracker.data

import com.gojungparkjo.routetracker.model.crosswalk.CrossWalkResponse
import com.gojungparkjo.routetracker.model.trafficlight.TrafficLightResponse
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.Query

interface RoadService {
    @GET("geoserver/seoul/wfs?request=GetFeature&version=1.1.1&typename=seoul%3AA004_A&outputFormat=application%2Fjson")
    suspend fun getRoadsInBound(@Query("bbox") bound:String): Response<CrossWalkResponse>

    @GET("geoserver/seoul/wfs?request=GetFeature&version=1.1.1&typename=seoul%3AA057_L&outputFormat=application%2Fjson")
    suspend fun getTrafficLightInBound(@Query("bbox") bound:String): Response<TrafficLightResponse>
}