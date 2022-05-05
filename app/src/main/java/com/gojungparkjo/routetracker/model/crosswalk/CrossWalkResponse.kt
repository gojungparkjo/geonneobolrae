package com.gojungparkjo.routetracker.model.crosswalk


import com.google.gson.annotations.SerializedName

data class CrossWalkResponse(
    @SerializedName("crs")
    val crs: Crs?,
    @SerializedName("features")
    val features: List<Feature>?,
    @SerializedName("totalFeatures")
    val totalFeatures: Int?,
    @SerializedName("type")
    val type: String?
)