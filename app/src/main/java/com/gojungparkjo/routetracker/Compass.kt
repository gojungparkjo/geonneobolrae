package com.gojungparkjo.routetracker

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager


class Compass(context: Context) : SensorEventListener {
    interface CompassListener {
        fun onNewAzimuth(azimuth: Float)
    }

    private var listener: CompassListener? = null
    private val sensorManager: SensorManager = context
        .getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val _accelerometer: Sensor
    private val _magneticSensor: Sensor
    private val _gravity = FloatArray(3)
    private val _geomagnetic = FloatArray(3)
    private val R = FloatArray(9)
    private val I = FloatArray(9)
    private var azimuth = 0f
    private var azimuthFix = 0f
    fun start() {
        sensorManager.registerListener(
            this, _accelerometer,
            SensorManager.SENSOR_DELAY_GAME
        )
        sensorManager.registerListener(
            this, _magneticSensor,
            SensorManager.SENSOR_DELAY_GAME
        )
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    fun setAzimuthFix(fix: Float) {
        azimuthFix = fix
    }

    fun resetAzimuthFix() {
        setAzimuthFix(0f)
    }

    fun setListener(l: CompassListener) {
        listener = l
    }

    override fun onSensorChanged(event: SensorEvent) {
        val alpha = 0.96f
        synchronized(this) {
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                _gravity[0] = alpha * _gravity[0] + (1 - alpha) * event.values[0]
                _gravity[1] = alpha * _gravity[1] + (1 - alpha) * event.values[1]
                _gravity[2] = alpha * _gravity[2] + (1 - alpha) * event.values[2]

                // mGravity = event.values;

                // Log.e(TAG, Float.toString(mGravity[0]));
            }
            if (event.sensor.type == Sensor.TYPE_MAGNETIC_FIELD) {
                // mGeomagnetic = event.values;
                _geomagnetic[0] = alpha * _geomagnetic[0] + (1 - alpha) * event.values[0]
                _geomagnetic[1] = alpha * _geomagnetic[1] + (1 - alpha) * event.values[1]
                _geomagnetic[2] = alpha * _geomagnetic[2] + (1 - alpha) * event.values[2]
                // Log.e(TAG, Float.toString(event.values[0]));
            }
            val success = SensorManager.getRotationMatrix(
                R, I, _gravity,
                _geomagnetic
            )
            if (success) {
                val orientation = FloatArray(3)
                SensorManager.getOrientation(R, orientation)
                // Log.d(TAG, "azimuth (rad): " + azimuth);
                azimuth =
                    Math.toDegrees(orientation[0].toDouble()).toFloat() // orientation
                azimuth = (azimuth + azimuthFix + 360) % 360
                // Log.d(TAG, "azimuth (deg): " + azimuth);
                if (listener != null) {
                    listener!!.onNewAzimuth(azimuth)
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}

    companion object {
        private const val TAG = "Compass"
    }

    init {
        _accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        _magneticSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    }
}