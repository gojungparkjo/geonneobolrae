package com.gojungparkjo.routetracker

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.PointF
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.speech.RecognizerIntent
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.gojungparkjo.routetracker.ProjUtil.toLatLng
import com.gojungparkjo.routetracker.data.RoadRepository
import com.gojungparkjo.routetracker.databinding.ActivityMainBinding
import com.gojungparkjo.routetracker.model.FeedBackDialog
import com.gojungparkjo.routetracker.model.TTS_Module
import com.gojungparkjo.routetracker.model.crosswalk.CrossWalkResponse
import com.gojungparkjo.routetracker.model.trafficlight.TrafficLightResponse
import com.google.android.gms.location.*
import com.google.android.gms.tasks.CancellationTokenSource
import com.google.android.material.snackbar.Snackbar
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import com.naver.maps.geometry.LatLng
import com.naver.maps.geometry.LatLngBounds
import com.naver.maps.map.*
import com.naver.maps.map.overlay.LocationOverlay
import com.naver.maps.map.overlay.Marker
import com.naver.maps.map.overlay.OverlayImage
import com.naver.maps.map.overlay.PolygonOverlay
import com.naver.maps.map.util.FusedLocationSource
import com.naver.maps.map.util.MarkerIcons
import kotlinx.coroutines.*
import kotlinx.coroutines.tasks.await
import org.locationtech.proj4j.ProjCoordinate
import java.time.format.DateTimeFormatter
import kotlin.math.atan2

//speech branch2

class MainActivity : AppCompatActivity(), SensorEventListener,
    OnMapReadyCallback {
    private val TAG = "MainActivity"

    //    private var tts: TextToSpeech? = null
//    private var text = ""
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    // 네이버지도 fusedlocationsourece
    private lateinit var locationSource: FusedLocationSource
    private val repository = RoadRepository()
    private var buttonSpeak: Button? = null
    private var tv: TextView? = null
    var currentSteps = 0
    val db = Firebase.firestore
    val dateTimeFormatter = DateTimeFormatter.ofPattern("yyMMdd-hhmmss.S")
    lateinit var tts: TTS_Module
    lateinit var compass: Compass
    val locationPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        when {
            permissions.getOrDefault(Manifest.permission.ACCESS_FINE_LOCATION, false) -> {
                // Precise location access granted.
                startTracking(fusedLocationClient)
            }
            permissions.getOrDefault(Manifest.permission.ACCESS_COARSE_LOCATION, false) -> {
                // Only approximate location access granted.
                Toast.makeText(this, "정밀 권한을 주세요", Toast.LENGTH_SHORT).show()
            }
            else -> {
                // No location access granted.
                Toast.makeText(this, "위치 권한이 거부되었습니다.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // locationsource 퍼미션 ㅇ더어오기 추후 통합해야할듯??
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (locationSource.onRequestPermissionsResult(
                requestCode, permissions,
                grantResults
            )
        ) {
            if (!locationSource.isActivated) { // //권한 거부됨
                naverMap.locationTrackingMode = LocationTrackingMode.None
            }
            return
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    val stepPermissionRequest = fun() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.ACTIVITY_RECOGNITION),
            0
        )
        Toast.makeText(this, "걸음수 권환획득.", Toast.LENGTH_SHORT).show()
    }


    val locationCallback by lazy {
        object : LocationCallback() {
            @SuppressLint("SetTextI18n")
            override fun onLocationResult(locationResult: LocationResult) {
                super.onLocationResult(locationResult)
                for (location in locationResult.locations) {
                    binding.textView.setText(
                        "위도 ${location.latitude}" +
                                "경도 ${location.longitude}" +
                                "속도 ${location.speed}"
                    )
                }
                for (location in locationResult.locations) {
                    naverMap.let {
                        val coord = LatLng(location)

                        it.locationOverlay.isVisible = true
                        it.locationOverlay.position = coord

                        if (binding.trackingSwitch.isChecked) it.moveCamera(
                            CameraUpdate.scrollTo(
                                coord
                            )
                        )
                    }
                }
            }
        }
    }


    lateinit var binding: ActivityMainBinding

    private lateinit var naverMap: NaverMap

    private var requesting = false

    var job: Job? = null

    lateinit var sensorManager: SensorManager

    lateinit var cancellationTokenSource: CancellationTokenSource

    private val polygonList = mutableListOf<PolygonOverlay>()
    private val trafficLightMarkerList = mutableListOf<Marker>()

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        savedInstanceState?.let {
            if (it.keySet().contains(REQUESTING_CODE)) {
                requesting = it.getBoolean(REQUESTING_CODE)
            }
        }
        tts = TTS_Module(this)

        binding = ActivityMainBinding.inflate(LayoutInflater.from(this))
        setContentView(binding.root)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        locationSource = FusedLocationSource(this, LOCATION_PERMISSION_REQUEST_CODE)
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        bindView()
        initMap()
        setupCompass()

        sttTest()

    }

    private fun setupCompass() {
        compass = Compass(this)
        compass.setListener(object : Compass.CompassListener {
            override fun onNewAzimuth(degree: Float) {
                binding.compassTextView.text = degree.toInt().toString()
                paintTrafficSignAndCrosswalkInSight(degree.toDouble())
            }
        })
    }

    var ep = 0.000011
    private fun bindView() {
        binding.compassTextView.setOnClickListener {
            naverMap.locationOverlay.position = naverMap.cameraPosition.target
        }

        binding.trackingButton.setOnClickListener {
            if (requesting) {
                stopTracking()
                it.setBackgroundColor(ContextCompat.getColor(this, R.color.purple_200))
            } else {
                checkPermissions()
            }
        }
        binding.infoTextView.setOnClickListener {
            it.visibility = View.GONE
        }
    }

    private fun initMap() {
        val fm = supportFragmentManager
        val mapFragment = fm.findFragmentById(R.id.map) as MapFragment?
            ?: MapFragment.newInstance().also {
                fm.beginTransaction().add(R.id.map, it).commit()
            }
        mapFragment.getMapAsync(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return
        if (event.sensor.type == Sensor.TYPE_STEP_DETECTOR) {
            if (event.values[0] == 1.0f) {
                currentSteps++;
                binding.trackingSteps.setText(currentSteps.toString())
            }
        }
    }

    override fun onAccuracyChanged(p0: Sensor?, p1: Int) {
        Log.d(TAG, "onAccuracyChanged: " + p0.toString())
        Log.d(TAG, "onAccuracyChanged: $p1")
    }

    var colorJob: Job? = null

    private fun paintTrafficSignAndCrosswalkInSight(degree: Double) {
        if (job?.isActive == true) return

        if (this::naverMap.isInitialized) {
            colorJob?.let {
                if (it.isActive) return
            }
            colorJob = MainScope().launch {
                naverMap.let { map ->
                    map.locationOverlay.bearing = degree.toFloat()
                    trafficLightMarkerList.forEach {
                        val temp = atan2(
                            it.position.longitude - map.locationOverlay.position.longitude,
                            it.position.latitude - map.locationOverlay.position.latitude
                        ).toDegree()
                        val diff = temp.toInt() - degree.toInt()
                        val dist = it.position.distanceTo(map.locationOverlay.position)
                        it.captionText = "diff: $diff"
                        it.iconTintColor = if (diff in -20..20 && dist < 10) Color.BLACK else Color.GREEN
                        it.icon = MarkerIcons.BLACK
                        if (diff in -20..20 && dist < 10 && tts.tts.isSpeaking.not()) {
                            tts.speakOut(it.tag.toString())
                        }
                    }
                    polygonList.forEach { polygon ->
                        var nearest = Double.MAX_VALUE
                        var flag = false
                        polygon.coords.forEach {
                            val temp = it.distanceTo(map.locationOverlay.position)
                            if (temp < nearest && temp < 10) {
                                nearest = temp
                                val temp2 = atan2(
                                    it.longitude - map.locationOverlay.position.longitude,
                                    it.latitude - map.locationOverlay.position.latitude
                                ).toDegree()
                                val diff = temp2.toInt() - degree.toInt()
                                flag = diff in -20..20
                            }
                        }
                        polygon.color = if (flag) Color.RED else Color.WHITE
                    }
                }

            }
        }
    }

    override fun onResume() {
        super.onResume()
        compass.start()
        sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)?.also { stepDetect ->
            sensorManager.registerListener(
                this,
                stepDetect,
                SensorManager.SENSOR_DELAY_FASTEST
            )
        }
    }

    override fun onStart() {
        super.onStart()
        compass.start()
    }

    override fun onPause() {
        super.onPause()
        compass.stop()
        sensorManager.unregisterListener(this)
    }

    override fun onStop() {
        super.onStop()
        compass.stop()
    }

    override fun onMapReady(naverMap: NaverMap) {
        naverMap.addOnCameraIdleListener {
            addMarkersWithInBound(naverMap.contentBounds)
        }
        naverMap.apply {
            locationOverlay.icon =
                OverlayImage.fromResource(com.naver.maps.map.R.drawable.navermap_default_location_overlay_sub_icon_cone)
            locationOverlay.iconWidth = LocationOverlay.SIZE_AUTO
            locationOverlay.iconHeight = LocationOverlay.SIZE_AUTO
            locationOverlay.anchor = PointF(0.5f, 1f)
        }
        this.naverMap = naverMap

    }

    fun addMarkersWithInBound(bound: LatLngBounds?) {
        job?.let {
            if (it.isActive) it.cancel()
        }
        colorJob?.let { if (it.isActive) it.cancel() }
        job = CoroutineScope(Dispatchers.Main).launch {
            binding.loadingView.visibility = View.VISIBLE
            withContext(Dispatchers.Default) {
                Log.d(TAG, "addMarkersWithInBound")
                if (bound == null) return@withContext
                if (naverMap.cameraPosition.zoom > 16) {
                    val response = repository.getRoadInBound(bound)
                    removeAllPolygon()
                    response?.let { addPolygonFromCrossWalkResponse(it) }
                    val trafficLights = repository.getTrafficLightInBound(bound)
                    removeTrafficLightMarker()
                    trafficLights?.let { addMarkerFromTrafficLightResponse(trafficLights) }
                    withContext(Dispatchers.Main) {
                        polygonList.forEach {
                            it.map = naverMap
                        }
                        trafficLightMarkerList.forEach {
                            it.map = naverMap
                        }
                    }
                }
            }

            binding.loadingView.visibility = View.GONE
        }

    }

    fun addPolygonFromCrossWalkResponse(response: CrossWalkResponse) {
        response.features?.forEach { feature ->
            if (feature.properties?.vIEWCDE != "002" || feature.properties.eVECDE != "001") return@forEach
            feature.geometry?.coordinates?.forEach {
                val list = mutableListOf<LatLng>()
                it.forEach {
                    list.add(ProjCoordinate(it[0], it[1]).toLatLng())
                }
                if (list.size > 2) {
                    polygonList.add(PolygonOverlay().apply {
                        coords = list; color = Color.WHITE
                        outlineWidth = 5; outlineColor = Color.GREEN
                        tag = feature.properties.toString()
                        setOnClickListener {
                            binding.infoTextView.text = it.tag.toString()
                            binding.infoTextView.visibility = View.VISIBLE
                            true
                        }
                    }.also {
                        Log.d(TAG, "call convex hull")
                        polygonList.add(it.getConvexHull().apply {
                            color = Color.TRANSPARENT
                            outlineColor = Color.RED
                            outlineWidth = 5
                            zIndex = 20000
                        })
                    })
                }
            }
        }
    }

    suspend fun addMarkerFromTrafficLightResponse(response: TrafficLightResponse) {
        response.features?.forEach { feature ->
            if (feature.properties?.vIEWCDE != "002" || feature.properties.eVECDE != "001") return@forEach
            feature.properties.let { property ->
                if (property.sNLPKNDCDE == "007" && property.xCE != null && property.yCE != null) {
                    feature.geometry?.coordinates?.first {
                        val info = db.collection("trafficSignGuideInfo").document(property.mGRNU!!).get().await()
                        trafficLightMarkerList.add(
                            Marker(
                                ProjCoordinate(
                                    it[0],
                                    it[1]
                                ).toLatLng()
                            ).apply {
                                tag = info.get("line")?:"신호등 정보가 없습니다."
                            }
                        )
                    }
                }
            }
        }
    }

    suspend fun removeAllPolygon() = withContext(Dispatchers.Main) {
        polygonList.forEach {
            it.map = null
        }
        polygonList.clear()
    }

    suspend fun removeTrafficLightMarker() = withContext(Dispatchers.Main) {
        trafficLightMarkerList.forEach {
            it.map = null
        }
        trafficLightMarkerList.clear()
    }

    fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startTracking(fusedLocationClient)
        } else {
            locationPermissionRequest.launch(
                arrayOf(
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.ACCESS_FINE_LOCATION
                )
            )
        }
    }

    @SuppressLint("MissingPermission")
    fun startTracking(fusedLocationProviderClient: FusedLocationProviderClient) {
        binding.trackingButton.setBackgroundColor(Color.RED)
        requesting = true
        fusedLocationProviderClient.requestLocationUpdates(
            createLocationRequest(),
            locationCallback,
            mainLooper
        )
    }

    fun stopTracking() {
        requesting = false
        stopLocationUpdates()
    }

    fun createLocationRequest() = LocationRequest.create().apply {
        interval = 1000
        priority = LocationRequest.PRIORITY_HIGH_ACCURACY
    }

    private fun stopLocationUpdates() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }


    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean(REQUESTING_CODE, requesting)
        super.onSaveInstanceState(outState)
    }

    companion object {
        val REQUESTING_CODE = "100"
        private const val LOCATION_PERMISSION_REQUEST_CODE = 1000
    }

    // tts part
    override fun onBackPressed() {
        val dig = FeedBackDialog(this)
        dig.show(this)
        tts.speakOut("우리 어플을 평가해주세요")
    }
//    private fun speakOut() {
//        text = "우리어플을 평가해주세요"
//        tts!!.speak(text, TextToSpeech.QUEUE_FLUSH, null,"")
//    }


    fun sttTest(){
        binding.btnStt.setOnClickListener(View.OnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show()


            val intent = Intent()
            intent.action = RecognizerIntent.ACTION_RECOGNIZE_SPEECH
            intent.putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )

            val language = "ko-KR"

            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
            startActivityForResult(intent, 2)

        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            val results = data
                ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            val str = results!![0]
            Toast.makeText(baseContext, str, Toast.LENGTH_SHORT).show()
            val tv = findViewById<TextView>(R.id.tv_stt)
            tv.text = str
        }
    }



}




