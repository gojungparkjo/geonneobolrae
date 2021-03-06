package com.gojungparkjo.routetracker

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.PointF
import android.os.Bundle
import android.speech.SpeechRecognizer
import android.telephony.SmsManager
import android.util.Log
import android.view.LayoutInflater
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.gojungparkjo.routetracker.ProjUtil.toEPSG5186
import com.gojungparkjo.routetracker.ProjUtil.toLatLng
import com.gojungparkjo.routetracker.activity.destinationsetting.DestinationSettingActivity
import com.gojungparkjo.routetracker.data.RoadRepository
import com.gojungparkjo.routetracker.data.TmapDirectionRepository
import com.gojungparkjo.routetracker.data.TmapLabelRepository
import com.gojungparkjo.routetracker.databinding.ActivityMainBinding
import com.gojungparkjo.routetracker.model.crosswalk.CrossWalkResponse
import com.gojungparkjo.routetracker.model.intersection.InterSectionResponse
import com.gojungparkjo.routetracker.model.pedestrianroad.PedestrianRoadResponse
import com.gojungparkjo.routetracker.model.trafficlight.TrafficLightResponse
import com.gojungparkjo.routetracker.model.tmapdirection.TmapDirection
import com.google.android.gms.location.*
import com.naver.maps.geometry.LatLng
import com.naver.maps.geometry.LatLngBounds
import com.naver.maps.map.CameraUpdate
import com.naver.maps.map.MapFragment
import com.naver.maps.map.NaverMap
import com.naver.maps.map.OnMapReadyCallback
import com.naver.maps.map.overlay.*
import com.naver.maps.map.util.FusedLocationSource
import kotlinx.coroutines.*
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.LineSegment
import org.locationtech.proj4j.ProjCoordinate
import kotlin.math.atan2
import kotlin.math.log
import kotlin.math.min


class MainActivity : AppCompatActivity(),
    OnMapReadyCallback {

    // ??????????????? fusedLocationSource
    private lateinit var locationSource: FusedLocationSource
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private val tmapLabelRepository = TmapLabelRepository()
    private val roadRepository = RoadRepository()
    private val tmapDirectionRepository = TmapDirectionRepository()
    private lateinit var tts: TTS_Module
    private lateinit var compass: Compass
    private lateinit var stepCounter: StepCounter
    private lateinit var stt: Stt
    private lateinit var sharedPref: SharedPreferences
    private lateinit var naverMap: NaverMap
    private  var Azimuth: Float = 0.0f
    lateinit var binding: ActivityMainBinding

    private var flag = false
    private var requesting = false
    private var guideMode = false
    var lastAnnounceTime = 0L

    private var fetchAndMakeJob: Job = Job().apply { complete() }
    private var addShapeJob: Job = Job()
    private var findJob: Job = Job()

    private val polygonMap =
        HashMap<String, Triple<PolygonOverlay, Pair<LatLng, LatLng>, Pair<String, String>>>()
    private val pedestrianRoadMap = HashMap<String, PolygonOverlay>()
    private val polylineMap = HashMap<String, PolylineOverlay>()
    private val trafficLightMap = HashMap<String, Marker>()
    private val interSectionMap = HashMap<String, String>()

    // ??????, ?????????, ????????????
    private var tmapDirectionMap = mutableListOf<Marker>()
    private var tmapDirectionMapMent = mutableListOf<String>()

    private var validBound: LatLngBounds? = null
    private var validBoundPolygon: PolygonOverlay? = null

    private val destinationSettingLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()){
        when(it.resultCode){
            DESTINATION_SUCCESS ->{
                it.data?.let{ //????????? lat, lng ????????? ??????
                    val lat = it.getDoubleExtra("lat",0.0)
                    val lng = it.getDoubleExtra("lng",0.0)
                    Log.d(TAG, "latlng $lat: $lng")
                    MainScope().launch {
                        getCurrentPosition()?.let { it1 -> saveDirection(it1, LatLng(lng,lat)) }
                    }
                }
            }
            DESTINATION_ERROR->{ //????????????

            }
        }
    }

    private val locationPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { result -> result.value == true }) {
            startTracking(fusedLocationClient)
        } else {
            showToast("?????? ????????? ?????????")
            MainScope().launch {
                delay(FINISH_DELAY)
                finish()
            }
        }
    }

    private val locationCallback by lazy {
        object : LocationCallback() {
            @SuppressLint("SetTextI18n")
            override fun onLocationResult(locationResult: LocationResult) {
                super.onLocationResult(locationResult)
                for (location in locationResult.locations) {
                    naverMap.let {
                        val coordinate = LatLng(location)
                        Log.d("??????", coordinate.longitude.toString())
                        Log.d("??????", coordinate.latitude.toString())
                        // ???????????? ??????????????? ????????? ??????????????????. ?????? ????????? ????????? ?????? ???????????
                        it.locationOverlay.isVisible = true
                        it.locationOverlay.position = coordinate
                        getDirection(coordinate, Azimuth)
                        if (binding.trackingSwitch.isChecked) {
                            it.moveCamera(CameraUpdate.scrollTo(coordinate))
                            it.moveCamera(CameraUpdate.zoomTo(18.0))   //?????? ???????????? ??????
                        }
                    }
                }
            }
        }
    }

    private var phoneNumber = "01033433317" // ??????????????? 120??? ???????????? ???
    var checkAddFix: Boolean = false // true: add, false: fix
    var mapMode: Boolean = false // true: mapMode, false: buttonMode

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sharedPref = super.getPreferences(Context.MODE_PRIVATE)
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

        bindView()
        initMap()
        checkPermissions()
        setupCompass()
        setupStepCounter()
        smsSttSetup()
    }


    private fun smsSttSetup() {

        stt = Stt(this, packageName)

        stt.onReadyForSpeech = {
            showToast("??????????????? ???????????????.")
        }

        stt.onResults = { results ->
            val matches: ArrayList<String> =
                results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION) as ArrayList<String>
            val smsManager: SmsManager = SmsManager.getDefault()
            var symptom = ""
            for (i in 0 until matches.size) {
                symptom = matches[i]
            }
            val reportType = if (checkAddFix) "?????? ??????" else "?????? ??????"
            val myMsg = "??????????????? ${reportType}?????????.\n" +
                    "????????? ${symptom}."
            val reportLocation = "https://m.map.naver.com/appLink.naver?app=Y&pinType=site&" +
                    "lng=${naverMap.locationOverlay.position.longitude}&appMenu=location&menu=location&version=2&" +
                    "lat=${naverMap.locationOverlay.position.latitude}"
//                checkMsgSend(symptom)

            // ?????? ????????? ????????? ???????????? ?????? ????????? ??????
            smsManager.sendTextMessage(phoneNumber, null, myMsg, null, null)
            smsManager.sendTextMessage(phoneNumber, null, reportLocation, null, null)
            showToast("???????????? ???????????????.")
        }

        stt.onError = { errorMessage ->
            showToast("?????? ??????: $errorMessage")
        }

    }

    // msp ????????????????????? ????????? ????????? ??????????????? ???????????? ?????? ??????, ?????? ?????????
    fun checkMsgSend(symptom: String) {
        tts.speakOut("?????? ????????? ??????????")
        val msp = MsgSendPreview(this, symptom)
        msp.show(this)
    }

// azimuth ??? ?????? ???????????? 0?????? 360
    private fun setupCompass() {
        compass = Compass(this)
        compass.setListener(object : Compass.CompassListener {
            override fun onNewAzimuth(azimuth: Float) {
                Azimuth = azimuth
                binding.compassTextView.text = azimuth.toInt().toString()
                if (::naverMap.isInitialized) naverMap.locationOverlay.bearing = azimuth
                if (System.currentTimeMillis() - lastAnnounceTime > ANNOUNCE_INTERVAL) {
                    findTrafficSignAndCrosswalkInSight(azimuth.toDouble())
                }
            }
        })
    }

    private fun setupStepCounter() {
        stepCounter = StepCounter(this)
        stepCounter.onNewStepCount = {
            binding.trackingSteps.text = it.toString()
        }
        binding.trackingSteps.setOnClickListener({
            tts.speakOut(binding.trackingSteps.text.toString() + "???????????????.")
        })
    }

    // ?????? ?????? ??? ????????? ??????
    private fun bindView() {
        binding.compassTextView.setOnClickListener {
            naverMap.locationOverlay.position = naverMap.cameraPosition.target
        }

        binding.logoImageView.setOnClickListener {
            destinationSettingLauncher.launch(Intent(this,DestinationSettingActivity::class.java))
        }

        binding.trackingButton.setOnClickListener {
            if (guideMode) {
                guideMode = false
                binding.trackingButton.setBackgroundColor(Color.parseColor("#80008000"))
                binding.trackingButton.text = "?????? ??????"
                showToast("????????? ???????????????.")
            } else {
                checkPermissions()
                binding.trackingButton.setBackgroundColor(Color.parseColor("#80FF0000"))
                binding.trackingButton.text = "?????? ??????"
                showToast("????????? ???????????????.")
                guideMode = true
            }
        }
        // ??????????????? ?????? ??????/?????? ??????
        binding.fixButton.setOnClickListener {
            checkAddFix = false
            tts.speakOut("?????? ?????? ????????? ??????????????????. ????????? ??????????????????.")
            MainScope().launch {
                delay(STT_DELAY)
                stt.startSpeech()
            }
        }
        binding.addButton.setOnClickListener {
            checkAddFix = true
            tts.speakOut("?????? ?????? ????????? ??????????????????. ????????? ??????????????????.")
            MainScope().launch {
                delay(STT_DELAY)
                stt.startSpeech()
            }
        }
        // ?????????(?????? ??????) ??????
        binding.bugReportButton.setOnClickListener {
            val bundle = Bundle().apply {
                putParcelable("position", getCurrentPosition())
            }
            BugReportFragment().apply { arguments = bundle }
                .show(supportFragmentManager, "BugReport")
        }
        // ?????? ??????/?????? ??????
        binding.mapButton.setOnClickListener {
            if (!mapMode) {
                binding.addButton.toInvisible()
                binding.fixButton.toInvisible()
                binding.trackingButton.toInvisible()
                binding.mapButton.text = "?????? ??????"
                mapMode = true
            } else {
                binding.addButton.toVisible()
                binding.fixButton.toVisible()
                binding.trackingButton.toVisible()
                binding.mapButton.text = "?????? ??????"
                mapMode = false
            }
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

    override fun onMapReady(naverMap: NaverMap) {
        naverMap.addOnCameraIdleListener {
            fetchDataWithInBound(naverMap.contentBounds)
            addShapesWithInBound(naverMap.contentBounds)
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
    // test




    private fun findTrafficSignAndCrosswalkInSight(degree: Double) {
        if (!guideMode) return
        if (fetchAndMakeJob.isActive) return
        if (this::naverMap.isInitialized) {
            if (findJob.isActive) return
            findJob = MainScope().launch {
                naverMap.let { map ->
                    var nearestEntryPointDistanceInSight = Double.MAX_VALUE
                    var nearestEntryPointIntersectionCode = ""
                    var nearestEntryPointCrossWalkCode = ""
                    var first = false
                    polygonMap.forEach { (_, polygon) -> //???????????? ????????? ??????
                        val firstPointDistance = //???????????? ???????????? ???????????? ??????
                            polygon.second.first.distanceTo(map.locationOverlay.position)
                        val secondPointDistance = //???????????? ???????????? ?????? ???????????? ??????
                            polygon.second.second.distanceTo(map.locationOverlay.position)
                        //??? ??? ????????? ???
                        val minPointDistance = min(firstPointDistance, secondPointDistance)
                        //??? ??? ????????? ?????? 10m ???????????? ??????
                        if (minPointDistance > 10) return@forEach
                        //??? ??? ????????? ?????? ??????
                        val nearPoint =
                            if (firstPointDistance < secondPointDistance) polygon.second.first else polygon.second.second
                        //????????? ????????? ??????g
                        val angle = atan2(
                            nearPoint.longitude - map.locationOverlay.position.longitude,
                            nearPoint.latitude - map.locationOverlay.position.latitude
                        ).toDegree()
                        val diff = angle.toInt() - degree.toInt()
                        // ?????? ???????????? ?????? -20~20??? ????????? ????????? ??????
                        if (diff !in -20..20) return@forEach
                        // ???????????? ??????????????? ???????????? ????????? ???????????? ??????????????? ??????
                        val crossWalkAngle =
                            if (firstPointDistance < secondPointDistance) atan2(
                                polygon.second.second.longitude - polygon.second.first.longitude,
                                polygon.second.second.latitude - polygon.second.first.latitude
                            ).toDegree() else atan2(
                                polygon.second.first.longitude - polygon.second.second.longitude,
                                polygon.second.first.latitude - polygon.second.second.latitude
                            ).toDegree()
                        // ??????????????? ??????????????? ???????????? ?????? ????????? -20~20??? ????????? ????????? ??????
                        if (crossWalkAngle.toInt() - degree.toInt() !in -20..20) return@forEach
                        // ?????? 10m ?????? ??????, -20~20??? ????????? ?????? ????????? ?????????,
                        // ?????? ?????? ??? ?????? ????????? ?????? ?????? ?????? ??????, ?????? ??? ??????????????? ??????
                        if (minPointDistance < nearestEntryPointDistanceInSight) {
                            // ?????? ???????????? ???????????? ???????????? ??????
                            nearestEntryPointDistanceInSight = minPointDistance
                            // ???????????? ?????? ??????
                            (polygon.first.tag as? List<String>)?.also {
                                nearestEntryPointIntersectionCode = it[0]
                                nearestEntryPointCrossWalkCode = it[1]
                            }
                            // ????????? ????????? ????????? ????????? ?????? ??????
                            first = firstPointDistance < secondPointDistance
                        }
                    }
                    if (nearestEntryPointDistanceInSight != Double.MAX_VALUE) {
//                        Log.d(
//                            TAG,
//                            "findTrafficSignAndCrosswalkInSight: " + "$nearestEntryPointIntersectionCode ${interSectionMap[nearestEntryPointIntersectionCode]} " +
//                                    "${polygonMap[nearestEntryPointCrossWalkCode]?.third?.let { if (first) it.first else it.second }} ?????? ???????????? ?????????."
//                        )
                        if (interSectionMap[nearestEntryPointIntersectionCode] != null) {
                            tts.speakOut("10?????? ????????? ${interSectionMap[nearestEntryPointIntersectionCode]} ${polygonMap[nearestEntryPointCrossWalkCode]?.third?.let { if (first) it.second else it.first }} ?????? ???????????? ?????????.")
                        } else {
                            tts.speakOut("10?????? ????????? ${polygonMap[nearestEntryPointCrossWalkCode]?.third?.let { if (first) it.second else it.first }} ?????? ???????????? ?????????.")
                        }
                        lastAnnounceTime = System.currentTimeMillis()
                    }
                }

            }
        }
    }

    private fun fetchDataWithInBound(bound: LatLngBounds?) {
        if (naverMap.cameraPosition.zoom < 16) return
        if (validBound?.contains(naverMap.locationOverlay.position) == true) return
        if (fetchAndMakeJob.isActive) return
        if (addShapeJob.isActive) addShapeJob.cancel()
        fetchAndMakeJob = CoroutineScope(Dispatchers.IO).launch fetch@{
            if (bound == null) return@fetch
            val job = launch {
                launch {
                    val interSectionResponse = roadRepository.getIntersectionInBound(bound)
                    interSectionResponse?.let { getInterSectionNameInBound(it) }
                }
                launch {
                    val trafficLights = roadRepository.getTrafficLightInBound(bound)
                    trafficLights?.let { addMarkerFromTrafficLightResponse(it) }
                }
                launch {
                    val pedestrianRoadResponse = roadRepository.getPedestrianRoadInBound(bound)
                    pedestrianRoadResponse?.let { addPolygonFromPedestrianRoadResponse(it) }
                }
                launch {
                    val trafficIslandResponse = roadRepository.getTrafficIslandInBound(bound)
                    trafficIslandResponse?.let { addPolygonFromPedestrianRoadResponse(it) }
                }
            }
            val crossWalkResponse = async { roadRepository.getRoadInBound(bound) }
            job.join()
            crossWalkResponse.await()?.let { addPolygonFromCrossWalkResponse(it) }
            validBound = bound.scaleByWeight(-0.15)
            MainScope().launch {
                validBoundPolygon?.map = null
                validBoundPolygon = validBound?.vertexes
                    ?.let {
                        PolygonOverlay(it.toList()).apply {
                            outlineWidth = 3;color = Color.TRANSPARENT
                            outlineColor = Color.parseColor("#80FF0000");zIndex = 20003
                        }
                    }
                validBoundPolygon?.map = naverMap
            }
        }
        addShapesWithInBound(bound)
    }

    fun LatLngBounds.scaleByWeight(w: Double): LatLngBounds {
        val width = vertexes[2].longitude - vertexes[0].longitude
        val height = vertexes[1].latitude - vertexes[0].latitude
        return LatLngBounds(
            LatLng(vertexes[0].latitude - height * w, vertexes[0].longitude - width * w),
            LatLng(vertexes[2].latitude + height * w, vertexes[2].longitude + width * w)
        )
    }

    private suspend fun addPolygonFromPedestrianRoadResponse(response: PedestrianRoadResponse) =
        withContext(Dispatchers.Default) {
            response.features?.forEach { feature ->
                if (feature.properties?.vIEWCDE != "002" || pedestrianRoadMap.containsKey(
                        feature.properties.mGRNU
                    )
                ) return@forEach
                feature.geometry?.coordinates?.forEach {
                    val list = mutableListOf<LatLng>()
                    it.forEach { point ->
                        list.add(ProjCoordinate(point[0], point[1]).toLatLng())
                    }
                    if (list.size > 2) {
                        pedestrianRoadMap[feature.properties.mGRNU] = PolygonOverlay().apply {
                            coords = list; color = Color.WHITE
                            outlineWidth = 5; outlineColor = Color.RED
                            tag = feature.properties.toString()
                        }
                    }
                }
            }
        }

    private suspend fun addPolygonFromCrossWalkResponse(response: CrossWalkResponse) =
        withContext(Dispatchers.Default) {
            response.features?.forEach { feature ->
                if (feature.properties?.vIEWCDE != "002" || feature.properties.eVECDE != "001" || polygonMap.containsKey(
                        feature.properties.mGRNU
                    )
                ) return@forEach
                feature.geometry?.coordinates?.forEach {
                    val list = mutableListOf<LatLng>()
                    it.forEach { point ->
                        list.add(ProjCoordinate(point[0], point[1]).toLatLng())
                    }
                    if (list.size <= 2) return@forEach

                    val crossWalkPolygon = PolygonOverlay().apply {
                        coords = list; color = Color.WHITE
                        outlineWidth = 5; outlineColor = Color.GREEN
                        tag = listOf(
                            feature.properties.cSSCDE.toString().trim(),
                            feature.properties.mGRNU
                        )
                    }

                    // ?????? ?????????
                    val rectanglePolygon = crossWalkPolygon.minimumRectangle()
                    // ????????? ?????????
                    val rectangleCoord = rectanglePolygon.coords.map { it.toCoordinate() }
                    // ??? ?????? ?????? ??????
                    val middlePoints = mutableListOf<LatLng>()
                    for (i in 0 until rectangleCoord.size - 1) {
                        val middlePoint =
                            LineSegment.midPoint(rectangleCoord[i], rectangleCoord[i + 1])
                        middlePoints.add(middlePoint.toLatLng())

                    }
                    // ????????? ?????? ?????????
                    val delta = 0.00003
                    val midLine1 =
                        lengthenLine(middlePoints[0], middlePoints[2], delta)
                    val midLine2 =
                        lengthenLine(middlePoints[1], middlePoints[3], delta)

                    val geometryFactory = GeometryFactory()
                    val midLine1Geometry = LineSegment(
                        midLine1[0].toCoordinate(),
                        midLine1[1].toCoordinate()
                    ).toGeometry(geometryFactory)
                    val midLine2Geometry = LineSegment(
                        midLine2[0].toCoordinate(),
                        midLine2[1].toCoordinate()
                    ).toGeometry(geometryFactory)

                    // ???????????? ?????? ?????? ??????
                    var midLine1Intersect = 0
                    var midLine2Intersect = 0

                    withContext(Dispatchers.Main) {
                        pedestrianRoadMap.forEach { _, polygon ->
                            val polygonBoundary =
                                geometryFactory.createPolygon(polygon.coords.map { it.toCoordinate() }
                                    .toTypedArray()).boundary
                            if (polygonBoundary.intersects(midLine1Geometry)) midLine1Intersect++
                            if (polygonBoundary.intersects(midLine2Geometry)) midLine2Intersect++
                        }
                    }

                    val mainMidLine: List<LatLng>

                    mainMidLine = if (midLine1Intersect > midLine2Intersect) {
                        lengthenLine(middlePoints[0], middlePoints[2], 0.00020)
                    } else if (midLine1Intersect < midLine2Intersect) {
                        lengthenLine(middlePoints[1], middlePoints[3], 0.00020)
                    } else {
                        listOf(rectangleCoord[0].toLatLng(), rectangleCoord[2].toLatLng())
                    }

                    val label1: String =
                        tmapLabelRepository.getLabelFromLatLng(mainMidLine[0])?.poiInfo?.name
                            ?: ""
                    val label2: String =
                        tmapLabelRepository.getLabelFromLatLng(mainMidLine[1])?.poiInfo?.name
                            ?: ""
                    Log.d(TAG, "addPolygonFromCrossWalkResponse: $label1")
                    Log.d(TAG, "addPolygonFromCrossWalkResponse: $label2")
                    Log.d(TAG, "addPolygonFromCrossWalkResponse: #######")

                    val mainMidLineGeometry = LineSegment(
                        mainMidLine[0].toCoordinate(),
                        mainMidLine[1].toCoordinate()
                    ).toGeometry(geometryFactory)

                    val intersectionPoint =
                        geometryFactory.createPolygon(crossWalkPolygon.coords.map { it.toCoordinate() }
                            .toTypedArray())
                            .intersection(mainMidLineGeometry).coordinates
                    Log.d(
                        TAG,
                        "addPolygonFromCrossWalkResponse: ${intersectionPoint.joinToString(" ")}"
                    )
                    polygonMap[feature.properties.mGRNU] = Triple(
                        crossWalkPolygon,
                        Pair(intersectionPoint[0].toLatLng(), intersectionPoint[1].toLatLng()),
                        Pair(label1, label2)
                    )

                    polylineMap[feature.properties.mGRNU + "L1"] =
                        PolylineOverlay(midLine1).apply {
                            zIndex = 20003
                            if (midLine1Intersect > midLine2Intersect) color =
                                Color.MAGENTA
                        }

                    polylineMap[feature.properties.mGRNU + "L2"] =
                        PolylineOverlay(midLine2).apply {
                            zIndex = 20003
                            if (midLine1Intersect < midLine2Intersect) color =
                                Color.MAGENTA
                        }
                }
            }
        }

    private suspend fun addMarkerFromTrafficLightResponse(response: TrafficLightResponse) =
        withContext(Dispatchers.Default) {
            response.features?.forEach { feature ->
                if (feature.properties?.vIEWCDE != "002" || feature.properties.eVECDE != "001" || trafficLightMap.containsKey(
                        feature.properties.mGRNU
                    )
                ) return@forEach
                feature.properties.let { property ->
                    if (property.sNLPKNDCDE == "007" && property.xCE != null && property.yCE != null) {
                        feature.geometry?.coordinates?.first {
                            trafficLightMap[feature.properties.mGRNU] =
                                Marker(ProjCoordinate(it[0], it[1]).toLatLng())
                            true
                        }
                    }
                }
            }
        }

    private suspend fun getInterSectionNameInBound(response: InterSectionResponse) {
        // ??? ????????? ??????????????? css ?????????
        withContext(Dispatchers.Default) {
            response.features.forEach { feature ->
                if (interSectionMap.containsKey(feature.properties.CSS_NUM.toString().trim()))
                    return@forEach
                feature.properties.let { property ->
                    interSectionMap[property.CSS_NUM.toString().trim()] = property.CSS_NAM
                }
            }
        }
    }

    private fun addShapesWithInBound(bound: LatLngBounds?) {
        if (bound == null) return
        if (addShapeJob.isActive) addShapeJob.cancel()
        if (findJob.isActive) findJob.cancel()
        addShapeJob = CoroutineScope(Dispatchers.Main).launch {
            binding.loadingView.toVisible()

            if (fetchAndMakeJob.isActive) fetchAndMakeJob.join()
            if (naverMap.cameraPosition.zoom < MINIMUM_ZOOM_LEVEL) {
                binding.loadingView.toGone()
                return@launch
            }

            polygonMap.forEach { (_, crosswalk) ->
                crosswalk.first.map =
                    if (bound.contains(crosswalk.first.bounds) || bound.intersects(crosswalk.first.bounds)) naverMap else null
            }
            pedestrianRoadMap.forEach { (_, pedestrianRoad) ->
                pedestrianRoad.map =
                    if (bound.contains(pedestrianRoad.bounds) || bound.intersects(pedestrianRoad.bounds)) naverMap else null
            }
            trafficLightMap.forEach { (_, trafficLight) ->
                trafficLight.map = if (bound.contains(trafficLight.position)) naverMap else null
            }
            polylineMap.forEach { (_, polyline) ->
                polyline.map =
                    if (bound.contains(polyline.bounds) || bound.intersects(polyline.bounds)) naverMap else null
            }

            binding.loadingView.toGone()
        }
    }

    private fun getCurrentPosition(): LatLng? {
        if (!::naverMap.isInitialized || !naverMap.locationOverlay.isVisible) return null
        return naverMap.locationOverlay.position
    }

        // ????????? ?????? ????????????
        private suspend fun saveDirection(currentpoisition: LatLng, destinationpoisition: LatLng) {
            val tmapDirectionResponse = tmapDirectionRepository.getDirectionFromDepToDes(
                // latlng ????????? ????????? ???????????? ????????? ??????
                currentpoisition.longitude,
                currentpoisition.latitude,
                destinationpoisition.latitude,
                destinationpoisition.longitude
            )
            withContext(Dispatchers.Default) {
                tmapDirectionResponse?.features?.forEach { feature ->
                    if(feature.geometry.type=="LineString"){
                        val linePoints = (feature.geometry.coordinates as List<List<Double>>).map { LatLng(it[1],it[0]) }
                        if(::naverMap.isInitialized){
                            MainScope().launch {
                                PolylineOverlay(linePoints).map = naverMap
                            }

                        }
                    }else{
                        if(::naverMap.isInitialized){
                            val point = (feature.geometry.coordinates as List<Double>)
                            val latlng = LatLng(point[1],point[0])
                            MainScope().launch {
                                Marker(latlng).apply{captionText = feature.properties.description}.map = naverMap
                                tmapDirectionMap.add(Marker(latlng))
                                tmapDirectionMapMent.add(feature.properties.description)
                            }
                        }
                    }
                            }
                }
            }

    private fun getDirection(position: LatLng , azimuth: Float){
//        val crossWalkAngle =
//            if (firstPointDistance < secondPointDistance) atan2(
//                polygon.second.second.longitude - polygon.second.first.longitude,
//                polygon.second.second.latitude - polygon.second.first.latitude
//            ).toDegree() else atan2(
//                polygon.second.first.longitude - polygon.second.second.longitude,
//                polygon.second.first.latitude - polygon.second.second.latitude
//            ).toDegree()
//        // ??????????????? ??????????????? ???????????? ?????? ????????? -20~20??? ????????? ????????? ??????
//        if (crossWalkAngle.toInt() - degree.toInt() !in -20..20) return@forEach
        CoroutineScope(Dispatchers.IO).launch {
            if (tmapDirectionMap.isEmpty() != true && tmapDirectionMap.get(0).position.distanceTo(position) <5){
                val markerangle = atan2(tmapDirectionMap.get(1).position.longitude - tmapDirectionMap.get(0).position.longitude,
                    tmapDirectionMap.get(1).position.latitude - tmapDirectionMap.get(0).position.latitude
                    ).toDegree()
                Log.d(TAG,  "????????????"+markerangle.toString())
                Log.d(TAG, "?????? ?????? ??????"+azimuth.toString())
                if(markerangle.toInt() - azimuth.toInt() > 5) {
                    tts.speakOut("????????? ???????????? ???????????????."+tmapDirectionMapMent.get(0))
                }
                else if(markerangle.toInt() - azimuth.toInt() < -5){
                    tts.speakOut("?????????????????? ???????????????." + tmapDirectionMapMent.get(0))
                }else{
                    tts.speakOut("????????? ?????? ???????????????."+tmapDirectionMapMent.get(0))
                }

                // ?????? ???????????? ?????? ????????? ???????????? ??? ?????????
//                tts.speakOut(tmapDirectionMapMent.get(0))
                tmapDirectionMap.removeAt(0)
                tmapDirectionMapMent.removeAt(0)
            }
        }
    }


    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED && ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACTIVITY_RECOGNITION
            ) == PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.SEND_SMS
            ) == PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startTracking(fusedLocationClient)
        } else {
            locationPermissionRequest.launch(
                arrayOf(
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACTIVITY_RECOGNITION,
                    Manifest.permission.SEND_SMS,
                    Manifest.permission.RECORD_AUDIO
                )
            )
        }
    }

    @SuppressLint("MissingPermission")
    fun startTracking(fusedLocationProviderClient: FusedLocationProviderClient) {
        requesting = true
        fusedLocationProviderClient.requestLocationUpdates(
            createLocationRequest(),
            locationCallback,
            mainLooper
        )
    }

    private fun stopTracking() {
        requesting = false
        stopLocationUpdates()
    }

    private fun createLocationRequest() = LocationRequest.create().apply {
        interval = 1000
        priority = LocationRequest.PRIORITY_HIGH_ACCURACY
    }

    private fun stopLocationUpdates() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }

    @SuppressLint("MissingPermission")
    override fun onResume() {
        super.onResume()
        compass.start()
        stepCounter.start()
    }

    override fun onStart() {
        super.onStart()
        compass.start()
        stepCounter.start()
    }

    override fun onPause() {
        super.onPause()
        compass.stop()
        stepCounter.stop()
    }

    override fun onStop() {
        super.onStop()
        compass.stop()
        stepCounter.stop()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean(REQUESTING_CODE, requesting)
        super.onSaveInstanceState(outState)
    }

    private fun showToast(msg: String) {
        MainScope().launch {
        Toast.makeText(applicationContext, msg, Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        const val MINIMUM_ZOOM_LEVEL = 16
        const val STT_DELAY = 5000L
        const val FINISH_DELAY = 3000L
        const val ANNOUNCE_INTERVAL = 5000
        const val REQUESTING_CODE = "100"
        const val DESTINATION_SUCCESS = 200
        const val DESTINATION_ERROR = 400
        private const val LOCATION_PERMISSION_REQUEST_CODE = 1000
        private const val TAG = "MainActivity"
    }

    // tts part
    override fun onBackPressed() {
        val dig = FeedBackDialog(this)
        dig.show(this)
        val a = sharedPref.getString("flag", false.toString())
        flag = a.toBoolean()
        Toast.makeText(this, a, Toast.LENGTH_SHORT).show()
        if (!flag) {
            tts.speakOut("????????? ??????????????????")
        }

    }
}


