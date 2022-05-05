package com.gojungparkjo.routetracker.model

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import java.util.*

class TTS_Module(context:Context):TextToSpeech.OnInitListener {
    private var buttonSpeak: Button? = null
    private var text =""

    lateinit var tts:TextToSpeech

    init {
        tts = TextToSpeech(context,this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.KOREAN)

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS","The Language specified is not supported!")
            } else {
//                buttonSpeak!!.isEnabled = true
            }

        } else {
            Log.e("TTS", "Initilization Failed!")
        }
    }

    fun speakOut(text:String){
        tts.speak(text,TextToSpeech.QUEUE_FLUSH,null,"")
    }
}
