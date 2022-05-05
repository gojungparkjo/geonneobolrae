package com.gojungparkjo.routetracker.model

import android.app.Dialog
import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Window
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.gojungparkjo.routetracker.MainActivity
import com.gojungparkjo.routetracker.databinding.FeedbackdialogBinding
import java.util.*
import kotlin.system.exitProcess

class FeedBackDialog(private val context : AppCompatActivity) {

    private lateinit var binding : FeedbackdialogBinding
    private val dlg = Dialog(context)   //부모 액티비티의 context 가 들어감
//    private lateinit var listener : Feedbackdialog
//    tts = TextToSpeech(this, TextToSpeech.OnInitListener {
//        fun onInit(status: Int) {
//            if (status == TextToSpeech.SUCCESS) {
//                // set US English as language for tts
//                val result = tts!!.setLanguage(Locale.US)
//
//                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
//                    Log.e("TTS","The Language specified is not supported!")
//                } else {
//                    buttonSpeak!!.isEnabled = true
//                }
//
//            } else {
//                Log.e("TTS", "Initilization Failed!")
//            }
//        }
//    })
    fun show(content: MainActivity) {
        binding = FeedbackdialogBinding.inflate(context.layoutInflater)

        dlg.requestWindowFeature(Window.FEATURE_NO_TITLE)   //타이틀바 제거
        dlg.setContentView(binding.root)     //다이얼로그에 사용할 xml 파일을 불러옴
        dlg.setCancelable(false)    //다이얼로그의 바깥 화면을 눌렀을 때 다이얼로그가 닫히지 않도록 함


        //ok 버튼 동작
        binding.yesBtn.setOnClickListener {
            dlg.dismiss()
            ActivityCompat.finishAffinity(context)
            exitProcess(0)
        }

        //cancel 버튼 동작
        binding.mic.setOnClickListener {
//
            dlg.dismiss()
        }

        dlg.show()
    }

//    fun setOnOKClickedListener(listener: (String) -> Unit) {
//        this.listener = object: MyDialogOKClickedListener {
//            override fun onOKClicked(content: String) {
//                listener(content)
//            }
//        }
//    }

//
//    interface MyDialogOKClickedListener {
//        fun onOKClicked(content : String)
//    }

}