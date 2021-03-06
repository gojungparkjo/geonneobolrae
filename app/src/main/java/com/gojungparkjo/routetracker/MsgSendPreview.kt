package com.gojungparkjo.routetracker

import android.app.Dialog
import android.view.Window
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.gojungparkjo.routetracker.databinding.MsgSendPreviewBinding
import kotlin.system.exitProcess

class MsgSendPreview(private val context: AppCompatActivity, private val symptom: String) {

    private lateinit var binding : MsgSendPreviewBinding
    private val dlg = Dialog(context)   //부모 액티비티의 context 가 들어감
    var rts : Boolean = false
    fun show(content: MainActivity) {
        binding = MsgSendPreviewBinding.inflate(context.layoutInflater)

        dlg.requestWindowFeature(Window.FEATURE_NO_TITLE)   //타이틀바 제거
        dlg.setContentView(binding.root)     //다이얼로그에 사용할 xml 파일을 불러옴
        dlg.setCancelable(false)    //다이얼로그의 바깥 화면을 눌렀을 때 다이얼로그가 닫히지 않도록 함
        binding.textView2.text = symptom

        binding.sendButton.setOnClickListener {
            rts = true
            dlg.dismiss()
        }
        binding.cancelButton.setOnClickListener {
            dlg.dismiss()
        }

        dlg.show()
    }
    fun readyToSend(): Boolean {
        return rts
    }
}