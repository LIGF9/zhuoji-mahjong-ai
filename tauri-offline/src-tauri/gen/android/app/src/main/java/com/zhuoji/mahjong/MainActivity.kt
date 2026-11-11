package com.zhuoji.mahjong

import android.content.pm.ActivityInfo
import android.content.res.Configuration
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

/**
 * 全屏横屏游戏宿主。
 *
 * 四项 Android 端行为（2026-11-10 需求）：
 *  1. **隐藏状态栏/导航栏**：进入游戏即沉浸式全屏，从屏幕边缘下滑只临时显示
 *     「瞬时条」，松手自动隐藏（`BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE`）。
 *     厂商 ROM 在 `onWindowFocusChanged` 时会恢复系统栏，所以每次获得焦点都重放一遍。
 *  2. **保持亮屏 5 分钟**：有触摸就续期 `FLAG_KEEP_SCREEN_ON`；连续 5 分钟无触摸
 *     撤掉该标记，交回系统息屏策略（省电优先，且不会永久阻止锁屏）。
 *  3. **允许安卓小窗/分屏**：`resizeableActivity=true` + 进入多窗口时把方向请求
 *     临时放开为 `UNSPECIFIED`（固定方向的 Activity 在小窗里会被 letterbox，
 *     这是「横屏游戏开不了小窗」的根因），退出多窗口再锁回横屏。
 *  4. 不复制、不修改任何游戏 HTML/CSS/JS：WebView 内容仍由服务端（或离线内核）提供。
 */
class MainActivity : TauriActivity() {

    private val idleHandler = Handler(Looper.getMainLooper())
    private var screenOnFlagged = false

    private val clearScreenOn = Runnable {
        if (screenOnFlagged) {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            screenOnFlagged = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        applyImmersive()
        applyOrientation(isInMultiWindowMode)
        touchKeepScreenOn()
    }

    override fun onResume() {
        super.onResume()
        applyImmersive()
        touchKeepScreenOn()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) applyImmersive()
    }

    override fun onMultiWindowModeChanged(isInMultiWindowMode: Boolean, newConfig: Configuration) {
        super.onMultiWindowModeChanged(isInMultiWindowMode, newConfig)
        applyOrientation(isInMultiWindowMode)
        applyImmersive()
    }

    /** 任何触摸都续期亮屏；`dispatchTouchEvent` 只观察不消费，不影响 WebView 手势。 */
    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.actionMasked == MotionEvent.ACTION_DOWN ||
            ev.actionMasked == MotionEvent.ACTION_MOVE
        ) {
            touchKeepScreenOn()
        }
        return super.dispatchTouchEvent(ev)
    }

    private fun touchKeepScreenOn() {
        if (!screenOnFlagged) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            screenOnFlagged = true
        }
        idleHandler.removeCallbacks(clearScreenOn)
        idleHandler.postDelayed(clearScreenOn, IDLE_TIMEOUT_MS)
    }

    private fun applyImmersive() {
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        controller.hide(WindowInsetsCompat.Type.systemBars())
        WindowCompat.setDecorFitsSystemWindows(window, false)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            window.attributes = window.attributes.apply {
                layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
        }
    }

    /** 多窗口时放开方向，单窗口时锁横屏（游戏是横屏布局）。 */
    private fun applyOrientation(inMultiWindow: Boolean) {
        requestedOrientation = if (inMultiWindow) {
            ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        } else {
            ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        }
    }

    override fun onDestroy() {
        idleHandler.removeCallbacks(clearScreenOn)
        super.onDestroy()
    }

    private companion object {
        const val IDLE_TIMEOUT_MS = 5 * 60 * 1000L
    }
}
