package com.zhuoji.mahjong

import android.content.pm.ActivityInfo
import android.content.res.Configuration
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.WebView
import androidx.activity.OnBackPressedCallback
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
 *  5. **返回键回选择页**（2026-11-12 用户反馈）：`TauriActivity` 把
 *     `handleBackNavigation` 关成 false，Wry 不注册返回键回调，安卓返回键会
 *     直接 finish Activity。这里自己注册回调，判据是**当前 URL**：
 *     选择页根（`http://tauri.localhost/`，无 hash）才退出，其余（在线面板 /
 *     离线面板 / 游戏页 / 任意局域网地址）一律 `loadUrl` 回选择页。
 *     面板状态由启动页写进 URL hash（`#online` / `#offline`），原生同步可读，
 *     不依赖异步 `evaluateJavascript`。
 */
class MainActivity : TauriActivity() {

    /**
     * **关掉 Wry 自带的返回键处理**（2026-11-12 根因）。
     *
     * `WryActivity.handleBackNavigation` 是个 `open val`（默认 `true`），Wry 在
     * `setWebView()` 里据此把自己的 `OnBackPressedCallback` 注册到
     * `onBackPressedDispatcher` 上。而 `setWebView()` 发生在 onCreate **之后**
     * （WebView 建好时），所以 Wry 的 callback 永远在栈顶，我们在 onCreate 里注册的
     * callback 被压在下面 —— **一次都不会被调用**（logcat 里连日志都没有）。
     *
     * 表现就是：面板页面按返回能回选择页（那是 Wry 的 `webView.goBack()` 干的），
     * 但选择页本身没有历史，Wry 的 `goBack()` 落空后**不 finish**，于是
     * 「按返回没反应、再按也没反应」。
     *
     * 覆盖成 false 后 Wry 干脆不注册，我们自己的 callback 是唯一处理器。
     */
    override val handleBackNavigation: Boolean = false

    private val idleHandler = Handler(Looper.getMainLooper())
    private var screenOnFlagged = false
    private var gameWebView: WebView? = null

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

        // handleBackNavigation=false（见类头）→ Wry 不注册回调，这里是唯一的处理器：
        // 选择页根（无 hash）退出；面板 / 游戏页 / 在线网页版一律回选择页。
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                // 2026-11-12 修复「在线模式返回直接退 App」：
                // Tauri 导航到局域网地址时可能重建 WebView，缓存的 gameWebView 会 detach
                // （url 读到空串 → 被误判为「已在选择页」→ finish）。这里每次现取。
                val wv = currentWebView()
                if (wv == null) {
                    exitApp()
                    return
                }
                // 2026-11-12 定稿：面板状态同步写在 URL hash 里，原生只读 `wv.url`。
                // 之前用 evaluateJavascript 问 currentPane 是异步的，页面刚好在重载时
                // 回调会丢 → 首屏按一次返回没反应、再按一次才退出（模拟器复现）。
                //   选择页      http://tauri.localhost/           → 退出 App
                //   在线面板    http://tauri.localhost/#online    → 回选择页
                //   离线面板    http://tauri.localhost/#offline   → 回选择页
                //   游戏页/外链  任意其它地址                      → 回选择页
                val url = wv.url.orEmpty()
                val onChooser = url.startsWith(CHOOSER_URL) || url.startsWith("about:blank")
                val atRoot = onChooser && !url.contains("#")
                Log.w(TAG, "back: url=$url onChooser=$onChooser atRoot=$atRoot")
                if (atRoot) {
                    exitApp()
                    return
                }
                // 面板或游戏页：一律 loadUrl 回选择页首屏并清历史。
                // 不用 goBack：外链历史是「选择页 → 在线面板 → 外链」，
                // 后退只会退到在线面板，用户要按两次才回得去。
                wv.loadUrl(CHOOSER_URL)
                wv.clearHistory()
            }

            /**
             * 退出 App：**直接 finishAndRemoveTask()**，不走 OnBackPressedDispatcher。
             *
             * 原先写的是 `isEnabled = false; dispatcher.onBackPressed()`（标准写法），
             * 实机（模拟器 API 34）复现：选择页按返回没有任何反应，连按多次才退。
             * 原因是 Tauri/Wry 在 Activity 里另注册了 back callback，我们的回调
             * disable 之后事件落到它手上被消费掉，`onBackPressed()` 没有走到
             * Activity 的默认 finish。直接 finish 一步到位，行为可预测。
             */
            private fun exitApp() {
                finishAndRemoveTask()
            }
        })
    }

    /** 取当前真正挂载的 WebView：Tauri 换页后旧引用可能已从视图树 detach。 */
    private fun currentWebView(): WebView? {
        val found = ArrayList<WebView>(2)
        fun walk(v: View) {
            if (v is WebView) {
                found.add(v)
                return
            }
            if (v is ViewGroup) {
                for (i in 0 until v.childCount) walk(v.getChildAt(i))
            }
        }
        walk(window.decorView)
        return found.lastOrNull() ?: gameWebView
    }

    /** Wry 在 WebView 建好时回调；保存引用供返回键导航使用。 */
    override fun onWebViewCreate(webView: WebView) {
        super.onWebViewCreate(webView)
        gameWebView = webView
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
        const val TAG = "DushanBack"
        const val IDLE_TIMEOUT_MS = 5 * 60 * 1000L
        const val CHOOSER_URL = "http://tauri.localhost/"
    }
}
