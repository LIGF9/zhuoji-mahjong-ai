# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile
# 导出 → 系统分享面板（2026-11-12）：
# MainActivity.shareFile 只被 Rust 侧通过 JNI 调用（loadClass + CallStaticObjectMethod），
# Java/Kotlin 代码里没有任何引用 → R8 在 release 包里把它当死代码删掉，运行时
# 直接 NoSuchMethodError（实测 APK 的 classes.dex 里搜不到 shareFile）。
-keepclassmembers class com.zhuoji.mahjong.MainActivity {
    public static java.lang.String shareFile(android.content.Context, java.lang.String);
}
