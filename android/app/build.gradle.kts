plugins {
    id("com.android.application")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.plugin_test"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources {
        noCompress += "tflite"
    }

    defaultConfig {
        applicationId = "com.example.plugin_test"
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

// Force one TFLite stack — exclude old org.tensorflow packages that clash
configurations.all {
    exclude(group = "org.tensorflow", module = "tensorflow-lite")
    exclude(group = "org.tensorflow", module = "tensorflow-lite-gpu")
    exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
}

flutter {
    source = "../.."
}