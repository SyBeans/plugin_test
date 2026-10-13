# Signify — Plugin Test App

Standalone Flutter app to verify the two landmark plugins (face + hands) and
the two TFLite models (sign + emotion) before integrating them into the main
Signify app.

## What This App Does

- Runs the **front or back camera** in real time
- Detects **468 face landmarks** (`face_detection_tflite`)
- Detects **21 hand landmarks × 2 hands** (`hand_landmarker`)
- Draws landmarks as colored dots over the preview
- Runs **sign_model.tflite** on hand landmark sequences (motion-based)
- Runs **emotion_model.tflite** on face crop + landmarks
- Displays predictions live on screen

## Requirements

| Tool | Version |
|---|---|
| **Flutter** | 3.44.9 (stable channel) |
| **Dart** | 3.12.2 |
| **Android SDK** | Platform 36 (compileSdk = 36) |
| **Java JDK** | 17 |
| **Device** | Android 8.0+ with camera (phone or emulator) |

Verify:

```bash
flutter --version
flutter doctor -v






Repository Layout
plugin_test/
├── android/                   # Android host app (Gradle config)
├── assets/
│   └── models/
│       ├── sign_model.tflite         # 3.12 MB, 105 sign classes
│       ├── emotion_model.tflite      # 7.48 MB, 4 emotion classes
│       └── labels.json               # 105 sign label names
├── lib/
│   ├── main.dart                     # main screen (camera + inference)
│   └── services/
│       ├── sign_model_service.dart   # TFLite wrapper for sign model
│       └── emotion_model_service.dart# TFLite wrapper for emotion model
├── pubspec.yaml
└── README.md



First-Time Setup
# 1. Clone or copy the plugin_test folder
cd ~/plugin_test

# 2. Get packages
flutter pub get

# 3. Plug in the Android device (USB debugging enabled)
flutter devices

# 4. Build + install
flutter run
First build takes ~5 minutes (Gradle downloads + compiles native libs).
After that, rebuilds are ~30 seconds.