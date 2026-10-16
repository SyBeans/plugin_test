import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

import 'services/sign_model_service.dart';
import 'services/emotion_model_service.dart';

const bool kSwapDims = true;
const bool kUseContain = false;
const double kOffsetShiftX = 0.0;
const double kOffsetShiftY = 0.0;
const double kHandScale = 0.55;
const double kFaceDotRadius = 1.0;
const double kHandTipRadius = 3.0;
const double kHandJointRadius = 1.8;
const double kFaceDotOpacity = 0.55;
const int kFrameSkip = 1;
const bool kShowDebugFrame = false;
const int kBackRotationMode = 1;
const double kMinHandSpan = 0.12;

// ═══════════════════════════════════════════════════════════
// SIGN — NATURAL CONTINUOUS SIGNING
// Motion gate only suppresses STILL hands. While actively
// signing, predictions run continuously and new signs commit
// as soon as their labels stabilize in a rolling vote window.
// ═══════════════════════════════════════════════════════════
const int kSignBufferLen = 30; // model input window
const int kSignKeepFrames = 40; // rolling buffer cap
const int kSignFeaturesPerFrame = 126;
const double kSignMinConfidence = 0.50;

// Motion gate
const int kSignMotionHistoryLen = 10;
const double kSignMotionStartThreshold = 0.015;

// Inference cadence while active
const int kSignPredictEveryNFrames = 3;

// Label stabilization
const int kSignVoteWindow = 6;
const int kSignStableVoteCount = 4;
const int kSignCooldownFrames = 8;

// Hand dropout tolerance
const int kSignHandDropDelay = 6;

// ═══════════════════════════════════════════════════════════
// EMOTION
// ═══════════════════════════════════════════════════════════
const int kEmotionEveryNFrames = 5;
const double kEmotionMinConfidence = 0.40;
const Duration kEmotionCropTimeout = Duration(milliseconds: 400);

const List<double> kEmotionClassWeights = [1.0, 0.50, 1.0, 1.50];
const int kEmotionSmoothingLen = 8;

const int kDefaultResolutionIndex = 1;

class Landmark {
  final double x, y, z;
  Landmark(double? x, double? y, double? z)
    : x = x ?? 0.0,
      y = y ?? 0.0,
      z = z ?? 0.0;
  Map<String, double> toJson() => {'x': x, 'y': y, 'z': z};
}

class FrameResult {
  final List<Landmark> faceLandmarks;
  final List<Landmark> handLandmarks;
  final int inferenceTimeMs;
  FrameResult({
    required this.faceLandmarks,
    required this.handLandmarks,
    required this.inferenceTimeMs,
  });
}

class ResolutionOption {
  final String label;
  final ResolutionPreset preset;
  final IconData icon;
  const ResolutionOption(this.label, this.preset, this.icon);
}

const List<ResolutionOption> kResolutionOptions = [
  ResolutionOption('LOW', ResolutionPreset.low, Icons.sd_card_outlined),
  ResolutionOption('MED', ResolutionPreset.medium, Icons.sd_card),
  ResolutionOption('HIGH', ResolutionPreset.high, Icons.sd_card_alert_outlined),
];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Plugin Test',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
      ),
      home: CameraScreen(cameras: cameras),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraScreen({super.key, required this.cameras});
  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  FaceDetector? _faceDetector;
  HandLandmarkerPlugin? _handPlugin;
  StreamSubscription<List<Hand>>? _handSub;
  bool _isDetecting = false;
  bool _showOverlay = true;
  FrameResult? _lastResult;
  int _frameCount = 0;
  double _fps = 0;
  int _lastFpsTime = DateTime.now().millisecondsSinceEpoch;
  int _framesSinceFps = 0;
  List<Landmark> _lastHandLandmarks = [];
  int _frameSkipCounter = 0;
  double _lastHandSpan = 0;
  bool _handTooSmall = false;

  late CameraDescription _currentCamera;
  int _cameraIndex = 0;
  bool _isSwitching = false;
  int _resIndex = kDefaultResolutionIndex;

  bool get _isFrontCamera =>
      _currentCamera.lensDirection == CameraLensDirection.front;

  String _sensorOrientation = '?';
  String _previewSize = '?';
  String _frameSize = '?';

  final SignModelService _signService = SignModelService();
  final EmotionModelService _emotionService = EmotionModelService();
  bool _modelsReady = false;

  // ── Sign: continuous streaming state ──
  final List<List<double>> _posBuffer = [];
  List<double>? _lastSignPositions;
  final List<double> _signMotionHistory = [];
  bool _signActive = false;
  int _signNoHandFrames = 0;
  int _signPredictCounter = 0;
  int _signCooldown = 0;
  final List<String> _signVoteBuffer = [];

  String _lastSign = '—';
  double _lastSignConf = 0.0;
  final List<String> _sentenceBuffer = [];
  String _bufferState = 'idle';

  // ── Emotion state ──
  int _emotionFrameCounter = 0;
  String _lastEmotion = '—';
  double _lastEmotionConf = 0.0;
  String? _emotionError;
  double _lastCropMean = 0.0;
  double _lastLmMean = 0.0;
  bool _emotionInFlight = false;
  final List<String> _emotionVoteBuffer = [];

  // ── Debug data ──
  int _cropRotation = 0;
  List<double> _lastEmotionProbs = [0, 0, 0, 0];
  List<double> _lastSignProbs = [];
  double _lastMotion = 0.0;
  Uint8List? _lastCropPng;

  @override
  void initState() {
    super.initState();
    _cameraIndex = widget.cameras.indexWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
    );
    if (_cameraIndex < 0) _cameraIndex = 0;
    _currentCamera = widget.cameras[_cameraIndex];
    _loadModels();
    _initializeCamera();
  }

  Future<void> _loadModels() async {
    try {
      await _signService.load();
      await _emotionService.load();
      if (mounted) setState(() => _modelsReady = true);
      debugPrint('✅ Both models loaded');
    } catch (e, st) {
      debugPrint('Model load error: $e\n$st');
    }
  }

  Future<void> _initializeCamera() async {
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) return;

    if (_controller != null) {
      try {
        await _controller!.dispose();
      } catch (_) {}
      _controller = null;
    }

    _sensorOrientation = '${_currentCamera.sensorOrientation}°';
    final resOption = kResolutionOptions[_resIndex];

    _controller = CameraController(
      _currentCamera,
      resOption.preset,
      enableAudio: false,
    );

    try {
      await _controller!.initialize();
    } catch (e) {
      debugPrint('Camera init failed: $e');
      return;
    }

    final pv = _controller!.value.previewSize;
    final previewStr = pv == null
        ? 'null'
        : '${pv.width.toInt()} × ${pv.height.toInt()}';

    _faceDetector ??= await FaceDetector.create();
    _handPlugin ??= HandLandmarkerPlugin.create(
      numHands: 2,
      minHandDetectionConfidence: 0.3,
      delegate: HandLandmarkerDelegate.gpu,
    );

    _handSub ??= _handPlugin!.landmarkStream.listen((hands) {
      final lms = <Landmark>[];
      for (final hand in hands) {
        for (final lm in hand.landmarks) {
          lms.add(Landmark(lm.x, lm.y, lm.z));
        }
      }
      _lastHandLandmarks = lms;
      if (lms.length >= 21) {
        final xs = lms.map((l) => l.x).toList();
        final span =
            xs.reduce((a, b) => a > b ? a : b) -
            xs.reduce((a, b) => a < b ? a : b);
        _lastHandSpan = span;
        _handTooSmall = span < kMinHandSpan;
      } else {
        _lastHandSpan = 0;
        _handTooSmall = false;
      }
    });

    if (mounted) {
      setState(() {
        _previewSize = previewStr;
        _isSwitching = false;
      });
    }
    _controller!.startImageStream(_processCameraImage);
  }

  Future<void> _cycleResolution() async {
    if (_isSwitching) return;
    setState(() {
      _isSwitching = true;
      _lastResult = null;
      _lastHandLandmarks = [];
      _resetSignStream();
      _emotionVoteBuffer.clear();
      _resIndex = (_resIndex + 1) % kResolutionOptions.length;
    });
    try {
      await _controller?.stopImageStream();
    } catch (_) {}
    await _initializeCamera();
  }

  Future<void> _switchCamera() async {
    if (_isSwitching || widget.cameras.length < 2) return;
    setState(() {
      _isSwitching = true;
      _lastResult = null;
      _lastHandLandmarks = [];
      _resetSignStream();
      _emotionVoteBuffer.clear();
    });
    try {
      await _controller?.stopImageStream();
    } catch (_) {}
    _cameraIndex = (_cameraIndex + 1) % widget.cameras.length;
    _currentCamera = widget.cameras[_cameraIndex];
    await _initializeCamera();
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isDetecting) return;
    if (_isSwitching) return;
    _frameSkipCounter++;
    if (_frameSkipCounter % (kFrameSkip + 1) != 0) return;

    _isDetecting = true;
    final stopwatch = Stopwatch()..start();

    try {
      try {
        _handPlugin?.processFrame(
          image,
          _controller!.description.sensorOrientation,
        );
      } catch (e) {
        debugPrint('Hand error: $e');
      }

      final faces = await _faceDetector?.detectFacesFromCameraImage(image);
      final faceLandmarks = <Landmark>[];
      final frameW = image.width.toDouble();
      final frameH = image.height.toDouble();

      if (faces != null) {
        for (final face in faces) {
          final mesh = face.mesh;
          if (mesh != null) {
            for (final point in mesh.points) {
              final px = point.x ?? 0.0;
              final py = point.y ?? 0.0;
              final pz = point.z ?? 0.0;
              final nx = (px > 1.5) ? (px / frameW) : px;
              final ny = (py > 1.5) ? (py / frameH) : py;
              faceLandmarks.add(Landmark(nx, ny, pz));
            }
          }
        }
      }

      final inferenceTime = stopwatch.elapsedMilliseconds;
      _framesSinceFps++;
      final now = DateTime.now().millisecondsSinceEpoch;
      if (now - _lastFpsTime > 1000) {
        _fps = _framesSinceFps * 1000 / (now - _lastFpsTime);
        _framesSinceFps = 0;
        _lastFpsTime = now;
      }

      if (mounted) {
        setState(() {
          _lastResult = FrameResult(
            faceLandmarks: faceLandmarks,
            handLandmarks: List.from(_lastHandLandmarks),
            inferenceTimeMs: inferenceTime,
          );
          _frameSize = '${image.width} × ${image.height}';
          _frameCount++;
        });
      }

      if (_modelsReady) _updateSignStream();

      if (_modelsReady && faceLandmarks.length >= 400 && !_emotionInFlight) {
        _emotionFrameCounter++;
        if (_emotionFrameCounter % kEmotionEveryNFrames == 0) {
          _emotionInFlight = true;
          _runEmotionInference(
            image,
            faceLandmarks,
          ).whenComplete(() => _emotionInFlight = false);
        }
      }
    } catch (e, st) {
      debugPrint('Detection error: $e\n$st');
    } finally {
      _isDetecting = false;
    }
  }

  // ═══════════════════════════════════════════════════════════
  // SIGN — CONTINUOUS NATURAL STREAMING
  // ═══════════════════════════════════════════════════════════
  void _updateSignStream() {
    // ── Hand absent ──
    if (_lastHandLandmarks.isEmpty) {
      _signNoHandFrames++;
      if (_signNoHandFrames >= kSignHandDropDelay) {
        _signActive = false;
        _signVoteBuffer.clear();
        _bufferState = 'idle (no hand)';
      }
      return;
    }
    _signNoHandFrames = 0;

    // ── Build 126-feature frame ──
    final frame = List<double>.filled(kSignFeaturesPerFrame, 0.0);
    final handCount = _lastHandLandmarks.length ~/ 21;
    for (int h = 0; h < math.min(handCount, 2); h++) {
      for (int i = 0; i < 21; i++) {
        final lm = _lastHandLandmarks[h * 21 + i];
        final base = h * 63 + i * 3;
        frame[base + 0] = lm.x;
        frame[base + 1] = lm.y;
        frame[base + 2] = lm.z;
      }
    }

    // ── Motion (Euclidean norm vs previous frame) ──
    double motion = 0.0;
    if (_lastSignPositions != null) {
      double sumSq = 0.0;
      for (int i = 0; i < kSignFeaturesPerFrame; i++) {
        final d = frame[i] - _lastSignPositions![i];
        sumSq += d * d;
      }
      motion = math.sqrt(sumSq);
    }
    _lastMotion = motion;
    _lastSignPositions = frame;

    _signMotionHistory.add(motion);
    if (_signMotionHistory.length > kSignMotionHistoryLen) {
      _signMotionHistory.removeAt(0);
    }

    // ── Active if any recent frame had motion ──
    final anyMotion = _signMotionHistory.any(
      (m) => m > kSignMotionStartThreshold,
    );

    if (!anyMotion) {
      _signActive = false;
      _signVoteBuffer.clear();
      _bufferState = 'idle (still)';
      return;
    }

    _signActive = true;

    // ── Push to rolling buffer ──
    _posBuffer.add(frame);
    if (_posBuffer.length > kSignKeepFrames) _posBuffer.removeAt(0);

    if (_posBuffer.length < kSignBufferLen) {
      _bufferState = 'filling ${_posBuffer.length}/30';
      return;
    }

    // ── Cooldown after a commit ──
    if (_signCooldown > 0) {
      _signCooldown--;
      _bufferState = 'cooldown $_signCooldown';
      return;
    }

    // ── Predict every N active frames ──
    _signPredictCounter++;
    if (_signPredictCounter % kSignPredictEveryNFrames != 0) {
      _bufferState = 'active buf=${_posBuffer.length}';
      return;
    }

    _bufferState = 'active (predict)';
    _runContinuousInference();
  }

  void _runContinuousInference() {
    // Build (30, 252) from the last kSignBufferLen frames
    final int n = _posBuffer.length;
    final List<List<double>> window = _posBuffer.sublist(n - kSignBufferLen);

    final sequence = <List<double>>[];
    for (int i = 0; i < kSignBufferLen; i++) {
      final frame = List<double>.filled(252, 0.0);
      for (int j = 0; j < kSignFeaturesPerFrame; j++) {
        frame[j] = window[i][j];
      }
      if (i > 0) {
        for (int j = 0; j < kSignFeaturesPerFrame; j++) {
          frame[kSignFeaturesPerFrame + j] = window[i][j] - window[i - 1][j];
        }
      }
      sequence.add(frame);
    }

    final result = _signService.predict(sequence);
    if (result == null) return;

    final (label, conf, probs) = result;
    setState(() {
      _lastSignProbs = probs;
      _lastSign = label;
      _lastSignConf = conf;
    });

    if (conf < kSignMinConfidence) return;

    // ── Vote window ──
    _signVoteBuffer.add(label);
    if (_signVoteBuffer.length > kSignVoteWindow) {
      _signVoteBuffer.removeAt(0);
    }
    if (_signVoteBuffer.length < kSignVoteWindow) return;

    final counts = <String, int>{};
    for (final l in _signVoteBuffer) {
      counts[l] = (counts[l] ?? 0) + 1;
    }
    final winner = counts.entries.reduce((a, b) => a.value >= b.value ? a : b);

    if (winner.value < kSignStableVoteCount) return;

    // Only commit if different from last committed word
    if (_sentenceBuffer.isEmpty || _sentenceBuffer.last != winner.key) {
      debugPrint(
        'SIGN: committed ${winner.key} '
        '(${winner.value}/$kSignVoteWindow votes)',
      );
      setState(() {
        _sentenceBuffer.add(winner.key);
        _signVoteBuffer.clear();
        _signCooldown = kSignCooldownFrames;
      });
    }
  }

  void _resetSignStream() {
    _posBuffer.clear();
    _lastSignPositions = null;
    _signMotionHistory.clear();
    _signNoHandFrames = 0;
    _signPredictCounter = 0;
    _signCooldown = 0;
    _signActive = false;
    _signVoteBuffer.clear();
    _bufferState = 'idle';
  }

  // ═══════════════════════════════════════════════════════════
  // EMOTION (unchanged)
  // ═══════════════════════════════════════════════════════════
  Future<void> _runEmotionInference(
    CameraImage image,
    List<Landmark> faceLandmarks,
  ) async {
    String? localError;
    try {
      final preview = <Landmark>[];
      for (final lm in faceLandmarks) {
        if (_isFrontCamera) {
          preview.add(Landmark(lm.y, 1.0 - lm.x, lm.z));
        } else {
          preview.add(Landmark(1.0 - lm.y, lm.x, lm.z));
        }
      }

      double xMin = preview.map((l) => l.x).reduce(math.min);
      double xMax = preview.map((l) => l.x).reduce(math.max);
      double yMin = preview.map((l) => l.y).reduce(math.min);
      double yMax = preview.map((l) => l.y).reduce(math.max);

      final double cx = (xMin + xMax) / 2;
      final double cy = (yMin + yMax) / 2;
      final double size = math.max(xMax - xMin, yMax - yMin) * 1.55;

      final double x0 = (cx - size * 0.50).clamp(0.0, 1.0);
      final double x1 = (cx + size * 0.50).clamp(0.0, 1.0);
      final double y0 = (cy - size * 0.65).clamp(0.0, 1.0);
      final double y1 = (cy + size * 0.35).clamp(0.0, 1.0);

      final yPlane = image.planes[0];
      final bytes = yPlane.bytes;
      final int rowStride = yPlane.bytesPerRow;
      final int pixelStride = yPlane.bytesPerPixel ?? 1;

      final raw = List<double>.filled(48 * 48, 0.0);
      double rawMin = double.infinity;
      double rawMax = -double.infinity;
      double rawSum = 0.0;

      for (int yy = 0; yy < 48; yy++) {
        for (int xx = 0; xx < 48; xx++) {
          final double px = x0 + (xx / 48.0) * (x1 - x0);
          final double py = y0 + (yy / 48.0) * (y1 - y0);

          double rawX, rawY;
          if (_isFrontCamera) {
            rawX = 1.0 - py;
            rawY = px;
          } else {
            rawX = py;
            rawY = 1.0 - px;
          }

          final int srcX = (rawX * image.width).toInt().clamp(
            0,
            image.width - 1,
          );
          final int srcY = (rawY * image.height).toInt().clamp(
            0,
            image.height - 1,
          );
          final int idx = srcY * rowStride + srcX * pixelStride;
          double v = 0.0;
          if (idx >= 0 && idx < bytes.length) {
            v = bytes[idx] / 255.0;
          }
          raw[yy * 48 + xx] = v;
          if (v < rawMin) rawMin = v;
          if (v > rawMax) rawMax = v;
          rawSum += v;
        }
      }
      final rawMean = rawSum / raw.length;

      final crop = List<double>.filled(48 * 48, 0.0);
      for (int y = 0; y < 48; y++) {
        for (int x = 0; x < 48; x++) {
          int srcIdx;
          switch (_cropRotation) {
            case 90:
              srcIdx = x * 48 + (47 - y);
              break;
            case 180:
              srcIdx = (47 - y) * 48 + (47 - x);
              break;
            case 270:
              srcIdx = (47 - x) * 48 + y;
              break;
            default:
              srcIdx = y * 48 + x;
          }
          crop[y * 48 + x] = raw[srcIdx];
        }
      }

      final lmVec = <double>[];
      for (final lm in preview) {
        lmVec.add(lm.x);
        lmVec.add(lm.y);
      }
      double lmMin = lmVec.reduce(math.min);
      double lmMax = lmVec.reduce(math.max);
      double lmSum = 0.0;
      for (final v in lmVec) {
        lmSum += v;
      }
      final lmMean = lmSum / lmVec.length;

      debugPrint(
        'EMO: crop min=${rawMin.toStringAsFixed(3)} '
        'max=${rawMax.toStringAsFixed(3)} '
        'mean=${rawMean.toStringAsFixed(3)} | '
        'lm min=${lmMin.toStringAsFixed(3)} '
        'max=${lmMax.toStringAsFixed(3)} '
        'mean=${lmMean.toStringAsFixed(3)}',
      );

      final result = _emotionService.predict(crop, lmVec);
      if (result == null) {
        localError = 'predict → null';
        debugPrint('EMO: predict returned null');
      } else {
        final (_, _, rawProbs) = result;

        final weighted = List<double>.filled(4, 0.0);
        double wSum = 0.0;
        for (int i = 0; i < 4; i++) {
          weighted[i] = rawProbs[i] * kEmotionClassWeights[i];
          wSum += weighted[i];
        }
        if (wSum > 0) {
          for (int i = 0; i < 4; i++) {
            weighted[i] /= wSum;
          }
        }

        int frameIdx = 0;
        double frameTop = weighted[0];
        for (int i = 1; i < 4; i++) {
          if (weighted[i] > frameTop) {
            frameTop = weighted[i];
            frameIdx = i;
          }
        }
        final frameLabel = EmotionModelService.classNames[frameIdx];

        if (frameTop >= kEmotionMinConfidence) {
          _emotionVoteBuffer.add(frameLabel);
          if (_emotionVoteBuffer.length > kEmotionSmoothingLen) {
            _emotionVoteBuffer.removeAt(0);
          }
        }

        String smoothedLabel = frameLabel;
        double smoothedConf = frameTop;
        if (_emotionVoteBuffer.isNotEmpty) {
          final counts = <String, int>{};
          for (final lbl in _emotionVoteBuffer) {
            counts[lbl] = (counts[lbl] ?? 0) + 1;
          }
          final winner = counts.entries.reduce(
            (a, b) => a.value >= b.value ? a : b,
          );
          smoothedLabel = winner.key;

          final smoothedIdx = EmotionModelService.classNames.indexOf(
            smoothedLabel,
          );
          if (smoothedIdx >= 0) {
            smoothedConf = weighted[smoothedIdx];
          }
        }

        debugPrint(
          'EMO: raw=${rawProbs.map((p) => p.toStringAsFixed(2)).toList()} '
          'weighted=${weighted.map((p) => p.toStringAsFixed(2)).toList()} '
          'frame=$frameLabel(${(frameTop * 100).toStringAsFixed(0)}%) '
          'smoothed=$smoothedLabel(${(smoothedConf * 100).toStringAsFixed(0)}%) '
          'votes=${_emotionVoteBuffer.length}',
        );

        Uint8List? png;
        try {
          png = await _cropToPng(crop).timeout(kEmotionCropTimeout);
        } catch (e) {
          debugPrint('EMO: cropToPng failed: $e');
        }

        if (!mounted) return;
        setState(() {
          _lastEmotionProbs = weighted;
          _lastCropPng = png ?? _lastCropPng;
          _lastCropMean = rawMean;
          _lastLmMean = lmMean;
          _emotionError = null;
          _lastEmotion = smoothedLabel;
          _lastEmotionConf = smoothedConf;
        });
      }
    } catch (e, st) {
      localError = e.toString();
      debugPrint('EMO: THREW $e\n$st');
    }

    if (localError != null && mounted) {
      setState(() {
        _emotionError = localError;
      });
    }
  }

  Future<Uint8List> _cropToPng(List<double> crop) {
    final rgba = Uint8List(48 * 48 * 4);
    for (int i = 0; i < 48 * 48; i++) {
      final v = (crop[i] * 255).clamp(0, 255).toInt();
      rgba[i * 4 + 0] = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = 255;
    }
    final completer = Completer<Uint8List>();
    ui.decodeImageFromPixels(rgba, 48, 48, ui.PixelFormat.rgba8888, (
      img,
    ) async {
      try {
        final bd = await img.toByteData(format: ui.ImageByteFormat.png);
        if (bd == null) {
          completer.completeError('toByteData returned null');
          return;
        }
        completer.complete(bd.buffer.asUint8List());
      } catch (e, st) {
        completer.completeError(e, st);
      }
    });
    return completer.future;
  }

  @override
  void dispose() {
    _handSub?.cancel();
    _controller?.dispose();
    _faceDetector?.dispose();
    _handPlugin?.dispose();
    _signService.dispose();
    _emotionService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Plugin Test — Streaming',
          style: TextStyle(fontSize: 16),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.cameraswitch),
            tooltip: 'Flip camera',
            onPressed: _isSwitching ? null : _switchCamera,
          ),
          IconButton(
            icon: Icon(_showOverlay ? Icons.visibility : Icons.visibility_off),
            tooltip: 'Toggle landmarks',
            onPressed: () => setState(() => _showOverlay = !_showOverlay),
          ),
          IconButton(
            icon: const Icon(Icons.delete_sweep),
            tooltip: 'Clear sentence',
            onPressed: () => setState(() {
              _sentenceBuffer.clear();
              _resetSignStream();
              _emotionVoteBuffer.clear();
            }),
          ),
        ],
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: ClipRect(
              child: FittedBox(
                fit: BoxFit.cover,
                clipBehavior: Clip.hardEdge,
                child: SizedBox(
                  width: ctrl.value.previewSize!.height,
                  height: ctrl.value.previewSize!.width,
                  child: CameraPreview(ctrl),
                ),
              ),
            ),
          ),

          if (_showOverlay && _lastResult != null)
            Positioned.fill(
              child: CustomPaint(
                painter: LandmarkPainter(
                  faceLandmarks: _lastResult!.faceLandmarks,
                  handLandmarks: _lastResult!.handLandmarks,
                  previewSize: ctrl.value.previewSize!,
                  mirror: _isFrontCamera,
                  isFrontCamera: _isFrontCamera,
                  sensorOrientation: _currentCamera.sensorOrientation,
                ),
              ),
            ),

          Positioned(top: 20, left: 20, child: _buildStats()),

          if (_modelsReady)
            Positioned(top: 20, right: 20, child: _buildDebugStrip()),

          if (_modelsReady)
            Positioned(
              bottom: 20,
              left: 20,
              right: 20,
              child: _buildSignPanel(),
            ),

          if (_handTooSmall && _lastHandSpan > 0)
            Positioned(
              bottom: 220,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.orange.withOpacity(0.85),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text(
                    '✋ Move closer to camera',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),

          if (_isSwitching)
            const Positioned.fill(
              child: ColoredBox(
                color: Colors.black54,
                child: Center(child: CircularProgressIndicator()),
              ),
            ),

          if (!_modelsReady)
            const Positioned(
              top: 20,
              left: 0,
              right: 0,
              child: Center(
                child: Text(
                  'Loading models…',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildDebugStrip() {
    return Container(
      width: 178,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.82),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.orangeAccent, width: 1.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'CROP',
            style: TextStyle(
              color: Colors.orangeAccent,
              fontSize: 9,
              fontWeight: FontWeight.bold,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.white24, width: 1),
                  borderRadius: BorderRadius.circular(4),
                  color: Colors.black,
                ),
                child: _lastCropPng == null
                    ? const Center(
                        child: Text(
                          '—',
                          style: TextStyle(color: Colors.white38, fontSize: 9),
                        ),
                      )
                    : Image.memory(
                        _lastCropPng!,
                        width: 60,
                        height: 60,
                        fit: BoxFit.contain,
                        filterQuality: FilterQuality.none,
                        gaplessPlayback: true,
                      ),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: GestureDetector(
                  onTap: () {
                    setState(() {
                      _cropRotation = (_cropRotation + 90) % 360;
                    });
                  },
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 4,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.rotate_right,
                          size: 14,
                          color: Colors.yellowAccent,
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '${_cropRotation}°',
                          style: const TextStyle(
                            color: Colors.yellowAccent,
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
          if (_lastCropMean > 0)
            Padding(
              padding: const EdgeInsets.only(top: 3),
              child: Text(
                'crop mean ${_lastCropMean.toStringAsFixed(2)}'
                '  lm mean ${_lastLmMean.toStringAsFixed(2)}',
                style: const TextStyle(color: Colors.white54, fontSize: 8),
              ),
            ),
          const SizedBox(height: 6),
          const Divider(color: Colors.white24, height: 1),
          const SizedBox(height: 6),

          const Text(
            'EMOTION',
            style: TextStyle(
              color: Colors.pinkAccent,
              fontSize: 9,
              fontWeight: FontWeight.bold,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 3),
          _tinyProb('Pos', _lastEmotionProbs[0]),
          _tinyProb('Neg', _lastEmotionProbs[1]),
          _tinyProb('Sur', _lastEmotionProbs[2]),
          _tinyProb('Neu', _lastEmotionProbs[3]),
          const SizedBox(height: 4),
          Text(
            _emotionError == null
                ? '→ $_lastEmotion '
                      '(${(_lastEmotionConf * 100).toStringAsFixed(0)}%)'
                : '⚠ $_emotionError',
            style: TextStyle(
              color: _emotionError == null ? Colors.white70 : Colors.redAccent,
              fontSize: 9,
            ),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          const Divider(color: Colors.white24, height: 8),

          const Text(
            'SIGN (NATURAL)',
            style: TextStyle(
              color: Colors.cyanAccent,
              fontSize: 9,
              fontWeight: FontWeight.bold,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            _bufferState,
            style: const TextStyle(color: Colors.white70, fontSize: 9),
          ),
          Text(
            'motion ${_lastMotion.toStringAsFixed(3)}  '
            'votes ${_signVoteBuffer.length}/$kSignVoteWindow',
            style: const TextStyle(color: Colors.white70, fontSize: 9),
          ),
          const SizedBox(height: 4),
          if (_lastSignProbs.isNotEmpty)
            ..._topN(_lastSignProbs, 3).map((e) => _tinyTopLabel(e.$1, e.$2)),
          const SizedBox(height: 2),
          Text(
            '→ $_lastSign (${(_lastSignConf * 100).toStringAsFixed(0)}%)',
            style: const TextStyle(color: Colors.white70, fontSize: 9),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }

  Widget _tinyProb(String label, double p) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: Row(
        children: [
          SizedBox(
            width: 24,
            child: Text(
              label,
              style: const TextStyle(color: Colors.white, fontSize: 9),
            ),
          ),
          SizedBox(
            width: 28,
            child: Text(
              p.toStringAsFixed(2),
              style: const TextStyle(color: Colors.white, fontSize: 9),
            ),
          ),
          Expanded(
            child: LayoutBuilder(
              builder: (_, c) => Stack(
                children: [
                  Container(
                    height: 5,
                    decoration: BoxDecoration(
                      color: Colors.white12,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                  Container(
                    height: 5,
                    width: c.maxWidth * p.clamp(0.0, 1.0),
                    decoration: BoxDecoration(
                      color: p > 0.4 ? Colors.green : Colors.white38,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  List<(String, double)> _topN(List<double> probs, int n) {
    final indexed = List.generate(probs.length, (i) => (i, probs[i]));
    indexed.sort((a, b) => b.$2.compareTo(a.$2));
    return indexed.take(n).map((e) {
      final label = e.$1 < _signService.labels.length
          ? _signService.labels[e.$1]
          : '?';
      return (label, e.$2);
    }).toList();
  }

  Widget _tinyTopLabel(String label, double p) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label.length > 8 ? label.substring(0, 8) : label,
              style: const TextStyle(color: Colors.white, fontSize: 9),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          Text(
            p.toStringAsFixed(2),
            style: TextStyle(
              color: p > 0.15 ? Colors.cyanAccent : Colors.white38,
              fontSize: 9,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStats() {
    final resOption = kResolutionOptions[_resIndex];
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.7),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'FPS: ${_fps.toStringAsFixed(1)}',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
          Text(
            'Face: ${_lastResult?.faceLandmarks.length ?? 0}',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
          Text(
            'Hands: ${_lastResult?.handLandmarks.length ?? 0}',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
          Text(
            'Latency: ${_lastResult?.inferenceTimeMs ?? 0} ms',
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
          const SizedBox(height: 6),
          InkWell(
            onTap: _isSwitching ? null : _cycleResolution,
            borderRadius: BorderRadius.circular(20),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.12),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(resOption.icon, size: 14, color: Colors.cyanAccent),
                  const SizedBox(width: 4),
                  Text(
                    resOption.label,
                    style: const TextStyle(
                      color: Colors.cyanAccent,
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Preview: $_previewSize',
            style: const TextStyle(color: Colors.yellow, fontSize: 10),
          ),
          Text(
            'Frame: $_frameSize',
            style: const TextStyle(color: Colors.pink, fontSize: 10),
          ),
        ],
      ),
    );
  }

  Widget _buildSignPanel() {
    final confPct = (_lastSignConf * 100).toStringAsFixed(0);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.75),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.greenAccent, width: 2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Text(
                'SIGN: ',
                style: TextStyle(
                  color: Colors.greenAccent,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
              Expanded(
                child: Text(
                  _lastSign.toUpperCase(),
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 22,
                  ),
                ),
              ),
              Text(
                '$confPct%',
                style: const TextStyle(color: Colors.greenAccent),
              ),
            ],
          ),
          if (_sentenceBuffer.isNotEmpty) ...[
            const SizedBox(height: 6),
            const Divider(color: Colors.white24, height: 1),
            const SizedBox(height: 6),
            Text(
              _sentenceBuffer.join(' '),
              style: const TextStyle(color: Colors.white70, fontSize: 14),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ],
      ),
    );
  }
}

// ============================================================
// LANDMARK PAINTER
// ============================================================
class LandmarkPainter extends CustomPainter {
  final List<Landmark> faceLandmarks;
  final List<Landmark> handLandmarks;
  final Size previewSize;
  final bool mirror;
  final bool isFrontCamera;
  final int sensorOrientation;

  LandmarkPainter({
    required this.faceLandmarks,
    required this.handLandmarks,
    required this.previewSize,
    required this.mirror,
    required this.isFrontCamera,
    required this.sensorOrientation,
  });

  static const _handConnections = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17],
  ];

  Offset _rotate(double x, double y) {
    if (isFrontCamera) {
      final t = x;
      return Offset(y, 1.0 - t);
    } else {
      switch (kBackRotationMode) {
        case 0:
          final t = x;
          return Offset(y, 1.0 - t);
        case 1:
          final t = x;
          return Offset(1.0 - y, t);
        case 2:
          return Offset(1.0 - x, 1.0 - y);
        case 3:
          return Offset(x, y);
        default:
          final t = x;
          return Offset(1.0 - y, t);
      }
    }
  }

  Offset _mapFace(Landmark lm, double sw, double sh, double ox, double oy) {
    var p = _rotate(lm.x, lm.y);
    double nx = p.dx;
    double ny = p.dy;
    if (mirror) nx = 1.0 - nx;
    nx += kOffsetShiftX;
    ny += kOffsetShiftY;
    return Offset(nx * sw + ox, ny * sh + oy);
  }

  Offset _mapHand(
    Landmark lm,
    double sw,
    double sh,
    double ox,
    double oy,
    Offset anchor,
    double scale,
  ) {
    final sx = anchor.dx + (lm.x - anchor.dx) * scale;
    final sy = anchor.dy + (lm.y - anchor.dy) * scale;
    var p = _rotate(sx, sy);
    double nx = p.dx;
    double ny = p.dy;
    if (mirror) nx = 1.0 - nx;
    nx += kOffsetShiftX;
    ny += kOffsetShiftY;
    return Offset(nx * sw + ox, ny * sh + oy);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final frameW = kSwapDims ? previewSize.height : previewSize.width;
    final frameH = kSwapDims ? previewSize.width : previewSize.height;

    double scale;
    if (kUseContain) {
      scale = (size.width / frameW) < (size.height / frameH)
          ? (size.width / frameW)
          : (size.height / frameH);
    } else {
      scale = (size.width / frameW) > (size.height / frameH)
          ? (size.width / frameW)
          : (size.height / frameH);
    }

    final scaledW = frameW * scale;
    final scaledH = frameH * scale;
    final offsetX = (size.width - scaledW) / 2;
    final offsetY = (size.height - scaledH) / 2;

    if (kShowDebugFrame) {
      canvas.drawRect(
        Rect.fromLTWH(0, 0, size.width, size.height),
        Paint()
          ..color = Colors.green
          ..strokeWidth = 4
          ..style = PaintingStyle.stroke,
      );
    }

    final paintFace = Paint()
      ..color = Colors.red.withOpacity(kFaceDotOpacity)
      ..style = PaintingStyle.fill;
    for (final lm in faceLandmarks) {
      final p = _mapFace(lm, scaledW, scaledH, offsetX, offsetY);
      canvas.drawCircle(p, kFaceDotRadius, paintFace);
    }

    final paintHand = Paint()
      ..color = Colors.green
      ..strokeWidth = 2.0
      ..style = PaintingStyle.fill;
    final paintLine = Paint()
      ..color = Colors.green.withOpacity(0.7)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    const perHand = 21;
    const double rejectRadius = 0.35;

    for (int h = 0; h + perHand <= handLandmarks.length; h += perHand) {
      final slice = handLandmarks.sublist(h, h + perHand);

      double cx = 0, cy = 0;
      for (final lm in slice) {
        cx += lm.x;
        cy += lm.y;
      }
      cx /= slice.length;
      cy /= slice.length;
      final anchor = Offset(cx, cy);

      final pts = <Offset?>[];
      for (final lm in slice) {
        final dx = (lm.x - cx).abs();
        final dy = (lm.y - cy).abs();
        if (dx > rejectRadius || dy > rejectRadius) {
          pts.add(null);
          continue;
        }
        pts.add(
          _mapHand(lm, scaledW, scaledH, offsetX, offsetY, anchor, kHandScale),
        );
      }

      for (final conn in _handConnections) {
        final a = pts[conn[0]];
        final b = pts[conn[1]];
        if (a != null && b != null) canvas.drawLine(a, b, paintLine);
      }
      for (int i = 0; i < pts.length; i++) {
        final p = pts[i];
        if (p == null) continue;
        final isTip =
            i == 0 || i == 4 || i == 8 || i == 12 || i == 16 || i == 20;
        canvas.drawCircle(
          p,
          isTip ? kHandTipRadius : kHandJointRadius,
          paintHand,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
