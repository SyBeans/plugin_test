import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

import 'services/sign_model_service.dart';
import 'services/emotion_model_service.dart';

// ============================================================
// TUNING FLAGS
// ============================================================
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

// Sign inference — buffer & motion
const int kSignBufferLen = 30;
const int kSignFeaturesPerFrame = 126;
const double kMotionStart = 0.02;
const double kMotionEnd = 0.008;
const int kSignCooldownFrames = 30;
const double kSignMinConfidence = 0.40;

// Emotion inference
const int kEmotionEveryNFrames = 5;
const double kEmotionMinConfidence = 0.35;
// ============================================================

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
      title: 'Plugin Test — Live Inference',
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
  final List<FrameResult> _capturedFrames = [];
  int _lastFpsTime = DateTime.now().millisecondsSinceEpoch;
  int _framesSinceFps = 0;
  List<Landmark> _lastHandLandmarks = [];
  int _frameSkipCounter = 0;
  double _lastHandSpan = 0;
  bool _handTooSmall = false;

  late CameraDescription _currentCamera;
  int _cameraIndex = 0;
  bool _isSwitching = false;

  bool get _isFrontCamera =>
      _currentCamera.lensDirection == CameraLensDirection.front;

  String _sensorOrientation = '?';
  String _previewSize = '?';
  String _frameSize = '?';

  // ---- Model services ----
  final SignModelService _signService = SignModelService();
  final EmotionModelService _emotionService = EmotionModelService();
  bool _modelsReady = false;

  // ---- Sign inference state ----
  final List<List<double>> _posBuffer = [];
  bool _handWasMoving = false;
  int _signCooldown = 0;
  String _lastSign = '—';
  double _lastSignConf = 0.0;
  final List<String> _sentenceBuffer = [];

  // ---- Emotion inference state ----
  int _emotionFrameCounter = 0;
  String _lastEmotion = '—';
  double _lastEmotionConf = 0.0;

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

    _controller = CameraController(
      _currentCamera,
      ResolutionPreset.medium,
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

  Future<void> _switchCamera() async {
    if (_isSwitching || widget.cameras.length < 2) return;
    setState(() {
      _isSwitching = true;
      _lastResult = null;
      _lastHandLandmarks = [];
      _posBuffer.clear();
      _handWasMoving = false;
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

      // ---- Run sign inference every frame ----
      if (_modelsReady) {
        _updateSignBuffer();
      }

      // ---- Run emotion inference every Nth frame ----
      if (_modelsReady && faceLandmarks.length >= 400) {
        _emotionFrameCounter++;
        if (_emotionFrameCounter % kEmotionEveryNFrames == 0) {
          _runEmotionInference(image, faceLandmarks);
        }
      }
    } catch (e, st) {
      debugPrint('Detection error: $e\n$st');
    } finally {
      _isDetecting = false;
    }
  }

  // ============================================================
  // SIGN INFERENCE
  // ============================================================
  void _updateSignBuffer() {
    if (_lastHandLandmarks.isEmpty) {
      _posBuffer.clear();
      _handWasMoving = false;
      return;
    }

    // Build 126-feature frame (2 hands × 63)
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

    // Motion detection
    if (_posBuffer.isNotEmpty) {
      final prev = _posBuffer.last;
      double motion = 0.0;
      for (int i = 0; i < kSignFeaturesPerFrame; i++) {
        motion += (frame[i] - prev[i]).abs();
      }
      motion /= kSignFeaturesPerFrame;

      if (motion > kMotionStart) _handWasMoving = true;

      if (_handWasMoving && motion < kMotionEnd && _signCooldown == 0) {
        if (_posBuffer.length >= kSignBufferLen) {
          _runSignInference();
        }
        _posBuffer.clear();
        _handWasMoving = false;
        _signCooldown = kSignCooldownFrames;
      }
    }

    _posBuffer.add(frame);
    if (_posBuffer.length > kSignBufferLen) {
      _posBuffer.removeAt(0);
    }

    if (_signCooldown > 0) _signCooldown--;
  }

  void _runSignInference() {
    // Build (30, 252): [position | velocity]
    final sequence = <List<double>>[];
    for (int i = 0; i < kSignBufferLen; i++) {
      final frame = List<double>.filled(252, 0.0);
      // Position (first 126)
      for (int j = 0; j < kSignFeaturesPerFrame; j++) {
        frame[j] = _posBuffer[i][j];
      }
      // Velocity (last 126) — zero for frame 0
      if (i > 0) {
        for (int j = 0; j < kSignFeaturesPerFrame; j++) {
          frame[kSignFeaturesPerFrame + j] =
              _posBuffer[i][j] - _posBuffer[i - 1][j];
        }
      }
      sequence.add(frame);
    }

    final result = _signService.predict(sequence);
    if (result == null) return;

    final (label, conf, _) = result;
    if (conf >= kSignMinConfidence) {
      setState(() {
        _lastSign = label;
        _lastSignConf = conf;
        _sentenceBuffer.add(label);
      });
      debugPrint('SIGN: $label (${(conf * 100).toStringAsFixed(1)}%)');
    }
  }

  // ============================================================
  // EMOTION INFERENCE
  // ============================================================
  void _runEmotionInference(CameraImage image, List<Landmark> faceLandmarks) {
    // Rotate landmarks to preview orientation
    final preview = <Landmark>[];
    for (final lm in faceLandmarks) {
      if (_isFrontCamera) {
        // Front (sensor 270): preview(x,y) = (lm.y, 1 - lm.x)
        preview.add(Landmark(lm.y, 1.0 - lm.x, lm.z));
      } else {
        // Back (sensor 90): different rotation — mirror of front
        preview.add(Landmark(1.0 - lm.y, lm.x, lm.z));
      }
    }

    // Compute crop box in preview space (matches training pipeline)
    double xMin = preview.map((l) => l.x).reduce(math.min);
    double xMax = preview.map((l) => l.x).reduce(math.max);
    double yMin = preview.map((l) => l.y).reduce(math.min);
    double yMax = preview.map((l) => l.y).reduce(math.max);

    double cx = (xMin + xMax) / 2;
    double cy = (yMin + yMax) / 2;
    double size = math.max(xMax - xMin, yMax - yMin) * 1.55;

    double x0 = (cx - size * 0.50).clamp(0.0, 1.0);
    double x1 = (cx + size * 0.50).clamp(0.0, 1.0);
    double y0 = (cy - size * 0.65).clamp(0.0, 1.0);
    double y1 = (cy + size * 0.35).clamp(0.0, 1.0);

    // Extract 48×48 crop from Y plane (with rotation back to raw coords)
    final yPlane = image.planes[0];
    final bytes = yPlane.bytes;
    final rowStride = yPlane.bytesPerRow;
    final pixelStride = yPlane.bytesPerPixel ?? 1;

    final crop = List<double>.filled(48 * 48, 0.0);
    for (int yy = 0; yy < 48; yy++) {
      for (int xx = 0; xx < 48; xx++) {
        double px = x0 + (xx / 48.0) * (x1 - x0);
        double py = y0 + (yy / 48.0) * (y1 - y0);

        // Map preview → raw
        double rawX, rawY;
        if (_isFrontCamera) {
          // preview (px, py) → raw (1 - py, px)
          rawX = 1.0 - py;
          rawY = px;
        } else {
          // Back cam: mirror of that
          rawX = py;
          rawY = 1.0 - px;
        }

        int srcX = (rawX * image.width).toInt().clamp(0, image.width - 1);
        int srcY = (rawY * image.height).toInt().clamp(0, image.height - 1);
        int idx = srcY * rowStride + srcX * pixelStride;

        if (idx >= 0 && idx < bytes.length) {
          crop[yy * 48 + xx] = bytes[idx] / 255.0;
        }
      }
    }

    // Landmarks for model: 936 = 468 × 2 (x, y only)
    final lmVec = <double>[];
    for (final lm in faceLandmarks) {
      lmVec.add(lm.x);
      lmVec.add(lm.y);
    }

    final result = _emotionService.predict(crop, lmVec);
    if (result == null) return;

    final (label, conf, _) = result;
    if (conf >= kEmotionMinConfidence) {
      setState(() {
        _lastEmotion = label;
        _lastEmotionConf = conf;
      });
      debugPrint('EMOTION: $label (${(conf * 100).toStringAsFixed(1)}%)');
    }
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
        title: Text('Plugin Test (${_isFrontCamera ? "Front" : "Back"})'),
        actions: [
          IconButton(
            icon: const Icon(Icons.cameraswitch),
            onPressed: _isSwitching ? null : _switchCamera,
          ),
          IconButton(
            icon: Icon(_showOverlay ? Icons.visibility : Icons.visibility_off),
            onPressed: () => setState(() => _showOverlay = !_showOverlay),
          ),
          IconButton(
            icon: const Icon(Icons.delete_sweep),
            tooltip: 'Clear sentence',
            onPressed: () => setState(() => _sentenceBuffer.clear()),
          ),
        ],
      ),
      body: Stack(
        children: [
          Positioned.fill(child: CameraPreview(ctrl)),

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

          // ---- Stats top-left ----
          Positioned(top: 20, left: 20, child: _buildStats()),

          // ---- Emotion top-right ----
          if (_modelsReady)
            Positioned(top: 20, right: 20, child: _buildEmotionPanel()),

          // ---- Sign bottom ----
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

  Widget _buildStats() {
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
          const SizedBox(height: 4),
          Text(
            'Preview: $_previewSize',
            style: const TextStyle(color: Colors.yellow, fontSize: 10),
          ),
        ],
      ),
    );
  }

  Widget _buildEmotionPanel() {
    final icons = {
      'Positive': '😊',
      'Negative': '😢',
      'Surprise': '😲',
      'Neutral': '😐',
    };
    final icon = icons[_lastEmotion] ?? '❓';
    final confPct = (_lastEmotionConf * 100).toStringAsFixed(0);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.75),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.pinkAccent, width: 2),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(icon, style: const TextStyle(fontSize: 30)),
          Text(
            _lastEmotion,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 13,
            ),
          ),
          Text(
            '$confPct%',
            style: const TextStyle(color: Colors.pinkAccent, fontSize: 11),
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
        if (a != null && b != null) {
          canvas.drawLine(a, b, paintLine);
        }
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
