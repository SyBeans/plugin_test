import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

// ============================================================
// TUNING FLAGS
// ============================================================
const bool kSwapDims = true;
const bool kUseContain = false;

// Position fine-tune
const double kOffsetShiftX = 0.0;
const double kOffsetShiftY = 0.0;

// Hand scale
const double kHandScale = 0.55;

// Dot sizes
const double kFaceDotRadius = 1.0;
const double kHandTipRadius = 3.0;
const double kHandJointRadius = 1.8;
const double kFaceDotOpacity = 0.55;

// Performance
const int kFrameSkip = 1;

// Debug
const bool kShowDebugFrame = false;

// Back camera rotation mode — try 0,1,2,3 if dots are wrong
// 0: same as front (270° style)
// 1: 90° CW
// 2: 180°
// 3: no rotation
const int kBackRotationMode = 1;
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
  final List<FrameResult> _capturedFrames = [];
  int _lastFpsTime = DateTime.now().millisecondsSinceEpoch;
  int _framesSinceFps = 0;
  List<Landmark> _lastHandLandmarks = [];
  int _frameSkipCounter = 0;

  late CameraDescription _currentCamera;
  int _cameraIndex = 0;
  bool _isSwitching = false;

  bool get _isFrontCamera =>
      _currentCamera.lensDirection == CameraLensDirection.front;

  String _sensorOrientation = '?';
  String _previewSize = '?';

  @override
  void initState() {
    super.initState();

    _cameraIndex = widget.cameras.indexWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
    );
    if (_cameraIndex < 0) _cameraIndex = 0;
    _currentCamera = widget.cameras[_cameraIndex];

    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) {
      debugPrint('Camera permission denied');
      return;
    }

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
      minHandDetectionConfidence: 0.5,
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
          _frameCount++;
        });
      }
    } catch (e) {
      debugPrint('Detection error: $e');
    } finally {
      _isDetecting = false;
    }
  }

  @override
  void dispose() {
    _handSub?.cancel();
    _controller?.dispose();
    _faceDetector?.dispose();
    _handPlugin?.dispose();
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
            tooltip: 'Switch camera',
            onPressed: _isSwitching ? null : _switchCamera,
          ),
          IconButton(
            icon: Icon(_showOverlay ? Icons.visibility : Icons.visibility_off),
            onPressed: () => setState(() => _showOverlay = !_showOverlay),
          ),
          IconButton(
            icon: const Icon(Icons.camera),
            onPressed: () {
              if (_lastResult != null) {
                _capturedFrames.add(_lastResult!);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Captured ${_capturedFrames.length}')),
                );
              }
            },
          ),
          IconButton(icon: const Icon(Icons.save), onPressed: _saveFrames),
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

          if (_isSwitching)
            const Positioned.fill(
              child: ColoredBox(
                color: Colors.black54,
                child: Center(child: CircularProgressIndicator()),
              ),
            ),

          Positioned(top: 20, left: 20, child: _buildStats()),
        ],
      ),
    );
  }

  Widget _buildStats() {
    return Container(
      padding: const EdgeInsets.all(12),
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
            style: const TextStyle(color: Colors.white),
          ),
          Text(
            'Face: ${_lastResult?.faceLandmarks.length ?? 0}',
            style: const TextStyle(color: Colors.white),
          ),
          Text(
            'Hands: ${_lastResult?.handLandmarks.length ?? 0}',
            style: const TextStyle(color: Colors.white),
          ),
          Text(
            'Latency: ${_lastResult?.inferenceTimeMs ?? 0} ms',
            style: const TextStyle(color: Colors.white),
          ),
          const Divider(color: Colors.white24, height: 12),
          Text(
            'Cam: ${_isFrontCamera ? "Front" : "Back"}',
            style: const TextStyle(color: Colors.cyan, fontSize: 12),
          ),
          Text(
            'Sensor: $_sensorOrientation',
            style: const TextStyle(color: Colors.yellow, fontSize: 12),
          ),
          Text(
            'Preview: $_previewSize',
            style: const TextStyle(color: Colors.yellow, fontSize: 12),
          ),
        ],
      ),
    );
  }

  Future<void> _saveFrames() async {
    if (_capturedFrames.isEmpty) return;

    final directory = Directory('/storage/emulated/0/Download');
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }

    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final path = '${directory.path}/signify_landmarks_$timestamp.json';

    final file = File(path);
    final data = _capturedFrames
        .map(
          (f) => {
            'face': f.faceLandmarks.map((l) => l.toJson()).toList(),
            'hands': f.handLandmarks.map((l) => l.toJson()).toList(),
          },
        )
        .toList();

    try {
      await file.writeAsString(jsonEncode(data));
      if (mounted) {
        showDialog(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text('Saved'),
            content: SelectableText(
              '$path\n\nFrames: ${_capturedFrames.length}',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Save failed: $e')));
      }
    }
  }
}

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

  /// Apply rotation based on sensor orientation.
  /// Front cam (270°): standard rotation
  /// Back cam (90°): different rotation — controlled by kBackRotationMode
  Offset _rotate(double x, double y) {
    if (isFrontCamera) {
      // Front camera: 270° sensor
      final t = x;
      return Offset(y, 1.0 - t);
    } else {
      // Back camera: 90° sensor — different rotation
      switch (kBackRotationMode) {
        case 0:
          // Same as front
          final t = x;
          return Offset(y, 1.0 - t);
        case 1:
          // 90° CW
          final t = x;
          return Offset(1.0 - y, t);
        case 2:
          // 180°
          return Offset(1.0 - x, 1.0 - y);
        case 3:
          // No rotation
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
    // Scale around hand anchor — but rotation must be applied AFTER
    // scaling, so we scale in raw space first, then rotate.
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
