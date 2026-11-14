import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

class EmotionModelService {
  Interpreter? _interpreter;
  bool _loaded = false;
  bool _isGpu = false; // 🆕 tracks if GPU delegate succeeded

  static const int imgSize = 48;
  static const int numLandmarks = 936;
  static const int numClasses = 4;

  static const List<String> classNames = [
    'Positive',
    'Negative',
    'Surprise',
    'Neutral',
  ];

  bool get isLoaded => _loaded;
  bool get isGpu => _isGpu; // 🆕 expose to UI

  Future<void> load() async {
    if (_loaded) return;
    final modelData = await rootBundle.load(
      'assets/models/emotion_model.tflite',
    );

    try {
      final options = InterpreterOptions();
      options.addDelegate(GpuDelegateV2());
      debugPrint('⏳ EMO-MODEL: Attempting GPU Delegate...');

      _interpreter = Interpreter.fromBuffer(
        modelData.buffer.asUint8List(),
        options: options,
      );
      _isGpu = true;
      debugPrint('✅ EMO-MODEL: GPU Delegate enabled successfully');
    } catch (e) {
      debugPrint('⚠️ EMO-MODEL: GPU failed. Falling back to CPU...');

      final cpuOptions = InterpreterOptions();
      _interpreter = Interpreter.fromBuffer(
        modelData.buffer.asUint8List(),
        options: cpuOptions,
      );
      _isGpu = false;
      debugPrint('✅ EMO-MODEL: Loaded on CPU instead');
    }

    _interpreter!.allocateTensors();
    _loaded = true;
  }

  (String, double, List<double>)? predict(
    List<double> image,
    List<double> landmarks,
  ) {
    final interp = _interpreter;
    if (!_loaded || interp == null) return null;

    try {
      final in0 = interp.getInputTensor(0);
      final in1 = interp.getInputTensor(1);
      final out0 = interp.getOutputTensor(0);

      final imgInput = image.reshape([1, imgSize, imgSize, 1]);
      final lmInput = landmarks.reshape([1, numLandmarks]);

      in0.setTo(imgInput);
      in1.setTo(lmInput);

      interp.invoke();

      final outData = out0.data;
      final probs = List<double>.from(
        outData.buffer.asFloat32List(
          outData.offsetInBytes,
          outData.lengthInBytes ~/ 4,
        ),
      );

      int bestIdx = 0;
      double bestProb = probs[0];
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > bestProb) {
          bestProb = probs[i];
          bestIdx = i;
        }
      }
      return (classNames[bestIdx], bestProb, probs);
    } catch (e) {
      debugPrint('EMO-PRED error: $e');
      return null;
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loaded = false;
  }
}
