import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

class EmotionModelService {
  Interpreter? _interpreter;
  bool _loaded = false;
  bool _isGpu = false;

  static const int imgSize = 48;
  static const int numLandmarks = 936;
  static const int numEngineered = 26; // 🆕 v12.3
  static const int numClasses = 4;

  static const List<String> classNames = [
    'Positive',
    'Negative',
    'Surprise',
    'Neutral',
  ];

  bool get isLoaded => _loaded;
  bool get isGpu => _isGpu;

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
      debugPrint('✅ EMO-MODEL: GPU Delegate enabled');
    } catch (e) {
      debugPrint('⚠️ EMO-MODEL: GPU failed. Falling back to CPU...');

      final cpuOptions = InterpreterOptions();
      _interpreter = Interpreter.fromBuffer(
        modelData.buffer.asUint8List(),
        options: cpuOptions,
      );
      _isGpu = false;
      debugPrint('✅ EMO-MODEL: Loaded on CPU');
    }

    _interpreter!.allocateTensors();

    // 🆕 Sanity check — confirm the loaded model is 3-input v12.3
    final inCount = _interpreter!.getInputTensors().length;
    debugPrint('📊 EMO-MODEL: $inCount inputs detected');
    if (inCount != 3) {
      throw StateError(
        'Emotion model has $inCount inputs — expected 3 (v12.3). '
        'Did you forget to copy emotion_model_v12_3.tflite?',
      );
    }

    _loaded = true;
  }

  /// 3-input predict for v12.3:
  ///   image       : 48×48 grayscale, length 2304, values 0..1
  ///   landmarks   : 936 face landmarks (x,y pairs, already rotated/mirrored)
  ///   engineered  : 26 features from engineeredFeatures()
  ///
  /// Returns (label, topProb, allProbs). No post-softmax weighting —
  /// v12.3 was trained with class weights at training time only.
  (String, double, List<double>)? predict(
    List<double> image,
    List<double> landmarks,
    List<double> engineered, // 🆕
  ) {
    final interp = _interpreter;
    if (!_loaded || interp == null) return null;
    if (image.length != imgSize * imgSize) return null;
    if (landmarks.length != numLandmarks) return null;
    if (engineered.length != numEngineered) return null;

    try {
      final in0 = interp.getInputTensor(0); // image
      final in1 = interp.getInputTensor(1); // landmarks
      final in2 = interp.getInputTensor(2); // engineered 🆕
      final out0 = interp.getOutputTensor(0);

      in0.setTo(image.reshape([1, imgSize, imgSize, 1]));
      in1.setTo(landmarks.reshape([1, numLandmarks]));
      in2.setTo(engineered.reshape([1, numEngineered])); // 🆕

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
