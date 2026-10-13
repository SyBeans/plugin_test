import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Runs emotion_model.tflite — inputs: (1, 48, 48, 1) + (1, 936) → output (1, 4)
class EmotionModelService {
  Interpreter? _interpreter;
  bool _loaded = false;

  static const int imgSize = 48;
  static const int numLandmarks = 936;
  static const List<String> classNames = [
    'Positive',
    'Negative',
    'Surprise',
    'Neutral',
  ];

  bool get isLoaded => _loaded;

  Future<void> load() async {
    if (_loaded) return;

    final modelData = await rootBundle.load(
      'assets/models/emotion_model.tflite',
    );
    _interpreter = Interpreter.fromBuffer(modelData.buffer.asUint8List());

    print('Emotion model loaded:');
    print(
      '  inputs:  ${_interpreter!.getInputTensor(0).shape} / ${_interpreter!.getInputTensor(1).shape}',
    );
    print('  outputs: ${_interpreter!.getOutputTensor(0).shape}');

    _loaded = true;
  }

  /// Predict emotion.
  /// [image] must be a flat list of 48*48 = 2304 floats in [0,1].
  /// [landmarks] must be 936 floats.
  /// Returns (label, confidence, allProbs).
  (String, double, List<double>)? predict(
    List<double> image,
    List<double> landmarks,
  ) {
    if (!_loaded) return null;
    if (image.length != imgSize * imgSize) return null;
    if (landmarks.length != numLandmarks) return null;

    // Input 1: image (1, 48, 48, 1)
    final imgTensor = image.reshape([1, imgSize, imgSize, 1]);

    // Input 2: landmarks (1, 936)
    final lmTensor = [landmarks]; // wrap in batch

    // Output: (1, 4)
    final output = List.filled(4, 0.0).reshape([1, 4]);

    _interpreter!.run([imgTensor, lmTensor], [output]);

    final probs = List<double>.from(output[0]);
    int bestIdx = 0;
    double bestProb = probs[0];
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestIdx = i;
      }
    }

    return (classNames[bestIdx], bestProb, probs);
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loaded = false;
  }
}
