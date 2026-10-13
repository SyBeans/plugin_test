import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Runs sign_model.tflite — input (1, 30, 252), output (1, 105)
class SignModelService {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _loaded = false;

  bool get isLoaded => _loaded;
  List<String> get labels => _labels;

  static const int numFrames = 30;
  static const int numFeatures = 252;

  Future<void> load() async {
    if (_loaded) return;

    // Load labels
    final labelsJson = await rootBundle.loadString('assets/models/labels.json');
    _labels = List<String>.from(jsonDecode(labelsJson));

    // Load model
    final modelData = await rootBundle.load('assets/models/sign_model.tflite');
    final interpreterOptions = InterpreterOptions();

    // Sign model uses Flex ops — need to enable it
    // (loaded via native delegate on Android — see build.gradle.kts)

    _interpreter = Interpreter.fromBuffer(
      modelData.buffer.asUint8List(),
      options: interpreterOptions,
    );

    // Print shapes for verification
    print('Sign model loaded:');
    print('  inputs:  ${_interpreter!.getInputTensor(0).shape}');
    print('  outputs: ${_interpreter!.getOutputTensor(0).shape}');
    print('  labels:  ${_labels.length}');

    _loaded = true;
  }

  /// Predict from a (30, 252) feature sequence.
  /// Returns (label, confidence, allProbs) or null if not loaded.
  (String, double, List<double>)? predict(List<List<double>> sequence) {
    if (!_loaded) return null;
    if (sequence.length != numFrames) return null;
    if (sequence[0].length != numFeatures) return null;

    // Build input tensor: shape (1, 30, 252)
    final input = [sequence]; // wrap in batch dim

    // Output: shape (1, 105)
    final output = List.filled(
      1 * _labels.length,
      0.0,
    ).reshape([1, _labels.length]);

    _interpreter!.run(input, output);

    final probs = List<double>.from(output[0]);
    int bestIdx = 0;
    double bestProb = probs[0];
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestIdx = i;
      }
    }

    return (_labels[bestIdx], bestProb, probs);
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loaded = false;
  }
}
