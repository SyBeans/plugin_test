import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

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

    final labelsJson = await rootBundle.loadString('assets/models/labels.json');
    _labels = List<String>.from(jsonDecode(labelsJson));

    final modelData = await rootBundle.load('assets/models/sign_model.tflite');

    // ── Try GPU First, Fallback to CPU ──
    try {
      final options = InterpreterOptions();
      options.addDelegate(GpuDelegateV2());
      debugPrint('⏳ SIGN-MODEL: Attempting GPU Delegate...');

      _interpreter = Interpreter.fromBuffer(
        modelData.buffer.asUint8List(),
        options: options,
      );
      debugPrint('✅ SIGN-MODEL: GPU Delegate enabled successfully');
    } catch (e) {
      debugPrint(
        '⚠️ SIGN-MODEL: GPU failed (BiLSTMs usually lack GPU support). Falling back to CPU...',
      );

      // Fallback to CPU
      final cpuOptions = InterpreterOptions();
      _interpreter = Interpreter.fromBuffer(
        modelData.buffer.asUint8List(),
        options: cpuOptions,
      );
      debugPrint('✅ SIGN-MODEL: Loaded on CPU instead');
    }

    debugPrint('Sign model loaded:');
    debugPrint('  inputs:  ${_interpreter!.getInputTensor(0).shape}');
    debugPrint('  outputs: ${_interpreter!.getOutputTensor(0).shape}');
    debugPrint('  labels:  ${_labels.length}');

    _loaded = true;
  }

  (String, double, List<double>)? predict(List<List<double>> sequence) {
    if (!_loaded) return null;
    if (sequence.length != numFrames) return null;
    if (sequence[0].length != numFeatures) return null;

    final input = [sequence];
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
