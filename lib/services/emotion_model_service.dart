import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Runs emotion_model.tflite — inputs: (1, 48, 48, 1) + (1, 936) → (1, 4)
///
/// Uses getInputTensor(i).setTo(reshaped) + invoke(), NOT run([...]).
/// Passing flat Float32List to run() triggers tflite_flutter's auto-resize
/// path, which corrupts the tensor shape from [1,48,48,1] to [2,2304].
/// setTo() with a correctly-shaped nested list writes bytes directly.
class EmotionModelService {
  Interpreter? _interpreter;
  bool _loaded = false;

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

  Future<void> load() async {
    if (_loaded) return;

    final modelData = await rootBundle.load(
      'assets/models/emotion_model.tflite',
    );
    _interpreter = Interpreter.fromBuffer(modelData.buffer.asUint8List());

    // Allocate tensors ONCE. Calling allocateTensors during run() is what
    // triggered the "Bad state: failed precondition".
    _interpreter!.allocateTensors();

    debugPrint('EMO-MODEL: loaded bytes = ${modelData.lengthInBytes}');
    debugPrint('EMO-MODEL: loaded');
    final inTensors = _interpreter!.getInputTensors();
    for (int i = 0; i < inTensors.length; i++) {
      final t = inTensors[i];
      debugPrint(
        'EMO-MODEL in[$i] shape=${t.shape} type=${t.type} '
        'name="${t.name}" bytes=${t.data.length}',
      );
    }
    final outTensors = _interpreter!.getOutputTensors();
    for (int i = 0; i < outTensors.length; i++) {
      final t = outTensors[i];
      debugPrint(
        'EMO-MODEL out[$i] shape=${t.shape} type=${t.type} '
        'name="${t.name}" bytes=${t.data.length}',
      );
    }

    _loaded = true;
  }

  /// [image] flat 48*48 = 2304 floats in [0,1].
  /// [landmarks] flat 936 floats.
  (String, double, List<double>)? predict(
    List<double> image,
    List<double> landmarks,
  ) {
    final interp = _interpreter;
    if (!_loaded || interp == null) {
      debugPrint('EMO-PRED: interpreter not loaded');
      return null;
    }
    if (image.length != imgSize * imgSize) {
      debugPrint('EMO-PRED: bad image len ${image.length}');
      return null;
    }
    if (landmarks.length != numLandmarks) {
      debugPrint('EMO-PRED: bad lm len ${landmarks.length}');
      return null;
    }

    try {
      final in0 = interp.getInputTensor(0);
      final in1 = interp.getInputTensor(1);
      final out0 = interp.getOutputTensor(0);

      final int in0Elems = in0.shape.reduce((a, b) => a * b);
      final int in1Elems = in1.shape.reduce((a, b) => a * b);
      final int outElems = out0.shape.reduce((a, b) => a * b);

      debugPrint(
        'EMO-PRED feed: in0.shape=${in0.shape} (${in0Elems}e) '
        'in1.shape=${in1.shape} (${in1Elems}e) '
        'out0.shape=${out0.shape} (${outElems}e)',
      );

      if (in0Elems != image.length || in1Elems != landmarks.length) {
        debugPrint('EMO-PRED: element count mismatch, bailing');
        return null;
      }

      // ── Build correctly-shaped nested lists ──────────────────────────────
      // `.reshape([...])` from tflite_flutter produces nested lists whose
      // structure matches the target tensor, so setTo() does a plain byte
      // copy with no resize.
      final imgInput = image.reshape([1, imgSize, imgSize, 1]);
      final lmInput = landmarks.reshape([1, numLandmarks]);
      final outBuf = List<double>.filled(
        numClasses,
        0.0,
      ).reshape([1, numClasses]);

      // ── Write inputs via setTo ───────────────────────────────────────────
      in0.setTo(imgInput);
      in1.setTo(lmInput);

      // ── Run ──────────────────────────────────────────────────────────────
      interp.invoke();

      // ── Read output ──────────────────────────────────────────────────────
      final outData = out0.data; // Uint8List view
      final probs = List<double>.from(
        outData.buffer.asFloat32List(
          outData.offsetInBytes,
          outData.lengthInBytes ~/ 4,
        ),
      );

      // Manual sanity: verify probs look like a softmax distribution
      final double sum = probs.fold<double>(0.0, (a, b) => a + b);
      if (sum < 0.5 || sum > 1.5) {
        debugPrint('EMO-PRED: suspicious prob sum = $sum, probs=$probs');
      }

      int bestIdx = 0;
      double bestProb = probs[0];
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > bestProb) {
          bestProb = probs[i];
          bestIdx = i;
        }
      }
      return (classNames[bestIdx], bestProb, probs);
    } catch (e, st) {
      debugPrint('EMO-PRED threw: $e\n$st');
      return null;
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _loaded = false;
  }
}
