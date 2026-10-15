import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Runs emotion_model.tflite — inputs: image + landmarks → 4 classes.
///
/// Feeds flat Float32List buffers via Interpreter.run(). The typed-list path
/// in tflite_flutter copies raw bytes into the tensor backing store, skipping
/// the nested-list shape-guessing that produced a 5D tensor on Conv2D input.
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
  /// Returns (label, confidence, allProbs) or null on any failure.
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
      debugPrint(
        'EMO-PRED: bad image len ${image.length} '
        '(expected ${imgSize * imgSize})',
      );
      return null;
    }
    if (landmarks.length != numLandmarks) {
      debugPrint(
        'EMO-PRED: bad lm len ${landmarks.length} '
        '(expected $numLandmarks)',
      );
      return null;
    }

    try {
      // ── Sanity: verify tensor byte counts before handing over buffers ──
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

      if (in0Elems != image.length) {
        debugPrint(
          'EMO-PRED: tensor0 expects $in0Elems elems, '
          'we have ${image.length}',
        );
        return null;
      }
      if (in1Elems != landmarks.length) {
        debugPrint(
          'EMO-PRED: tensor1 expects $in1Elems elems, '
          'we have ${landmarks.length}',
        );
        return null;
      }
      if (outElems != numClasses) {
        debugPrint('EMO-PRED: out tensor expects $outElems elems');
        return null;
      }

      // ── Flat typed buffers ──────────────────────────────────────────────
      final imgInput = Float32List.fromList(image); // 2304 floats
      final lmInput = Float32List.fromList(landmarks); // 936 floats
      final outBuf = Float32List(numClasses); // 4 floats

      // run(List, List) internally dispatches to runForMultipleInputs with a
      // Map<int,Object> for outputs. This is the correct entry point.
      interp.run([imgInput, lmInput], [outBuf]);

      final probs = outBuf.toList();

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
