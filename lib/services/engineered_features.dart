// lib/services/engineered_features.dart
//
// Exact Dart port of engineered_features_numpy() from precompute_engineered_v13.py.
// The v12.3 model was trained on these 26 values in this exact order with this
// exact normalization (center → maxdist divide → distances with +1e-6 epsilon).
//
// DO NOT reorder. DO NOT change the +1e-6 placement. DO NOT skip the epsilon.
// If any of those drift, the engineered input becomes OOD → emotion accuracy
// collapses to ~25% (random) on-device, even though desktop test looked fine.
//
// Input: l936 = flat [x0, y0, x1, y1, ..., x467, y467]  (936 values)
// Output: 26 doubles, matching Python feature order exactly.

import 'dart:math' as math;

List<double> engineeredFeatures(List<double> l936) {
  assert(l936.length == 936, 'Expected 936 landmarks, got ${l936.length}');
  const int n = 468;

  // ── Step 1: center (mean of x, mean of y) ──
  double cx = 0.0, cy = 0.0;
  for (int i = 0; i < n; i++) {
    cx += l936[i * 2];
    cy += l936[i * 2 + 1];
  }
  cx /= n;
  cy /= n;

  // ── Step 2: centered coords (flat) ──
  final centered = List<double>.filled(n * 2, 0.0);
  for (int i = 0; i < n; i++) {
    centered[i * 2] = l936[i * 2] - cx;
    centered[i * 2 + 1] = l936[i * 2 + 1] - cy;
  }

  // ── Step 3: max distance from center + epsilon ──
  double maxD = 0.0;
  for (int i = 0; i < n; i++) {
    final dx = centered[i * 2];
    final dy = centered[i * 2 + 1];
    final d = math.sqrt(dx * dx + dy * dy);
    if (d > maxD) maxD = d;
  }
  maxD += 1e-6;

  // ── Step 4: normalize (divide by maxD) ──
  final norm = List<double>.filled(n * 2, 0.0);
  for (int i = 0; i < n * 2; i++) {
    norm[i] = centered[i] / maxD;
  }

  // ── Helpers matching Python's dist() and y() ──
  double dist(int a, int b) {
    final dx = norm[a * 2] - norm[b * 2];
    final dy = norm[a * 2 + 1] - norm[b * 2 + 1];
    return math.sqrt(dx * dx + dy * dy) + 1e-6;
  }

  double y(int i) => norm[i * 2 + 1];

  // ── Feature computation (must match Python EXACTLY) ──
  final leftEarNum = dist(159, 145);
  final leftEarDen = dist(33, 133);
  final rightEarNum = dist(386, 374);
  final rightEarDen = dist(362, 263);
  final leftEar = leftEarNum / leftEarDen;
  final rightEar = rightEarNum / rightEarDen;

  final mouthH = dist(13, 14);
  final mouthW = dist(61, 291);
  final mar = mouthH / mouthW;

  final mouthCenterY = (y(13) + y(14)) / 2.0;
  final cornersAvg = (y(61) + y(291)) / 2.0;
  final mouthCurve = mouthCenterY - cornersAvg;

  final lbAbove = y(159) - y(105);
  final rbAbove = y(386) - y(334);

  final faceAspect = dist(10, 152) / dist(234, 454);

  final eyeAvgY = (y(159) + y(386)) / 2.0;
  final noseVsEyes = y(1) - eyeAvgY;

  return [
    leftEar, // 1
    rightEar, // 2
    mar, // 3
    mouthCurve, // 4
    dist(61, 133), // 5
    dist(291, 362), // 6
    dist(105, 159), // 7
    dist(334, 386), // 8
    dist(107, 336), // 9
    lbAbove, // 10
    rbAbove, // 11
    faceAspect, // 12
    dist(1, 13), // 13
    dist(152, 14), // 14
    dist(172, 397), // 15
    dist(93, 323), // 16
    noseVsEyes, // 17
    (leftEar - rightEar).abs(), // 18
    (y(61) - y(291)).abs(), // 19
    (lbAbove - rbAbove).abs(), // 20
    leftEarDen / rightEarDen, // 21
    cornersAvg - y(1), // 22
    (dist(107, 1) + dist(336, 1)) / 2.0, // 23
    mouthW, // 24
    mouthH, // 25
    leftEarDen, // 26
  ];
}
