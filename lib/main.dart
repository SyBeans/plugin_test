import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'services/sign_model_service.dart';
import 'services/emotion_model_service.dart';

void main() {
  runApp(const MaterialApp(home: ModelLoadTest()));
}

class ModelLoadTest extends StatefulWidget {
  const ModelLoadTest({super.key});
  @override
  State<ModelLoadTest> createState() => _ModelLoadTestState();
}

class _ModelLoadTestState extends State<ModelLoadTest> {
  String _signStatus = 'Loading...';
  String _emotionStatus = 'Loading...';

  final _sign = SignModelService();
  final _emotion = EmotionModelService();

  @override
  void initState() {
    super.initState();
    _loadModels();
  }

  Future<void> _loadModels() async {
    try {
      await _sign.load();
      setState(
        () => _signStatus = '✅ Sign loaded (${_sign.labels.length} labels)',
      );
    } catch (e, st) {
      print('Sign load error: $e\n$st');
      setState(() => _signStatus = '❌ Sign error: $e');
    }

    try {
      await _emotion.load();
      setState(() => _emotionStatus = '✅ Emotion loaded');
    } catch (e, st) {
      print('Emotion load error: $e\n$st');
      setState(() => _emotionStatus = '❌ Emotion error: $e');
    }
  }

  @override
  void dispose() {
    _sign.dispose();
    _emotion.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('TFLite Model Load Test')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_signStatus, style: const TextStyle(fontSize: 16)),
            const SizedBox(height: 20),
            Text(_emotionStatus, style: const TextStyle(fontSize: 16)),
          ],
        ),
      ),
    );
  }
}
