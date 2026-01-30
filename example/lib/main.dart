import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:llmcpp/llmcpp.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final LlmModelIsolated _llmcppPlugin = LlmModelIsolated(
    LlmConfig(
      temp: 0.7,
      nGpuLayers: 4,
      nCtx: 2048,
      nThreads: 4,
      nPredict: 256,
      topP: 0.9,
      penaltyRepeat: 1.1,
    ),
  );

  final _modelUrl =
      'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf';
  String? modelPath;
  TextEditingController controller = TextEditingController(
    text: 'What is capital of Poland?',
  );
  String text = 'Initial text';
  bool modelFileExists = false;
  bool isDownloading = false;
  double downloadProgress = 0;

  // Performance metrics
  String performanceMetrics = '';
  bool showMetrics = true;

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    if (_llmcppPlugin.isInitialized) {
      _llmcppPlugin.clean();
    }
    _llmcppPlugin.dispose();
    controller.dispose();
    super.dispose();
  }

  Future<String?> _downloadFile() async {
    try {
      final pathDirectory = await getApplicationDocumentsDirectory();
      if (File('${pathDirectory.path}/model.gguf').existsSync()) {
        setState(() {
          modelFileExists = true;
        });
        return '${pathDirectory.path}/model.gguf';
      }
      setState(() {
        isDownloading = true;
      });
      final httpClient = HttpClient();
      final request = await httpClient.getUrl(Uri.parse(_modelUrl));
      final response = await request.close();

      if (response.statusCode == 200) {
        final contentLength = response.contentLength;
        final filePath = '${pathDirectory.path}/model.gguf';
        final file = File(filePath);
        final sink = file.openWrite();

        int downloadedBytes = 0;

        await for (var chunk in response) {
          sink.add(chunk);
          downloadedBytes += chunk.length;

          if (contentLength > 0) {
            final progress = (downloadedBytes / contentLength * 100);
            setState(() {
              downloadProgress = progress;
            });
          }
        }

        await sink.close();
        if (File('${pathDirectory.path}/model.gguf').existsSync()) {
          setState(() {
            modelFileExists = true;
          });
        }

        setState(() {
          isDownloading = false;
        });
        return filePath;
      } else {
        print('File downloading error: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      setState(() {
        isDownloading = false;
      });
      print('Exception during file download: $e');
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
          title: Text('Example'),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: .center,
            children: modelFileExists
                ? [
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: SizedBox(
                        height: 60,
                        child: TextField(
                          controller: controller,
                          decoration: const InputDecoration(
                            hintText: 'Enter prompt...',
                          ),
                        ),
                      ),
                    ),
                    const Divider(),
                    // Performance metrics display
                    if (performanceMetrics.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.all(16),
                        child: Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.blue.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                              color: Colors.blue.withOpacity(0.3),
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Row(
                                children: [
                                  Icon(
                                    Icons.speed,
                                    size: 20,
                                    color: Colors.blue,
                                  ),
                                  SizedBox(width: 8),
                                  Text(
                                    '⚡ Performance Metrics (Live)',
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: 16,
                                      color: Colors.blue,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Text(
                                performanceMetrics,
                                style: const TextStyle(
                                  fontFamily: 'monospace',
                                  fontSize: 13,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    Expanded(
                      child: SingleChildScrollView(
                        padding: const EdgeInsets.all(16),
                        child: Text(
                          text.isEmpty ? 'Press send to generate...' : text,
                          style: Theme.of(context).textTheme.bodyLarge,
                        ),
                      ),
                    ),
                  ]
                : [
                    isDownloading
                        ? Column(
                            children: [
                              const Text('Downloading model...'),
                              Text('${downloadProgress.toStringAsFixed(2)} %'),
                              CircularProgressIndicator(
                                value: downloadProgress / 100,
                              ),
                            ],
                          )
                        : ElevatedButton(
                            onPressed: () async {
                              _downloadFile().then((path) {
                                setState(() {
                                  modelPath = path;
                                });
                              });
                            },
                            child: const Text('Download Model'),
                          ),
                  ],
          ),
        ),
        floatingActionButton: modelFileExists
            ? FloatingActionButton(
                onPressed: () async {
                  setState(() {
                    text = '';
                    performanceMetrics = '';
                  });

                  await _llmcppPlugin.loadModel(modelPath!);

                  // Use streaming with live metrics
                  _llmcppPlugin
                      .sendPromptStream(controller.text)
                      .listen(
                        (chunk) {
                          setState(() {
                            // Append text from chunk
                            if (chunk.text.isNotEmpty) {
                              text = text + chunk.text;
                            }

                            // Update metrics in real-time
                            if (chunk.metrics != null) {
                              final m = chunk.metrics!;
                              performanceMetrics =
                                  '''
Tokens Generated: ${m.tokensGenerated}
Duration: ${m.durationMs}ms
Speed: ${m.tokensPerSecond.toStringAsFixed(2)} t/s
Time per token: ${m.msPerToken.toStringAsFixed(2)}ms
''';
                            }
                          });
                        },
                        onError: (error) {
                          setState(() {
                            text = 'Error: $error';
                          });
                        },
                      );
                },
                tooltip: 'Generate',
                child: const Icon(Icons.send),
              )
            : null,
      ),
    );
  }
}
