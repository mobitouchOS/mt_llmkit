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
  Llmcpp _llmcppPlugin = Llmcpp(
    temp: 0.7,
    nGpuLayers: 4,
    nCtx: 2048,
    nThreads: 4,
    nPredict: 256,
    topP: 0.9,
    penaltyRepeat: 1.1,
  );

  final _modelUrl =
      'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf';
  String? modelPath;
  TextEditingController controller = TextEditingController(
    text: '<s>[INST] What is the capital of France? [/INST]',
  );
  String text = 'Initial text';
  bool modelFileExists = false;
  bool isDownloading = false;
  double downloadProgress = 0;

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    _llmcppPlugin.clean();
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
                          decoration: InputDecoration(
                            hintText: 'Enter prompt...',
                          ),
                        ),
                      ),
                    ),
                    const Divider(),
                    Expanded(
                      child: SingleChildScrollView(
                        child: Text(
                          text,
                          style: Theme.of(context).textTheme.headlineMedium,
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
                                value: downloadProgress,
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
                    text = 'Generating...';
                  });
                  await _llmcppPlugin.loadIsolated(modelPath!);
                  _llmcppPlugin.sendPromptIsolated(controller.text)?.listen((
                    event,
                  ) {
                    setState(() {
                      text = text + event;
                    });
                  });

                  // _llmcppPlugin = Llmcpp();
                  // await _llmcppPlugin.loadModel(modelPath!);
                  // _llmcppPlugin.sendPrompt(controller.text)?.listen((event) {
                  //   setState(() {
                  //     text = text + event;
                  //   });
                  // });
                },
                tooltip: 'Generate',
                child: const Icon(Icons.send),
              )
            : null,
      ),
    );
  }
}
