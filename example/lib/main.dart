// example/lib/main.dart

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:llmcpp/llmcpp.dart';
import 'package:path_provider/path_provider.dart';

import 'rag_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'llmcpp Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MainPage(),
    );
  }
}

// ── Nawigacja główna ────────────────────────────────────────────────────────

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  int _selectedIndex = 0;

  static const _pages = [LlmDemoPage(), RagPage()];

  static const _labels = [
    NavigationDestination(icon: Icon(Icons.chat_outlined), selectedIcon: Icon(Icons.chat), label: 'LLM'),
    NavigationDestination(icon: Icon(Icons.search_outlined), selectedIcon: Icon(Icons.search), label: 'RAG'),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(_selectedIndex == 0 ? 'llmcpp — LLM Demo' : 'llmcpp — RAG Demo'),
      ),
      body: IndexedStack(
        index: _selectedIndex,
        children: _pages,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: _labels,
      ),
    );
  }
}

// ── Strona LLM (bez zmian, wydzielona do osobnej klasy) ────────────────────

enum ProviderType { localGGUF, openAI }

class LlmDemoPage extends StatefulWidget {
  const LlmDemoPage({super.key});

  @override
  State<LlmDemoPage> createState() => _LlmDemoPageState();
}

class _LlmDemoPageState extends State<LlmDemoPage> {
  LlmPlugin? _plugin;
  StreamSubscription<StreamingChunk>? _streamSubscription;

  ProviderType _selectedProvider = ProviderType.localGGUF;
  String? _modelPath;
  bool _isDownloading = false;
  double _downloadProgress = 0;
  bool _isModelReady = false;
  bool _isGenerating = false;

  final StringBuffer _outputBuffer = StringBuffer();
  String _output = '';
  String _metricsText = '';
  String _statusMessage = '';

  final _promptController = TextEditingController(
    text: 'Jaka jest stolica Polski? Opisz ją w 2 zdaniach.',
  );
  final _apiKeyController = TextEditingController();
  final _scrollController = ScrollController();

  static const _modelUrl =
      'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf';

  @override
  void initState() {
    super.initState();
    _checkModelExists();
  }

  @override
  void dispose() {
    _streamSubscription?.cancel();
    _plugin?.dispose();
    _promptController.dispose();
    _apiKeyController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _checkModelExists() async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/model.gguf');
    if (file.existsSync()) {
      setState(() {
        _modelPath = file.path;
        _isModelReady = true;
        _statusMessage = 'Model gotowy: Llama-3.2-1B-Instruct Q4_K_M';
      });
    }
  }

  Future<void> _downloadModel() async {
    setState(() {
      _isDownloading = true;
      _downloadProgress = 0;
      _statusMessage = 'Pobieranie modelu...';
    });

    try {
      final dir = await getApplicationDocumentsDirectory();
      final filePath = '${dir.path}/model.gguf';
      final httpClient = HttpClient();
      final request = await httpClient.getUrl(Uri.parse(_modelUrl));
      final response = await request.close();

      if (response.statusCode == 200) {
        final contentLength = response.contentLength;
        final file = File(filePath);
        final sink = file.openWrite();
        int downloadedBytes = 0;

        await for (final chunk in response) {
          sink.add(chunk);
          downloadedBytes += chunk.length;
          if (contentLength > 0) {
            setState(() {
              _downloadProgress = downloadedBytes / contentLength;
              _statusMessage = 'Pobieranie: ${(_downloadProgress * 100).toStringAsFixed(1)}%';
            });
          }
        }

        await sink.close();
        setState(() {
          _modelPath = filePath;
          _isModelReady = true;
          _isDownloading = false;
          _statusMessage = 'Model pobrany pomyślnie';
        });
      } else {
        throw HttpException('HTTP ${response.statusCode}');
      }
    } catch (e) {
      setState(() {
        _isDownloading = false;
        _statusMessage = '';
      });
      _showError('Błąd pobierania: $e');
    }
  }

  Future<bool> _initializePlugin() async {
    await _plugin?.dispose();
    _plugin = null;

    try {
      switch (_selectedProvider) {
        case ProviderType.localGGUF:
          _plugin = LlmPlugin.localGGUF(
            modelPath: _modelPath!,
            config: const LlmConfig(temp: 0.7, nGpuLayers: 4, nCtx: 2048, nBatch: 512, nThreads: 4, nPredict: 256),
          );
        case ProviderType.openAI:
          _plugin = LlmPlugin.openAI(apiKey: _apiKeyController.text, model: 'gpt-4o-mini');
      }
      await _plugin!.initialize();
      return true;
    } on UnimplementedError catch (e) {
      _showError('${e.message}');
      _plugin = null;
      return false;
    } catch (e) {
      _showError('Błąd inicjalizacji: $e');
      _plugin = null;
      return false;
    }
  }

  Future<void> _sendPrompt() async {
    if (_isGenerating) return;
    setState(() {
      _isGenerating = true;
      _output = '';
      _metricsText = '';
      _statusMessage = 'Inicjalizowanie...';
      _outputBuffer.clear();
    });

    if (_plugin == null || !_plugin!.isInitialized) {
      final success = await _initializePlugin();
      if (!success) {
        setState(() => _isGenerating = false);
        return;
      }
    }

    setState(() => _statusMessage = 'Generowanie...');

    _streamSubscription = _plugin!.sendPromptStream(_promptController.text).listen(
      (chunk) {
        setState(() {
          if (chunk.text.isNotEmpty) {
            _outputBuffer.write(chunk.text);
            _output = _outputBuffer.toString();
          }
          if (chunk.metrics != null) {
            final m = chunk.metrics!;
            _metricsText = 'Tokeny: ${m.tokensGenerated} │ ${m.tokensPerSecond.toStringAsFixed(1)} t/s │ ${m.msPerToken.toStringAsFixed(0)} ms/token';
          }
          if (chunk.isFinal) {
            _isGenerating = false;
            _statusMessage = 'Gotowe';
          }
        });
      },
      onError: (Object error) {
        setState(() {
          _output = 'Błąd: $error';
          _isGenerating = false;
          _statusMessage = 'Błąd';
        });
      },
      onDone: () {
        if (mounted) setState(() => _isGenerating = false);
      },
    );
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red.shade700),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: SegmentedButton<ProviderType>(
            segments: const [
              ButtonSegment(value: ProviderType.localGGUF, label: Text('Lokalny GGUF'), icon: Icon(Icons.computer)),
              ButtonSegment(value: ProviderType.openAI, label: Text('OpenAI (szkielet)'), icon: Icon(Icons.cloud_outlined)),
            ],
            selected: {_selectedProvider},
            onSelectionChanged: (selected) {
              setState(() {
                _selectedProvider = selected.first;
                _plugin?.dispose();
                _plugin = null;
              });
            },
          ),
        ),
        if (_selectedProvider == ProviderType.localGGUF)
          _buildLocalGGUFSection()
        else
          _buildOpenAISection(),
        const Divider(height: 1),
        if (_isModelReady || _selectedProvider == ProviderType.openAI)
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
            child: TextField(
              controller: _promptController,
              maxLines: 3,
              minLines: 1,
              decoration: const InputDecoration(labelText: 'Prompt', border: OutlineInputBorder()),
            ),
          ),
        if (_metricsText.isNotEmpty)
          Container(
            color: Colors.blue.shade50,
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            child: Text(_metricsText, style: TextStyle(fontSize: 11, color: Colors.blue.shade700, fontFamily: 'monospace')),
          ),
        Expanded(
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_output.isNotEmpty)
                  SelectableText(_output, style: Theme.of(context).textTheme.bodyLarge)
                else if (!_isGenerating && _isModelReady)
                  Text('Naciśnij ▶ aby wygenerować...', style: TextStyle(color: Colors.grey.shade400)),
                if (_isGenerating)
                  const Padding(
                    padding: EdgeInsets.only(top: 12),
                    child: SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
                  ),
              ],
            ),
          ),
        ),
        if (_statusMessage.isNotEmpty)
          Container(
            color: Colors.grey.shade100,
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            child: Text(_statusMessage, style: const TextStyle(fontSize: 11, color: Colors.black54)),
          ),
      ],
    );
  }

  Widget _buildLocalGGUFSection() {
    if (_isModelReady) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        child: Row(
          children: [
            Icon(Icons.check_circle, color: Colors.green.shade600, size: 18),
            const SizedBox(width: 8),
            const Expanded(child: Text('Llama-3.2-1B-Instruct Q4_K_M', style: TextStyle(fontSize: 13))),
            if (_isGenerating)
              IconButton(icon: const Icon(Icons.stop_circle_outlined), onPressed: () async {
                await _streamSubscription?.cancel();
                setState(() => _isGenerating = false);
              })
            else
              IconButton(icon: const Icon(Icons.send), onPressed: _sendPrompt),
          ],
        ),
      );
    }
    if (_isDownloading) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Pobieranie: ${(_downloadProgress * 100).toStringAsFixed(1)}%', style: const TextStyle(fontSize: 13)),
            const SizedBox(height: 6),
            LinearProgressIndicator(value: _downloadProgress),
          ],
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: ElevatedButton.icon(onPressed: _downloadModel, icon: const Icon(Icons.download), label: const Text('Pobierz model GGUF (~800MB)')),
    );
  }

  Widget _buildOpenAISection() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: Column(
        children: [
          TextField(
            controller: _apiKeyController,
            obscureText: true,
            decoration: const InputDecoration(labelText: 'OpenAI API Key', hintText: 'sk-...', prefixIcon: Icon(Icons.key), border: OutlineInputBorder()),
          ),
          const SizedBox(height: 6),
          const Text('Szkielet — metody rzucają UnimplementedError.', style: TextStyle(fontSize: 11, color: Colors.orange)),
        ],
      ),
    );
  }
}
