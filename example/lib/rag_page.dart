// example/lib/rag_page.dart
//
// Demonstracja pełnego pipeline'u RAG:
//
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │  1. Modele: model generowania + model embeddingów (osobne GGUF)     │
//  │  2. Dokumenty: PDF/TXT → ekstrakcja tekstu → TextChunker → chunks   │
//  │  3. Ingestia: chunks → LlamaEmbeddingProvider → InMemoryVectorStore  │
//  │  4. Zapytanie: query → embed → search → augment prompt → stream      │
//  └─────────────────────────────────────────────────────────────────────┘

import 'dart:async';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:llmcpp/llmcpp.dart';
import 'package:path_provider/path_provider.dart';
import 'package:read_pdf_text/read_pdf_text.dart';

// ── Modele downloadów ──────────────────────────────────────────────────────

class _ModelSpec {
  final String name;
  final String url;
  final String filename;
  final String description;

  const _ModelSpec({
    required this.name,
    required this.url,
    required this.filename,
    required this.description,
  });
}

const _generationModel = _ModelSpec(
  name: 'Llama-3.2-1B-Instruct Q4_K_M',
  url:
      'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
  filename: 'model.gguf',
  description: 'Model generowania (~800MB)',
);

const _embeddingModel = _ModelSpec(
  name: 'nomic-embed-text-v1.5 Q4_K_M',
  url:
      'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf',
  filename: 'embed_model.gguf',
  description: 'Model embeddingów (~270MB)',
);

// ── Widok dokumentu w liście ───────────────────────────────────────────────

class _IndexedDoc {
  final Document document;
  final int chunkCount;
  _IndexedDoc(this.document, this.chunkCount);
}

// ── Strona RAG ─────────────────────────────────────────────────────────────

class RagPage extends StatefulWidget {
  const RagPage({super.key});

  @override
  State<RagPage> createState() => _RagPageState();
}

class _RagPageState extends State<RagPage> {
  // ── State: modele ──────────────────────────────────────────────────────
  String? _generationModelPath;
  String? _embeddingModelPath;
  bool _generationModelReady = false;
  bool _embeddingModelReady = false;
  bool _isDownloadingGen = false;
  bool _isDownloadingEmbed = false;
  double _downloadProgressGen = 0;
  double _downloadProgressEmbed = 0;

  // ── State: pipeline ────────────────────────────────────────────────────
  LlamaRagCoordinator? _coordinator;
  RagPipeline? _pipeline;
  bool _pipelineReady = false;

  // ── State: dokumenty ───────────────────────────────────────────────────
  final List<_IndexedDoc> _indexedDocs = [];
  bool _isIngesting = false;
  RagIngestionProgress? _ingestionProgress;

  // ── State: zapytanie ───────────────────────────────────────────────────
  bool _isQuerying = false;
  String _answer = '';
  List<VectorSearchResult> _sources = [];
  String _metricsText = '';
  StreamSubscription<StreamingChunk>? _querySubscription;

  // ── Kontrolery ─────────────────────────────────────────────────────────
  final _queryController = TextEditingController();
  final _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _checkExistingModels();
  }

  @override
  void dispose() {
    _querySubscription?.cancel();
    _pipeline?.dispose();
    _coordinator?.dispose();
    _queryController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ── Sprawdzenie istniejących modeli ────────────────────────────────────

  Future<void> _checkExistingModels() async {
    final dir = await getApplicationDocumentsDirectory();
    final genFile = File('${dir.path}/${_generationModel.filename}');
    final embedFile = File('${dir.path}/${_embeddingModel.filename}');

    setState(() {
      if (genFile.existsSync()) {
        _generationModelPath = genFile.path;
        _generationModelReady = true;
      }
      if (embedFile.existsSync()) {
        _embeddingModelPath = embedFile.path;
        _embeddingModelReady = true;
      }
    });

    // Wczytaj wcześniej zapisany indeks jeśli oba modele gotowe
    if (_generationModelReady && _embeddingModelReady) {
      await _tryLoadSavedIndex();
    }
  }

  Future<void> _tryLoadSavedIndex() async {
    final dir = await getApplicationDocumentsDirectory();
    final indexPath = '${dir.path}/rag_index.json';
    if (File(indexPath).existsSync()) {
      // Indeks zostanie załadowany przy initializacji pipeline
      _showSnack(
        'Znaleziono zapisany indeks — zostanie wczytany po zainicjowaniu.',
      );
    }
  }

  // ── Pobieranie modeli ──────────────────────────────────────────────────

  Future<void> _downloadModel(_ModelSpec spec) async {
    final isGen = spec == _generationModel;
    setState(() {
      if (isGen) {
        _isDownloadingGen = true;
        _downloadProgressGen = 0;
      } else {
        _isDownloadingEmbed = true;
        _downloadProgressEmbed = 0;
      }
    });

    try {
      final dir = await getApplicationDocumentsDirectory();
      final filePath = '${dir.path}/${spec.filename}';
      final httpClient = HttpClient();
      final request = await httpClient.getUrl(Uri.parse(spec.url));
      final response = await request.close();

      if (response.statusCode == 200) {
        final contentLength = response.contentLength;
        final sink = File(filePath).openWrite();
        int downloaded = 0;

        await for (final chunk in response) {
          sink.add(chunk);
          downloaded += chunk.length;
          if (contentLength > 0) {
            final progress = downloaded / contentLength;
            setState(() {
              if (isGen) {
                _downloadProgressGen = progress;
              } else {
                _downloadProgressEmbed = progress;
              }
            });
          }
        }
        await sink.close();

        setState(() {
          if (isGen) {
            _generationModelPath = filePath;
            _generationModelReady = true;
            _isDownloadingGen = false;
          } else {
            _embeddingModelPath = filePath;
            _embeddingModelReady = true;
            _isDownloadingEmbed = false;
          }
        });
      } else {
        throw HttpException('HTTP ${response.statusCode}');
      }
    } catch (e) {
      setState(() {
        if (isGen) {
          _isDownloadingGen = false;
        } else {
          _isDownloadingEmbed = false;
        }
      });
      _showError('Błąd pobierania ${spec.name}: $e');
    }
  }

  // ── Inicjalizacja pipeline ─────────────────────────────────────────────

  /// Inicjalizuje RagPipeline przez [LlamaRagCoordinator].
  ///
  /// Koordynator ładuje oba modele (embeddingi + generowanie) w JEDNYM
  /// worker isolate — eliminuje crash "Cannot invoke native callback from a
  /// different isolate" spowodowany przez wyścig `llama_log_set` w dwóch
  /// izolowanych kontekstach.
  Future<void> _initializePipeline() async {
    if (!_generationModelReady || !_embeddingModelReady) return;

    try {
      final dir = await getApplicationDocumentsDirectory();
      final indexPath = '${dir.path}/rag_index.json';

      // Jeden izolat zarządza oboma modelami — brak wyścigu log callback
      final coordinator = await LlamaRagCoordinator.create(
        embedModelPath: _embeddingModelPath!,
        genModelPath: _generationModelPath!,
        genConfig: const LlmConfig(
          temp: 0.2,
          topP: 0.85,
          topK: 30,
          nBatch: 512,
          penaltyRepeat: 1.15,
          nCtx: 4096,
          nGpuLayers: 4,
          nThreads: 4,
          nPredict: 384,
        ),
        embedNCtx: 1024,
      );

      // VectorStore z auto-save i wczytaniem wcześniejszego indeksu
      final vectorStore = InMemoryVectorStore(autoSavePath: indexPath);
      await vectorStore.load(indexPath);

      final pipeline = RagPipeline(
        embeddingProvider: coordinator.embeddingProvider,
        vectorStore: vectorStore,
        generationPlugin: coordinator.generationPlugin,
        chunker: const TextChunker(chunkSize: 400, chunkOverlap: 80),
      );

      setState(() {
        _coordinator = coordinator;
        _pipeline = pipeline;
        _pipelineReady = true;
      });

      if (vectorStore.size > 0) {
        _showSnack('Wczytano indeks: ${vectorStore.indexedSize} chunków');
      }
    } catch (e) {
      _showError('Błąd inicjalizacji pipeline: $e');
    }
  }

  // ── Dodawanie dokumentu ────────────────────────────────────────────────

  Future<void> _pickAndIngestFile() async {
    print('picking');
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'txt'],
      allowMultiple: false,
    );
    print('picked: $result');
    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;
    final path = file.path;
    if (path == null) return;

    final extension = file.extension?.toLowerCase();
    late final String content;

    try {
      content = extension == 'pdf'
          ? await _extractPdfText(path)
          : await File(path).readAsString();
    } catch (e) {
      _showError('Błąd odczytu pliku: $e');
      return;
    }

    if (content.trim().isEmpty) {
      _showError('Plik jest pusty lub nie można wyekstrahować tekstu.');
      return;
    }

    final document = extension == 'pdf'
        ? Document.fromPdf(content, source: path)
        : Document.fromText(content, source: path);

    await _ingestDocument(document);
  }

  /// Ekstrakcja tekstu z PDF przy użyciu syncfusion_flutter_pdf.
  Future<String> _extractPdfText(String pdfPath) async {
    // final bytes = await File(pdfPath).readAsBytes();
    // final pdfDocument = PdfDocument(inputBytes: bytes);
    // final extractor = PdfTextExtractor(pdfDocument);
    final text = await ReadPdfText.getPDFtextPaginated(pdfPath);

    final buffer = StringBuffer();
    for (int i = 0; i < text.length; i++) {
      // final pageText = extractor.extractText(
      //   startPageIndex: i,
      //   endPageIndex: i,
      // );
      if (text[i].isNotEmpty) {
        buffer.writeln(text[i]);
      }
    }

    // text.dispose();
    return buffer.toString();
  }

  /// Ingestion dokumentu: chunk → embed → store.
  ///
  /// Stream [RagPipeline.ingestDocument] informuje o postępie po każdym
  /// zaembeddowanym chunku — aktualizujemy UI w czasie rzeczywistym.
  Future<void> _ingestDocument(Document document) async {
    if (_pipeline == null) return;

    setState(() {
      _isIngesting = true;
      _ingestionProgress = null;
    });

    int chunkCount = 0;
    try {
      await for (final progress in _pipeline!.ingestDocument(document)) {
        setState(() {
          _ingestionProgress = progress;
          chunkCount = progress.embeddedChunks;
        });
      }

      setState(() {
        _indexedDocs.add(_IndexedDoc(document, chunkCount));
        _isIngesting = false;
        _ingestionProgress = null;
      });

      _showSnack('Zaindeksowano "${document.title}" ($chunkCount chunków)');
    } catch (e) {
      setState(() => _isIngesting = false);
      _showError('Błąd ingestii: $e');
    }
  }

  Future<void> _removeDocument(_IndexedDoc doc) async {
    await _pipeline!.vectorStore.removeDocument(doc.document.id);
    setState(() => _indexedDocs.remove(doc));
    _showSnack('Usunięto "${doc.document.title}" z indeksu');
  }

  // ── Zapytanie RAG ──────────────────────────────────────────────────────

  /// Odpytuje pipeline RAG i wyświetla streaming odpowiedzi z sourcami.
  ///
  /// Flow:
  /// 1. embed(query) → queryEmbedding
  /// 2. vectorStore.search(queryEmbedding) → relevantChunks
  /// 3. buildPrompt(context, question) → augmentedPrompt
  /// 4. generationPlugin.sendPromptStream(augmentedPrompt) → `Stream<StreamingChunk>`
  Future<void> _sendQuery() async {
    final question = _queryController.text.trim();
    if (question.isEmpty || _pipeline == null || _isQuerying) return;

    await _querySubscription?.cancel();

    setState(() {
      _isQuerying = true;
      _answer = '';
      _sources = [];
      _metricsText = '';
    });

    try {
      // Pobierz relevantne chunki (do wyświetlenia jako źródła)
      _sources = await _pipeline!.findRelevant(
        question,
        topK: 4,
        minSimilarity: 0.25,
      );
      setState(() {});

      // Streaming odpowiedzi z RAG
      final answerBuffer = StringBuffer();
      _querySubscription = _pipeline!
          .query(question, topK: 4, minSimilarity: 0.25)
          .listen(
            (chunk) {
              setState(() {
                if (chunk.text.isNotEmpty) {
                  answerBuffer.write(chunk.text);
                  _answer = answerBuffer.toString();
                }
                if (chunk.metrics != null) {
                  final m = chunk.metrics!;
                  _metricsText =
                      '${m.tokensGenerated} tokenów │ '
                      '${m.tokensPerSecond.toStringAsFixed(1)} t/s │ '
                      '${(m.durationMs / 1000).toStringAsFixed(1)}s';
                }
                if (chunk.isFinal) {
                  _isQuerying = false;
                }
              });

              WidgetsBinding.instance.addPostFrameCallback((_) {
                if (_scrollController.hasClients) {
                  _scrollController.animateTo(
                    _scrollController.position.maxScrollExtent,
                    duration: const Duration(milliseconds: 100),
                    curve: Curves.easeOut,
                  );
                }
              });
            },
            onError: (Object error) {
              setState(() {
                _answer = 'Błąd: $error';
                _isQuerying = false;
              });
            },
            onDone: () {
              if (mounted) setState(() => _isQuerying = false);
            },
          );
    } catch (e) {
      setState(() {
        _answer = 'Błąd: $e';
        _isQuerying = false;
      });
    }
  }

  Future<void> _stopQuery() async {
    await _querySubscription?.cancel();
    setState(() => _isQuerying = false);
  }

  // ── UI helpers ─────────────────────────────────────────────────────────

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: Colors.red.shade700),
    );
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  // ── Build ──────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        controller: _scrollController,
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Sekcja: Modele ──────────────────────────────────────────
            _SectionHeader(title: '1. Modele', icon: Icons.memory),
            _buildModelRow(
              _generationModel,
              _generationModelReady,
              _isDownloadingGen,
              _downloadProgressGen,
            ),
            const SizedBox(height: 8),
            _buildModelRow(
              _embeddingModel,
              _embeddingModelReady,
              _isDownloadingEmbed,
              _downloadProgressEmbed,
            ),
            const SizedBox(height: 8),

            if (_generationModelReady &&
                _embeddingModelReady &&
                !_pipelineReady)
              ElevatedButton.icon(
                onPressed: _initializePipeline,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Inicjalizuj Pipeline RAG'),
              ),

            if (_pipelineReady)
              const _StatusChip(label: 'Pipeline gotowy', color: Colors.green),

            const SizedBox(height: 20),

            // ── Sekcja: Dokumenty ───────────────────────────────────────
            _SectionHeader(title: '2. Baza Wiedzy', icon: Icons.folder_open),

            if (!_pipelineReady)
              const Text(
                'Najpierw zainicjalizuj pipeline.',
                style: TextStyle(color: Colors.grey),
              ),

            if (_pipelineReady) ...[
              ElevatedButton.icon(
                onPressed: _isIngesting ? null : _pickAndIngestFile,
                icon: const Icon(Icons.attach_file),
                label: const Text('Dodaj plik (PDF / TXT)'),
              ),

              // Progress bar ingestii
              if (_isIngesting && _ingestionProgress != null) ...[
                const SizedBox(height: 12),
                _IngestionProgressBar(progress: _ingestionProgress!),
              ],

              // Lista zaindeksowanych dokumentów
              if (_indexedDocs.isNotEmpty) ...[
                const SizedBox(height: 12),
                ...List.generate(_indexedDocs.length, (i) {
                  final doc = _indexedDocs[i];
                  return _DocumentTile(
                    doc: doc,
                    onRemove: () => _removeDocument(doc),
                  );
                }),
              ],
            ],

            const SizedBox(height: 20),

            // ── Sekcja: Zapytanie RAG ────────────────────────────────────
            _SectionHeader(title: '3. Zapytaj', icon: Icons.question_answer),

            if (_pipelineReady && _pipeline!.vectorStore.indexedSize > 0) ...[
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _queryController,
                      maxLines: 2,
                      minLines: 1,
                      decoration: const InputDecoration(
                        hintText: 'Wpisz pytanie do dokumentów...',
                        border: OutlineInputBorder(),
                      ),
                      onSubmitted: (_) => _sendQuery(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  if (_isQuerying)
                    IconButton(
                      icon: const Icon(Icons.stop_circle),
                      onPressed: _stopQuery,
                      tooltip: 'Zatrzymaj',
                    )
                  else
                    IconButton.filled(
                      icon: const Icon(Icons.send),
                      onPressed: _sendQuery,
                      tooltip: 'Wyślij zapytanie',
                    ),
                ],
              ),
              const SizedBox(height: 12),

              // Metryki
              if (_metricsText.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.blue.shade50,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.speed, size: 14, color: Colors.blue.shade700),
                      const SizedBox(width: 6),
                      Text(
                        _metricsText,
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.blue.shade700,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ],
                  ),
                ),

              // Odpowiedź
              if (_answer.isNotEmpty) ...[
                const SizedBox(height: 12),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade50,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey.shade200),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.smart_toy, size: 16),
                          const SizedBox(width: 6),
                          const Text(
                            'Odpowiedź',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          if (_isQuerying) ...[
                            const SizedBox(width: 8),
                            const SizedBox(
                              width: 12,
                              height: 12,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                          ],
                        ],
                      ),
                      const SizedBox(height: 8),
                      SelectableText(
                        _answer,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                ),
              ],

              // Źródła (relevantne chunki)
              if (_sources.isNotEmpty) ...[
                const SizedBox(height: 12),
                const Text(
                  'Źródła (relevantne fragmenty):',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
                ),
                const SizedBox(height: 6),
                ..._sources.map((r) => _SourceChip(result: r)),
              ],
            ] else if (_pipelineReady) ...[
              const Text(
                'Dodaj dokumenty do bazy wiedzy, żeby móc zadawać pytania.',
                style: TextStyle(color: Colors.grey),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildModelRow(
    _ModelSpec spec,
    bool isReady,
    bool isDownloading,
    double progress,
  ) {
    if (isReady) {
      return Row(
        children: [
          Icon(Icons.check_circle, color: Colors.green.shade600, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(spec.name, style: const TextStyle(fontSize: 13)),
          ),
          Text(
            spec.description,
            style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
          ),
        ],
      );
    }
    if (isDownloading) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Pobieranie ${spec.name}: ${(progress * 100).toStringAsFixed(1)}%',
            style: const TextStyle(fontSize: 13),
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(value: progress),
        ],
      );
    }
    return TextButton.icon(
      onPressed: () => _downloadModel(spec),
      icon: const Icon(Icons.download, size: 18),
      label: Text('Pobierz ${spec.name}'),
    );
  }
}

// ── Pomocnicze widgety ─────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final String title;
  final IconData icon;
  const _SectionHeader({required this.title, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Icon(icon, size: 20, color: Theme.of(context).colorScheme.primary),
          const SizedBox(width: 8),
          Text(
            title,
            style: Theme.of(
              context,
            ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  final String label;
  final Color color;
  const _StatusChip({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(Icons.check, size: 16, color: color),
      label: Text(label, style: TextStyle(color: color, fontSize: 12)),
      backgroundColor: color.withValues(alpha: 0.1),
      side: BorderSide(color: color.withValues(alpha: 0.3)),
      padding: EdgeInsets.zero,
    );
  }
}

class _DocumentTile extends StatelessWidget {
  final _IndexedDoc doc;
  final VoidCallback onRemove;
  const _DocumentTile({required this.doc, required this.onRemove});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      child: ListTile(
        dense: true,
        leading: Icon(
          doc.document.type == 'pdf'
              ? Icons.picture_as_pdf
              : Icons.text_snippet,
          color: doc.document.type == 'pdf' ? Colors.red : Colors.blue,
        ),
        title: Text(doc.document.title, style: const TextStyle(fontSize: 13)),
        subtitle: Text(
          '${doc.chunkCount} chunków • ${doc.document.contentLength} znaków',
          style: const TextStyle(fontSize: 11),
        ),
        trailing: IconButton(
          icon: const Icon(Icons.delete_outline, size: 20),
          onPressed: onRemove,
          tooltip: 'Usuń z indeksu',
        ),
      ),
    );
  }
}

class _IngestionProgressBar extends StatelessWidget {
  final RagIngestionProgress progress;
  const _IngestionProgressBar({required this.progress});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Embedding ${progress.embeddedChunks}/${progress.totalChunks} chunków...',
          style: const TextStyle(fontSize: 12),
        ),
        const SizedBox(height: 4),
        LinearProgressIndicator(value: progress.fraction),
        if (progress.currentPreview.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              '"${progress.currentPreview}..."',
              style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
      ],
    );
  }
}

class _SourceChip extends StatelessWidget {
  final VectorSearchResult result;
  const _SourceChip({required this.result});

  @override
  Widget build(BuildContext context) {
    final pct = (result.similarity * 100).toStringAsFixed(0);
    final title =
        result.chunk.metadata['documentTitle'] as String? ??
        result.chunk.documentId;

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.green.shade50,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.green.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.format_quote, size: 14, color: Colors.green.shade700),
              const SizedBox(width: 4),
              Text(
                '#${result.rank} • $title • $pct% zbieżność',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  color: Colors.green.shade700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            result.chunk.preview,
            style: const TextStyle(fontSize: 11),
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
