// example/lib/rest_api_tab.dart

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:mt_llmkit/mt_llmkit.dart';

/// Self-contained tab for interacting with cloud AI providers via REST API.
///
/// Supports OpenAI, Gemini, Claude, and Mistral through the shared
/// [AIChatProvider] abstraction. The user selects a provider, enters an API
/// key, an optional model name, and a prompt, then streams the response.
class RestApiTab extends StatefulWidget {
  const RestApiTab({super.key});

  @override
  State<RestApiTab> createState() => _RestApiTabState();
}

class _RestApiTabState extends State<RestApiTab> {
  // ── Provider selection ──────────────────────────────────────────────────────

  AIChatProviderType _providerType = AIChatProviderType.openai;

  static const _providerLabels = {
    AIChatProviderType.openai: 'OpenAI',
    AIChatProviderType.gemini: 'Gemini',
    AIChatProviderType.claude: 'Claude',
    AIChatProviderType.mistral: 'Mistral',
  };

  static const _defaultModels = {
    AIChatProviderType.openai: 'gpt-4o-mini',
    AIChatProviderType.gemini: 'gemini-1.5-flash',
    AIChatProviderType.claude: 'claude-haiku-4-5-20251001',
    AIChatProviderType.mistral: 'mistral-small-latest',
  };

  // ── Controllers ─────────────────────────────────────────────────────────────

  final _apiKeyController = TextEditingController();
  final _modelController = TextEditingController();
  final _promptController = TextEditingController(
    text: 'What is the capital of Poland? Describe it in 2 sentences.',
  );
  final _scrollController = ScrollController();

  // ── Runtime state ────────────────────────────────────────────────────────────

  AIChatProvider? _provider;
  StreamSubscription<String>? _streamSub;
  final _outputBuffer = StringBuffer();

  bool _isGenerating = false;
  String _output = '';
  String _statusMessage = '';

  // ── Lifecycle ────────────────────────────────────────────────────────────────

  @override
  void dispose() {
    _streamSub?.cancel();
    _provider?.dispose();
    _apiKeyController.dispose();
    _modelController.dispose();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ── Actions ──────────────────────────────────────────────────────────────────

  Future<void> _send() async {
    if (_isGenerating) return;

    // Dispose the previous provider instance before creating a new one.
    await _provider?.dispose();
    _provider = null;

    final apiKey = _apiKeyController.text.trim();
    final model = _modelController.text.trim();
    final prompt = _promptController.text.trim();

    if (prompt.isEmpty) {
      _showError('Prompt cannot be empty.');
      return;
    }

    final config = <String, dynamic>{'apiKey': apiKey};
    if (model.isNotEmpty) config['model'] = model;

    // Initialize the provider — ArgumentError thrown for missing/empty apiKey.
    try {
      _provider = AIChatProviderFactory.create(_providerType);
      await _provider!.initialize(config);
    } on ArgumentError catch (e) {
      _showError('Configuration error: ${e.message}');
      return;
    } catch (e) {
      _showError('Initialization failed: $e');
      return;
    }

    setState(() {
      _isGenerating = true;
      _outputBuffer.clear();
      _output = '';
      _statusMessage = 'Generating…';
    });

    _streamSub = _provider!
        .sendMessageStream(prompt)
        .listen(
          (token) {
            setState(() {
              _outputBuffer.write(token);
              _output = _outputBuffer.toString();
            });
            // Auto-scroll to bottom as tokens arrive.
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (_scrollController.hasClients) {
                _scrollController.jumpTo(
                  _scrollController.position.maxScrollExtent,
                );
              }
            });
          },
          onError: (Object error) {
            if (!mounted) return;
            setState(() {
              _isGenerating = false;
              _statusMessage = 'Error';
            });
            _showError('$error');
          },
          onDone: () {
            if (!mounted) return;
            setState(() {
              _isGenerating = false;
              _statusMessage = 'Done';
            });
          },
        );
  }

  Future<void> _stop() async {
    await _streamSub?.cancel();
    _streamSub = null;
    if (mounted) {
      setState(() {
        _isGenerating = false;
        _statusMessage = 'Stopped';
      });
    }
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red.shade700),
    );
  }

  // ── Build ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final hintModel = _defaultModels[_providerType]!;

    return Column(
      children: [
        // ── Configuration fields ──────────────────────────────────────────────
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Provider selector
              DropdownButtonFormField<AIChatProviderType>(
                initialValue: _providerType,
                decoration: const InputDecoration(
                  labelText: 'Provider',
                  prefixIcon: Icon(Icons.cloud_outlined),
                  border: OutlineInputBorder(),
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 14,
                  ),
                ),
                items: AIChatProviderType.values
                    .map(
                      (t) => DropdownMenuItem(
                        value: t,
                        child: Text(_providerLabels[t]!),
                      ),
                    )
                    .toList(),
                onChanged: _isGenerating
                    ? null
                    : (value) {
                        if (value != null) {
                          setState(() => _providerType = value);
                        }
                      },
              ),
              const SizedBox(height: 8),

              // API key
              TextField(
                controller: _apiKeyController,
                obscureText: true,
                enabled: !_isGenerating,
                decoration: const InputDecoration(
                  labelText: 'API Key',
                  prefixIcon: Icon(Icons.key),
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 8),

              // Model (optional)
              TextField(
                controller: _modelController,
                enabled: !_isGenerating,
                decoration: InputDecoration(
                  labelText: 'Model (optional)',
                  hintText: hintModel,
                  prefixIcon: const Icon(Icons.model_training),
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 8),

              // Prompt
              TextField(
                controller: _promptController,
                maxLines: 3,
                minLines: 1,
                enabled: !_isGenerating,
                decoration: const InputDecoration(
                  labelText: 'Prompt',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 8),

              // Send / Stop row
              Row(
                children: [
                  if (_isGenerating)
                    FilledButton.icon(
                      onPressed: _stop,
                      icon: const Icon(Icons.stop_circle_outlined),
                      label: const Text('Stop'),
                      style: FilledButton.styleFrom(
                        backgroundColor: Colors.red.shade600,
                      ),
                    )
                  else
                    FilledButton.icon(
                      onPressed: _send,
                      icon: const Icon(Icons.send),
                      label: const Text('Send'),
                    ),
                  const SizedBox(width: 12),
                  if (_isGenerating)
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  if (_statusMessage.isNotEmpty) ...[
                    const SizedBox(width: 8),
                    Text(
                      _statusMessage,
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.black54,
                      ),
                    ),
                  ],
                ],
              ),
            ],
          ),
        ),

        const Divider(height: 16),

        // ── Response output ───────────────────────────────────────────────────
        Expanded(
          child: SingleChildScrollView(
            controller: _scrollController,
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: _output.isEmpty
                ? Text(
                    _isGenerating ? '' : 'Press Send to generate a response…',
                    style: TextStyle(color: Colors.grey.shade400),
                  )
                : SelectableText(
                    _output,
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
          ),
        ),
      ],
    );
  }
}
