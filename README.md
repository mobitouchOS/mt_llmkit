# llmcpp

A Flutter plugin for running Large Language Models (LLMs) locally on Android and iOS using [llama.cpp](https://github.com/ggerganov/llama.cpp) via FFI. Also provides a unified interface for cloud AI chat providers (OpenAI, Gemini, Claude, Mistral) and a fully local RAG (Retrieval-Augmented Generation) pipeline.

---

## Table of Contents

1. [Installation](#installation)
2. [Local LLM Inference (GGUF)](#local-llm-inference-gguf)
   - [Quick start](#quick-start)
   - [Backends](#backends)
   - [Configuration](#configuration)
   - [Generation methods](#generation-methods)
   - [Prompt formats](#prompt-formats)
   - [Performance metrics](#performance-metrics)
3. [Cloud API Providers](#cloud-api-providers)
   - [Supported providers](#supported-providers)
   - [Basic usage](#basic-usage)
   - [Multi-turn conversations](#multi-turn-conversations)
   - [Streaming](#streaming)
   - [Provider-specific config](#provider-specific-config)
   - [Error handling](#error-handling)
4. [Local RAG Pipeline](#local-rag-pipeline)
   - [How it works](#how-it-works)
   - [Quick start](#quick-start-1)
   - [Document ingestion](#document-ingestion)
   - [Querying](#querying)
   - [Index persistence](#index-persistence)
   - [Advanced: custom prompt template](#advanced-custom-prompt-template)

---

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  llmcpp: ^1.0.0
```

Then run:

```bash
flutter pub get
```

Import the library:

```dart
import 'package:llmcpp/llmcpp.dart';
```

---

## Local LLM Inference (GGUF)

Run quantized GGUF models entirely on-device — no internet connection required.

### Quick start

```dart
final plugin = GgufPlugin(
  config: LlmConfig(temp: 0.7, nCtx: 2048, nGpuLayers: 4),
);

await plugin.loadModel('/path/to/model.gguf');

// Stream tokens as they are generated
plugin.sendPrompt('What is Flutter?').listen((token) {
  stdout.write(token);
});

plugin.dispose();
```

### Backends

`GgufPlugin` supports two backends controlled by the `backend` parameter:

| Backend | Class | When to use |
|---|---|---|
| `GGUFBackend.isolate` *(default)* | `LlmModelIsolated` | Production. Runs in a Dart Isolate — no UI jank. Required when loading multiple models (e.g. RAG). |
| `GGUFBackend.inProcess` | `LlmModelStandard` | Lighter startup cost. Supports `clean()` to reset context without reloading the model. |

```dart
// Isolate backend (default)
final plugin = GgufPlugin(backend: GGUFBackend.isolate);

// In-process backend
final plugin = GgufPlugin(backend: GGUFBackend.inProcess);
plugin.clean(); // reset context — only available with inProcess
```

> **Note:** `clean()` throws `UnsupportedError` when called on the `isolate` backend.

### Configuration

All parameters are optional; sensible defaults are applied automatically.

```dart
final config = LlmConfig(
  promptFormat: ChatMLFormat(),  // default
  nGpuLayers: 4,    // GPU layers offloaded (default: 64)
  nCtx: 2048,       // context window in tokens (default: 8192)
  nBatch: 512,      // batch size (default: 4096)
  nPredict: 1024,   // max tokens to generate (default: 8192)
  nThreads: 4,      // CPU threads (default: 6)
  temp: 0.7,        // temperature (default: 0.72)
  topK: 40,         // top-K sampling (default: 64)
  topP: 0.9,        // top-P sampling (default: 0.95)
  penaltyRepeat: 1.1, // repetition penalty (default: 1.1)
);
```

### Generation methods

Three methods are available on `GgufPlugin` (and any `LlmInterface` implementation):

| Method | Return type | Description |
|---|---|---|
| `sendPrompt(prompt)` | `Stream<String>` | Raw token stream. Lowest overhead. |
| `sendPromptComplete(prompt)` | `Future<String>` | Waits for the full response and returns it as a single string. |
| `sendPromptStream(prompt)` | `Stream<StreamingChunk>` | **Recommended.** Token stream with live performance metrics. |

```dart
// 1. Raw token stream
plugin.sendPrompt('Hello').listen(stdout.write);

// 2. Full response at once
final response = await plugin.sendPromptComplete('Hello');
print(response);

// 3. Streaming with live metrics (recommended)
plugin.sendPromptStream('Hello').listen((chunk) {
  stdout.write(chunk.text);

  if (chunk.isFinal && chunk.metrics != null) {
    final m = chunk.metrics!;
    print('\n--- ${m.tokensGenerated} tokens, ${m.tokensPerSecond.toStringAsFixed(1)} t/s ---');
  }
});
```

`StreamingChunk` fields:

| Field | Type | Description |
|---|---|---|
| `text` | `String` | The generated text fragment. |
| `isFinal` | `bool` | `true` on the last chunk of the response. |
| `metrics` | `PerformanceMetrics?` | Available on every chunk; most useful on the final one. |

`PerformanceMetrics` fields: `tokensGenerated`, `durationMs`, `tokensPerSecond`, `msPerToken`.

### Prompt formats

The format wraps the raw prompt in model-specific delimiters. Set it via `LlmConfig.promptFormat`:

```dart
// Built-in formats
LlmConfig(promptFormat: ChatMLFormat())   // default — Llama 3, Mistral, etc.
LlmConfig(promptFormat: AlpacaFormat())   // Alpaca instruction format
LlmConfig(promptFormat: GemmaFormat())    // Google Gemma

// Custom format (shipped in this package)
LlmConfig(promptFormat: HarmonyFormat())
```

### Performance metrics

`PerformanceMetrics` is updated incrementally with every `StreamingChunk`:

```dart
plugin.sendPromptStream('Explain Dart isolates in detail.').listen((chunk) {
  stdout.write(chunk.text);

  if (chunk.metrics != null) {
    final m = chunk.metrics!;
    // Update UI progress indicator
    print('${m.tokensGenerated} tokens | ${m.tokensPerSecond.toStringAsFixed(2)} t/s');
  }
});
```
---

## Cloud API Providers

`llmcpp` includes a unified `AIChatProvider` interface for four cloud LLM providers. All providers share the same API surface, making it easy to swap backends.

### Supported providers

| Provider | Enum value | Default model |
|---|---|---|
| OpenAI | `AIChatProviderType.openai` | `gpt-4o-mini` |
| Google Gemini | `AIChatProviderType.gemini` | `gemini-1.5-flash` |
| Anthropic Claude | `AIChatProviderType.claude` | `claude-haiku-4-5-20251001` |
| Mistral AI | `AIChatProviderType.mistral` | `mistral-small-latest` |

### Basic usage

Use `AIChatProviderFactory` to create a provider without importing the concrete class:

```dart
// Create and initialize in one step
final provider = await AIChatProviderFactory.createAndInitialize(
  AIChatProviderType.openai,
  {'apiKey': 'sk-...'},
);

final response = await provider.sendMessage('What is Flutter?');
print(response.message.content);
print('Tokens used: ${response.inputTokens} in / ${response.outputTokens} out');

await provider.dispose();
```

Or manage the lifecycle manually:

```dart
final provider = AIChatProviderFactory.create(AIChatProviderType.gemini);
await provider.initialize({'apiKey': 'AIza...'});

final response = await provider.sendMessage('Hello!');
print(response.message.content);

await provider.dispose();
```

### Multi-turn conversations

Build conversation history with `ChatMessage`:

```dart
final history = <ChatMessage>[
  ChatMessage.system('You are a concise assistant. Reply in three sentences max.'),
  ChatMessage.user('What is a Dart isolate?'),
  ChatMessage.assistant('A Dart isolate is an independent thread of execution...'),
];

// Continue the conversation
final r = await provider.sendMessage('Give me a code example.', history: history);
print(r.message.content);

// Append the reply to keep history growing
history.add(r.message);
```

For full control, pass a complete message list directly:

```dart
final response = await provider.sendChatMessages([
  ChatMessage.system('You are a poet.'),
  ChatMessage.user('Write a haiku about Flutter.'),
]);
print(response.message.content);
```

### Streaming

All providers support token streaming via `sendMessageStream`:

```dart
await for (final token in provider.sendMessageStream('Tell me a story.')) {
  stdout.write(token);
}
```

With conversation history:

```dart
final history = [ChatMessage.system('Reply only in Spanish.')];

await for (final token in provider.sendMessageStream('Hello!', history: history)) {
  stdout.write(token);
}
```

### Provider-specific config

#### OpenAI

```dart
await provider.initialize({
  'apiKey': 'sk-...',
  'model': 'gpt-4o',               // optional, default: gpt-4o-mini
  'baseUrl': 'https://...',        // optional, for Azure OpenAI or proxies
});
```

#### Google Gemini

```dart
await provider.initialize({
  'apiKey': 'AIza...',
  'model': 'gemini-1.5-pro',       // optional
});
```

#### Anthropic Claude

```dart
await provider.initialize({
  'apiKey': 'sk-ant-...',
  'model': 'claude-opus-4-6',      // optional
});
```

#### Mistral AI

```dart
await provider.initialize({
  'apiKey': '...',
  'model': 'mistral-large-latest', // optional
});
```

### Error handling

All providers throw typed exceptions from `chat_exceptions.dart`:

| Exception | Cause |
|---|---|
| `APIKeyException` | Invalid or missing API key (HTTP 401/403) |
| `NetworkException` | Transport error (timeout, DNS, connection reset) |
| `RateLimitException` | Quota exceeded (HTTP 429); contains `retryAfter` |
| `AIChatException` | Base class for any other API error |

Network and rate-limit errors are **automatically retried** up to 3 times with exponential back-off.

```dart
try {
  final response = await provider.sendMessage('Hello');
  print(response.message.content);
} on APIKeyException catch (e) {
  print('Check your API key: $e');
} on RateLimitException catch (e) {
  print('Rate limited. Retry after ${e.retryAfter?.inSeconds}s');
} on AIChatException catch (e) {
  print('API error: $e');
}
```

---

## Local RAG Pipeline

`RagPlugin` provides a fully on-device Retrieval-Augmented Generation pipeline. Documents are chunked, embedded with a local embedding model, stored in an in-memory vector store, and retrieved at query time to ground the generation model's response.

### How it works

```
Ingestion:  Document → TextChunker → chunks → EmbeddingModel → VectorStore
Query:      question → EmbeddingModel → VectorStore.search() → prompt + context → GenerationModel
```

Both the embedding model and the generation model run inside a **single worker isolate**, avoiding the native callback crash caused by llama.cpp's global log pointer.

### Quick start

You need two GGUF models:
- A **generation model** (e.g. Llama 3, Mistral) for producing answers.
- An **embedding model** (e.g. nomic-embed-text) for vectorising text.

```dart
final rag = RagPlugin(
  genModelPath: '/path/to/llama.gguf',
  embedModelPath: '/path/to/nomic-embed.gguf',
  genConfig: LlmConfig(temp: 0.3, nCtx: 4096, nGpuLayers: 4),
);

await rag.initialize();
```

### Document ingestion

Create a `Document` from text or PDF content and ingest it into the vector store:

```dart
// From plain text
final doc = Document.fromText(
  'Flutter is Google's UI toolkit for building natively compiled applications...',
  source: 'flutter_intro.txt',
);

// From extracted PDF text (use a PDF parser to extract the string first)
final pdfDoc = Document.fromPdf(
  extractedText,
  source: '/path/to/manual.pdf',
  pageCount: 42,
);

// Ingest — stream progress for UI updates
await for (final progress in rag.ingestDocument(doc)) {
  print('${progress.embeddedChunks}/${progress.totalChunks} — ${progress.currentPreview}');
  // progress.fraction gives 0.0–1.0 for a progress bar
}

print('Indexed: ${rag.indexedSize} chunks across ${rag.documentIds.length} documents');
```

Manage the index:

```dart
// Remove a single document (all its chunks)
await rag.removeDocument(doc.id);

// Clear everything
await rag.clearIndex();
```

### Querying

```dart
// Stream the generated answer
await for (final chunk in rag.query('What is Flutter?')) {
  stdout.write(chunk.text);

  if (chunk.isFinal && chunk.metrics != null) {
    print('\n${chunk.metrics!.tokensPerSecond.toStringAsFixed(1)} t/s');
  }
}
```

Optional query parameters:

```dart
rag.query(
  'Explain the rendering pipeline.',
  topK: 3,              // number of context chunks (default: 5)
  minSimilarity: 0.35,  // minimum cosine similarity 0.0–1.0 (default: 0.25)
)
```

Retrieve relevant chunks without generating an answer (useful for inspection):

```dart
final results = await rag.findRelevant('rendering pipeline', topK: 3);
for (final r in results) {
  print('${(r.similarity * 100).toStringAsFixed(0)}% — ${r.chunk.text.substring(0, 80)}');
}
```

### Index persistence

Pass `indexPath` to automatically save and restore the vector index between sessions:

```dart
final dir = await getApplicationDocumentsDirectory();

final rag = RagPlugin(
  genModelPath: '/path/to/llama.gguf',
  embedModelPath: '/path/to/nomic-embed.gguf',
  indexPath: '${dir.path}/rag_index.json',  // auto-save on every change
);

await rag.initialize(); // loads existing index if the file exists
```

When `indexPath` is `null`, the store is in-memory only and data is lost when the app restarts.

### Advanced: custom prompt template

The default template instructs the model to answer only from the provided context. Override it if you need different behaviour:

```dart
final rag = RagPlugin(
  genModelPath: '...',
  embedModelPath: '...',
  promptTemplate:
    'You are a helpful assistant. Use the context below to answer.\n\n'
    'CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:',
);
```

The template must contain `{context}` and `{question}` placeholders.

### Cleanup

```dart
rag.dispose(); // releases the worker isolate and both models; safe to call twice
```
