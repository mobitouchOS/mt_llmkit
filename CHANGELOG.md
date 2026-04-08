## 0.0.1-beta.1

Initial beta release of **mt_llmkit**.

### Features

- **Local GGUF inference** — run quantized LLMs entirely on-device via [llamadart](https://pub.dev/packages/llamadart), with no internet connection required
- **Two execution backends** — `ModelBackend.isolate` (default, Dart Isolate, no UI jank) and `ModelBackend.inProcess` (lighter startup, supports `clean()`)
- **Three generation methods** on `LocalModel` / `LlmInterface`:
  - `sendPrompt` — raw token stream
  - `sendPromptComplete` — full response as a single `String`
  - `sendPromptStream` — token stream with live `PerformanceMetrics` (recommended)
- **Vision / multimodal** — supports LLaVA, Gemma 3, Qwen VL, SmolVLM and any `libmtmd`-compatible model via `LlmConfig.mmprojPath` and `LlamaImageContent`
- **Performance metrics** — `PerformanceMetrics` with `tokensGenerated`, `durationMs`, `tokensPerSecond`, `msPerToken` updated on every `StreamingChunk`
- **Cloud AI chat providers** — unified `AIChatProvider` interface with implementations for OpenAI, Google Gemini, Anthropic Claude, and Mistral AI; automatic retry with exponential back-off
- **Local RAG pipeline** — fully on-device `RagEngine` with document chunking, embedding (via a separate CPU isolate), cosine-similarity vector search, and optional index persistence
