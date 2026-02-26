// lib/llmcpp.dart

// ── External prompt formats (from llama_cpp_dart) ───────────────────────────
export 'package:llama_cpp_dart/llama_cpp_dart.dart'
    show PromptFormat, ChatMLFormat, AlpacaFormat, GemmaFormat;

// ── REST API providers (prompt-based) ────────────────────────────────────────
export 'src/api/openai_provider.dart';
export 'src/api/rest_api_provider.dart';
// ── AI Chat providers (conversation-based) ───────────────────────────────────
export 'src/api/ai_chat_provider.dart';
export 'src/api/ai_chat_provider_factory.dart';
export 'src/api/chat_exceptions.dart';
export 'src/api/chat_models.dart';
export 'src/api/claude_chat_provider.dart';
export 'src/api/gemini_chat_provider.dart';
export 'src/api/mistral_chat_provider.dart';
export 'src/api/openai_chat_provider.dart';
// ── Core ─────────────────────────────────────────────────────────────────────
export 'src/core/generation_result.dart';
export 'src/core/llm_config.dart';
export 'src/core/llm_interface.dart';
export 'src/core/performance_metrics.dart';
export 'src/core/streaming_result.dart';
// ── Prompt formats ────────────────────────────────────────────────────────────
export 'src/formats/harmony_format.dart';
export 'src/formats/prompt_format.dart';
// ── GGUF (local model plugin) ─────────────────────────────────────────────────
export 'src/gguf/gguf_plugin.dart';
// ── Models (low-level, for advanced use) ─────────────────────────────────────
export 'src/models/llm_model_base.dart';
export 'src/models/llm_model_isolated.dart';
export 'src/models/llm_model_standard.dart';
// ── RAG (Retrieval-Augmented Generation) ─────────────────────────────────────
export 'src/rag/chunking/text_chunker.dart';
export 'src/rag/document/document.dart';
export 'src/rag/document/document_chunk.dart';
export 'src/rag/embeddings/embedding_provider.dart';
export 'src/rag/embeddings/llama_embedding_provider.dart';
export 'src/rag/llama_rag_coordinator.dart';
export 'src/rag/rag_pipeline.dart';
export 'src/rag/rag_plugin.dart';
export 'src/rag/vector_store/in_memory_vector_store.dart';
export 'src/rag/vector_store/vector_store.dart';
