// lib/llmcpp.dart

// ── External prompt formats (from llama_cpp_dart) ───────────────────────────
export 'package:llama_cpp_dart/llama_cpp_dart.dart'
    show PromptFormat, ChatMLFormat, AlpacaFormat, GemmaFormat;

// ── Core (typy danych, metryki, wyniki) ──────────────────────────────────────
export 'src/core/generation_result.dart';
export 'src/core/llm_config.dart';
export 'src/core/llm_interface.dart';
export 'src/core/performance_metrics.dart';
export 'src/core/streaming_result.dart';
// ── Data (provider implementations) ─────────────────────────────────────────
export 'src/data/providers/local_gguf_provider.dart';
export 'src/data/providers/openai_provider.dart';
// ── Domain (abstrakcje) ───────────────────────────────────────────────────────
export 'src/domain/providers/llm_provider.dart';
// ── Prompt formats ────────────────────────────────────────────────────────────
export 'src/formats/harmony_format.dart';
export 'src/formats/prompt_format.dart';
// ── Legacy (backwards compatibility) ─────────────────────────────────────────
export 'src/models/llm_model_base.dart';
export 'src/models/llm_model_isolated.dart';
export 'src/models/llm_model_standard.dart';
// ── Presentation (main plugin class) ─────────────────────────────────────────
export 'src/presentation/llm_plugin.dart';
export 'src/rag/chunking/text_chunker.dart';
// ── RAG (Retrieval-Augmented Generation) ─────────────────────────────────────
export 'src/rag/document/document.dart';
export 'src/rag/document/document_chunk.dart';
export 'src/rag/embeddings/embedding_provider.dart';
export 'src/rag/embeddings/llama_embedding_provider.dart';
export 'src/rag/llama_rag_coordinator.dart';
export 'src/rag/rag_pipeline.dart';
export 'src/rag/rag_plugin.dart';
export 'src/rag/vector_store/in_memory_vector_store.dart';
export 'src/rag/vector_store/vector_store.dart';
