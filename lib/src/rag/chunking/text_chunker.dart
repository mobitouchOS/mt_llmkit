// lib/src/rag/chunking/text_chunker.dart

import 'dart:math' as math;

import '../document/document.dart';
import '../document/document_chunk.dart';

/// Dzieli tekst dokumentu na nakładające się fragmenty (chunki).
///
/// ## Algorytm sliding-window z inteligentnym podziałem
///
/// 1. Okno przesuwa się o `chunkSize - chunkOverlap` znaków na iterację
/// 2. Na granicy każdego chunku szukamy naturalnego miejsca podziału
///    (koniec zdania: `.`, `!`, `?`, `\n`) w ostatniej 1/3 okna
/// 3. Jeśli nie znaleziono — używamy granicy słowa (spacja)
/// 4. Chunki krótsze niż `minChunkSize` są pomijane
///
/// ## Parametry
///
/// - `chunkSize` (domyślnie 500 znaków ≈ 100–150 tokenów) — docelowy rozmiar
/// - `chunkOverlap` (domyślnie 100 znaków) — nakładka dla ciągłości kontekstu
/// - `minChunkSize` (domyślnie 50 znaków) — minimum dla chunka, poniżej pomijany
///
/// ## Przykład
///
/// ```dart
/// final chunker = TextChunker(chunkSize: 500, chunkOverlap: 100);
/// final chunks = chunker.chunk(document);
/// print('Podzielono na ${chunks.length} chunków');
/// ```
class TextChunker {
  /// Docelowa liczba znaków w jednym chunku
  final int chunkSize;

  /// Liczba znaków nakładki między sąsiednimi chunkami
  final int chunkOverlap;

  /// Minimalna liczba znaków dla chunka (poniżej — pomijany)
  final int minChunkSize;

  const TextChunker({
    this.chunkSize = 500,
    this.chunkOverlap = 100,
    this.minChunkSize = 50,
  }) : assert(
          chunkOverlap < chunkSize,
          'chunkOverlap musi być mniejszy niż chunkSize',
        );

  /// Dzieli [document] na listę [DocumentChunk].
  ///
  /// Zwraca pustą listę jeśli dokument jest pusty.
  List<DocumentChunk> chunk(Document document) {
    final text = document.content.trim();
    if (text.isEmpty) return [];

    final chunks = <DocumentChunk>[];
    int start = 0;
    int chunkIndex = 0;

    while (start < text.length) {
      final rawEnd = math.min(start + chunkSize, text.length);

      // Znajdź naturalne miejsce podziału (nie rób tego na końcu tekstu)
      final end = rawEnd < text.length
          ? _findSplitPoint(text, start, rawEnd)
          : rawEnd;

      final chunkText = text.substring(start, end).trim();

      if (chunkText.length >= minChunkSize) {
        chunks.add(DocumentChunk(
          id: '${document.id}_chunk_$chunkIndex',
          documentId: document.id,
          text: chunkText,
          chunkIndex: chunkIndex,
          startChar: start,
          endChar: end,
          metadata: {
            'documentTitle': document.title,
            'documentSource': document.source,
          },
        ));
        chunkIndex++;
      }

      // Przesuń okno z uwzględnieniem nakładki
      final step = end - start - chunkOverlap;
      if (step <= 0) {
        // Zabezpieczenie przed nieskończoną pętlą dla bardzo małych tekstów
        break;
      }
      start += step;
    }

    return chunks;
  }

  /// Szuka najlepszego punktu podziału w zakresie `[rawEnd * 2/3, rawEnd]`.
  ///
  /// Priorytet:
  /// 1. Koniec zdania (`.`, `!`, `?`) + spacja lub nowa linia
  /// 2. Nowa linia (`\n`)
  /// 3. Granica słowa (ostatnia spacja w oknie)
  /// 4. Twarda granica `rawEnd` (fallback)
  int _findSplitPoint(String text, int start, int rawEnd) {
    // Szukaj tylko w ostatniej 1/3 okna — zachowaj większość chunkSize
    final searchStart = start + (rawEnd - start) * 2 ~/ 3;
    final candidate = text.substring(searchStart, rawEnd);

    // 1. Koniec zdania z następującą spacją lub nową linią
    final sentenceEnd = RegExp(r'[.!?][\s\n]');
    final sentenceMatch = sentenceEnd.allMatches(candidate).lastOrNull;
    if (sentenceMatch != null) {
      return searchStart + sentenceMatch.end;
    }

    // 2. Nowa linia
    final newlineIdx = candidate.lastIndexOf('\n');
    if (newlineIdx >= 0) {
      return searchStart + newlineIdx + 1;
    }

    // 3. Ostatnia spacja (granica słowa)
    final spaceIdx = candidate.lastIndexOf(' ');
    if (spaceIdx >= 0) {
      return searchStart + spaceIdx + 1;
    }

    // 4. Twarda granica
    return rawEnd;
  }
}
