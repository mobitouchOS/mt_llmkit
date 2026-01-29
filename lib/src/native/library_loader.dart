import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

class LibraryLoader {
  static void initialize() {
    if (Platform.isAndroid) {
      // libmtmd
      Llama.libraryPath = 'libllama.so';
    } else if (Platform.isLinux) {
      _loadLinuxLibraries();
    }
  }

  static void _loadLinuxLibraries() {
    try {
      final libs = [
        'libggml-base.so',
        'libggml-cpu.so',
        'libggml-opencl.so',
        'libggml-vulkan.so',
        'libggml.so',
        'libllama.so',
        'libmtmd.so',
      ];
      for (final lib in libs) {
        ffi.DynamicLibrary.open(lib);
      }
    } catch (e) {
      if (kDebugMode) print('ggml preload failed: $e');
    }
    Llama.libraryPath = 'libmtmd.so';
  }
}
