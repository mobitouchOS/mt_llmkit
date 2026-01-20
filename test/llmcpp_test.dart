import 'package:flutter_test/flutter_test.dart';
import 'package:llmcpp/llmcpp.dart';
import 'package:llmcpp/llmcpp_platform_interface.dart';
import 'package:llmcpp/llmcpp_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockLlmcppPlatform
    with MockPlatformInterfaceMixin
    implements LlmcppPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final LlmcppPlatform initialPlatform = LlmcppPlatform.instance;

  test('$MethodChannelLlmcpp is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelLlmcpp>());
  });

  test('getPlatformVersion', () async {
    Llmcpp llmcppPlugin = Llmcpp();
    MockLlmcppPlatform fakePlatform = MockLlmcppPlatform();
    LlmcppPlatform.instance = fakePlatform;

    expect(await llmcppPlugin.getPlatformVersion(), '42');
  });
}
