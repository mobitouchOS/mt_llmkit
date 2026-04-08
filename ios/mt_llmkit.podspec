#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint llmcpp.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'mt_llmkit'
  s.version          = '0.0.1'
  s.summary          = 'Run Large Language Models locally on iOS with Flutter.'
  s.description      = <<-DESC
mt_llmkit enables running Large Language Models locally on iOS using llama.cpp. This package provides real-time streaming inference, performance metrics, cloud AI chat providers, and a fully local RAG pipeline — all from Flutter.
                       DESC
  s.homepage         = 'https://github.com/mobitouchOS/mt_llmkit'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Mobitouch' => 'mobitouch.net@gmail.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '16.4'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # If your plugin requires a privacy manifest, for example if it uses any
  # required reason APIs, update the PrivacyInfo.xcprivacy file to describe your
  # plugin's privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  # s.resource_bundles = {'llmcpp_privacy' => ['Resources/PrivacyInfo.xcprivacy']}
end
