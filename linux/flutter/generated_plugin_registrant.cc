//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <face_detection_tflite/face_detection_tflite_plugin.h>

void fl_register_plugins(FlPluginRegistry* registry) {
  g_autoptr(FlPluginRegistrar) face_detection_tflite_registrar =
      fl_plugin_registry_get_registrar_for_plugin(registry, "FaceDetectionTflitePlugin");
  face_detection_tflite_plugin_register_with_registrar(face_detection_tflite_registrar);
}
