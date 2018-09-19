/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/posit8.h"
#include "tensorflow/core/lib/posit8/posit8.h"

namespace tensorflow {

void FloatToPosit8(const float* src, posit8* dst, int64 size) {
  for (int64 i = 0; i < size; i++) {
    dst[i] = posit8(src[i]);
  }
}

void Posit8ToFloat(const posit8* src, float* dst, int64 size) {
  for (int64 i = 0; i < size; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

}  // end namespace tensorflow
