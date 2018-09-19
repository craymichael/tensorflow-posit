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

#ifndef TENSORFLOW_FRAMEWORK_POSIT8_H_
#define TENSORFLOW_FRAMEWORK_POSIT8_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/types.h"

#if defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/windows/cpu_info.h"
#endif

// This file must be compilable by nvcc.
//
// The type is defined in framework/numeric_types.h.

namespace tensorflow {

// Conversion routines between an array of float and posit8 of
// "size".
void FloatToPosit8(const float* src, posit8* dst, int64 size);
void Posit8ToFloat(const posit8* src, float* dst, int64 size);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_POSIT8_H_
