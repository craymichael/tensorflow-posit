/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER8(UnaryOp, CPU, "Expm1", functor::expm1, float, Eigen::half, double,
          complex64, complex128, posit8, posit16, posit32);
#if GOOGLE_CUDA
REGISTER3(UnaryOp, GPU, "Expm1", functor::expm1, float, Eigen::half, double);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER2(UnaryOp, SYCL, "Expm1", functor::expm1, float, double);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
