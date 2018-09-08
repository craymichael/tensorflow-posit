/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/posit16/posit16.h"

#include "third_party/eigen3/Eigen/Core"
#include "softposit.h"

namespace tensorflow {

P16_DEVICE_FUNC posit16::posit16(const float val) {
  posit16_t p = convertFloatToP16(val);
  this->value = p.v;
}

P16_DEVICE_FUNC posit16::posit16(const double val) {
  posit16_t p = convertDoubleToP16(val);
  this->value = p.v;
}

P16_DEVICE_FUNC posit16::operator Eigen::half() const {
  return static_cast<Eigen::half>(float(*this));
}
}  // end namespace tensorflow
