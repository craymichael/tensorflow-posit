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

#include "tensorflow/core/lib/posit32/posit32.h"

#include "third_party/eigen3/Eigen/Core"
#include "softposit.h"
#include <cmath>
#include <ostream>

namespace tensorflow {

P32_DEVICE_FUNC posit32::posit32(const float val) {
  posit32_t p = convertFloatToP32(val);
  this->value = p.v;
}

P32_DEVICE_FUNC posit32::posit32(const double val) {
  posit32_t p = convertDoubleToP32(val);
  this->value = p.v;
}

P32_DEVICE_FUNC posit32::operator float() const {
  posit32_t p = { .v=this->value };
  return static_cast<float>(convertP32ToDouble(p));
}

P32_DEVICE_FUNC posit32::operator double() const {
  posit32_t p = { .v=this->value };
  return convertP32ToDouble(p);
}

P32_DEVICE_FUNC posit32::operator Eigen::half() const {
  return static_cast<Eigen::half>(float(*this));
}

P32_DEVICE_FUNC std::ostream& operator<<(std::ostream& os, const posit32& dt) {
  if (dt.value == posit32::NAR_VALUE) {
    os << "nar";
  } else {
    posit32_t p = { .v=dt.value };
    os << convertP32ToDouble(p);
  }
  return os;
}

P32_DEVICE_FUNC posit32 operator+(posit32 a, posit32 b) {
  posit32_t pa = { .v=a.value };
  posit32_t pb = { .v=b.value };
  posit32_t pc = p32_add(pa, pb);
  posit32 c;
  c.value = pc.v;
  return c;
}
P32_DEVICE_FUNC posit32 operator-(posit32 a, posit32 b) {
  posit32_t pa = { .v=a.value };
  posit32_t pb = { .v=b.value };
  posit32_t pc = p32_sub(pa, pb);
  posit32 c;
  c.value = pc.v;
  return c;
}
P32_DEVICE_FUNC posit32 operator*(posit32 a, posit32 b) {
  posit32_t pa = { .v=a.value };
  posit32_t pb = { .v=b.value };
  posit32_t pc = p32_mul(pa, pb);
  posit32 c;
  c.value = pc.v;
  return c;
}
P32_DEVICE_FUNC posit32 operator/(posit32 a, posit32 b) {
  posit32_t pa = { .v=a.value };
  posit32_t pb = { .v=b.value };
  posit32_t pc = p32_div(pa, pb);
  posit32 c;
  c.value = pc.v;
  return c;
}
}  // end namespace tensorflow

namespace std {
using tensorflow::posit32;

bool isinf(const posit32& a) {
  return a.value == posit32::NAR_VALUE;
}

bool isnan(const posit32& a) {
  return a.value == posit32::NAR_VALUE;
}

bool isfinite(const posit32& a) {
  return a.value != posit32::NAR_VALUE;
}

posit32 abs(const posit32& a) {
  posit32 r;
  r.value = (a.value <= 0x7FFFFFFFU) ? a.value : -a.value;
  return r;
}

posit32 exp(const posit32& a) {
  return posit32(std::exp(float(a)));
}

posit32 log(const posit32& a) {
  return posit32(std::log(float(a)));
}

posit32 log10(const posit32& a) {
  return posit32(std::log10(float(a)));
}

posit32 sqrt(const posit32& a) {
  posit32_t pa = { .v=a.value };
  posit32_t pr = p32_sqrt(pa);
  posit32 r;
  r.value = pr.v;
  return r;
}

posit32 pow(const posit32& a, const posit32& b) {
  return posit32(std::pow(float(a), float(b)));
}

posit32 sin(const posit32& a) {
  return posit32(std::sin(float(a)));
}

posit32 cos(const posit32& a) {
  return posit32(std::cos(float(a)));
}

posit32 tan(const posit32& a) {
  return posit32(std::tan(float(a)));
}

posit32 tanh(const posit32& a) {
  return posit32(std::tanh(float(a)));
}

posit32 floor(const posit32& a) {
  return posit32(std::floor(float(a)));
}

posit32 ceil(const posit32& a) {
  return posit32(std::ceil(float(a)));
}
}  // namespace std
