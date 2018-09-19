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

#include "tensorflow/core/lib/posit8/posit8.h"

#include "third_party/eigen3/Eigen/Core"
#include "softposit.h"
#include <cmath>
#include <ostream>

namespace tensorflow {

P8_DEVICE_FUNC posit8::posit8(const float val) {
  posit8_t p = convertDoubleToP8((double)val);
  this->value = p.v;
}

P8_DEVICE_FUNC posit8::posit8(const double val) {
  posit8_t p = convertDoubleToP8(val);
  this->value = p.v;
}

P8_DEVICE_FUNC posit8::operator float() const {
  posit8_t p = { .v=this->value };
  return static_cast<float>(convertP8ToDouble(p));
}

P8_DEVICE_FUNC posit8::operator double() const {
  posit8_t p = { .v=this->value };
  return convertP8ToDouble(p);
}

P8_DEVICE_FUNC posit8::operator Eigen::half() const {
  return static_cast<Eigen::half>(float(*this));
}

P8_DEVICE_FUNC std::ostream& operator<<(std::ostream& os, const posit8& dt) {
  if (dt.value == posit8::NAR_VALUE) {
    os << "nar";
  } else {
    posit8_t p = { .v=dt.value };
    os << convertP8ToDouble(p);
  }
  return os;
}

P8_DEVICE_FUNC posit8 operator+(posit8 a, posit8 b) {
  posit8_t pa = { .v=a.value };
  posit8_t pb = { .v=b.value };
  posit8_t pc = p8_add(pa, pb);
  posit8 c;
  c.value = pc.v;
  return c;
}
P8_DEVICE_FUNC posit8 operator-(posit8 a, posit8 b) {
  posit8_t pa = { .v=a.value };
  posit8_t pb = { .v=b.value };
  posit8_t pc = p8_sub(pa, pb);
  posit8 c;
  c.value = pc.v;
  return c;
}
P8_DEVICE_FUNC posit8 operator*(posit8 a, posit8 b) {
  posit8_t pa = { .v=a.value };
  posit8_t pb = { .v=b.value };
  posit8_t pc = p8_mul(pa, pb);
  posit8 c;
  c.value = pc.v;
  return c;
}
P8_DEVICE_FUNC posit8 operator/(posit8 a, posit8 b) {
  posit8_t pa = { .v=a.value };
  posit8_t pb = { .v=b.value };
  posit8_t pc = p8_div(pa, pb);
  posit8 c;
  c.value = pc.v;
  return c;
}
}  // end namespace tensorflow

namespace std {
using tensorflow::posit8;

bool isinf(const posit8& a) {
  return a.value == posit8::NAR_VALUE;
}

bool isnan(const posit8& a) {
  return a.value == posit8::NAR_VALUE;
}

bool isfinite(const posit8& a) {
  return a.value != posit8::NAR_VALUE;
}

posit8 abs(const posit8& a) {
  posit8 r;
  r.value = (a.value <= 0x7Fu) ? a.value : -a.value;
  return r;
}

posit8 exp(const posit8& a) {
  return posit8(std::exp(float(a)));
}

posit8 log(const posit8& a) {
  return posit8(std::log(float(a)));
}

posit8 log10(const posit8& a) {
  return posit8(std::log10(float(a)));
}

posit8 sqrt(const posit8& a) {
  posit8_t pa = { .v=a.value };
  posit8_t pr = p8_sqrt(pa);
  posit8 r;
  r.value = pr.v;
  return r;
}

posit8 pow(const posit8& a, const posit8& b) {
  return posit8(std::pow(float(a), float(b)));
}

posit8 sin(const posit8& a) {
  return posit8(std::sin(float(a)));
}

posit8 cos(const posit8& a) {
  return posit8(std::cos(float(a)));
}

posit8 tan(const posit8& a) {
  return posit8(std::tan(float(a)));
}

posit8 tanh(const posit8& a) {
  return posit8(std::tanh(float(a)));
}

posit8 floor(const posit8& a) {
  return posit8(std::floor(float(a)));
}

posit8 ceil(const posit8& a) {
  return posit8(std::ceil(float(a)));
}
}  // namespace std
