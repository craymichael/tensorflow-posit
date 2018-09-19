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
#include <cmath>
#include <ostream>

namespace tensorflow {

P16_DEVICE_FUNC posit16::posit16(const float val) {
  posit16_t p = convertFloatToP16(val);
  this->value = p.v;
}

P16_DEVICE_FUNC posit16::posit16(const double val) {
  posit16_t p = convertDoubleToP16(val);
  this->value = p.v;
}

P16_DEVICE_FUNC posit16::operator float() const {
  posit16_t p = { .v=this->value };
  return static_cast<float>(convertP16ToDouble(p));
}

P16_DEVICE_FUNC posit16::operator double() const {
  posit16_t p = { .v=this->value };
  return convertP16ToDouble(p);
}

P16_DEVICE_FUNC posit16::operator Eigen::half() const {
  return static_cast<Eigen::half>(float(*this));
}

P16_DEVICE_FUNC std::ostream& operator<<(std::ostream& os, const posit16& dt) {
  if (dt.value == posit16::NAR_VALUE) {
    os << "nar";
  } else {
    posit16_t p = { .v=dt.value };
    os << convertP16ToDouble(p);
  }
  return os;
}

P16_DEVICE_FUNC posit16 operator+(posit16 a, posit16 b) {
  posit16_t pa = { .v=a.value };
  posit16_t pb = { .v=b.value };
  posit16_t pc = p16_add(pa, pb);
  posit16 c;
  c.value = pc.v;
  return c;
}
P16_DEVICE_FUNC posit16 operator-(posit16 a, posit16 b) {
  posit16_t pa = { .v=a.value };
  posit16_t pb = { .v=b.value };
  posit16_t pc = p16_sub(pa, pb);
  posit16 c;
  c.value = pc.v;
  return c;
}
P16_DEVICE_FUNC posit16 operator*(posit16 a, posit16 b) {
  posit16_t pa = { .v=a.value };
  posit16_t pb = { .v=b.value };
  posit16_t pc = p16_mul(pa, pb);
  posit16 c;
  c.value = pc.v;
  return c;
}
P16_DEVICE_FUNC posit16 operator/(posit16 a, posit16 b) {
  posit16_t pa = { .v=a.value };
  posit16_t pb = { .v=b.value };
  posit16_t pc = p16_div(pa, pb);
  posit16 c;
  c.value = pc.v;
  return c;
}
}  // end namespace tensorflow

namespace std {
using tensorflow::posit16;

bool isinf(const posit16& a) {
  return a.value == posit16::NAR_VALUE;
}

bool isnan(const posit16& a) {
  return a.value == posit16::NAR_VALUE;
}

bool isfinite(const posit16& a) {
  return a.value != posit16::NAR_VALUE;
}

posit16 abs(const posit16& a) {
  posit16 r;
  r.value = (a.value <= 0x7FFFU) ? a.value : -a.value;
  return r;
}

posit16 exp(const posit16& a) {
  return posit16(std::exp(float(a)));
}

posit16 log(const posit16& a) {
  return posit16(std::log(float(a)));
}

posit16 log10(const posit16& a) {
  return posit16(std::log10(float(a)));
}

posit16 sqrt(const posit16& a) {
  posit16_t pa = { .v=a.value };
  posit16_t pr = p16_sqrt(pa);
  posit16 r;
  r.value = pr.v;
  return r;
}

posit16 pow(const posit16& a, const posit16& b) {
  return posit16(std::pow(float(a), float(b)));
}

posit16 sin(const posit16& a) {
  return posit16(std::sin(float(a)));
}

posit16 cos(const posit16& a) {
  return posit16(std::cos(float(a)));
}

posit16 tan(const posit16& a) {
  return posit16(std::tan(float(a)));
}

posit16 tanh(const posit16& a) {
  return posit16(std::tanh(float(a)));
}

posit16 floor(const posit16& a) {
  return posit16(std::floor(float(a)));
}

posit16 ceil(const posit16& a) {
  return posit16(std::ceil(float(a)));
}
}  // namespace std
