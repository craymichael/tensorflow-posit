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

#ifndef TENSORFLOW_CORE_LIB_POSIT16_POSIT16_H_
#define TENSORFLOW_CORE_LIB_POSIT16_POSIT16_H_

#include <cmath>
#include <complex>

#include "tensorflow/core/platform/byte_order.h"

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define P16_DEVICE_FUNC __host__ __device__

#else
#define P16_DEVICE_FUNC

#endif

namespace Eigen {
struct half;
}

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// see framework/posit16.h for description.
struct posit16 {
  P16_DEVICE_FUNC posit16() {}

  P16_DEVICE_FUNC explicit posit16(const float val);
  P16_DEVICE_FUNC explicit posit16(const double val);

  // Following the convention of numpy, converting between complex and
  // float will lead to loss of imag value.
  P16_DEVICE_FUNC explicit posit16(const complex64& val)
      : posit16(val.real()) {}

  P16_DEVICE_FUNC explicit posit16(const complex128& val)
      : posit16(static_cast<float>(val.real())) {}

  P16_DEVICE_FUNC explicit posit16(const unsigned short val)
      : posit16(static_cast<float>(val)) {}

  P16_DEVICE_FUNC explicit posit16(const unsigned int val)
      : posit16(static_cast<float>(val)) {}

  P16_DEVICE_FUNC explicit posit16(const int val)
      : posit16(static_cast<float>(val)) {}

  P16_DEVICE_FUNC explicit posit16(const long val)
      : posit16(static_cast<float>(val)) {}

  P16_DEVICE_FUNC explicit posit16(const long long val)
      : posit16(static_cast<float>(val)) {}

  template <class T>
  P16_DEVICE_FUNC explicit posit16(const T& val)
      : posit16(static_cast<float>(val)) {}

  P16_DEVICE_FUNC explicit operator float() const;

  P16_DEVICE_FUNC explicit operator bool() const {
    return static_cast<bool>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator Eigen::half() const;

  P16_DEVICE_FUNC explicit operator short() const {
    return static_cast<short>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator int() const {
    return static_cast<int>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator long() const {
    return static_cast<long>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator char() const {
    return static_cast<char>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator signed char() const {
    return static_cast<signed char>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator unsigned char() const {
    return static_cast<unsigned char>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator unsigned short() const {
    return static_cast<unsigned short>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator unsigned int() const {
    return static_cast<unsigned int>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator unsigned long() const {
    return static_cast<unsigned long>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator unsigned long long() const {
    return static_cast<unsigned long long>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator long long() const {
    return static_cast<long long>(float(*this));
  }

  P16_DEVICE_FUNC explicit operator double() const;

  P16_DEVICE_FUNC explicit operator complex64() const {
    return complex64(float(*this), float(0.0));
  }

  P16_DEVICE_FUNC explicit operator complex128() const {
    return complex128(double(*this), double(0.0));
  }

  union FP32 {
    unsigned int u;
    float f;
  };

  // Converts a float point to posit16, with round-nearest-to-even as rounding
  // method.
  P16_DEVICE_FUNC static posit16 round_to_posit16(float v) {
    uint32_t input;
    FP32 f;
    f.f = v;
    input = f.u;
    posit16 output;
    // FIXME: posit: convertFloatToP16
    output.value = v;
    return output;
  }

  static posit16 epsilon() {
    posit16 x;
    x.value = 0x0001;
    return x;
  }

  static posit16 highest() {
    posit16 x;
    x.value = 0x7FFF;
    return x;
  }

  static posit16 lowest() {
    posit16 x;
    x.value = 0x0001;
    return x;
  }

  static posit16 nar() {
    posit16 x;
    x.value = 0x8000;
    return x;
  }

  uint16_t value;

  // A value that represents "not a real".
  static const uint16_t NAR_VALUE = 0x8000;

 private:
  P16_DEVICE_FUNC static bool float_isnan(const float& x) {
#ifdef __CUDA_ARCH__
    return ::isnan(x);
#else
    return std::isnan(x);
#endif
  }
};

P16_DEVICE_FUNC inline std::ostream& operator<<(std::ostream& os,
                                                const posit16& dt) {
  os << static_cast<float>(dt);
  return os;
}

P16_DEVICE_FUNC inline posit16 operator+(posit16 a, posit16 b) {
  return posit16(static_cast<float>(a) + static_cast<float>(b));
}
P16_DEVICE_FUNC inline posit16 operator+(posit16 a, int b) {
  return posit16(static_cast<float>(a) + static_cast<float>(b));
}
P16_DEVICE_FUNC inline posit16 operator+(int a, posit16 b) {
  return posit16(static_cast<float>(a) + static_cast<float>(b));
}
P16_DEVICE_FUNC inline posit16 operator-(posit16 a, posit16 b) {
  return posit16(static_cast<float>(a) - static_cast<float>(b));
}
P16_DEVICE_FUNC inline posit16 operator*(posit16 a, posit16 b) {
  return posit16(static_cast<float>(a) * static_cast<float>(b));
}
P16_DEVICE_FUNC inline posit16 operator/(posit16 a, posit16 b) {
  return posit16(static_cast<float>(a) / static_cast<float>(b));
}
P16_DEVICE_FUNC inline posit16 operator-(posit16 a) {
  a.value = -a.value;
  return a;
}
P16_DEVICE_FUNC inline bool operator<(posit16 a, posit16 b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
P16_DEVICE_FUNC inline bool operator<=(posit16 a, posit16 b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
P16_DEVICE_FUNC inline bool operator==(posit16 a, posit16 b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
P16_DEVICE_FUNC inline bool operator!=(posit16 a, posit16 b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
P16_DEVICE_FUNC inline bool operator>(posit16 a, posit16 b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
P16_DEVICE_FUNC inline bool operator>=(posit16 a, posit16 b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}
P16_DEVICE_FUNC inline posit16& operator+=(posit16& a, posit16 b) {
  a = a + b;
  return a;
}
P16_DEVICE_FUNC inline posit16& operator-=(posit16& a, posit16 b) {
  a = a - b;
  return a;
}
P16_DEVICE_FUNC inline posit16 operator++(posit16& a) {
  a += posit16(1);
  return a;
}
P16_DEVICE_FUNC inline posit16 operator--(posit16& a) {
  a -= posit16(1);
  return a;
}
P16_DEVICE_FUNC inline posit16 operator++(posit16& a, int) {
  posit16 original_value = a;
  ++a;
  return original_value;
}
P16_DEVICE_FUNC inline posit16 operator--(posit16& a, int) {
  posit16 original_value = a;
  --a;
  return original_value;
}
P16_DEVICE_FUNC inline posit16& operator*=(posit16& a, posit16 b) {
  a = a * b;
  return a;
}
P16_DEVICE_FUNC inline posit16& operator/=(posit16& a, posit16 b) {
  a = a / b;
  return a;
}
}  // end namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::posit16> {
  size_t operator()(const tensorflow::posit16& v) const {
    return hash<float>()(static_cast<float>(v));
  }
};

using tensorflow::posit16;
inline bool isinf(const posit16& a) { return std::isinf(float(a)); }
inline bool isnan(const posit16& a) { return std::isnan(float(a)); }
inline bool isfinite(const posit16& a) { return std::isfinite(float(a)); }
inline posit16 abs(const posit16& a) { return posit16(std::abs(float(a))); }
inline posit16 exp(const posit16& a) { return posit16(std::exp(float(a))); }
inline posit16 log(const posit16& a) { return posit16(std::log(float(a))); }
inline posit16 log10(const posit16& a) {
  return posit16(std::log10(float(a)));
}
inline posit16 sqrt(const posit16& a) {
  return posit16(std::sqrt(float(a)));
}
inline posit16 pow(const posit16& a, const posit16& b) {
  return posit16(std::pow(float(a), float(b)));
}
inline posit16 sin(const posit16& a) { return posit16(std::sin(float(a))); }
inline posit16 cos(const posit16& a) { return posit16(std::cos(float(a))); }
inline posit16 tan(const posit16& a) { return posit16(std::tan(float(a))); }
inline posit16 tanh(const posit16& a) {
  return posit16(std::tanh(float(a)));
}
inline posit16 floor(const posit16& a) {
  return posit16(std::floor(float(a)));
}
inline posit16 ceil(const posit16& a) {
  return posit16(std::ceil(float(a)));
}
}  // namespace std

#endif  // TENSORFLOW_CORE_LIB_POSIT16_POSIT16_H_
