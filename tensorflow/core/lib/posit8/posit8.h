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

#ifndef TENSORFLOW_CORE_LIB_POSIT8_POSIT8_H_
#define TENSORFLOW_CORE_LIB_POSIT8_POSIT8_H_

#include <complex>

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define P8_DEVICE_FUNC __host__ __device__

#else
#define P8_DEVICE_FUNC

#endif

namespace Eigen {
struct half;
}

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// see framework/posit8.h for description.
struct posit8 {
  P8_DEVICE_FUNC posit8() {}

  P8_DEVICE_FUNC explicit posit8(const float val);
  P8_DEVICE_FUNC explicit posit8(const double val);

  // Following the convention of numpy, converting between complex and
  // float will lead to loss of imag value.
  P8_DEVICE_FUNC explicit posit8(const complex64& val)
      : posit8(val.real()) {}

  P8_DEVICE_FUNC explicit posit8(const complex128& val)
      : posit8(static_cast<float>(val.real())) {}

  P8_DEVICE_FUNC explicit posit8(const unsigned short val)
      : posit8(static_cast<float>(val)) {}

  P8_DEVICE_FUNC explicit posit8(const unsigned int val)
      : posit8(static_cast<float>(val)) {}

  P8_DEVICE_FUNC explicit posit8(const int val)
      : posit8(static_cast<float>(val)) {}

  P8_DEVICE_FUNC explicit posit8(const long val)
      : posit8(static_cast<float>(val)) {}

  P8_DEVICE_FUNC explicit posit8(const long long val)
      : posit8(static_cast<float>(val)) {}

  template <class T>
  P8_DEVICE_FUNC explicit posit8(const T& val)
      : posit8(static_cast<float>(val)) {}

  P8_DEVICE_FUNC explicit operator float() const;

  P8_DEVICE_FUNC explicit operator bool() const {
    return static_cast<bool>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator Eigen::half() const;

  P8_DEVICE_FUNC explicit operator short() const {
    return static_cast<short>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator int() const {
    return static_cast<int>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator long() const {
    return static_cast<long>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator char() const {
    return static_cast<char>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator signed char() const {
    return static_cast<signed char>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator unsigned char() const {
    return static_cast<unsigned char>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator unsigned short() const {
    return static_cast<unsigned short>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator unsigned int() const {
    return static_cast<unsigned int>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator unsigned long() const {
    return static_cast<unsigned long>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator unsigned long long() const {
    return static_cast<unsigned long long>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator long long() const {
    return static_cast<long long>(float(*this));
  }

  P8_DEVICE_FUNC explicit operator double() const;

  P8_DEVICE_FUNC explicit operator complex64() const {
    return complex64(float(*this), float(0.0));
  }

  P8_DEVICE_FUNC explicit operator complex128() const {
    return complex128(double(*this), double(0.0));
  }

  static posit8 epsilon() {
    posit8 x;
    x.value = 0x01u;
    return x;
  }

  static posit8 highest() {
    posit8 x;
    x.value = 0x7Fu;
    return x;
  }

  static posit8 lowest() {
    posit8 x;
    x.value = 0x01u;
    return x;
  }

  static posit8 nar() {
    posit8 x;
    x.value = NAR_VALUE;
    return x;
  }

  uint8_t value;

  // A value that represents "not a real".
  static const uint8_t NAR_VALUE = 0x80u;
  static const uint8_t ONE_VALUE = 0x40u;
  static const uint8_t ZERO_VALUE = 0x00u;
};

P8_DEVICE_FUNC std::ostream& operator<<(std::ostream& os, const posit8& dt);

P8_DEVICE_FUNC posit8 operator+(posit8 a, posit8 b);
P8_DEVICE_FUNC inline posit8 operator+(posit8 a, int b) {
  return a + posit8(b);
}
P8_DEVICE_FUNC inline posit8 operator+(int a, posit8 b) {
  return posit8(a) + b;
}
P8_DEVICE_FUNC posit8 operator-(posit8 a, posit8 b);
P8_DEVICE_FUNC posit8 operator*(posit8 a, posit8 b);
P8_DEVICE_FUNC posit8 operator/(posit8 a, posit8 b);
P8_DEVICE_FUNC inline posit8 operator-(posit8 a) {
  a.value = -a.value;
  return a;
}
P8_DEVICE_FUNC inline bool operator<(posit8 a, posit8 b) {
  union { uint8_t u; int8_t i; } va = { .u=a.value };
  union { uint8_t u; int8_t i; } vb = { .u=b.value };
  return va.i < vb.i;
}
P8_DEVICE_FUNC inline bool operator<=(posit8 a, posit8 b) {
  union { uint8_t u; int8_t i; } va = { .u=a.value };
  union { uint8_t u; int8_t i; } vb = { .u=b.value };
  return va.i <= vb.i;
}
P8_DEVICE_FUNC inline bool operator==(posit8 a, posit8 b) {
  return a.value == b.value;
}
P8_DEVICE_FUNC inline bool operator!=(posit8 a, posit8 b) {
  return !(a == b);
}
P8_DEVICE_FUNC inline bool operator>(posit8 a, posit8 b) {
  return b < a;
}
P8_DEVICE_FUNC inline bool operator>=(posit8 a, posit8 b) {
  return b <= a;
}
P8_DEVICE_FUNC inline posit8& operator+=(posit8& a, posit8 b) {
  a = a + b;
  return a;
}
P8_DEVICE_FUNC inline posit8& operator-=(posit8& a, posit8 b) {
  a = a - b;
  return a;
}
P8_DEVICE_FUNC inline posit8 operator++(posit8& a) {
  posit8 one;
  one.value = posit8::ONE_VALUE;
  a += one;
  return a;
}
P8_DEVICE_FUNC inline posit8 operator--(posit8& a) {
  posit8 one;
  one.value = posit8::ONE_VALUE;
  a -= one;
  return a;
}
P8_DEVICE_FUNC inline posit8 operator++(posit8& a, int) {
  posit8 original_value = a;
  ++a;
  return original_value;
}
P8_DEVICE_FUNC inline posit8 operator--(posit8& a, int) {
  posit8 original_value = a;
  --a;
  return original_value;
}
P8_DEVICE_FUNC inline posit8& operator*=(posit8& a, posit8 b) {
  a = a * b;
  return a;
}
P8_DEVICE_FUNC inline posit8& operator/=(posit8& a, posit8 b) {
  a = a / b;
  return a;
}
}  // end namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::posit8> {
  size_t operator()(const tensorflow::posit8& v) const {
    return hash<float>()(static_cast<float>(v));
  }
};

using tensorflow::posit8;
bool isinf(const posit8& a);
bool isnan(const posit8& a);
bool isfinite(const posit8& a);
posit8 abs(const posit8& a);
posit8 exp(const posit8& a);
posit8 log(const posit8& a);
posit8 log10(const posit8& a);
posit8 sqrt(const posit8& a);
posit8 pow(const posit8& a, const posit8& b);
posit8 sin(const posit8& a);
posit8 cos(const posit8& a);
posit8 tan(const posit8& a);
posit8 tanh(const posit8& a);
posit8 floor(const posit8& a);
posit8 ceil(const posit8& a);
}  // namespace std

#endif  // TENSORFLOW_CORE_LIB_POSIT8_POSIT8_H_
