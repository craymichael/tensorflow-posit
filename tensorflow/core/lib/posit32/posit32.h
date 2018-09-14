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

#ifndef TENSORFLOW_CORE_LIB_POSIT32_POSIT32_H_
#define TENSORFLOW_CORE_LIB_POSIT32_POSIT32_H_

#include <complex>

#ifdef __CUDACC__
// All functions callable from CUDA code must be qualified with __device__
#define P32_DEVICE_FUNC __host__ __device__

#else
#define P32_DEVICE_FUNC

#endif

namespace Eigen {
struct half;
}

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// see framework/posit32.h for description.
// FIXME: posit: proper conversions.
struct posit32 {
  P32_DEVICE_FUNC posit32() {}

  P32_DEVICE_FUNC explicit posit32(const float val);
  P32_DEVICE_FUNC explicit posit32(const double val);

  // Following the convention of numpy, converting between complex and
  // float will lead to loss of imag value.
  P32_DEVICE_FUNC explicit posit32(const complex64& val)
      : posit32(val.real()) {}

  P32_DEVICE_FUNC explicit posit32(const complex128& val)
      : posit32(static_cast<float>(val.real())) {}

  P32_DEVICE_FUNC explicit posit32(const unsigned short val)
      : posit32(static_cast<float>(val)) {}

  P32_DEVICE_FUNC explicit posit32(const unsigned int val)
      : posit32(static_cast<float>(val)) {}

  P32_DEVICE_FUNC explicit posit32(const int val)
      : posit32(static_cast<float>(val)) {}

  P32_DEVICE_FUNC explicit posit32(const long val)
      : posit32(static_cast<float>(val)) {}

  P32_DEVICE_FUNC explicit posit32(const long long val)
      : posit32(static_cast<float>(val)) {}

  template <class T>
  P32_DEVICE_FUNC explicit posit32(const T& val)
      : posit32(static_cast<float>(val)) {}

  P32_DEVICE_FUNC explicit operator float() const;

  P32_DEVICE_FUNC explicit operator bool() const {
    return static_cast<bool>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator Eigen::half() const;

  P32_DEVICE_FUNC explicit operator short() const {
    return static_cast<short>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator int() const {
    return static_cast<int>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator long() const {
    return static_cast<long>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator char() const {
    return static_cast<char>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator signed char() const {
    return static_cast<signed char>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator unsigned char() const {
    return static_cast<unsigned char>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator unsigned short() const {
    return static_cast<unsigned short>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator unsigned int() const {
    return static_cast<unsigned int>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator unsigned long() const {
    return static_cast<unsigned long>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator unsigned long long() const {
    return static_cast<unsigned long long>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator long long() const {
    return static_cast<long long>(float(*this));
  }

  P32_DEVICE_FUNC explicit operator double() const;

  P32_DEVICE_FUNC explicit operator complex64() const {
    return complex64(float(*this), float(0.0));
  }

  P32_DEVICE_FUNC explicit operator complex128() const {
    return complex128(double(*this), double(0.0));
  }

  static posit32 epsilon() {
    posit32 x;
    x.value = 0x00000001U;
    return x;
  }

  static posit32 highest() {
    posit32 x;
    x.value = 0x7FFFFFFFU;
    return x;
  }

  static posit32 lowest() {
    posit32 x;
    x.value = 0x00000001U;
    return x;
  }

  static posit32 nar() {
    posit32 x;
    x.value = NAR_VALUE;
    return x;
  }

  uint32_t value;

  // A value that represents "not a real".
  static const uint32_t NAR_VALUE = 0x80000000U;
  static const uint32_t ONE_VALUE = 0x40000000U;
  static const uint32_t ZERO_VALUE = 0x00000000U;
};

P32_DEVICE_FUNC std::ostream& operator<<(std::ostream& os, const posit32& dt);

P32_DEVICE_FUNC posit32 operator+(posit32 a, posit32 b);
P32_DEVICE_FUNC inline posit32 operator+(posit32 a, int b) {
  return a + posit32(b);
}
P32_DEVICE_FUNC inline posit32 operator+(int a, posit32 b) {
  return posit32(a) + b;
}
P32_DEVICE_FUNC posit32 operator-(posit32 a, posit32 b);
P32_DEVICE_FUNC posit32 operator*(posit32 a, posit32 b);
P32_DEVICE_FUNC posit32 operator/(posit32 a, posit32 b);
P32_DEVICE_FUNC inline posit32 operator-(posit32 a) {
  a.value = -a.value;
  return a;
}
P32_DEVICE_FUNC inline bool operator<(posit32 a, posit32 b) {
  union { uint32_t u; int32_t i; } va = { .u=a.value };
  union { uint32_t u; int32_t i; } vb = { .u=b.value };
  return va.i < vb.i;
}
P32_DEVICE_FUNC inline bool operator<=(posit32 a, posit32 b) {
  union { uint32_t u; int32_t i; } va = { .u=a.value };
  union { uint32_t u; int32_t i; } vb = { .u=b.value };
  return va.i <= vb.i;
}
P32_DEVICE_FUNC inline bool operator==(posit32 a, posit32 b) {
  return a.value == b.value;
}
P32_DEVICE_FUNC inline bool operator!=(posit32 a, posit32 b) {
  return !(a == b);
}
P32_DEVICE_FUNC inline bool operator>(posit32 a, posit32 b) {
  return b < a;
}
P32_DEVICE_FUNC inline bool operator>=(posit32 a, posit32 b) {
  return b <= a;
}
P32_DEVICE_FUNC inline posit32& operator+=(posit32& a, posit32 b) {
  a = a + b;
  return a;
}
P32_DEVICE_FUNC inline posit32& operator-=(posit32& a, posit32 b) {
  a = a - b;
  return a;
}
P32_DEVICE_FUNC inline posit32 operator++(posit32& a) {
  posit32 one;
  one.value = posit32::ONE_VALUE;
  a += one;
  return a;
}
P32_DEVICE_FUNC inline posit32 operator--(posit32& a) {
  posit32 one;
  one.value = posit32::ONE_VALUE;
  a -= one;
  return a;
}
P32_DEVICE_FUNC inline posit32 operator++(posit32& a, int) {
  posit32 original_value = a;
  ++a;
  return original_value;
}
P32_DEVICE_FUNC inline posit32 operator--(posit32& a, int) {
  posit32 original_value = a;
  --a;
  return original_value;
}
P32_DEVICE_FUNC inline posit32& operator*=(posit32& a, posit32 b) {
  a = a * b;
  return a;
}
P32_DEVICE_FUNC inline posit32& operator/=(posit32& a, posit32 b) {
  a = a / b;
  return a;
}
}  // end namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::posit32> {
  size_t operator()(const tensorflow::posit32& v) const {
    return hash<float>()(static_cast<float>(v));
  }
};

using tensorflow::posit32;
bool isinf(const posit32& a);
bool isnan(const posit32& a);
bool isfinite(const posit32& a);
posit32 abs(const posit32& a);
posit32 exp(const posit32& a);
posit32 log(const posit32& a);
posit32 log10(const posit32& a);
posit32 sqrt(const posit32& a);
posit32 pow(const posit32& a, const posit32& b);
posit32 sin(const posit32& a);
posit32 cos(const posit32& a);
posit32 tan(const posit32& a);
posit32 tanh(const posit32& a);
posit32 floor(const posit32& a);
posit32 ceil(const posit32& a);
}  // namespace std

#endif  // TENSORFLOW_CORE_LIB_POSIT32_POSIT32_H_
