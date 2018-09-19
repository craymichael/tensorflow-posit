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

#ifndef TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
#define TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_

#include <complex>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// Disable clang-format to prevent 'FixedPoint' header from being included
// before 'Tensor' header on which it depends.
// clang-format off
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on

#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/posit8/posit8.h"
#include "tensorflow/core/lib/posit16/posit16.h"
#include "tensorflow/core/lib/posit32/posit32.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// We use Eigen's QInt implementations for our quantized int types.
typedef Eigen::QInt8 qint8;
typedef Eigen::QUInt8 quint8;
typedef Eigen::QInt32 qint32;
typedef Eigen::QInt16 qint16;
typedef Eigen::QUInt16 quint16;

}  // namespace tensorflow




static inline tensorflow::bfloat16 FloatToBFloat16(float float_val) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return *reinterpret_cast<tensorflow::bfloat16*>(
        reinterpret_cast<uint16_t*>(&float_val));
#else
    return *reinterpret_cast<tensorflow::bfloat16*>(
        &(reinterpret_cast<uint16_t*>(&float_val)[1]));
#endif
}

static inline tensorflow::posit8 FloatToPosit8(float float_val) {
    return static_cast<tensorflow::posit8>(float_val);
}

static inline tensorflow::posit16 FloatToPosit16(float float_val) {
    return static_cast<tensorflow::posit16>(float_val);
}

static inline tensorflow::posit32 FloatToPosit32(float float_val) {
    return static_cast<tensorflow::posit32>(float_val);
}
    
namespace Eigen {
// TODO(xpan): We probably need to overwrite more methods to have correct eigen
// behavior. E.g. epsilon(), dummy_precision, etc. See NumTraits.h in eigen.
template <>
struct NumTraits<tensorflow::bfloat16>
    : GenericNumTraits<tensorflow::bfloat16> {
  enum {
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0
  };
  static EIGEN_STRONG_INLINE tensorflow::bfloat16 highest() {
    return FloatToBFloat16(NumTraits<float>::highest());
  }

  static EIGEN_STRONG_INLINE tensorflow::bfloat16 lowest() {
    return FloatToBFloat16(NumTraits<float>::lowest());
  }

  static EIGEN_STRONG_INLINE tensorflow::bfloat16 infinity() {
    return FloatToBFloat16(NumTraits<float>::infinity());
  }

  static EIGEN_STRONG_INLINE tensorflow::bfloat16 quiet_NaN() {
    return FloatToBFloat16(NumTraits<float>::quiet_NaN());
  }
};

template <>
struct NumTraits<tensorflow::posit8>
    : GenericNumTraits<tensorflow::posit8> {
  enum {
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0
  };
  static EIGEN_STRONG_INLINE tensorflow::posit8 highest() {
    return tensorflow::posit8::highest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit8 lowest() {
    return tensorflow::posit8::lowest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit8 infinity() {
    return tensorflow::posit8::highest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit8 quiet_NaN() {
    return tensorflow::posit8::nar();
  }
};

template <>
struct NumTraits<tensorflow::posit16>
    : GenericNumTraits<tensorflow::posit16> {
  enum {
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0
  };
  static EIGEN_STRONG_INLINE tensorflow::posit16 highest() {
    return tensorflow::posit16::highest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit16 lowest() {
    return tensorflow::posit16::lowest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit16 infinity() {
    return tensorflow::posit16::highest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit16 quiet_NaN() {
    return tensorflow::posit16::nar();
  }
};

template <>
struct NumTraits<tensorflow::posit32>
    : GenericNumTraits<tensorflow::posit32> {
  enum {
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0
  };
  static EIGEN_STRONG_INLINE tensorflow::posit32 highest() {
    return tensorflow::posit32::highest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit32 lowest() {
    return tensorflow::posit32::lowest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit32 infinity() {
    return tensorflow::posit32::highest();
  }

  static EIGEN_STRONG_INLINE tensorflow::posit32 quiet_NaN() {
    return tensorflow::posit32::nar();
  }
};

using ::tensorflow::operator==;
using ::tensorflow::operator!=;

namespace numext {

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::bfloat16 log(
    const tensorflow::bfloat16& x) {
  return static_cast<tensorflow::bfloat16>(::logf(static_cast<float>(x)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit8 log(
    const tensorflow::posit8& x) {
  return std::log(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit16 log(
    const tensorflow::posit16& x) {
  return std::log(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit32 log(
    const tensorflow::posit32& x) {
  return std::log(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::bfloat16 exp(
    const tensorflow::bfloat16& x) {
  return static_cast<tensorflow::bfloat16>(::expf(static_cast<float>(x)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit8 exp(
    const tensorflow::posit8& x) {
  return std::exp(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit16 exp(
    const tensorflow::posit16& x) {
  return std::exp(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit32 exp(
    const tensorflow::posit32& x) {
  return std::exp(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::bfloat16 abs(
    const tensorflow::bfloat16& x) {
  return static_cast<tensorflow::bfloat16>(::fabsf(static_cast<float>(x)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit8 abs(
    const tensorflow::posit8& x) {
  return std::abs(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit16 abs(
    const tensorflow::posit16& x) {
  return std::abs(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit32 abs(
    const tensorflow::posit32& x) {
  return std::abs(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit8 sqrt(
    const tensorflow::posit8& x) {
  return std::sqrt(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit16 sqrt(
    const tensorflow::posit16& x) {
  return std::sqrt(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit32 sqrt(
    const tensorflow::posit32& x) {
  return std::sqrt(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool isinf(
    const tensorflow::posit8& x) {
  return std::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool isinf(
    const tensorflow::posit16& x) {
  return std::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool isinf(
    const tensorflow::posit32& x) {
  return std::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool isfinite(
    const tensorflow::posit8& x) {
  return std::isfinite(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool isfinite(
    const tensorflow::posit16& x) {
  return std::isfinite(x);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool isfinite(
    const tensorflow::posit32& x) {
  return std::isfinite(x);
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit8 log10(
    const tensorflow::posit8& x) {
  return std::log10(x);
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit16 log10(
    const tensorflow::posit16& x) {
  return std::log10(x);
}

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE tensorflow::posit32 log10(
    const tensorflow::posit32& x) {
  return std::log10(x);
}

}  // namespace numext
}  // namespace Eigen

#if defined(_MSC_VER) && !defined(__clang__)
namespace std {
template <>
struct hash<Eigen::half> {
  std::size_t operator()(const Eigen::half& a) const {
    return static_cast<std::size_t>(a.x);
  }
};
}  // namespace std
#endif  // _MSC_VER

#endif  // TENSORFLOW_FRAMEWORK_NUMERIC_TYPES_H_
