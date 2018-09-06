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

#include <array>

#include "tensorflow/python/lib/core/posit16.h"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace {

// Workarounds for Python 2 vs 3 API differences.
#if PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyString_FromString(s.c_str());
}

typedef long HashType;  // NOLINT

bool TfPyInt_Check(PyObject* object) { return PyInt_Check(object); }

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyInt_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyInt_AsLong(x);
}

#else  // PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyUnicode_FromString(s.c_str());
}

bool TfPyInt_Check(PyObject* object) {
  if (!PyLong_Check(object)) {
    return 0;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyLong_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyLong_AsLong(x);
}

typedef Py_hash_t HashType;

#endif  // PY_MAJOR_VERSION < 3

// Forward declaration.
extern PyTypeObject PyPosit16_Type;

// Representation of a Python posit16 object.
struct PyPosit16 {
  PyObject_HEAD;  // Python object header
  posit16 value;
};

// Returns true if 'object' is a PyPosit16.
bool PyPosit16_Check(PyObject* object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject*>(&PyPosit16_Type));
}

// Extracts the value of a PyPosit16 object.
posit16 PyPosit16_Posit16(PyObject* object) {
  return reinterpret_cast<PyPosit16*>(object)->value;
}

// Constructs a PyPosit16 object from a posit16.
Safe_PyObjectPtr PyPosit16_FromPosit16(posit16 x) {
  Safe_PyObjectPtr ref =
      make_safe(PyPosit16_Type.tp_alloc(&PyPosit16_Type, 0));
  PyPosit16* p = reinterpret_cast<PyPosit16*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a posit16 value. Returns true on success,
// returns false and reports a Python error on failure.
bool AsPosit16(PyObject* arg, posit16* output) {
  if (PyPosit16_Check(arg)) {
    *output = PyPosit16_Posit16(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = posit16(d);
    return true;
  }
  if (TfPyInt_Check(arg)) {
    long l = TfPyInt_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = posit16(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = posit16(f);
    return true;
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return false;
}

// Converts a PyPosit16 into a PyFloat.
PyObject* PyPosit16_Float(PyObject* self) {
  posit16 x = PyPosit16_Posit16(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyPosit16 into a PyInt.
PyObject* PyPosit16_Int(PyObject* self) {
  posit16 x = PyPosit16_Posit16(self);
  long y = static_cast<long>(x);  // NOLINT
  return TfPyInt_FromLong(y);
}

// Negates a PyPosit16.
PyObject* PyPosit16_Negative(PyObject* self) {
  posit16 x = PyPosit16_Posit16(self);
  return PyPosit16_FromPosit16(-x).release();
}

// Binary arithmetic operators on PyPosit16 values.
#define POSIT16_BINOP(name, op)                                  \
  PyObject* PyPosit16_##name(PyObject* a, PyObject* b) {          \
    posit16 x, y;                                                 \
    if (!AsPosit16(a, &x) || !AsPosit16(b, &y)) return nullptr;   \
    posit16 z = x op y;                                           \
    return PyPosit16_FromPosit16(z).release();                    \
  }
POSIT16_BINOP(Add, +)
POSIT16_BINOP(Subtract, -)
POSIT16_BINOP(Multiply, *)
POSIT16_BINOP(Divide, /)
#undef POSIT16_BINOP

// Python number methods for PyPosit16 objects.
PyNumberMethods PyPosit16_AsNumber = {
    PyPosit16_Add,        // nb_add
    PyPosit16_Subtract,   // nb_subtract
    PyPosit16_Multiply,   // nb_multiply
#if PY_MAJOR_VERSION < 3
    PyPosit16_Divide,   // nb_divide
#endif
    nullptr,              // nb_remainder
    nullptr,              // nb_divmod
    nullptr,              // nb_power
    PyPosit16_Negative,   // nb_negative
    nullptr,              // nb_positive
    nullptr,              // nb_absolute
    nullptr,              // nb_nonzero
    nullptr,              // nb_invert
    nullptr,              // nb_lshift
    nullptr,              // nb_rshift
    nullptr,              // nb_and
    nullptr,              // nb_xor
    nullptr,              // nb_or
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_coerce
#endif
    PyPosit16_Int,   // nb_int
#if PY_MAJOR_VERSION < 3
    PyPosit16_Int,   // nb_long
#else
    nullptr,  // reserved
#endif
    PyPosit16_Float,   // nb_float
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_oct
    nullptr,  // nb_hex
#endif

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_inplace_divide
#endif
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    nullptr,            // nb_floor_divide
    PyPosit16_Divide,   // nb_true_divide
    nullptr,            // nb_inplace_floor_divide
    nullptr,            // nb_inplace_true_divide
    nullptr,            // nb_index
};

// Constructs a new PyPosit16.
PyObject* PyPosit16_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to posit16 constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  if (PyPosit16_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else {
    posit16 value;
    if (!AsPosit16(arg, &value)) {
      return nullptr;
    }
    return PyPosit16_FromPosit16(value).release();
  }
}

// Comparisons on PyPosit16s.
PyObject* PyPosit16_RichCompare(PyObject* a, PyObject* b, int op) {
  posit16 x, y;
  if (!AsPosit16(a, &x) || !AsPosit16(b, &y)) return nullptr;
  bool result;
  switch (op) {
    case Py_LT:
      result = x < y;
      break;
    case Py_LE:
      result = x <= y;
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      result = x > y;
      break;
    case Py_GE:
      result = x >= y;
      break;
    default:
      LOG(FATAL) << "Invalid op type " << op;
  }
  return PyBool_FromLong(result);
}

// Implementation of repr() for PyPosit16.
PyObject* PyPosit16_Repr(PyObject* self) {
  posit16 x = reinterpret_cast<PyPosit16*>(self)->value;
  string v = strings::StrCat("posit16(", static_cast<float>(x), ")");
  return MakePyString(v);
}

// Implementation of str() for PyPosit16.
PyObject* PyPosit16_Str(PyObject* self) {
  posit16 x = reinterpret_cast<PyPosit16*>(self)->value;
  string v = strings::StrCat(static_cast<float>(x));
  return MakePyString(v);
}

// Hash function for PyPosit16. We use the identity function, which is a weak
// hash function.
HashType PyPosit16_Hash(PyObject* self) {
  posit16 x = reinterpret_cast<PyPosit16*>(self)->value;
  return x.value;
}

// Python type for PyPosit16 objects.
PyTypeObject PyPosit16_Type = {
#if PY_MAJOR_VERSION < 3
    PyObject_HEAD_INIT(nullptr) 0,  // ob_size
#else
    PyVarObject_HEAD_INIT(nullptr, 0)
#endif
    "posit16",                                 // tp_name
    sizeof(PyPosit16),                         // tp_basicsize
    0,                                         // tp_itemsize
    nullptr,                                   // tp_dealloc
    nullptr,                                   // tp_print
    nullptr,                                   // tp_getattr
    nullptr,                                   // tp_setattr
    nullptr,                                   // tp_compare / tp_reserved
    PyPosit16_Repr,                            // tp_repr
    &PyPosit16_AsNumber,                       // tp_as_number
    nullptr,                                   // tp_as_sequence
    nullptr,                                   // tp_as_mapping
    PyPosit16_Hash,                            // tp_hash
    nullptr,                                   // tp_call
    PyPosit16_Str,                             // tp_str
    nullptr,                                   // tp_getattro
    nullptr,                                   // tp_setattro
    nullptr,                                   // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
    "posit16 floating-point values",           // tp_doc
    nullptr,                                   // tp_traverse
    nullptr,                                   // tp_clear
    PyPosit16_RichCompare,                     // tp_richcompare
    0,                                         // tp_weaklistoffset
    nullptr,                                   // tp_iter
    nullptr,                                   // tp_iternext
    nullptr,                                   // tp_methods
    nullptr,                                   // tp_members
    nullptr,                                   // tp_getset
    nullptr,                                   // tp_base
    nullptr,                                   // tp_dict
    nullptr,                                   // tp_descr_get
    nullptr,                                   // tp_descr_set
    0,                                         // tp_dictoffset
    nullptr,                                   // tp_init
    nullptr,                                   // tp_alloc
    PyPosit16_New,                             // tp_new
    nullptr,                                   // tp_free
    nullptr,                                   // tp_is_gc
    nullptr,                                   // tp_bases
    nullptr,                                   // tp_mro
    nullptr,                                   // tp_cache
    nullptr,                                   // tp_subclasses
    nullptr,                                   // tp_weaklist
    nullptr,                                   // tp_del
    0,                                         // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyPosit16_ArrFuncs;

PyArray_Descr NPyPosit16_Descr = {
    PyObject_HEAD_INIT(nullptr) & PyPosit16_Type,  // typeobj
    // We must register posit16 with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != posit16.
    'P',  // kind
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    'k',                                                  // type
    '=',                                                  // byteorder
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,  // hasobject
    0,                                                    // type_num
    sizeof(posit16),                                      // elsize
    alignof(posit16),                                     // alignment
    nullptr,                                              // subarray
    nullptr,                                              // fields
    nullptr,                                              // names
    &NPyPosit16_ArrFuncs,                                 // f
};

// Registered numpy type ID. Global variable populated by the registration code.
int npy_posit16_ = -1;

// Implementations of NumPy array methods.

PyObject* NPyPosit16_GetItem(void* data, void* arr) {
  posit16 x;
  memcpy(&x, data, sizeof(posit16));
  return PyPosit16_FromPosit16(x).release();
}

int NPyPosit16_SetItem(PyObject* item, void* data, void* arr) {
  posit16 x;
  if (!AsPosit16(item, &x)) return -1;
  memcpy(data, &x, sizeof(posit16));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

void NPyPosit16_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                           npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (swap) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(uint16_t));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t)) {
    memcpy(dst, src, n * sizeof(uint16_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
    }
  }
}

void NPyPosit16_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyPosit16_NonZero(void* data, void* arr) {
  posit16 x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<posit16>(0);
}

int NPyPosit16_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  posit16* const buffer = reinterpret_cast<posit16*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<posit16>(start + i * delta);
  }
  return 0;
}

// NumPy casts

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const From* from = reinterpret_cast<From*>(from_void);
  To* to = reinterpret_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(from[i]);
  }
}

// Registers a cast between posit16 and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'. If 'cast_is_safe', registers that posit16 can be
// safely coerced to T.
template <typename T>
bool RegisterPosit16Cast(int numpy_type, bool cast_is_safe) {
  if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_posit16_,
                               NPyCast<T, posit16>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyPosit16_Descr, numpy_type,
                               NPyCast<posit16, T>) < 0) {
    return false;
  }
  if (cast_is_safe && PyArray_RegisterCanCast(&NPyPosit16_Descr, numpy_type,
                                              NPY_NOSCALAR) < 0) {
    return false;
  }
  return true;
}

template <typename InType, typename OutType, typename Functor>
void BinaryUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                 void* data) {
  const char* i0 = args[0];
  const char* i1 = args[1];
  char* o = args[2];
  for (npy_intp k = 0; k < *dimensions; k++) {
    InType x = *reinterpret_cast<const InType*>(i0);
    InType y = *reinterpret_cast<const InType*>(i1);
    *reinterpret_cast<OutType*>(o) = Functor()(x, y);
    i0 += steps[0];
    i1 += steps[1];
    o += steps[2];
  }
}

template <typename Functor>
void CompareUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                  void* data) {
  BinaryUFunc<posit16, npy_bool, Functor>(args, dimensions, steps, data);
}

struct Posit16EqFunctor {
  npy_bool operator()(posit16 a, posit16 b) { return a == b; }
};
struct Posit16NeFunctor {
  npy_bool operator()(posit16 a, posit16 b) { return a != b; }
};
struct Posit16LtFunctor {
  npy_bool operator()(posit16 a, posit16 b) { return a < b; }
};
struct Posit16GtFunctor {
  npy_bool operator()(posit16 a, posit16 b) { return a > b; }
};
struct Posit16LeFunctor {
  npy_bool operator()(posit16 a, posit16 b) { return a <= b; }
};
struct Posit16GeFunctor {
  npy_bool operator()(posit16 a, posit16 b) { return a >= b; }
};

// Initializes the module.
bool Initialize() {
  // It's critical to import umath to avoid crash in open source build.
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(MakePyString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  // We hit a mysterious crash if we haven't initialized numpy before this:
  PyPosit16_Type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&PyPosit16_Type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyPosit16_ArrFuncs);
  NPyPosit16_ArrFuncs.getitem = NPyPosit16_GetItem;
  NPyPosit16_ArrFuncs.setitem = NPyPosit16_SetItem;
  NPyPosit16_ArrFuncs.copyswapn = NPyPosit16_CopySwapN;
  NPyPosit16_ArrFuncs.copyswap = NPyPosit16_CopySwap;
  NPyPosit16_ArrFuncs.nonzero = NPyPosit16_NonZero;
  NPyPosit16_ArrFuncs.fill = NPyPosit16_Fill;

  Py_TYPE(&NPyPosit16_Descr) = &PyArrayDescr_Type;
  npy_posit16_ = PyArray_RegisterDataType(&NPyPosit16_Descr);
  if (npy_posit16_ < 0) return false;

  // Support dtype(posit16)
  if (PyDict_SetItemString(PyPosit16_Type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyPosit16_Descr)) <
      0) {
    return false;
  }

  // Register casts

  // We lie shamelessly and say that a cast from half to posit16 is safe.
  // Numpy frequently uses the smallest legal representation type for small
  // float constants (e.g., 1.0), which is often float16. Things break if these
  // cannot be converted transparently to posit16.
  if (!RegisterPosit16Cast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/true)) {
    return false;
  }

  if (!RegisterPosit16Cast<float>(NPY_FLOAT, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterPosit16Cast<double>(NPY_DOUBLE, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterPosit16Cast<int32>(NPY_INT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterPosit16Cast<int64>(NPY_INT64, /*cast_is_safe=*/false)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterPosit16Cast<complex64>(NPY_COMPLEX64, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterPosit16Cast<complex128>(NPY_COMPLEX128,
                                        /*cast_is_safe=*/true)) {
    return false;
  }

  // Register ufuncs
  auto register_ufunc = [&](const char* name, PyUFuncGenericFunction fn,
                            const std::array<int, 3>& types) {
    Safe_PyObjectPtr ufunc_obj =
        make_safe(PyObject_GetAttrString(numpy.get(), name));
    if (!ufunc_obj) {
      return false;
    }
    PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
    if (types.size() != ufunc->nargs) {
      PyErr_Format(PyExc_AssertionError,
                   "ufunc %s takes %d arguments, loop takes %lu", name,
                   ufunc->nargs, types.size());
      return false;
    }
    if (PyUFunc_RegisterLoopForType(ufunc, npy_posit16_, fn,
                                    const_cast<int*>(types.data()),
                                    nullptr) < 0) {
      return false;
    }
    return true;
  };

  // Comparisons
  const std::array<int, 3> compare_types = {
      {npy_posit16_, npy_posit16_, NPY_BOOL}};

  if (!register_ufunc("equal", CompareUFunc<Posit16EqFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("not_equal", CompareUFunc<Posit16NeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less", CompareUFunc<Posit16LtFunctor>, compare_types)) {
    return false;
  }
  if (!register_ufunc("greater", CompareUFunc<Posit16GtFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less_equal", CompareUFunc<Posit16LeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("greater_equal", CompareUFunc<Posit16GeFunctor>,
                      compare_types)) {
    return false;
  }
  return true;
}

}  // namespace

void RegisterNumpyPosit16() {
  if (npy_posit16_ >= 0) {
    // Already initialized.
    return;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load posit16 module.");
    }
    PyErr_Print();
  }
}

PyObject* Posit16PyType() {
  CHECK(PyPosit16_Type.tp_base != nullptr);
  Py_INCREF(&PyPosit16_Type);
  return reinterpret_cast<PyObject*>(&PyPosit16_Type);
}

int Posit16NumpyType() {
  CHECK_GE(npy_posit16_, 0);
  return npy_posit16_;
}

}  // namespace tensorflow
