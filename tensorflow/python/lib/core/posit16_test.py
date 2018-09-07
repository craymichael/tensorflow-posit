# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test cases for the posit16 Python type."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

# pylint: disable=unused-import,g-bad-import-order
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


posit16 = pywrap_tensorflow.TF_posit16_type()


class Posit16Test(test.TestCase):

  pass


class Posit16NumPyTest(test.TestCase):

  pass


if __name__ == "__main__":
  test.main()
