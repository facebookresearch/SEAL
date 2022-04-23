// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

%module fm_index

%include "std_string.i"
%include "std_vector.i"

namespace std {
   %template(IntVector) vector<unsigned long>;
   %template(IntVectorVector) vector<vector<unsigned long>>;
}

%{
#include "fm_index.hpp"
%}

%include "fm_index.hpp"