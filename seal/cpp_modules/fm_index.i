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