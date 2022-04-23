// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sdsl/suffix_arrays.hpp>


typedef sdsl::csa_wt_int<> fm_index_type;
// typedef fm_index_type::size_type size_type;
// typedef fm_index_type::value_type value_type;
// typedef fm_index_type::char_type char_type;
typedef unsigned long size_type;
typedef unsigned long value_type;
typedef unsigned long char_type;

class FMIndex {
    
    public:
     FMIndex();
     ~FMIndex();
     void initialize(const std::vector<char_type> &data);
     void initialize_from_file(const std::string file, int width);
     const std::vector<size_type> backward_search_multi(const std::vector<char_type> query);
     const std::vector<size_type> backward_search_step(char_type symbol, size_type low, size_type high);
     const std::vector<size_type> distinct(size_type low, size_type high);
     const std::vector<size_type> distinct_count(size_type low, size_type high);
     const std::vector<std::vector<size_type>> distinct_count_multi(std::vector<size_type> lows, std::vector<size_type> highs);
     size_type size();
     size_type locate(size_type row);
     const std::vector<char_type> extract_text(size_type begin, size_type end);
     void save(const std::string path);
     sdsl::csa_wt_int<> index;
     std::vector<value_type> chars;
     std::vector<size_type> rank_c_i;
     std::vector<size_type> rank_c_j;
    
    private:
     sdsl::int_vector<> query_;
};

FMIndex load_FMIndex(const std::string path);