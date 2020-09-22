//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <utility>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    struct VariableInfo
    {
        PartialShape data_shape;
        element::Type data_type;
        std::string variable_id;
    };

    class NGRAPH_API Variable
    {
    public:
        Variable() = default;

        explicit Variable(const VariableInfo& variable_info)
            : m_info(variable_info)
        {
        }

        VariableInfo get_info() { return m_info; }
        void update(const VariableInfo& variable_info) { m_info = variable_info; }
    private:
        VariableInfo m_info;
    };
}
