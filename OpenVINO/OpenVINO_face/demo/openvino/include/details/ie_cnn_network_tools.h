// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for CNNNetwork tools
 * 
 * @file ie_cnn_network_tools.h
 */
#pragma once
#include <vector>

#include "ie_common.h"
#include "ie_icnn_network.hpp"

namespace InferenceEngine {
namespace details {

INFERENCE_ENGINE_INTERNAL("Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1")
INFERENCE_ENGINE_API_CPP(std::vector<CNNLayerPtr>) CNNNetSortTopologically(const ICNNNetwork& network);

}  // namespace details
}  // namespace InferenceEngine
