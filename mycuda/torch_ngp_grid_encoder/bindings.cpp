/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <torch/extension.h>

#include "gridencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
}