// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file All.hpp
 * @brief Convenience header that includes all custom hipDNN data types.
 *
 * This header provides access to all portable floating-point types that
 * do not require the __HIPCC__ macro:
 *
 * - bfloat16: 16-bit brain floating point (1 sign, 8 exponent, 7 mantissa)
 * - half: 16-bit IEEE 754 half precision (1 sign, 5 exponent, 10 mantissa)
 * - fp8_e4m3: 8-bit floating point (1 sign, 4 exponent, 3 mantissa)
 * - fp8_e5m2: 8-bit floating point (1 sign, 5 exponent, 2 mantissa)
 *
 * All types use EXPLICIT constructors and conversion operators to prevent
 * silent precision loss and eliminate overload ambiguity issues.
 */

#include "Bfloat16.hpp"
#include "Fp8E4M3.hpp"
#include "Fp8E5M2.hpp"
#include "Half.hpp"
