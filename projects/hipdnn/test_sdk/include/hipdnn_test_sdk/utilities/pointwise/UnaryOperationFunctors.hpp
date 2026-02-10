// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <hipdnn_data_sdk/types/All.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceUtilities.hpp>
#include <limits>
#include <type_traits>

namespace hipdnn_test_sdk::utilities::pointwise
{

namespace detail
{

// Helper to convert ComputeType to OutputType, going through float for reduced precision types
template <typename OutputType, typename ComputeType>
OutputType toOutputType(ComputeType value)
{
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::fp8_e4m3;
    using hipdnn_data_sdk::types::fp8_e5m2;
    using hipdnn_data_sdk::types::half;

    // Our custom reduced-precision types have explicit constructors that take float
    constexpr bool IS_REDUCED_PRECISION_OUTPUT
        = std::is_same_v<OutputType, bfloat16> || std::is_same_v<OutputType, half>
          || std::is_same_v<OutputType, fp8_e4m3> || std::is_same_v<OutputType, fp8_e5m2>;

    if constexpr(IS_REDUCED_PRECISION_OUTPUT)
    {
        // Reduced precision types need explicit construction from float
        return OutputType(static_cast<float>(value));
    }
    else
    {
        // For float, double, int, etc., use direct static_cast
        return static_cast<OutputType>(value);
    }
}

} // namespace detail

// Unary operations with explicit ComputeType and OutputType
// ComputeType: The type used for intermediate calculations
// OutputType: The type returned from the operation

template <typename ComputeType = float, typename OutputType = ComputeType>
struct ReluForward
{
    ComputeType lowerClip;
    ComputeType upperClip;
    ComputeType lowerSlope;

    ReluForward(ComputeType lowerClipVal = ComputeType{0},
                ComputeType upperClipVal = std::numeric_limits<ComputeType>::max(),
                ComputeType lowerSlopeVal = ComputeType{0})
        : lowerClip(lowerClipVal)
        , upperClip(upperClipVal)
        , lowerSlope(lowerSlopeVal)
    {
    }

    template <typename X>
    OutputType operator()(const X& x) const
    {
        auto xCompute = static_cast<ComputeType>(x);

        ComputeType result;
        if(xCompute <= lowerClip)
        {
            result = (lowerSlope * (xCompute - lowerClip)) + lowerClip;
        }
        else if(xCompute >= upperClip)
        {
            result = upperClip;
        }
        else
        {
            result = xCompute;
        }
        return detail::toOutputType<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct SigmoidForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        auto xCompute = static_cast<ComputeType>(x);
        auto result = ComputeType{1} / (ComputeType{1} + std::exp(-xCompute));
        return detail::toOutputType<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct TanhForward
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        auto xCompute = static_cast<ComputeType>(x);
        auto result = std::tanh(xCompute);
        return detail::toOutputType<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct Identity
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        return detail::toOutputType<OutputType>(static_cast<ComputeType>(x));
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct AbsoluteValue
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        auto result = std::abs(static_cast<ComputeType>(x));
        return detail::toOutputType<OutputType>(result);
    }
};

template <typename ComputeType = float, typename OutputType = ComputeType>
struct Negation
{
    template <typename X>
    OutputType operator()(const X& x) const
    {
        auto result = -static_cast<ComputeType>(x);
        return detail::toOutputType<OutputType>(result);
    }
};

} // namespace hipdnn_test_sdk::utilities::pointwise
