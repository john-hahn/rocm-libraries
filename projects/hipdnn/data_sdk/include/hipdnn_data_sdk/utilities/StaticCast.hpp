// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>

namespace hipdnn_data_sdk::utilities
{

namespace detail
{
template <class T>
struct CastTo
{
    template <class S>
    static T from(S value)
    {
        return static_cast<T>(value);
    }
};

template <>
struct CastTo<hipdnn_data_sdk::types::bfloat16>
{
    template <class T>
    static hipdnn_data_sdk::types::bfloat16 from(T value)
    {
        return hipdnn_data_sdk::types::bfloat16(static_cast<float>(value));
    }

    static hipdnn_data_sdk::types::bfloat16 from(hipdnn_data_sdk::types::bfloat16 value)
    {
        return value;
    }
};

template <>
struct CastTo<hipdnn_data_sdk::types::half>
{
    template <class T>
    static hipdnn_data_sdk::types::half from(T value)
    {
        return hipdnn_data_sdk::types::half(static_cast<float>(value));
    }

    static hipdnn_data_sdk::types::half from(hipdnn_data_sdk::types::half value)
    {
        return value;
    }
};

} // namespace detail

template <class S, class T>
S staticCast(T value)
{
    return detail::CastTo<S>::from(value);
}

} // namespace hipdnn_data_sdk::utilities
