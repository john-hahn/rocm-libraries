// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/StaticCast.hpp>

using namespace hipdnn_data_sdk::utilities;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

namespace
{

template <class T, class S>
void testCastTo(S value)
{
    T expected = T(static_cast<float>(value));
    EXPECT_EQ(staticCast<T>(value), expected);
}

template <class T, class S>
void testCastToWithFloatIntermediate(S value)
{
    T expected = T(static_cast<float>(value));
    EXPECT_EQ(staticCast<T>(value), expected);
}

TEST(TestStaticCast, Correctness)
{
    testCastTo<bfloat16>(float());
    testCastToWithFloatIntermediate<bfloat16>(double());
    testCastTo<bfloat16>(half());
    testCastTo<bfloat16>(bfloat16());
    testCastToWithFloatIntermediate<bfloat16>(int());
    testCastToWithFloatIntermediate<bfloat16>(0U);
    testCastToWithFloatIntermediate<bfloat16>(uint64_t{0});
    testCastToWithFloatIntermediate<bfloat16>(int64_t{0});

    testCastTo<half>(float());
    testCastToWithFloatIntermediate<half>(double());
    testCastTo<half>(half());
    testCastTo<half>(bfloat16());
    testCastToWithFloatIntermediate<half>(int());
    testCastToWithFloatIntermediate<half>(0U);
    testCastToWithFloatIntermediate<half>(uint64_t{0});
    testCastToWithFloatIntermediate<half>(int64_t{0});
}

}
