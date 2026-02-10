// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types/All.hpp>

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::half;

namespace
{

// Test that static_cast works between custom types
TEST(TestStaticCast, BetweenCustomTypes)
{
    // bfloat16 to other types
    auto bf = bfloat16(1.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<half>(bf)), 1.5f);
    EXPECT_NEAR(static_cast<float>(static_cast<fp8_e4m3>(bf)), 1.5f, 0.1f);
    EXPECT_NEAR(static_cast<float>(static_cast<fp8_e5m2>(bf)), 1.5f, 0.25f);

    // half to other types
    auto h = half(2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<bfloat16>(h)), 2.0f);
    EXPECT_NEAR(static_cast<float>(static_cast<fp8_e4m3>(h)), 2.0f, 0.1f);
    EXPECT_NEAR(static_cast<float>(static_cast<fp8_e5m2>(h)), 2.0f, 0.25f);

    // fp8_e4m3 to other types
    auto f4 = fp8_e4m3(3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<bfloat16>(f4)), 3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<half>(f4)), 3.0f);
    EXPECT_NEAR(static_cast<float>(static_cast<fp8_e5m2>(f4)), 3.0f, 0.25f);

    // fp8_e5m2 to other types
    auto f5 = fp8_e5m2(4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<bfloat16>(f5)), 4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<half>(f5)), 4.0f);
    EXPECT_NEAR(static_cast<float>(static_cast<fp8_e4m3>(f5)), 4.0f, 0.1f);
}

// Test that static_cast works from standard types
TEST(TestStaticCast, FromStandardTypes)
{
    // From float
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<bfloat16>(1.5f)), 1.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<half>(1.5f)), 1.5f);

    // From double
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<bfloat16>(1.5)), 1.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<half>(1.5)), 1.5f);

    // From int
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<bfloat16>(2)), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(static_cast<half>(2)), 2.0f);
}

// Test that static_cast to float/double works
TEST(TestStaticCast, ToStandardTypes)
{
    auto bf = bfloat16(3.5f);
    auto h = half(3.5f);

    EXPECT_FLOAT_EQ(static_cast<float>(bf), 3.5f);
    EXPECT_DOUBLE_EQ(static_cast<double>(bf), 3.5);

    EXPECT_FLOAT_EQ(static_cast<float>(h), 3.5f);
    EXPECT_DOUBLE_EQ(static_cast<double>(h), 3.5);
}

}
