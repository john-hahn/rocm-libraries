// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/All.hpp>

// Import the custom types for use in tests
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::half;

// Import user-defined literals for convenience
using namespace hipdnn_data_sdk::types; // NOLINT(google-build-using-namespace) - for literals

TEST(TestUtilsFp16, BasicUsage)
{
    half h = 1.0_h;
    EXPECT_EQ(h, 1.0_h);
}

TEST(TestUtilsFp16, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_h), 1.0_h);
    EXPECT_EQ(std::fabs(1.0_h), 1.0_h);
}

TEST(TestUtilsFp16, Comparison)
{
    EXPECT_EQ(1.25_h, 1.25_h);
    ASSERT_NE(1.0_h, 2.0_h);
    ASSERT_LT(0.0156_h, 0.0176_h);
    ASSERT_LE(-0.0156_h, 0.0176_h);
    ASSERT_GT(0.0215_h, 0.0176_h);
    ASSERT_GE(0.0215_h, -0.0176_h);
}

TEST(TestUtilsFp16, Max)
{
    half a = 1.0_h;
    half b = 2.0_h;
    EXPECT_EQ(std::max(a, b), 2.0_h);
    EXPECT_EQ(std::max(b, a), 2.0_h);
}

TEST(TestUtilsBfp16, BasicUsage)
{
    bfloat16 bf = 1.0_bf16;
    EXPECT_EQ(bf, 1.0_bf16);
}

TEST(TestUtilsBfp16, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_bf16), 1.0_bf16);
    EXPECT_EQ(std::fabs(1.0_bf16), 1.0_bf16);
}

TEST(TestUtilsBfp16, Comparison)
{
    EXPECT_EQ(1.25_bf16, 1.25_bf16);
    ASSERT_NE(1.0_bf16, 2.0_bf16);
    ASSERT_LT(0.017_bf16, 0.018_bf16);
    ASSERT_LE(-0.017_bf16, 0.018_bf16);
    ASSERT_GT(0.022_bf16, 0.018_bf16);
    ASSERT_GE(0.022_bf16, -0.018_bf16);
}

TEST(TestUtilsBfp16, Max)
{
    bfloat16 a = 1.0_bf16;
    bfloat16 b = 2.0_bf16;
    EXPECT_EQ(std::max(a, b), 2.0_bf16);
    EXPECT_EQ(std::max(b, a), 2.0_bf16);
}

TEST(TestUtilsFp8, BasicUsage)
{
    fp8_e4m3 fp8 = 1.0_fp8e4;
    EXPECT_EQ(fp8, 1.0_fp8e4);
}

TEST(TestUtilsFp8, Negation)
{
    fp8_e4m3 fp8 = 1.0_fp8e4;
    EXPECT_EQ(-fp8, fp8_e4m3(-1.0f));
}

TEST(TestUtilsFp8, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_fp8e4), 1.0_fp8e4);
    EXPECT_EQ(std::fabs(1.0_fp8e4), 1.0_fp8e4);
}

TEST(TestUtilsFp8, Comparison)
{
    EXPECT_EQ(1.25_fp8e4, 1.25_fp8e4);
    ASSERT_NE(1.0_fp8e4, 2.0_fp8e4);
    ASSERT_LT(0.0156_fp8e4, 0.0176_fp8e4);
    ASSERT_LE(-0.0156_fp8e4, 0.0176_fp8e4);
    ASSERT_GT(0.0215_fp8e4, 0.0176_fp8e4);
    ASSERT_GE(0.0215_fp8e4, -0.0176_fp8e4);
}

TEST(TestUtilsFp8, Max)
{
    fp8_e4m3 a = 1.0_fp8e4;
    fp8_e4m3 b = 2.0_fp8e4;
    fp8_e4m3 nan = fp8_e4m3::from_bits(0x7F);
    EXPECT_EQ(std::max(a, b), 2.0_fp8e4);
    EXPECT_EQ(std::max(b, a), 2.0_fp8e4);
    EXPECT_EQ(std::max(a, nan), 1.0_fp8e4);
    EXPECT_EQ(std::max(nan, b), 2.0_fp8e4);
    EXPECT_TRUE(std::isnan(std::max(nan, nan)));
}

TEST(TestUtilsBfp8, BasicUsage)
{
    fp8_e5m2 bf = 1.0_fp8e5;
    EXPECT_EQ(bf, 1.0_fp8e5);
}

TEST(TestUtilsBfp8, Negation)
{
    fp8_e5m2 bf = 1.0_fp8e5;
    EXPECT_EQ(-bf, fp8_e5m2(-1.0f));
}

TEST(TestUtilsBfp8, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_fp8e5), 1.0_fp8e5);
    EXPECT_EQ(std::fabs(1.0_fp8e5), 1.0_fp8e5);
}

TEST(TestUtilsBfp8, Comparison)
{
    EXPECT_EQ(1.25_fp8e5, 1.25_fp8e5);
    ASSERT_NE(1.0_fp8e5, 2.0_fp8e5);
    ASSERT_LT(0.0001_fp8e5, 0.0002_fp8e5);
    ASSERT_LT(-0.0002_fp8e5, 0.0003_fp8e5);
    ASSERT_GT(0.0003_fp8e5, 0.0002_fp8e5);
    ASSERT_GE(0.0002_fp8e5, -0.0002_fp8e5);
}

TEST(TestUtilsBfp8, Max)
{
    fp8_e5m2 a = 1.0_fp8e5;
    fp8_e5m2 b = 2.0_fp8e5;
    fp8_e5m2 nan = fp8_e5m2::from_bits(0x7F);
    EXPECT_EQ(std::max(a, b), 2.0_fp8e5);
    EXPECT_EQ(std::max(b, a), 2.0_fp8e5);
    EXPECT_EQ(std::max(a, nan), 1.0_fp8e5);
    EXPECT_EQ(std::max(nan, b), 2.0_fp8e5);
    EXPECT_TRUE(std::isnan(std::max(nan, nan)));
}
