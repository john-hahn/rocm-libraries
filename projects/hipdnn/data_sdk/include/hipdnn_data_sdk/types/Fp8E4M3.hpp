// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <type_traits>

namespace hipdnn_data_sdk::types
{

namespace detail
{

// NOLINTBEGIN(readability-identifier-naming,readability-implicit-bool-conversion,modernize-use-auto)

// Convert float to FP8 E4M3 bits (OCP format: 1 sign, 4 exponent, 3 mantissa)
// Range: +/- 448, no infinity, NaN = 0x7F or 0xFF
inline uint8_t float_to_fp8_e4m3_bits(float f, bool saturate = true) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 24) & 0x80; // Extract sign to bit 7
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 7; // Rebias from float (127) to E4M3 (7)
    uint32_t mant = bits & 0x007FFFFF;

    // Handle NaN
    if(((bits >> 23) & 0xFF) == 255 && mant != 0)
    {
        return static_cast<uint8_t>(sign | 0x7F); // NaN
    }

    // Handle infinity or overflow
    if(exp >= 15 || ((bits >> 23) & 0xFF) == 255)
    {
        if(saturate)
        {
            // Saturate to max finite value (0x7E for positive, 0xFE for negative)
            return static_cast<uint8_t>(sign | 0x7E);
        }
        return static_cast<uint8_t>(sign | 0x7F); // NaN (no infinity in E4M3)
    }

    // Handle zero
    if(exp <= 0 && mant == 0)
    {
        return static_cast<uint8_t>(sign); // Signed zero
    }

    // Handle denormalized or underflow
    if(exp <= 0)
    {
        // Denormalized
        mant |= 0x00800000; // Add implicit 1
        uint32_t shift = static_cast<uint32_t>(1 - exp + 20); // 23 - 3 = 20 bits to shift
        if(shift >= 24)
        {
            return static_cast<uint8_t>(sign); // Too small, return zero
        }
        mant >>= shift;
        return static_cast<uint8_t>(sign | (mant & 0x07));
    }

    // Normal case: shift mantissa from 23 bits to 3 bits with rounding
    uint32_t fp8Mant = (mant >> 20) & 0x07;
    uint32_t remainder = mant & 0x000FFFFF;

    // Round to nearest even
    if(remainder > 0x80000 || (remainder == 0x80000 && (fp8Mant & 1)))
    {
        fp8Mant++;
        if(fp8Mant > 7)
        {
            fp8Mant = 0;
            exp++;
            if(exp >= 15)
            {
                if(saturate)
                {
                    return static_cast<uint8_t>(sign | 0x7E);
                }
                return static_cast<uint8_t>(sign | 0x7F); // NaN
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | fp8Mant);
}

// Convert FP8 E4M3 bits to float
inline float fp8_e4m3_bits_to_float(uint8_t bits) noexcept
{
    uint32_t sign = (static_cast<uint32_t>(bits) & 0x80) << 24;
    uint32_t exp = (bits >> 3) & 0x0F;
    uint32_t mant = bits & 0x07;

    // Handle NaN (0x7F or 0xFF)
    if((bits & 0x7F) == 0x7F)
    {
        uint32_t nanBits = sign | 0x7FC00000; // Quiet NaN
        float f;
        std::memcpy(&f, &nanBits, sizeof(float));
        return f;
    }

    // Handle zero
    if(exp == 0 && mant == 0)
    {
        float f;
        std::memcpy(&f, &sign, sizeof(float));
        return f;
    }

    // Handle denormalized
    if(exp == 0)
    {
        // Denormalized: value = (-1)^sign * 2^(-6) * (0.mantissa)
        float value = static_cast<float>(mant) / 8.0f * (1.0f / 64.0f);
        if(sign)
        {
            value = -value;
        }
        return value;
    }

    // Normal case: rebias from E4M3 (7) to float (127)
    exp = exp - 7 + 127;
    mant <<= 20; // Shift 3-bit mantissa to 23-bit position

    uint32_t floatBits = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

// NOLINTEND(readability-identifier-naming,readability-implicit-bool-conversion,modernize-use-auto)

} // namespace detail

/**
 * @brief Custom FP8 E4M3 type for hipDNN
 *
 * This type provides a portable FP8 E4M3 (1 sign, 4 exponent, 3 mantissa) implementation
 * that does not require the __HIPCC__ macro. Uses OCP E4M3 format.
 *
 * Binary layout: 1 sign bit, 4 exponent bits, 3 mantissa bits
 * Range: +/- 448 (max normal value)
 * No infinity representation (uses NaN for overflow without saturation)
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp8_e4m3
{
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp8_e4m3() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp8_e4m3(const fp8_e4m3&) = default;
    fp8_e4m3(fp8_e4m3&&) noexcept = default;
    fp8_e4m3& operator=(const fp8_e4m3&) = default;
    fp8_e4m3& operator=(fp8_e4m3&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp8_e4m3(float f) noexcept
        : data(detail::float_to_fp8_e4m3_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp8_e4m3(double d) noexcept
        : data(detail::float_to_fp8_e4m3_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e4m3(T value) noexcept
        : data(detail::float_to_fp8_e4m3_bits(static_cast<float>(value)))
    {
    }

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming) - using snake_case for factory function
    static constexpr fp8_e4m3 from_bits(uint8_t bits) noexcept
    {
        fp8_e4m3 val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp8_e4m3_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e4m3_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    fp8_e4m3 operator-() const noexcept
    {
        return from_bits(data ^ 0x80);
    }

    // Unary plus
    fp8_e4m3 operator+() const noexcept
    {
        return *this;
    }

    // Arithmetic operators (compute in float, return fp8_e4m3)
    friend fp8_e4m3 operator+(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return fp8_e4m3(static_cast<float>(a) + static_cast<float>(b));
    }

    friend fp8_e4m3 operator-(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return fp8_e4m3(static_cast<float>(a) - static_cast<float>(b));
    }

    friend fp8_e4m3 operator*(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return fp8_e4m3(static_cast<float>(a) * static_cast<float>(b));
    }

    friend fp8_e4m3 operator/(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return fp8_e4m3(static_cast<float>(a) / static_cast<float>(b));
    }

    // Compound assignment operators
    fp8_e4m3& operator+=(fp8_e4m3 other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    fp8_e4m3& operator-=(fp8_e4m3 other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    fp8_e4m3& operator*=(fp8_e4m3 other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    fp8_e4m3& operator/=(fp8_e4m3 other) noexcept
    {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (compare via float conversion)
    friend bool operator==(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    friend bool operator!=(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }

    friend bool operator<(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }

    friend bool operator>(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }

    friend bool operator<=(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }

    friend bool operator>=(fp8_e4m3 a, fp8_e4m3 b) noexcept
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp8_e4m3 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp8_e4m3) == sizeof(uint8_t), "fp8_e4m3 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e4m3>, "fp8_e4m3 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e4m3>, "fp8_e4m3 must be standard layout");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp8_e4m3 operator""_fp8e4(long double val)
{
    return fp8_e4m3(static_cast<float>(val));
}

} // namespace hipdnn_data_sdk::types

// std:: namespace math function overloads
namespace std
{

inline hipdnn_data_sdk::types::fp8_e4m3 abs(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3::from_bits(x.data & 0x7F);
}

inline hipdnn_data_sdk::types::fp8_e4m3 fabs(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3::from_bits(x.data & 0x7F);
}

inline bool isnan(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    // E4M3 NaN: 0x7F or 0xFF (all exponent and mantissa bits set)
    return (x.data & 0x7F) == 0x7F;
}

inline bool isinf(hipdnn_data_sdk::types::fp8_e4m3 /*x*/)
{
    // E4M3 has no infinity representation
    return false;
}

inline bool signbit(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return (x.data & 0x80) != 0;
}

inline bool isfinite(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return !std::isnan(x);
}

inline hipdnn_data_sdk::types::fp8_e4m3 max(hipdnn_data_sdk::types::fp8_e4m3 a,
                                            hipdnn_data_sdk::types::fp8_e4m3 b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x7F); // NaN
    }
    if(std::isnan(a))
    {
        return b;
    }
    if(std::isnan(b))
    {
        return a;
    }
    return a > b ? a : b;
}

inline hipdnn_data_sdk::types::fp8_e4m3 min(hipdnn_data_sdk::types::fp8_e4m3 a,
                                            hipdnn_data_sdk::types::fp8_e4m3 b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x7F); // NaN
    }
    if(std::isnan(a))
    {
        return b;
    }
    if(std::isnan(b))
    {
        return a;
    }
    return a < b ? a : b;
}

// Rounding functions
inline hipdnn_data_sdk::types::fp8_e4m3 floor(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::floor(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e4m3 ceil(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::ceil(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e4m3 round(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::round(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e4m3 trunc(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::trunc(static_cast<float>(x)));
}

// Math functions (compute in float)
inline hipdnn_data_sdk::types::fp8_e4m3 exp(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::exp(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e4m3 log(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::log(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e4m3 sqrt(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::sqrt(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e4m3 tanh(hipdnn_data_sdk::types::fp8_e4m3 x)
{
    return hipdnn_data_sdk::types::fp8_e4m3(std::tanh(static_cast<float>(x)));
}

} // namespace std

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp8_e4m3>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false; // E4M3 has no infinity
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 4; // 3 mantissa + 1 implicit
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 3;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -5;
    static constexpr int min_exponent10 = -1;
    static constexpr int max_exponent = 8;
    static constexpr int max_exponent10 = 2;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x08); // smallest positive normal: 2^-6
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 lowest() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0xFE); // -max = -448
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 max() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x7E); // max = 448
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 epsilon() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x20); // 2^-3 = 0.125
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 round_error() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x38); // 0.5
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 infinity() noexcept
    {
        // E4M3 has no infinity, return max
        return max();
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x7F);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x7F); // Same as quiet NaN
    }

    static constexpr hipdnn_data_sdk::types::fp8_e4m3 denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e4m3::from_bits(0x01); // smallest denormal
    }
};
// NOLINTEND(readability-identifier-naming)
