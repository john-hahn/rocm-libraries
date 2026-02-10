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

// Convert float to FP8 E5M2 bits (OCP format: 1 sign, 5 exponent, 2 mantissa)
// Range: +/- 57344, has infinity and NaN
inline uint8_t float_to_fp8_e5m2_bits(float f, bool saturate = true) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 24) & 0x80; // Extract sign to bit 7
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15; // Rebias from float (127) to E5M2 (15)
    uint32_t mant = bits & 0x007FFFFF;

    // Handle NaN
    if(((bits >> 23) & 0xFF) == 255 && mant != 0)
    {
        return static_cast<uint8_t>(sign | 0x7F); // NaN (all mantissa bits set)
    }

    // Handle infinity
    if(((bits >> 23) & 0xFF) == 255)
    {
        if(saturate)
        {
            return static_cast<uint8_t>(sign | 0x7B); // Saturate to max finite
        }
        return static_cast<uint8_t>(sign | 0x7C); // Infinity
    }

    // Handle overflow
    if(exp >= 31)
    {
        if(saturate)
        {
            return static_cast<uint8_t>(sign | 0x7B); // max finite value
        }
        return static_cast<uint8_t>(sign | 0x7C); // Infinity
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
        uint32_t shift = static_cast<uint32_t>(1 - exp + 21); // 23 - 2 = 21 bits to shift
        if(shift >= 24)
        {
            return static_cast<uint8_t>(sign); // Too small, return zero
        }
        mant >>= shift;
        return static_cast<uint8_t>(sign | (mant & 0x03));
    }

    // Normal case: shift mantissa from 23 bits to 2 bits with rounding
    uint32_t fp8Mant = (mant >> 21) & 0x03;
    uint32_t remainder = mant & 0x001FFFFF;

    // Round to nearest even
    if(remainder > 0x100000 || (remainder == 0x100000 && (fp8Mant & 1)))
    {
        fp8Mant++;
        if(fp8Mant > 3)
        {
            fp8Mant = 0;
            exp++;
            if(exp >= 31)
            {
                if(saturate)
                {
                    return static_cast<uint8_t>(sign | 0x7B);
                }
                return static_cast<uint8_t>(sign | 0x7C); // Infinity
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | fp8Mant);
}

// Convert FP8 E5M2 bits to float
inline float fp8_e5m2_bits_to_float(uint8_t bits) noexcept
{
    uint32_t sign = (static_cast<uint32_t>(bits) & 0x80) << 24;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mant = bits & 0x03;

    // Handle infinity (exp=31, mant=0)
    if(exp == 31 && mant == 0)
    {
        uint32_t infBits = sign | 0x7F800000;
        float f;
        std::memcpy(&f, &infBits, sizeof(float));
        return f;
    }

    // Handle NaN (exp=31, mant!=0)
    if(exp == 31)
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
        // Denormalized: value = (-1)^sign * 2^(-14) * (0.mantissa)
        float value = static_cast<float>(mant) / 4.0f * (1.0f / 16384.0f);
        if(sign)
        {
            value = -value;
        }
        return value;
    }

    // Normal case: rebias from E5M2 (15) to float (127)
    exp = exp - 15 + 127;
    mant <<= 21; // Shift 2-bit mantissa to 23-bit position

    uint32_t floatBits = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

// NOLINTEND(readability-identifier-naming,readability-implicit-bool-conversion,modernize-use-auto)

} // namespace detail

/**
 * @brief Custom FP8 E5M2 type for hipDNN
 *
 * This type provides a portable FP8 E5M2 (1 sign, 5 exponent, 2 mantissa) implementation
 * that does not require the __HIPCC__ macro. Uses OCP E5M2 format.
 *
 * Binary layout: 1 sign bit, 5 exponent bits, 2 mantissa bits
 * Range: +/- 57344 (max normal value)
 * Has infinity and NaN representations
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp8_e5m2
{
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp8_e5m2() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp8_e5m2(const fp8_e5m2&) = default;
    fp8_e5m2(fp8_e5m2&&) noexcept = default;
    fp8_e5m2& operator=(const fp8_e5m2&) = default;
    fp8_e5m2& operator=(fp8_e5m2&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit fp8_e5m2(float f) noexcept
        : data(detail::float_to_fp8_e5m2_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit fp8_e5m2(double d) noexcept
        : data(detail::float_to_fp8_e5m2_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit fp8_e5m2(T value) noexcept
        : data(detail::float_to_fp8_e5m2_bits(static_cast<float>(value)))
    {
    }

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming) - using snake_case for factory function
    static constexpr fp8_e5m2 from_bits(uint8_t bits) noexcept
    {
        fp8_e5m2 val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::fp8_e5m2_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp8_e5m2_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    fp8_e5m2 operator-() const noexcept
    {
        return from_bits(data ^ 0x80);
    }

    // Unary plus
    fp8_e5m2 operator+() const noexcept
    {
        return *this;
    }

    // Arithmetic operators (compute in float, return fp8_e5m2)
    friend fp8_e5m2 operator+(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return fp8_e5m2(static_cast<float>(a) + static_cast<float>(b));
    }

    friend fp8_e5m2 operator-(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return fp8_e5m2(static_cast<float>(a) - static_cast<float>(b));
    }

    friend fp8_e5m2 operator*(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return fp8_e5m2(static_cast<float>(a) * static_cast<float>(b));
    }

    friend fp8_e5m2 operator/(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return fp8_e5m2(static_cast<float>(a) / static_cast<float>(b));
    }

    // Compound assignment operators
    fp8_e5m2& operator+=(fp8_e5m2 other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    fp8_e5m2& operator-=(fp8_e5m2 other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    fp8_e5m2& operator*=(fp8_e5m2 other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    fp8_e5m2& operator/=(fp8_e5m2 other) noexcept
    {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (compare via float conversion)
    friend bool operator==(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    friend bool operator!=(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }

    friend bool operator<(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }

    friend bool operator>(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }

    friend bool operator<=(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }

    friend bool operator>=(fp8_e5m2 a, fp8_e5m2 b) noexcept
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, fp8_e5m2 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp8_e5m2) == sizeof(uint8_t), "fp8_e5m2 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp8_e5m2>, "fp8_e5m2 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp8_e5m2>, "fp8_e5m2 must be standard layout");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming)
inline fp8_e5m2 operator""_fp8e5(long double val)
{
    return fp8_e5m2(static_cast<float>(val));
}

} // namespace hipdnn_data_sdk::types

// std:: namespace math function overloads
namespace std
{

inline hipdnn_data_sdk::types::fp8_e5m2 abs(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2::from_bits(x.data & 0x7F);
}

inline hipdnn_data_sdk::types::fp8_e5m2 fabs(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2::from_bits(x.data & 0x7F);
}

inline bool isnan(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    // E5M2 NaN: exp=31 (0x7C mask after shifting), mantissa!=0
    return ((x.data & 0x7C) == 0x7C) && ((x.data & 0x03) != 0);
}

inline bool isinf(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    // E5M2 Infinity: exp=31, mantissa=0
    return (x.data & 0x7F) == 0x7C;
}

inline bool signbit(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return (x.data & 0x80) != 0;
}

inline bool isfinite(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return !std::isnan(x) && !std::isinf(x);
}

inline hipdnn_data_sdk::types::fp8_e5m2 max(hipdnn_data_sdk::types::fp8_e5m2 a,
                                            hipdnn_data_sdk::types::fp8_e5m2 b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x7F); // NaN
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

inline hipdnn_data_sdk::types::fp8_e5m2 min(hipdnn_data_sdk::types::fp8_e5m2 a,
                                            hipdnn_data_sdk::types::fp8_e5m2 b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x7F); // NaN
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
inline hipdnn_data_sdk::types::fp8_e5m2 floor(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::floor(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e5m2 ceil(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::ceil(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e5m2 round(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::round(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e5m2 trunc(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::trunc(static_cast<float>(x)));
}

// Math functions (compute in float)
inline hipdnn_data_sdk::types::fp8_e5m2 exp(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::exp(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e5m2 log(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::log(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e5m2 sqrt(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::sqrt(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::fp8_e5m2 tanh(hipdnn_data_sdk::types::fp8_e5m2 x)
{
    return hipdnn_data_sdk::types::fp8_e5m2(std::tanh(static_cast<float>(x)));
}

} // namespace std

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::fp8_e5m2>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true; // E5M2 has infinity
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 3; // 2 mantissa + 1 implicit
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x04); // smallest positive normal: 2^-14
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 lowest() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0xFB); // -max = -57344
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 max() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x7B); // max = 57344
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 epsilon() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x34); // 2^-2 = 0.25
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 round_error() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x38); // 0.5
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 infinity() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x7C); // +inf
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x7F);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x7D);
    }

    static constexpr hipdnn_data_sdk::types::fp8_e5m2 denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::fp8_e5m2::from_bits(0x01); // smallest denormal
    }
};
// NOLINTEND(readability-identifier-naming)
