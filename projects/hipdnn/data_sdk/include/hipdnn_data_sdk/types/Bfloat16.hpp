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

// NOLINTBEGIN(readability-identifier-naming) - using snake_case for internal detail functions

// Convert float to bfloat16 bits using truncation (matches HIP behavior)
inline uint16_t float_to_bfloat16_bits(float f) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    // Truncate lower 16 bits (simple truncation, no rounding)
    return static_cast<uint16_t>(bits >> 16);
}

// Convert bfloat16 bits to float
inline float bfloat16_bits_to_float(uint16_t bits) noexcept
{
    uint32_t floatBits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

// NOLINTEND(readability-identifier-naming)

} // namespace detail

/**
 * @brief Custom bfloat16 type for hipDNN
 *
 * This type provides a portable bfloat16 implementation that does not require
 * the __HIPCC__ macro. Both constructors from float/double and conversions TO
 * float/double are explicit to prevent silent precision loss and overload ambiguity.
 *
 * Binary layout is compatible with hip_bfloat16 (16-bit, same bit representation).
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase to match hip_bfloat16 convention
struct bfloat16
{
    uint16_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr bfloat16() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    bfloat16(const bfloat16&) = default;
    bfloat16(bfloat16&&) noexcept = default;
    bfloat16& operator=(const bfloat16&) = default;
    bfloat16& operator=(bfloat16&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit bfloat16(float f) noexcept
        : data(detail::float_to_bfloat16_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit bfloat16(double d) noexcept
        : data(detail::float_to_bfloat16_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit bfloat16(T value) noexcept
        : data(detail::float_to_bfloat16_bits(static_cast<float>(value)))
    {
    }

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming) - using snake_case for factory function
    static constexpr bfloat16 from_bits(uint16_t bits) noexcept
    {
        bfloat16 val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::bfloat16_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::bfloat16_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    bfloat16 operator-() const noexcept
    {
        return from_bits(data ^ 0x8000);
    }

    // Unary plus
    bfloat16 operator+() const noexcept
    {
        return *this;
    }

    // Arithmetic operators (compute in float, return bfloat16)
    friend bfloat16 operator+(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) + static_cast<float>(b));
    }

    friend bfloat16 operator-(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) - static_cast<float>(b));
    }

    friend bfloat16 operator*(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) * static_cast<float>(b));
    }

    friend bfloat16 operator/(bfloat16 a, bfloat16 b) noexcept
    {
        return bfloat16(static_cast<float>(a) / static_cast<float>(b));
    }

    // Compound assignment operators
    bfloat16& operator+=(bfloat16 other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    bfloat16& operator-=(bfloat16 other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    bfloat16& operator*=(bfloat16 other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    bfloat16& operator/=(bfloat16 other) noexcept
    {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (compare via float conversion)
    friend bool operator==(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    friend bool operator!=(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }

    friend bool operator<(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }

    friend bool operator>(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }

    friend bool operator<=(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }

    friend bool operator>=(bfloat16 a, bfloat16 b) noexcept
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, bfloat16 val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(bfloat16) == sizeof(uint16_t), "bfloat16 must be 2 bytes");
static_assert(std::is_trivially_copyable_v<bfloat16>, "bfloat16 must be trivially copyable");
static_assert(std::is_standard_layout_v<bfloat16>, "bfloat16 must be standard layout");

// User-defined literal
inline bfloat16 operator""_bf16(long double val)
{
    return bfloat16(static_cast<float>(val));
}

} // namespace hipdnn_data_sdk::types

// std:: namespace math function overloads
namespace std
{

// Basic math functions
inline hipdnn_data_sdk::types::bfloat16 abs(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16::from_bits(x.data & 0x7FFF);
}

inline hipdnn_data_sdk::types::bfloat16 fabs(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16::from_bits(x.data & 0x7FFF);
}

inline bool isnan(hipdnn_data_sdk::types::bfloat16 x)
{
    // NaN: exponent all 1s (0x7F80) and non-zero mantissa
    return (x.data & 0x7F80) == 0x7F80 && (x.data & 0x007F) != 0;
}

inline bool isinf(hipdnn_data_sdk::types::bfloat16 x)
{
    // Inf: exponent all 1s and zero mantissa
    return (x.data & 0x7FFF) == 0x7F80;
}

inline bool signbit(hipdnn_data_sdk::types::bfloat16 x)
{
    return (x.data & 0x8000) != 0;
}

inline bool isfinite(hipdnn_data_sdk::types::bfloat16 x)
{
    return !std::isnan(x) && !std::isinf(x);
}

inline hipdnn_data_sdk::types::bfloat16 copysign(hipdnn_data_sdk::types::bfloat16 x,
                                                 hipdnn_data_sdk::types::bfloat16 y)
{
    uint16_t xBits = x.data & 0x7FFF; // magnitude of x
    uint16_t ySign = y.data & 0x8000; // sign of y
    return hipdnn_data_sdk::types::bfloat16::from_bits(xBits | ySign);
}

// Min/max with NaN handling
inline hipdnn_data_sdk::types::bfloat16 max(hipdnn_data_sdk::types::bfloat16 a,
                                            hipdnn_data_sdk::types::bfloat16 b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        // Return canonical NaN
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x7FFF);
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

inline hipdnn_data_sdk::types::bfloat16 min(hipdnn_data_sdk::types::bfloat16 a,
                                            hipdnn_data_sdk::types::bfloat16 b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        // Return canonical NaN
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x7FFF);
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
inline hipdnn_data_sdk::types::bfloat16 floor(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::floor(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 ceil(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::ceil(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 round(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::round(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 trunc(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::trunc(static_cast<float>(x)));
}

// Exponential and logarithmic functions
inline hipdnn_data_sdk::types::bfloat16 exp(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::exp(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 exp2(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::exp2(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 log(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::log(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 log2(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::log2(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 log10(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::log10(static_cast<float>(x)));
}

// Power functions
inline hipdnn_data_sdk::types::bfloat16 sqrt(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::sqrt(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 rsqrt(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(1.0f / std::sqrt(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 pow(hipdnn_data_sdk::types::bfloat16 x,
                                            hipdnn_data_sdk::types::bfloat16 y)
{
    return hipdnn_data_sdk::types::bfloat16(std::pow(static_cast<float>(x), static_cast<float>(y)));
}

// Trigonometric functions
inline hipdnn_data_sdk::types::bfloat16 sin(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::sin(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 cos(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::cos(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 tan(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::tan(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 asin(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::asin(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 acos(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::acos(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 atan(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::atan(static_cast<float>(x)));
}

// Hyperbolic functions
inline hipdnn_data_sdk::types::bfloat16 sinh(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::sinh(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 cosh(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::cosh(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::bfloat16 tanh(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::tanh(static_cast<float>(x)));
}

// Error function
inline hipdnn_data_sdk::types::bfloat16 erf(hipdnn_data_sdk::types::bfloat16 x)
{
    return hipdnn_data_sdk::types::bfloat16(std::erf(static_cast<float>(x)));
}

// Floating-point manipulation
inline hipdnn_data_sdk::types::bfloat16 fmod(hipdnn_data_sdk::types::bfloat16 x,
                                             hipdnn_data_sdk::types::bfloat16 y)
{
    return hipdnn_data_sdk::types::bfloat16(
        std::fmod(static_cast<float>(x), static_cast<float>(y)));
}

// Fused multiply-add
inline hipdnn_data_sdk::types::bfloat16 fma(hipdnn_data_sdk::types::bfloat16 x,
                                            hipdnn_data_sdk::types::bfloat16 y,
                                            hipdnn_data_sdk::types::bfloat16 z)
{
    return hipdnn_data_sdk::types::bfloat16(
        std::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
}

} // namespace std

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::bfloat16>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 8; // 7 mantissa bits + 1 implicit
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::bfloat16 min() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x0080); // smallest positive normal
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 lowest() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0xFF7F); // -max
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 max() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x7F7F); // largest finite
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 epsilon() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x3C00); // 2^-7
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 round_error() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x3F00); // 0.5
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 infinity() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x7F80); // +inf
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x7FC0); // quiet NaN
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x7F81); // signaling NaN
    }

    static constexpr hipdnn_data_sdk::types::bfloat16 denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::bfloat16::from_bits(0x0001); // smallest denormal
    }
};
// NOLINTEND(readability-identifier-naming)
