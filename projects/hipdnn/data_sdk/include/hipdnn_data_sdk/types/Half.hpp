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

// NOLINTBEGIN(readability-identifier-naming,readability-else-after-return,
//              readability-implicit-bool-conversion,modernize-use-auto,
//              clang-diagnostic-sign-conversion)

// Convert float to fp16 bits using round-to-nearest-even
inline uint16_t float_to_half_bits(float f) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>(((bits >> 23) & 0xFF)) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFF;

    // Handle special cases
    if(exp <= 0)
    {
        if(exp < -10)
        {
            // Too small, return signed zero
            return static_cast<uint16_t>(sign);
        }
        // Denormalized number
        mant |= 0x00800000;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        mant >>= shift;
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if(exp == 0xFF - 127 + 15)
    {
        // Infinity or NaN
        if(mant == 0)
        {
            return static_cast<uint16_t>(sign | 0x7C00); // Infinity
        }
        return static_cast<uint16_t>(sign | 0x7C00 | (mant >> 13)); // NaN
    }
    if(exp > 30)
    {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    // Round to nearest even
    uint32_t halfMant = mant >> 13;
    uint32_t remainder = mant & 0x1FFF;
    if(remainder > 0x1000 || (remainder == 0x1000 && ((halfMant & 1) != 0U)))
    {
        halfMant++;
        if(halfMant > 0x3FF)
        {
            halfMant = 0;
            exp++;
            if(exp > 30)
            {
                return static_cast<uint16_t>(sign | 0x7C00); // Overflow
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

// Convert fp16 bits to float
inline float half_bits_to_float(uint16_t bits) noexcept
{
    uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000) << 16;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x03FF;

    if(exp == 0)
    {
        if(mant == 0)
        {
            // Signed zero
            float f;
            std::memcpy(&f, &sign, sizeof(float));
            return f;
        }
        // Denormalized
        while((mant & 0x0400) == 0)
        {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400U;
        exp = exp + 127 - 15;
        mant <<= 13;
    }
    else if(exp == 31)
    {
        // Infinity or NaN
        exp = 255;
        mant <<= 13;
    }
    else
    {
        exp = exp + 127 - 15;
        mant <<= 13;
    }

    uint32_t floatBits = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &floatBits, sizeof(float));
    return f;
}

// NOLINTEND(readability-identifier-naming,readability-else-after-return,
//           readability-implicit-bool-conversion,modernize-use-auto,
//           clang-diagnostic-sign-conversion)

} // namespace detail

/**
 * @brief Custom half (FP16) type for hipDNN
 *
 * This type provides a portable half-precision floating point implementation that does not require
 * the __HIPCC__ macro. Both constructors from float/double and conversions TO
 * float/double are explicit to prevent silent precision loss and overload ambiguity.
 *
 * Binary layout is compatible with __half (16-bit IEEE 754 half-precision).
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase to match half convention
struct half
{
    uint16_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr half() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    half(const half&) = default;
    half(half&&) noexcept = default;
    half& operator=(const half&) = default;
    half& operator=(half&&) noexcept = default;

    // EXPLICIT constructor from float
    explicit half(float f) noexcept
        : data(detail::float_to_half_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    explicit half(double d) noexcept
        : data(detail::float_to_half_bits(static_cast<float>(d)))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    explicit half(T value) noexcept
        : data(detail::float_to_half_bits(static_cast<float>(value)))
    {
    }

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming) - using snake_case for factory function
    static constexpr half from_bits(uint16_t bits) noexcept
    {
        half val;
        val.data = bits;
        return val;
    }

    // EXPLICIT conversion to float
    explicit operator float() const noexcept
    {
        return detail::half_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    explicit operator double() const noexcept
    {
        return static_cast<double>(detail::half_bits_to_float(data));
    }

    // Unary negation - XOR sign bit
    half operator-() const noexcept
    {
        return from_bits(data ^ 0x8000);
    }

    // Unary plus
    half operator+() const noexcept
    {
        return *this;
    }

    // Arithmetic operators (compute in float, return half)
    friend half operator+(half a, half b) noexcept
    {
        return half(static_cast<float>(a) + static_cast<float>(b));
    }

    friend half operator-(half a, half b) noexcept
    {
        return half(static_cast<float>(a) - static_cast<float>(b));
    }

    friend half operator*(half a, half b) noexcept
    {
        return half(static_cast<float>(a) * static_cast<float>(b));
    }

    friend half operator/(half a, half b) noexcept
    {
        return half(static_cast<float>(a) / static_cast<float>(b));
    }

    // Compound assignment operators
    half& operator+=(half other) noexcept
    {
        *this = *this + other;
        return *this;
    }

    half& operator-=(half other) noexcept
    {
        *this = *this - other;
        return *this;
    }

    half& operator*=(half other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    half& operator/=(half other) noexcept
    {
        *this = *this / other;
        return *this;
    }

    // Comparison operators (compare via float conversion)
    friend bool operator==(half a, half b) noexcept
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    friend bool operator!=(half a, half b) noexcept
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }

    friend bool operator<(half a, half b) noexcept
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }

    friend bool operator>(half a, half b) noexcept
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }

    friend bool operator<=(half a, half b) noexcept
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }

    friend bool operator>=(half a, half b) noexcept
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    // Stream output
    friend std::ostream& operator<<(std::ostream& os, half val)
    {
        return os << static_cast<float>(val);
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(half) == sizeof(uint16_t), "half must be 2 bytes");
static_assert(std::is_trivially_copyable_v<half>, "half must be trivially copyable");
static_assert(std::is_standard_layout_v<half>, "half must be standard layout");

// User-defined literal
// NOLINTNEXTLINE(readability-identifier-naming) - using _h suffix to match existing convention
inline half operator""_h(long double val)
{
    return half(static_cast<float>(val));
}

} // namespace hipdnn_data_sdk::types

// std:: namespace math function overloads
namespace std
{

// Basic math functions
inline hipdnn_data_sdk::types::half abs(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half::from_bits(x.data & 0x7FFF);
}

inline hipdnn_data_sdk::types::half fabs(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half::from_bits(x.data & 0x7FFF);
}

inline bool isnan(hipdnn_data_sdk::types::half x)
{
    // NaN: exponent all 1s (0x7C00) and non-zero mantissa
    return (x.data & 0x7C00) == 0x7C00 && (x.data & 0x03FF) != 0;
}

inline bool isinf(hipdnn_data_sdk::types::half x)
{
    // Inf: exponent all 1s and zero mantissa
    return (x.data & 0x7FFF) == 0x7C00;
}

inline bool signbit(hipdnn_data_sdk::types::half x)
{
    return (x.data & 0x8000) != 0;
}

inline bool isfinite(hipdnn_data_sdk::types::half x)
{
    return !std::isnan(x) && !std::isinf(x);
}

inline hipdnn_data_sdk::types::half copysign(hipdnn_data_sdk::types::half x,
                                             hipdnn_data_sdk::types::half y)
{
    uint16_t xBits = x.data & 0x7FFF; // magnitude of x
    uint16_t ySign = y.data & 0x8000; // sign of y
    return hipdnn_data_sdk::types::half::from_bits(xBits | ySign);
}

// Min/max with NaN handling
inline hipdnn_data_sdk::types::half max(hipdnn_data_sdk::types::half a,
                                        hipdnn_data_sdk::types::half b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        // Return canonical NaN
        return hipdnn_data_sdk::types::half::from_bits(0x7FFF);
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

inline hipdnn_data_sdk::types::half min(hipdnn_data_sdk::types::half a,
                                        hipdnn_data_sdk::types::half b)
{
    if(std::isnan(a) && std::isnan(b))
    {
        // Return canonical NaN
        return hipdnn_data_sdk::types::half::from_bits(0x7FFF);
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
inline hipdnn_data_sdk::types::half floor(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::floor(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half ceil(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::ceil(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half round(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::round(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half trunc(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::trunc(static_cast<float>(x)));
}

// Exponential and logarithmic functions
inline hipdnn_data_sdk::types::half exp(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::exp(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half exp2(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::exp2(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half log(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::log(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half log2(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::log2(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half log10(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::log10(static_cast<float>(x)));
}

// Power functions
inline hipdnn_data_sdk::types::half sqrt(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::sqrt(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half rsqrt(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(1.0f / std::sqrt(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half pow(hipdnn_data_sdk::types::half x,
                                        hipdnn_data_sdk::types::half y)
{
    return hipdnn_data_sdk::types::half(std::pow(static_cast<float>(x), static_cast<float>(y)));
}

// Trigonometric functions
inline hipdnn_data_sdk::types::half sin(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::sin(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half cos(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::cos(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half tan(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::tan(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half asin(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::asin(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half acos(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::acos(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half atan(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::atan(static_cast<float>(x)));
}

// Hyperbolic functions
inline hipdnn_data_sdk::types::half sinh(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::sinh(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half cosh(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::cosh(static_cast<float>(x)));
}

inline hipdnn_data_sdk::types::half tanh(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::tanh(static_cast<float>(x)));
}

// Error function
inline hipdnn_data_sdk::types::half erf(hipdnn_data_sdk::types::half x)
{
    return hipdnn_data_sdk::types::half(std::erf(static_cast<float>(x)));
}

// Floating-point manipulation
inline hipdnn_data_sdk::types::half fmod(hipdnn_data_sdk::types::half x,
                                         hipdnn_data_sdk::types::half y)
{
    return hipdnn_data_sdk::types::half(std::fmod(static_cast<float>(x), static_cast<float>(y)));
}

// Fused multiply-add
inline hipdnn_data_sdk::types::half fma(hipdnn_data_sdk::types::half x,
                                        hipdnn_data_sdk::types::half y,
                                        hipdnn_data_sdk::types::half z)
{
    return hipdnn_data_sdk::types::half(
        std::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
}

} // namespace std

// std::numeric_limits specialization
// NOLINTBEGIN(readability-identifier-naming) - standard library names must match exactly
template <>
class std::numeric_limits<hipdnn_data_sdk::types::half>
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
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11; // 10 mantissa bits + 1 implicit
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;

    static constexpr hipdnn_data_sdk::types::half min() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x0400); // smallest positive normal
    }

    static constexpr hipdnn_data_sdk::types::half lowest() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0xFBFF); // -max
    }

    static constexpr hipdnn_data_sdk::types::half max() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x7BFF); // largest finite (65504)
    }

    static constexpr hipdnn_data_sdk::types::half epsilon() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x1400); // 2^-10
    }

    static constexpr hipdnn_data_sdk::types::half round_error() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x3800); // 0.5
    }

    static constexpr hipdnn_data_sdk::types::half infinity() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x7C00); // +inf
    }

    static constexpr hipdnn_data_sdk::types::half quiet_NaN() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x7E00); // quiet NaN
    }

    static constexpr hipdnn_data_sdk::types::half signaling_NaN() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x7C01); // signaling NaN
    }

    static constexpr hipdnn_data_sdk::types::half denorm_min() noexcept
    {
        return hipdnn_data_sdk::types::half::from_bits(0x0001); // smallest denormal
    }
};
// NOLINTEND(readability-identifier-naming)
