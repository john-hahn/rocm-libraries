// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "benchmark_rocrand_occupancy_helper.hpp"
#include "benchmark_rocrand_utils.hpp"

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

template<typename State, typename Seed>
__global__
void init_states_kernel(State* states, Seed seed, unsigned long long offset)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    rocrand_init(seed, tid, offset, &states[tid]);
}

template<typename State, typename SobolType>
__global__
void init_sobol_kernel(State*     states,
                       SobolType* directions,
                       SobolType* scramble_constants,
                       size_t     offset)
{
    const unsigned int dimension = blockIdx.y;
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    State              state{};

    constexpr size_t elements_per_dim = sizeof(SobolType) * 8;

    if constexpr(std::is_same_v<State, rocrand_state_scrambled_sobol32>
                 || std::is_same_v<State, rocrand_state_scrambled_sobol64>)
    {
        rocrand_init(&directions[dimension * elements_per_dim],
                     scramble_constants[dimension],
                     offset + state_id,
                     &state);
    }
    else
    {
        rocrand_init(&directions[dimension * elements_per_dim], offset + state_id, &state);
    }

    states[dimension * gridDim.x * blockDim.x + state_id] = state;
}

template<typename State, typename T, typename Generator>
__global__
void generate_kernel(State* states, T* data, size_t size, Generator generator)
{
    const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    if constexpr(std::is_same_v<State, rocrand_state_sobol32>
                 || std::is_same_v<State, rocrand_state_sobol64>
                 || std::is_same_v<State, rocrand_state_scrambled_sobol32>
                 || std::is_same_v<State, rocrand_state_scrambled_sobol64>)
    {
        const unsigned int dimension = blockIdx.y;
        const unsigned int state_id  = tid;
        const size_t       offset    = dimension * size;

        const size_t state_base = gridDim.x * blockDim.x * dimension + state_id;

        State state = states[state_base];

        size_t index = state_id;
        while(index < size)
        {
            data[offset + index] = generator(&state);
            skipahead(stride - 1, &state);
            index += stride;
        }

        State final_state = states[state_base];
        skipahead(size, &final_state);
        states[state_base] = final_state;
    }
    else
    {
        State state = states[tid];

        for(size_t i = tid; i < size; i += stride)
            data[i] = generator(&state);

        states[tid] = state;
    }
}

enum distribution
{
    DISTRIBUTION_DEFAULT,
    DISTRIBUTION_UNIFORM,
    DISTRIBUTION_NORMAL,
    DISTRIBUTION_LOG_NORMAL,
    DISTRIBUTION_POISSON,
    DISTRIBUTION_DISCRETE_POISSON,
    DISTRIBUTION_DISCRETE_CUSTOM,
};

constexpr const char* distribution_name(distribution d)
{
    switch(d)
    {
        case DISTRIBUTION_DEFAULT: return "default";
        case DISTRIBUTION_UNIFORM: return "uniform";
        case DISTRIBUTION_NORMAL: return "normal";
        case DISTRIBUTION_LOG_NORMAL: return "log_normal";
        case DISTRIBUTION_POISSON: return "poisson";
        case DISTRIBUTION_DISCRETE_POISSON: return "discrete_poisson";
        case DISTRIBUTION_DISCRETE_CUSTOM: return "discrete_custom";
    }
    return "unknown";
}

constexpr size_t div_ceil(size_t numerator, size_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

constexpr size_t next_power2(size_t x)
{
    size_t power = 1;
    while(power < x)
        power *= 2;
    return power;
}

template<typename State, typename T, distribution Distribution>
struct rocrand_device_api_benchmark : public primbench::benchmark_interface
{
    rocrand_device_api_benchmark(rocrand_rng_type      engine,
                                 size_t                blocks,
                                 size_t                threads,
                                 size_t                dimensions,
                                 size_t                offset,
                                 std::optional<double> poisson_lambda = std::nullopt)
        : m_engine(engine)
        , m_blocks(blocks)
        , m_threads(threads)
        , m_dimensions(dimensions)
        , m_offset(offset)
        , m_poisson_lambda(poisson_lambda)
    {}

    primbench::json meta() const override
    {
        auto json = primbench::json{}
                        .add("algo", "rocrand_device_api")
                        .add("engine", engine_name(m_engine))
                        .add("type", primbench::name<T>())
                        .add("distribution", distribution_name(Distribution))
                        .add("blocks", m_blocks)
                        .add("threads", m_threads)
                        .add("dimensions", m_dimensions)
                        .add("offset", m_offset);

        if constexpr(Distribution == DISTRIBUTION_POISSON
                     || Distribution == DISTRIBUTION_DISCRETE_POISSON)
            json.add("poisson_lambda", *m_poisson_lambda);

        return json;
    }

    void run(primbench::state& state) override
    {
        const hipStream_t stream = state.stream;
        const size_t      items  = state.bytes / sizeof(T);
        const auto        seed   = state.seed;

        State* d_states{};
        T*     d_data{};

        allocate_states_and_data(items, d_states, d_data);
        init_states(stream, seed, d_states);

        rocrand_discrete_distribution discrete_dist{};
        create_discrete_distribution(discrete_dist);

        run_generation(stream, items, d_states, d_data, discrete_dist, state);

        if constexpr(Distribution == DISTRIBUTION_DISCRETE_POISSON
                     || Distribution == DISTRIBUTION_DISCRETE_CUSTOM)
        {
            ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_dist));
        }

        HIP_CHECK(hipFree(d_states));
        HIP_CHECK(hipFree(d_data));
    }

private:
    void allocate_states_and_data(size_t items, State*& d_states, T*& d_data)
    {
        HIP_CHECK(hipMalloc(&d_data, items * sizeof(T)));

        if constexpr(std::is_same_v<State, rocrand_state_sobol32>
                     || std::is_same_v<State, rocrand_state_sobol64>
                     || std::is_same_v<State, rocrand_state_scrambled_sobol32>
                     || std::is_same_v<State, rocrand_state_scrambled_sobol64>)
        {
            const size_t states_per_dim  = div_ceil(m_blocks, m_dimensions);
            const size_t padded_blocks_x = next_power2(states_per_dim);
            const size_t total_states    = padded_blocks_x * m_threads * m_dimensions;
            HIP_CHECK(hipMalloc(&d_states, total_states * sizeof(State)));
        }
        else
        {
            HIP_CHECK(hipMalloc(&d_states, m_blocks * m_threads * sizeof(State)));
        }
    }

    void init_states(hipStream_t stream, unsigned long long seed, State* d_states)
    {
        if constexpr(std::is_same_v<State, rocrand_state_mtgp32>)
        {
            const size_t states_size = std::min((size_t)200, m_blocks);
            ROCRAND_CHECK(
                rocrand_make_state_mtgp32(d_states, mtgp32dc_params_fast_11213, states_size, seed));
        }
        else if constexpr(std::is_same_v<State, rocrand_state_sobol32>
                          || std::is_same_v<State, rocrand_state_sobol64>
                          || std::is_same_v<State, rocrand_state_scrambled_sobol32>
                          || std::is_same_v<State, rocrand_state_scrambled_sobol64>)
        {
            init_sobol_states(stream, d_states);
        }
        else if constexpr(std::is_same_v<State, rocrand_state_lfsr113>)
        {
            const uint4 lfsr_seed{ROCRAND_LFSR113_DEFAULT_SEED_X,
                                  ROCRAND_LFSR113_DEFAULT_SEED_Y,
                                  ROCRAND_LFSR113_DEFAULT_SEED_Z,
                                  ROCRAND_LFSR113_DEFAULT_SEED_W};

            init_states_kernel<<<m_blocks, m_threads, 0, stream>>>(d_states, lfsr_seed, m_offset);
        }
        else
        {
            init_states_kernel<<<m_blocks, m_threads, 0, stream>>>(d_states, seed, m_offset);
        }
    }

    void init_sobol_states(hipStream_t stream, State* d_states)
    {
        constexpr size_t dir_bytes
            = std::is_same_v<State, rocrand_state_sobol32>
                      || std::is_same_v<State, rocrand_state_scrambled_sobol32>
                  ? 32
                  : 64;

        using dir_type = std::conditional_t<dir_bytes == 32, unsigned int, unsigned long long>;

        const dir_type* h_dirs{};
        const dir_type* h_scramble_consts{};

        if constexpr(std::is_same_v<State, rocrand_state_sobol32>)
            ROCRAND_CHECK(
                rocrand_get_direction_vectors32(&h_dirs, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        else if constexpr(std::is_same_v<State, rocrand_state_sobol64>)
            ROCRAND_CHECK(
                rocrand_get_direction_vectors64(&h_dirs, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        else if constexpr(std::is_same_v<State, rocrand_state_scrambled_sobol32>)
        {
            ROCRAND_CHECK(
                rocrand_get_direction_vectors32(&h_dirs,
                                                ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
            ROCRAND_CHECK(rocrand_get_scramble_constants32(&h_scramble_consts));
        }
        else
        {
            ROCRAND_CHECK(
                rocrand_get_direction_vectors64(&h_dirs,
                                                ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
            ROCRAND_CHECK(rocrand_get_scramble_constants64(&h_scramble_consts));
        }

        dir_type* d_dirs{};
        dir_type* d_scramble_consts{};

        HIP_CHECK(hipMalloc(&d_dirs, m_dimensions * dir_bytes * sizeof(dir_type)));
        HIP_CHECK(hipMemcpy(d_dirs,
                            h_dirs,
                            m_dimensions * dir_bytes * sizeof(dir_type),
                            hipMemcpyHostToDevice));

        if constexpr(std::is_same_v<State, rocrand_state_scrambled_sobol32>
                     || std::is_same_v<State, rocrand_state_scrambled_sobol64>)
        {
            HIP_CHECK(hipMalloc(&d_scramble_consts, m_dimensions * sizeof(dir_type)));
            HIP_CHECK(hipMemcpy(d_scramble_consts,
                                h_scramble_consts,
                                m_dimensions * sizeof(dir_type),
                                hipMemcpyHostToDevice));
        }

        const size_t states_per_dim  = div_ceil(m_blocks, m_dimensions);
        const size_t padded_blocks_x = next_power2(states_per_dim);

        init_sobol_kernel<State, dir_type>
            <<<dim3(padded_blocks_x, m_dimensions), dim3(m_threads), 0, stream>>>(d_states,
                                                                                  d_dirs,
                                                                                  d_scramble_consts,
                                                                                  m_offset);

        if(d_scramble_consts)
            HIP_CHECK(hipFree(d_scramble_consts));
        HIP_CHECK(hipFree(d_dirs));
    }

    void create_discrete_distribution(rocrand_discrete_distribution& dist)
    {
        if constexpr(Distribution == DISTRIBUTION_DISCRETE_POISSON)
        {
            ROCRAND_CHECK(rocrand_create_poisson_distribution(*m_poisson_lambda, &dist));
        }
        else if constexpr(Distribution == DISTRIBUTION_DISCRETE_CUSTOM)
        {
            std::vector<double> probs{10, 10, 1, 120, 8, 6, 140, 2, 150, 150, 10, 80};
            double              sum = std::accumulate(probs.begin(), probs.end(), 0.0);
            for(auto& p : probs)
                p /= sum;

            ROCRAND_CHECK(
                rocrand_create_discrete_distribution(probs.data(), probs.size(), m_offset, &dist));
        }
    }

    void run_generation(hipStream_t                   stream,
                        size_t                        items,
                        State*                        d_states,
                        T*                            d_data,
                        rocrand_discrete_distribution dist,
                        primbench::state&             state)
    {
        const double poisson_lambda
            = (Distribution == DISTRIBUTION_POISSON) ? *m_poisson_lambda : 0.0;

        auto gen = [=](auto* s) -> T
        {
            if constexpr(Distribution == DISTRIBUTION_DEFAULT && std::is_same_v<T, unsigned int>)
                return rocrand(s);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, float>)
                return rocrand_uniform(s);
            else if constexpr(Distribution == DISTRIBUTION_UNIFORM && std::is_same_v<T, double>)
                return rocrand_uniform_double(s);
            else if constexpr(Distribution == DISTRIBUTION_NORMAL && std::is_same_v<T, float>)
                return rocrand_normal(s);
            else if constexpr(Distribution == DISTRIBUTION_NORMAL && std::is_same_v<T, double>)
                return rocrand_normal_double(s);
            else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL && std::is_same_v<T, float>)
                return rocrand_log_normal(s, 0.0f, 1.0f);
            else if constexpr(Distribution == DISTRIBUTION_LOG_NORMAL && std::is_same_v<T, double>)
                return rocrand_log_normal_double(s, 0.0, 1.0);
            else if constexpr(Distribution == DISTRIBUTION_POISSON)
                return rocrand_poisson(s, poisson_lambda);
            else if constexpr(Distribution == DISTRIBUTION_DISCRETE_POISSON
                              || Distribution == DISTRIBUTION_DISCRETE_CUSTOM)
                return rocrand_discrete(s, dist);
            else
                static_assert(sizeof(T) == 0, "Unsupported distribution/type combination");
        };

        static_assert(std::is_trivially_copyable_v<decltype(gen)>);

        state.set_items(items);
        state.add_writes<T>(items);

        if constexpr(std::is_same_v<State, rocrand_state_sobol32>
                     || std::is_same_v<State, rocrand_state_sobol64>
                     || std::is_same_v<State, rocrand_state_scrambled_sobol32>
                     || std::is_same_v<State, rocrand_state_scrambled_sobol64>)
        {
            const size_t states_per_dim  = div_ceil(m_blocks, m_dimensions);
            const size_t padded_blocks_x = next_power2(states_per_dim);

            state.run(
                [&]
                {
                    generate_kernel<<<dim3(padded_blocks_x, m_dimensions),
                                      dim3(m_threads),
                                      0,
                                      stream>>>(d_states, d_data, items, gen);
                });
        }
        else
        {
            state.run(
                [&] {
                    generate_kernel<<<m_blocks, m_threads, 0, stream>>>(d_states,
                                                                        d_data,
                                                                        items,
                                                                        gen);
                });
        }
    }

private:
    rocrand_rng_type      m_engine;
    size_t                m_blocks;
    size_t                m_threads;
    size_t                m_dimensions;
    size_t                m_offset;
    std::optional<double> m_poisson_lambda;
};

#define QUEUE(T, State, engine, Dist, ...)                                   \
    executor.queue<rocrand_device_api_benchmark<State, T, Dist>>(engine,     \
                                                                 blocks,     \
                                                                 threads,    \
                                                                 dimensions, \
                                                                 offset,     \
                                                                 ##__VA_ARGS__)

#define QUEUE_DISTRIBUTIONS(State, engine)                                             \
    do                                                                                 \
    {                                                                                  \
        QUEUE(unsigned int, State, engine, DISTRIBUTION_DEFAULT);                      \
        QUEUE(float, State, engine, DISTRIBUTION_UNIFORM);                             \
        QUEUE(double, State, engine, DISTRIBUTION_UNIFORM);                            \
        QUEUE(float, State, engine, DISTRIBUTION_NORMAL);                              \
        QUEUE(double, State, engine, DISTRIBUTION_NORMAL);                             \
        QUEUE(float, State, engine, DISTRIBUTION_LOG_NORMAL);                          \
        QUEUE(double, State, engine, DISTRIBUTION_LOG_NORMAL);                         \
        for(double lambda : poisson_lambdas)                                           \
        {                                                                              \
            QUEUE(unsigned int, State, engine, DISTRIBUTION_POISSON, lambda);          \
            QUEUE(unsigned int, State, engine, DISTRIBUTION_DISCRETE_POISSON, lambda); \
        }                                                                              \
        QUEUE(unsigned int, State, engine, DISTRIBUTION_DISCRETE_CUSTOM);              \
    }                                                                                  \
    while(0)

int main(int argc, char* argv[])
{
    primbench::executor executor(argc, argv, 128 * primbench::MiB);

    auto blocks     = executor.get<size_t>("blocks", 256, "Number of blocks");
    auto threads    = executor.get<size_t>("threads", 256, "Threads per block");
    auto dimensions = executor.get<size_t>("dimensions", 1, "Number of quasi-random dimensions");
    auto offset     = executor.get<size_t>("offset", 0, "Offset of generated pseudo-random values");
    auto poisson_lambdas
        = executor.get<std::vector<double>>("lambda",
                                            {10.0},
                                            "Space-separated list of Poisson lambdas");

    QUEUE_DISTRIBUTIONS(rocrand_state_lfsr113, ROCRAND_RNG_PSEUDO_LFSR113);
    QUEUE_DISTRIBUTIONS(rocrand_state_mrg31k3p, ROCRAND_RNG_PSEUDO_MRG31K3P);
    QUEUE_DISTRIBUTIONS(rocrand_state_mrg32k3a, ROCRAND_RNG_PSEUDO_MRG32K3A);
    QUEUE_DISTRIBUTIONS(rocrand_state_philox4x32_10, ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry2x32_20, ROCRAND_RNG_PSEUDO_THREEFRY2_32_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry4x32_20, ROCRAND_RNG_PSEUDO_THREEFRY4_32_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry2x64_20, ROCRAND_RNG_PSEUDO_THREEFRY2_64_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_threefry4x64_20, ROCRAND_RNG_PSEUDO_THREEFRY4_64_20);
    QUEUE_DISTRIBUTIONS(rocrand_state_xorwow, ROCRAND_RNG_PSEUDO_XORWOW);

    QUEUE_DISTRIBUTIONS(rocrand_state_mtgp32, ROCRAND_RNG_PSEUDO_MTGP32);
    QUEUE_DISTRIBUTIONS(rocrand_state_sobol32, ROCRAND_RNG_QUASI_SOBOL32);
    QUEUE_DISTRIBUTIONS(rocrand_state_scrambled_sobol32, ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    QUEUE_DISTRIBUTIONS(rocrand_state_sobol64, ROCRAND_RNG_QUASI_SOBOL64);
    QUEUE_DISTRIBUTIONS(rocrand_state_scrambled_sobol64, ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64);

    executor.run();
}
