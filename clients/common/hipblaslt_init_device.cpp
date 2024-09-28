/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_xorwow.h>

#include "hipblaslt_init.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_random.hpp"
#include <hipblaslt/hipblaslt.h>


template<typename T, typename F>
void fill_batch(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, const F& f)
{
    thrust::for_each_n(thrust::device, thrust::counting_iterator(0), std::max(lda * N, stride) * batch_count, f);
}

__device__ uint32_t pseudo_random_device(size_t idx)
{
    rocrand_state_xorwow state;
    rocrand_init(idx, 0, 0, &state);
    return rocrand(&state);
}

template<typename T>
__device__ T random_generator_device(size_t idx)
{
    return T(pseudo_random_device(idx) % 10 + 1.f);
}

template<typename T>
__device__ T random_hpl_generator_device(size_t idx)
{
    auto r = pseudo_random_device(idx);
    return T(double(r) / double(std::numeric_limits<decltype(r)>::max()) - 0.5);
}

template<typename T>
__device__ T random_inf_generator_device(size_t idx)
{
    return T(pseudo_random_device(idx) & 1 ? -std::numeric_limits<double>::infinity() 
                                           : std::numeric_limits<double>::infinity());
}

template<typename T>
__device__ T random_zero_generator_device(size_t idx)
{
    return T(pseudo_random_device(idx) & 1 ? -0.0 : 0.0);
}

template <typename T>
void hipblaslt_init_device(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    fill_batch(A, M, N, lda, stride, batch_count,
        [A](size_t idx) { A[idx] = random_generator_device<T>(idx); });
}

template <typename T>
void hipblaslt_init_device_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    if constexpr(std::is_same<T,float>::value || 
                 std::is_same<T,double>::value ||
                 std::is_same<T,hipblasLtHalf>::value ||
                 std::is_same<T,int32_t>::value)
    {
        fill_batch(A, M, N, lda, stride, batch_count,
            [A](size_t idx) { A[idx] = T(random_generator_device<T>(idx) / 10.0f); } );
    }
    else
        hipblaslt_cerr << "Error type in hipblaslt_init_device_small" << std::endl;
}

template <typename T>
inline void hipblaslt_init_device_sin(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    fill_batch(A, M, N, lda, stride, batch_count,
        [A](size_t idx) { A[idx] = T(sin(double(idx))); } );
}

template <typename T>
inline void hipblaslt_init_device_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    stride = std::max(lda * N, stride);
    fill_batch(A, M, N, lda, stride, batch_count,
        [A,stride,lda](size_t idx) { 
            auto b = idx / stride;
            auto j = (idx - b * stride) / lda;
            auto i = (idx - b * stride) - j * lda;
            auto value = random_generator_device<T>(idx);
            A[idx] = (i ^ j) & 1 ? value : negate(value); 
        } );
}

template <typename T>
inline void hipblaslt_init_device_cos(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    fill_batch(A, M, N, lda, stride, batch_count,
        [A](size_t idx) { A[idx] = T(cos(double(idx))); } );
}

template <typename T>
inline void hipblaslt_init_device_hpl(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    fill_batch(A, M, N, lda, stride, batch_count,
        [A,stride,lda](size_t idx) { A[idx] = random_hpl_generator_device<T>(idx); } );
}

template <typename T>
inline void hipblaslt_init_device_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    // generate 100 random NaN's on host, and copy to device
    std::array<T,100> rand_nans;
    for(auto& r : rand_nans)
        r = T(hipblaslt_nan_rng());
    fill_batch(A, M, N, lda, stride, batch_count,
        [A,rand_nans](size_t idx) { A[idx] = rand_nans[pseudo_random_device(idx) % rand_nans.size()]; });
}

template <typename T>
inline void hipblaslt_init_device_inf(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    fill_batch(A, M, N, lda, stride, batch_count,
        [A](size_t idx) { A[idx] = random_inf_generator_device<T>(idx); } );
}

template <typename T>
inline void hipblaslt_init_device_zero(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    fill_batch(A, M, N, lda, stride, batch_count,
        [A](size_t idx) { A[idx] = T(0); } );
}

template <typename T>
inline void hipblaslt_init_device_alt_impl_big(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    const hipblasLtHalf ieee_half_max(65280.0);
    fill_batch(A, M, N, lda, stride, batch_count,
        [A,ieee_half_max](size_t idx) { A[idx] = T(ieee_half_max); } );
}

template <typename T>
inline void hipblaslt_init_device_alt_impl_small(
    T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    const hipblasLtHalf ieee_half_small(0.0000607967376708984375);
    fill_batch(A, M, N, lda, stride, batch_count,
        [A,ieee_half_small](size_t idx) { A[idx] = T(ieee_half_small); } );
}

template<typename T>
void hipblaslt_init_device(ABC         abc,
                           hipblaslt_initialization init,
                           bool        is_nan,
                           T*          A,
                           size_t      M,
                           size_t      N,
                           size_t      lda,
                           size_t      stride,
                           size_t      batch_count) 
{
    if(is_nan)
    {
        hipblaslt_init_device_nan<T>(A, M, N, lda, stride, batch_count);
    }
    else
    {
        switch(init)
        {
        case hipblaslt_initialization::rand_int:
            if(abc == ABC::A || abc == ABC::C)
                hipblaslt_init_device<T>(A, M, N, lda, stride, batch_count);
            else if(abc == ABC::B)
                hipblaslt_init_device_alternating_sign<T>(A, M, N, lda, stride, batch_count);
            break;
        case hipblaslt_initialization::trig_float:
            if(abc == ABC::A || abc == ABC::C)
                hipblaslt_init_device_sin(A, M, N, lda, stride, batch_count);
            else if(abc == ABC::B)
                hipblaslt_init_device_cos(A, M, N, lda, stride, batch_count);
            break;
        case hipblaslt_initialization::hpl:
            hipblaslt_init_device_hpl(A, M, N, lda, stride, batch_count);
            break;
        case hipblaslt_initialization::special:
            if (abc == ABC::A)
                hipblaslt_init_device_alt_impl_big(A, M, N, lda, stride, batch_count);
            else if(abc == ABC::B)
                hipblaslt_init_device_alt_impl_small(A, M, N, lda, stride, batch_count);
            else if(abc == ABC::C)
                hipblaslt_init_device(A, M, N, lda, stride, batch_count);
            break;
        case hipblaslt_initialization::zero:
            hipblaslt_init_device_zero(A, M, N, lda, stride, batch_count);
        default:
            hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
            break;
        }
    }
}

void hipblaslt_init_device(ABC         abc,
                           hipblaslt_initialization init,
                           bool        is_nan,
                           void*       A,
                           size_t      M,
                           size_t      N,
                           size_t      lda,
                           hipDataType type,
                           size_t      stride,
                           size_t      batch_count) 
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_device<float>(
            abc, init, is_nan, static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_device<double>(
            abc, init, is_nan, static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_device<hipblasLtHalf>(
            abc, init, is_nan, static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_device<hip_bfloat16>(
            abc, init, is_nan, static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_device<hipblaslt_f8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_device<hipblaslt_bf8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        hipblaslt_init_device<hipblaslt_f8>(
            abc, init, is_nan, static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_device<hipblaslt_bf8>(
            abc, init, is_nan, static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_device<int32_t>(
            abc, init, is_nan, static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_device<hipblasLtInt8>(
            abc, init, is_nan, static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
        break;
    }
}
