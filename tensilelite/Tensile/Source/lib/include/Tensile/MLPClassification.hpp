/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <array>
#include <vector>
#include <memory>

#if defined(TENSILE_USE_ONNX)
#include "onnxruntime_cxx_api.h"
#endif

#include "DataTypes_Half.hpp"

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup MLPClassification MLP Classification
     *
     * @brief Classification model using multilayer perceptron
     *
     * Neural net used to estimate efficiency values for solutions in the
     * library. Used for MLPClassificationLibrary.
     */

    /**
     * \ingroup MLPClassification
     */
    namespace MLPClassification
    {

        // using dtype = TensileLite::Half;
        using dtype = float;
        // using dtype = _Float16;   // very slow, -mavx512fp16
        // using dtype = __bf16;     // compiler errors, tried with -march=native

        struct StandardScaler
        {
            void operator()(std::vector<dtype>& F) const;

            std::vector<dtype> mean, scale;
        };

        struct WeightMatrix
        {
            WeightMatrix() = default;
            WeightMatrix(const std::vector<float>& W) : weight(W.begin(), W.end()) {}
            virtual ~WeightMatrix() = default;

            virtual void operator()(const std::vector<dtype>& F,
                                    std::vector<dtype>& Fout) const;

            std::vector<dtype> weight;
        };

        /*
         * Specifying matrix dimensions at compile time for better unrolling etc.?
         */
        template <int N_IN>
        struct WeightMatrixFixed : public WeightMatrix
        {
            WeightMatrixFixed() = default;
            WeightMatrixFixed(const std::vector<float>& W) : WeightMatrix(W) {}

            void operator()(const std::vector<dtype>& F,
                            std::vector<dtype>& Fout) const override;
        };

        struct DenseLayer
        {
            DenseLayer() = default;
            DenseLayer(const std::vector<float>& weights, std::vector<float>& bias);

            std::vector<dtype>
            operator()(const std::vector<dtype>& F) const
            {
                auto Fout = B;
                (*W)(F, Fout);
                return Fout;
            }

            std::vector<dtype> B;
            std::shared_ptr<WeightMatrix> W;
        };

        struct ResBlock
        {
            ResBlock() = default;

            std::vector<dtype>
            operator()(const std::vector<dtype>& F) const;

            DenseLayer linear1, linear2, res;
        };

        struct TunaNet
        {
            TunaNet() = default;

            std::vector<dtype> predict(std::vector<float> const& probkey) const;

#if defined(TENSILE_USE_ONNX)
            std::vector<dtype> predict_onnx(std::vector<float> const& probkey,
                                            const char* onnx_model_path) const
            {
                std::cout << "Using ONNX model" << std::endl;

                float M = probkey[0], N = probkey[1], B = probkey[2], K = probkey[3];
                float gflops = M * N * K / 1.e9, reads = (M*N + M*K + K*N) / 1.e6;
                std::vector<float> F =
                    {M, N, K, B, float(std::log(M * N)),
                     float(int(M) % 256), float(int(N) % 256), float(int(K) % 256), float(int(B) % 256),
                     gflops, reads, gflops/reads};
                scaler(F);

                static auto ort_env = std::make_unique<Ort::Env>();
                static auto ort_session = std::make_unique<Ort::Session>
                    (*ort_env, (std::string(onnx_model_path) + "/" + onnx_model).c_str(),
                     Ort::SessionOptions{nullptr});
                const char* in_names[] = {"Fin"};
                const char* out_names[] = {"solution_logits"};
                auto n_solvers = dense.B.size();
                std::vector<float> logits(n_solvers);
                std::array<int64_t, 2> in_shape{1, int64_t(F.size())}, out_shape{1, int64_t(n_solvers)};
                auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                Ort::Value in_tensor = Ort::Value::CreateTensor<float>
                    (mem_info, F.data(), F.size(), in_shape.data(), in_shape.size());
                Ort::Value out_tensor = Ort::Value::CreateTensor<float>
                    (mem_info, logits.data(), logits.size(), out_shape.data(), out_shape.size());
                ort_session->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, &out_tensor, 1);
                if(std::is_same_v<dtype,float>)
                    return logits;
                return std::vector<dtype>(logits.begin(), logits.end());
            }
#endif

            std::string description() const
            {
                return "TunaNet";
            }

            std::string onnx_model;

            std::vector<ResBlock> res_blocks;
            DenseLayer dense;
            StandardScaler scaler;
        };

     } // namespace MLPClassification
} // namespace TensileLite
