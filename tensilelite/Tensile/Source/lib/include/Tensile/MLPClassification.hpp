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
#include <functional>
#include <vector>
#include <algorithm>
#include <memory>

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
            void operator()(std::vector<dtype>& F) const
            {
                assert(mean.size() == F.size() && scale.size() == F.size());
                std::transform(F.begin(), F.end(), mean.begin(), F.begin(), std::minus{});
                std::transform(F.begin(), F.end(), scale.begin(), F.begin(), std::divides{});
            }

            std::vector<dtype> mean, scale;
        };

        inline std::vector<dtype>& activation(std::vector<dtype>& F)
        {
            for (auto& f : F)    // relu
                f = f > 0. ? f : 0.; // std::max(f, 0.f);
            return F;
        }

        inline std::vector<dtype> activation(std::vector<dtype>&& F)
        {
            return activation(F);
        }

        struct WeightMatrix
        {
            WeightMatrix() = default;
            WeightMatrix(const std::vector<float>& W) : weight(W.begin(), W.end()) {}
            virtual ~WeightMatrix() = default;

            virtual void operator()(const std::vector<dtype>& F,
                                    std::vector<dtype>& Fout) const
            {
                for (int i=0; i<Fout.size(); i++)
                    Fout[i] += std::inner_product
                        (F.begin(), F.end(), weight.begin()+i*F.size(), dtype(0.));
            }

            std::vector<dtype> weight;
        };

        /*
         * Specifying matrix dimensions at compile time for better unrolling etc.?
         */
        template <int N_IN, int N_OUT>
        struct WeightMatrixFixed : public WeightMatrix
        {
            WeightMatrixFixed() = default;
            WeightMatrixFixed(const std::vector<float>& W) : WeightMatrix(W) {}

            void operator()(const std::vector<dtype>& F,
                            std::vector<dtype>& Fout) const override
            {
                // assert(F.size() == N_IN && Fout.size() == N_OUT);
                // auto W = weight.data();
                // for (int i=0; i<N_OUT; i++) {
                //     dtype fi(0.);
                //     for (int j=0; j<N_IN; j++)
                //         fi += W[j] * F[j];
                //     W += N_IN;
                //     Fout[i] += fi;
                // }
                for (int i=0; i<N_OUT; i++)
                    Fout[i] += std::inner_product
                        (F.begin(), F.begin()+N_IN, weight.begin()+i*N_IN, dtype(0.));
            }
        };

        struct DenseLayer
        {
            DenseLayer() = default;

            DenseLayer(const std::vector<float>& weights, std::vector<float>& bias)
            {
                int n_out = bias.size();
                int n_in = weights.size() / n_out;
                     if (n_in ==  16 && n_out ==  16) W = std::make_shared<WeightMatrixFixed< 16, 16>>(weights);
                else if (n_in ==  64 && n_out == 128) W = std::make_shared<WeightMatrixFixed< 64,128>>(weights);
                else if (n_in == 128 && n_out == 256) W = std::make_shared<WeightMatrixFixed<128,256>>(weights);
                else if (n_in == 256 && n_out ==  64) W = std::make_shared<WeightMatrixFixed<256, 64>>(weights);
                else if (n_in ==  64 && n_out ==  64) W = std::make_shared<WeightMatrixFixed< 64, 64>>(weights);
                else if (n_in ==  64 && n_out ==  32) W = std::make_shared<WeightMatrixFixed< 64, 32>>(weights);
                else                                  W = std::make_shared<WeightMatrix>(weights);
                B.assign(bias.begin(), bias.end());
            }

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
            operator()(const std::vector<dtype>& F) const
            {
                auto Fout = linear2(activation(linear1(F)));
                auto Fres = res(F);
                std::transform(Fout.begin(), Fout.end(), Fres.begin(), Fout.begin(), std::plus{});
                return activation(Fout);
            }

            DenseLayer linear1, linear2, res;
        };

        struct TunaNet
        {
            TunaNet() = default;

            std::vector<dtype> predict(std::vector<float> const& probkey) const
            {
                dtype M = probkey[0], N = probkey[1], /*B = probkey[2],*/ K = probkey[3];
                dtype gflops = M * N * K / 1.e9, reads = (M*N + M*K + K*N) / 1.e6;
                std::vector<dtype> F =
                    {M, N, K,
                     dtype(std::log(M * N)),
                     dtype(int(M) % 256),
                     dtype(int(N) % 256),
                     dtype(int(K) % 256),
                     gflops, reads, gflops/reads};
                scaler(F);
                for (auto& res : res_blocks)
                    F = res(F);
                return dense(F);
            }

            std::string description() const
            {
                return "TunaNet";
            }

            std::vector<ResBlock> res_blocks;
            DenseLayer dense;
            StandardScaler scaler;
        };

     } // namespace MLPClassification
} // namespace TensileLite
