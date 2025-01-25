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

        struct StandardScaler
        {
            void operator()(std::vector<float>& F) const
            {
                assert(mean.size() == F.size() && scale.size() == F.size());
                std::transform(F.begin(), F.end(), mean.begin(), F.begin(), std::minus{});
                std::transform(F.begin(), F.end(), scale.begin(), F.begin(), std::divides{});
            }

            std::vector<float> mean, scale;
        };

        inline std::vector<float>& activation(std::vector<float>& F)
        {
            for (auto& f : F)    // relu
                f = std::max(f, 0.f);
            return F;
        }

        inline std::vector<float> activation(std::vector<float>&& F)
        {
            return activation(F);
        }

        struct DenseLayer
        {
            std::vector<float> operator()(const std::vector<float>& F) const
            {
                auto Fout = bias;
                const auto n_in = F.size(), n_out = Fout.size();
                #pragma omp parallel for if(n_out*n_in > 32*32)
                for (int i=0; i<n_out; i++) {
                    float fi = 0.;
                    for (int j=0; j<n_in; j++)
                        fi += weight[j+i*n_in] * F[j];
                    Fout[i] += fi;
                }
                return Fout;
            }

            std::vector<float> weight, bias;
        };

        struct ResBlock
        {
            std::vector<float> operator()(const std::vector<float>& F) const
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

            std::vector<float> predict(std::vector<float> const& probkey) const
            {
                float M = probkey[0], N = probkey[1], /*B = probkey[2],*/ K = probkey[3];
                float gflops = M * N * K / 1.e9, reads = (M*N + M*K + K*N) / 1.e6;
                std::vector<float> F =
                    {M, N, K, std::log(M * N),
                     float(int(M) % 256), float(int(N) % 256), float(int(K) % 256),
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
