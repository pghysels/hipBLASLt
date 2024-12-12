/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup MLPRegression MLP Regression
     *
     * @brief Regression model using multilayer perceptron
     *
     * Neural net used to estimate efficiency values for solutions in the
     * library. Used for MLPRegressionLibrary.
     */

    /**
     * \ingroup MLPRegression
     */
    namespace MLPRegression
    {

        struct StandardScaler
        {
            void transform(std::vector<float>& F) const
            {
                assert(mean.size() == F.size() && var.size() == F.size());
                std::transform(F.begin(), F.end(), mean.begin(), F.begin(), std::minus{});
                std::transform(F.begin(), F.end(), var.begin(), F.begin(), std::divides{});
            }

            std::vector<float> mean, var;
        };

        struct MLP
        {

            MLP() = default;

            std::vector<float> predict(std::vector<float> const& probkey) const
            {
                float M = probkey[0], N = probkey[1], /*B = probkey[2],*/ K = probkey[3];
                float gflops = M * N * K / 1.e9, reads = M*N + M*K + K*N;
                std::vector<float> F =
                    {std::log(M), std::log(N), std::log(K), std::log(M * N),
                     float(int(M) % 256), float(int(N) % 256), float(int(K) % 256),
                     gflops, reads, std::log(gflops/reads)};

                scaler.transform(F);

                const int layers = dims.size()-1;
                for (int l=0; l<dims.size()-1; l++)
                {
                    auto Ftmp = bias[l];
                    for (int i=0; i<dims[l+1]; i++)
                    {
                        for (int j=0; j<dims[l]; j++)
                            Ftmp[i] += weights[l][i+j*dims[l+1]] * F[j];
                            // Ftmp[i] += weights[l][j+i*dims[l]] * F[j];
                        if (l < layers-1)
                            Ftmp[i] = std::max(Ftmp[i], 0.f);
                    }
                    std::swap(Ftmp, F);
                }
                return F;
            }

            std::string description() const
            {
                return "MLPRegression";
            }

            std::vector<int> dims;
            std::vector<std::vector<float>> weights, bias;
            StandardScaler scaler;
        };

     } // namespace MLPRegression
} // namespace TensileLite
