/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <numeric>
#include <cassert>

#include <Tensile/MLPClassification.hpp>

namespace TensileLite
{
    namespace MLPClassification
    {
        void StandardScaler::operator()(std::vector<dtype>& F) const
        {
            assert(mean.size() == F.size() && scale.size() == F.size());
            std::transform(F.begin(), F.end(), mean.begin(), F.begin(), std::minus{});
            std::transform(F.begin(), F.end(), scale.begin(), F.begin(), std::divides{});
        }

        void WeightMatrix::operator()(const std::vector<dtype>& F,
                                      std::vector<dtype>& Fout) const
        {
            for (int i=0; i<Fout.size(); i++)
                Fout[i] += std::inner_product
                    (F.begin(), F.end(), weight.begin()+i*F.size(), dtype(0.));
        }

        template <int N_IN>
        void WeightMatrixFixed<N_IN>::operator()(const std::vector<dtype>& F,
                                                 std::vector<dtype>& Fout) const
        {
            assert(F.size() == N_IN);
            auto W = weight.data();
            for (int i=0; i<Fout.size(); i++) {
                dtype fi(0.);
                auto Fptr = F.data();
                #pragma clang loop unroll_count(N_IN)
                for (int j=0; j<N_IN; j++)
                    fi += (*W++) * (*Fptr++);
                Fout[i] += fi;
            }
        }

        DenseLayer::DenseLayer(const std::vector<float>& weights, std::vector<float>& bias)
        {
            int n_in = weights.size() / bias.size();
            switch (n_in) {
                case  10: W = std::make_shared<WeightMatrixFixed< 10>>(weights); break;
                case  16: W = std::make_shared<WeightMatrixFixed< 16>>(weights); break;
                case  32: W = std::make_shared<WeightMatrixFixed< 32>>(weights); break;
                case  64: W = std::make_shared<WeightMatrixFixed< 64>>(weights); break;
                case 128: W = std::make_shared<WeightMatrixFixed<128>>(weights); break;
                case 256: W = std::make_shared<WeightMatrixFixed<256>>(weights); break;
                default:  W = std::make_shared<WeightMatrix>(weights);
            }
            B.assign(bias.begin(), bias.end());
        }

        std::vector<dtype>& activation(std::vector<dtype>& F)
        {
            for (auto& f : F)    // relu
                f = f > 0. ? f : 0.; // std::max(f, 0.f);
            return F;
        }

        std::vector<dtype> activation(std::vector<dtype>&& F)
        {
            return activation(F);
        }

        std::vector<dtype> ResBlock::operator()(const std::vector<dtype>& F) const
        {
            auto Fout = linear2(activation(linear1(F)));
            auto Fres = res(F);
            std::transform(Fout.begin(), Fout.end(), Fres.begin(), Fout.begin(), std::plus{});
            return activation(Fout);
        }

        std::vector<dtype> TunaNet::predict(std::vector<float> const& probkey) const
        {
#if defined(TENSILE_USE_ONNX)
            static const char* onnx_model_path = std::getenv("TENSILE_ONNX_MODEL_PATH");
            if(!onnx_model.empty() && onnx_model_path)
                return predict_onnx(probkey, onnx_model_path);
#endif
            dtype M = probkey[0], N = probkey[1], B = probkey[2], K = probkey[3];
            dtype gflops = M * N * K / 1.e9, reads = (M*N + M*K + K*N) / 1.e6;
            std::vector<dtype> F =
                {M, N, K, B, dtype(std::log(M * N)),
                 dtype(int(M) % 256), dtype(int(N) % 256), dtype(int(K) % 256), dtype(int(B) % 256),
                 gflops, reads, gflops/reads};
            scaler(F);
            for (auto& res : res_blocks)
                F = res(F);
            return dense(F);
        }

    }
}
