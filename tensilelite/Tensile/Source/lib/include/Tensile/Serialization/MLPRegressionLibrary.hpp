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

#include <Tensile/Debug.hpp>
#include <Tensile/MLPRegressionLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace TensileLite
{
    namespace Serialization
    {
      
        template <typename IO>
        struct MappingTraits<Classification::Tree, IO>
        {
            using Tree = Classification::Tree;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Tree& tree)
            {
                iot::mapRequired(io, "left", tree.left);
                iot::mapRequired(io, "right", tree.right);
                iot::mapRequired(io, "feature_solution", tree.feature_solution);
                iot::mapRequired(io, "threshold", tree.threshold);
            }

            const static bool flow = false;
        };

        
        template <typename IO>
        struct MappingTraits<MLPRegression::StandardScaler, IO>
        {
            using Scaler = MLPRegression::StandardScaler;
            using iot    = IOTraits<IO>;

            static void mapping(IO& io, Scaler& scaler)
            {
                iot::mapRequired(io, "mean", scaler.mean);
                iot::mapRequired(io, "var", scaler.var);
            }

            const static bool flow = false;
        };
                
        template <typename IO>
        struct MappingTraits<MLPRegression::MLP, IO>
        {
            using MLP = MLPRegression::MLP;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, MLP& mlp)
            {
                iot::mapRequired(io, "dimensions", mlp.dims);
                iot::mapRequired(io, "weights", mlp.weights);
                iot::mapRequired(io, "bias", mlp.bias);
                iot::mapRequired(io, "scaler", mlp.scaler);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<MLPRegressionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = MLPRegressionLibrary<MyProblem, MySolution>;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "MLPRegressionLibrary requires that context be "
                                  "set to a SolutionMap.");
                }
                std::vector<int> mappingIndices;
                if(iot::outputting(io))
                {
                    mappingIndices.reserve(lib.solutionmap.size());

                    for(auto const& pair : lib.solutionmap)
                        mappingIndices.push_back(pair.first);

                    iot::mapRequired(io, "table", mappingIndices);
                }
                else
                {
                    iot::mapRequired(io, "table", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "MLPRegressionLibrary requires non empty "
                                      "mapping index set.");
                    
                    for(int index : mappingIndices)
                    {
                        auto slnIter = ctx->solutions->find(index);
                        if(slnIter == ctx->solutions->end())
                        {
                            iot::setError(
                                io,
                                concatenate("[MLPRegressionLibrary] Invalid solution index: ",
                                            index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            lib.solutionmap.insert(std::make_pair(index, solution));
                        }
                    }
                }

                using MLP = MLPRegression::MLP;
                std::shared_ptr<MLP> model;
                if(iot::outputting(io))
                {
                    model = std::dynamic_pointer_cast<MLP>(lib.model);
                }
                else
                {
                    model     = std::make_shared<MLP>();
                    lib.model = model;
                }
                iot::mapRequired(io, "mlp", *model);

                using Tree = Classification::Tree;
                std::shared_ptr<Tree> tree;
                if(iot::outputting(io))
                {
                    tree = std::dynamic_pointer_cast<Tree>(lib.tree);
                }
                else
                {
                    tree     = std::make_shared<Tree>();
                    lib.tree = tree;
                }
                iot::mapRequired(io, "tree", *tree);

                using ProblemFeatures
                    = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;
                ProblemFeatures probFeatures;
                if(iot::outputting(io))
                {
                    probFeatures = lib.probFeatures;
                }
                iot::mapOptional(io, "problemFeatures", probFeatures);
                lib.probFeatures = probFeatures;
            }
            const static bool flow = false;
        };

    } // namespace Serialization
} // namespace TensileLite
