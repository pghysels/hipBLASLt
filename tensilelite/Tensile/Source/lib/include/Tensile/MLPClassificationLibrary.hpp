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

#pragma once

#include <queue>
#include <set>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/MLFeatures.hpp>
#include <Tensile/MLPClassification.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/analytical/Utils.hpp>


namespace TensileLite
{
    /**
     * \ingroup SolutionLibrary
     *
     * Uses a small neural network to rank solutions for a given size.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct MLPClassificationLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using MLPNet           = MLPClassification::MLPNet;
        using SolutionFeatures = std::vector<std::shared_ptr<MLFeatures::MLFeature<MySolution>>>;
        using ProblemFeatures  = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;

        std::map<int, std::shared_ptr<MySolution>>      solutionmap;
        std::shared_ptr<MLPNet>                         model;
        SolutionFeatures                                solFeatures;
        ProblemFeatures                                 probFeatures;
        std::vector<TensileLite::analytical::TileTuple> tile_list;

        static std::string Type()
        {
            return "MLPClassification";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(model == nullptr)
                return concatenate(type(), ", MLPNet: nullptr");
            else
                return concatenate(type(), ": ", model->description());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // If the experimental library mode is not on treat it like it asserted out
                return nullptr;
            }
            // ;
            auto indexMatch = solutionmap.find(index);
            if(indexMatch != solutionmap.end())
                return indexMatch->second;
            return nullptr;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            SolutionVector<MySolution>  solutions = findTopSolutions(problem, hardware, 1);
            std::shared_ptr<MySolution> solution  = nullptr;
            if(solutions.size() > 0)
                solution = solutions[0];
            return solution;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }
            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            size_t                     m     = 1;
            size_t                     n     = 1;
            size_t                     k     = 1;
            size_t                     batch = 1;
            for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            {
                m *= problem.freeSizeA(i);
            }
            for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            {
                n *= problem.freeSizeB(i);
            }
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
            {
                k *= problem.boundSize(i);
            }
            for(size_t i = 0; i < problem.batchIndices().size(); ++i)
            {
                batch *= problem.batchSize(i);
            }

            bool                  debug   = Debug::Instance().printPropertyEvaluation();
            hip::HipAMDGPU const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            size_t                elementSizeA_bits
                = problem.a().elementBytes() * 8; // TODO update for A/B different types
            size_t elementSizeB_bits
                = problem.b().elementBytes() * 8; // TODO update for A/B different types
            size_t elementSizeC_bits
                = problem.c().elementBytes() * 8; // TODO update for A/B different types
            const analytical::Hardware& analaytical_hardware = *(pAMDGPU->analyticalHardware);
            int                         WGM
                = std::sqrt(std::floor(analaytical_hardware.N_CU / analaytical_hardware.NUM_XCD));
            auto selected_tiles = analytical::select_best_macro_tile_size(
                m,
                n,
                k,
                batch,
                problem.transA(),
                problem.transB(),
                *(pAMDGPU->analyticalHardware),
                tile_list,
                elementSizeA_bits,
                elementSizeA_bits,
                elementSizeC_bits,
                0, //mx_block_size -> MX Data types come from rocroller.
                0.8,
                debug,
                false,
                WGM);

            std::unordered_map<TensileLite::analytical::TileTuple, double> tile_latencies;
            tile_latencies.reserve(selected_tiles.size());
            for (const auto& tile : selected_tiles)
            {
                // std::cout << "tile latency: " << std::get<0>(tile)
                //           << " MT=" << std::get<1>(tile) << ", "
                //           << std::get<2>(tile) << ", "
                //           << std::get<3>(tile) << ", "
                //           << std::get<4>(tile) << ", "
                //           << std::get<5>(tile) << ", "
                //           << std::get<6>(tile) << std::endl;
                tile_latencies.insert({std::make_tuple(std::get<1>(tile),
                                                     std::get<2>(tile),
                                                     std::get<3>(tile),
                                                     std::get<4>(tile),
                                                     std::get<5>(tile),
                                                     std::get<6>(tile),
                                                     std::get<7>(tile)),
                                       std::get<0>(tile)});
            }

            std::vector<float> problemkey
                = ProblemKey::keyForProblem<std::vector<float>, MyProblem, float>(
                    problem, this->probFeatures);

            auto logits = model->predict(problemkey);
            assert(logits.size() == solutionmap.size());

            // used to sort solutions, first on Origami latency, then on logits
            std::vector<std::tuple<double,
                                   decltype(logits)::value_type,
                                   std::shared_ptr<MySolution>*>> solution_ranking;
            solution_ranking.reserve(solutionmap.size());
            for(auto& s : solutionmap)
            {
                // std::cout << "latency, logits: " << tile_latencies[std::make_tuple(
                //             s.second->sizeMapping.macroTile.x, // MT_M
                //             s.second->sizeMapping.macroTile.y, // MT_N
                //             s.second->sizeMapping.depthU, // MT_K
                //             s.second->sizeMapping.matrixInstruction[0], // MI_M
                //             s.second->sizeMapping.matrixInstruction[1], // MI_N
                //             s.second->sizeMapping.matrixInstruction[2])] << "  "
                //         << logits[s.second->libraryLogicIndex] << std::endl;
                solution_ranking.emplace_back(
                    tile_latencies[std::make_tuple(
                            s.second->sizeMapping.macroTile.x, // MT_M
                            s.second->sizeMapping.macroTile.y, // MT_N
                            s.second->sizeMapping.depthU, // MT_K
                            s.second->sizeMapping.matrixInstruction[0], // MI_M
                            s.second->sizeMapping.matrixInstruction[1], // MI_N
                            s.second->sizeMapping.matrixInstruction[2],
                            s.second->sizeMapping.CUOccupancy)],
                    -logits[s.second->libraryLogicIndex],
                    (std::shared_ptr<MySolution>*)(&s.second));
            }

            SolutionVector<MySolution> rv;
            int numToSort = std::min(numSolutions, int(solution_ranking.size()));
            rv.reserve(numToSort);
            auto it = solution_ranking.begin(), it_end = solution_ranking.end();
            while(it != it_end && numToSort)
            {
                std::partial_sort(it, it + numToSort, it_end);
                for(; it != it + numToSort; it++)
                {
                    auto& solution = *std::get<2>(*it);
                    if((*solution->hardwarePredicate)(hardware) &&
                       (*solution->problemPredicate)(problem))
                    {
                        // std::cout << "SORTED: " << std::get<0>(*it) << " " << -std::get<1>(*it) << std::endl;
                        rv.emplace_back(solution);
                        numToSort--;
                    }
                }
            }
            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            const bool experimental = Debug::Instance().useExperimentalSelection();
            if(!experimental)
            {
                // Skip the search for solutions if the environment variable
                // that enables the experimental method is not set
                SolutionSet<MySolution> rv;
                return rv;
            }

            SolutionSet<MySolution> rv;
            for(auto const& row : solutionmap)
                rv.insert(row.second);

            return rv;
        }
    };

} // namespace TensileLite
