/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/ProblemKey.hpp>
#include <Tensile/MLPRegression.hpp>
#include <Tensile/ClassificationTree.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    /**
     * \ingroup SolutionLibrary
     *
     * Uses a small neural network to rank solutions for a given size.
     */

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct MLPRegressionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using MLP              = MLPRegression::MLP;
        using Tree             = Classification::Tree;
        using SolutionFeatures = std::vector<std::shared_ptr<MLFeatures::MLFeature<MySolution>>>;
        using ProblemFeatures  = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;

        std::map<int, std::shared_ptr<MySolution>> solutionmap;
        std::shared_ptr<MLP>                       model;
        std::shared_ptr<Tree>                      tree;
        SolutionFeatures                           solFeatures;
        ProblemFeatures                            probFeatures;

        static std::string Type()
        {
            return "MLPRegression";
        }
        virtual std::string type() const override
        {
            return Type();
        }
        virtual std::string description() const override
        {
            if(model == nullptr)
                return concatenate(type(), ", MLP: nullptr");
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
            // TODO this isn't used?
            std::vector<float> problemkey
                = ProblemKey::keyForProblem<std::vector<float>, MyProblem, float>(
                    problem, this->probFeatures);

            // auto effs = model->predict(problemkey);
            // auto sol = solutionmap.begin();
            // std::advance(sol, 
            //              std::distance(effs.cbegin(), 
            //                            std::max_element(effs.cbegin(), effs.cend())));
            // return sol->second;

            std::cout << "MLPRegressionLibrary::findBestSolution" << std::endl;

            return solutionmap.find(tree->predict(problemkey))->second;
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType != SolutionLibrarySearchType::DEFAULT)
            {
                // if the solution library search is not default then return an empty
                // set of solutions.
                SolutionSet<MySolution> rv;
                return rv;
            }

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
            std::cout << "MLPRegressionLibrary::findTopSolution(numSolutions=" 
                      << numSolutions << ")" << std::endl;
            
            if(numSolutions == 1)
                return SolutionVector<MySolution>({findBestSolution(problem, hardware)});

            std::vector<float> problemkey
                = ProblemKey::keyForProblem<std::vector<float>, MyProblem, float>(
                    problem, this->probFeatures);

            auto effs = model->predict(problemkey);

            std::vector<std::pair<float, int>> solutionRank;
            solutionRank.reserve(solutionmap.size());
            int i = 0;
            for(auto& s : solutionmap)
                solutionRank.emplace_back(effs[i++], s.first);

            numSolutions = std::min(numSolutions, int(solutionmap.size()));
            std::partial_sort
                (solutionRank.begin(), solutionRank.begin() + numSolutions,
                 solutionRank.end(), std::greater{});

            SolutionVector<MySolution> rv;
            rv.reserve(numSolutions);
            for(int i=0; i<numSolutions; i++)
            {
                auto indexMatch = solutionmap.find(solutionRank[i].second);
                rv.push_back(indexMatch->second);
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(searchType != SolutionLibrarySearchType::DEFAULT)
            {
                // if the solution library search is notSolutionSet default then return an empty
                // set of solutions
                SolutionSet<MySolution> rv;
                return rv;
            }

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
