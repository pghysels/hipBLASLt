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
     * \defgroup Classification
     *
     * @brief Classification model for solution selection
     *
     */

    /**
     * \ingroup ClassificationTree
     */
    namespace Classification
    {
        
        struct Tree
        {
            Tree() = default;

            int predict(std::vector<float> const& probkey) const
            {
                // remove batch dimension
                std::array<float,3> key = {probkey[0], probkey[1], probkey[3]};

                int node = 0;
                while (left[node] >= 0)
                    node = (key[feature_solution[node]] <= threshold[node]) ?
                        left[node] : right[node];
                
                return feature_solution[node];
            }
            
            std::string description() const
            {
                return "Tree";
            }

            std::vector<int> left, right, feature_solution;
            std::vector<float> threshold;
        };

     } // namespace Classification
} // namespace TensileLite
