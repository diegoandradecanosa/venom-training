/*
 * Copyright (C) 2024 Diego Teijeiro Paredes (diego.teijeiro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//#include "../spmm/spmm_op.h"
//#include "../spmm/spmm_library_decl.h"
#include "controlled_wait_library.cu"

#include <torch/extension.h>

using namespace wait_kernel;



torch::Tensor controlled_wait(torch::Tensor input,
                        torch::Tensor output,
                        int size,
                        unsigned long long wait_time_in_microseconds)
{
    return controlled_wait_cuda(input, output, size, wait_time_in_microseconds);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("controlled_wait",  &controlled_wait,  "Controlled wait kernel");
}