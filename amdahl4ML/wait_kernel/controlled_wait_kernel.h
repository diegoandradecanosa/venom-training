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






__device__ __forceinline__
extern "C" __global__ void wait_kernel(float* input, float* output, int size, unsigned long long wait_time_in_microseconds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned long long start_time = clock64();

        float value = input[idx];
        output[idx] = value; // Simulate a simple operation
        
        // Calculate stop time based on clock rate and wait_time_in_microseconds
        //unsigned long long wait_time_in_cycles = wait_time_in_microseconds * (clockRate / 1000000ULL);
        unsigned long long wait_time_in_cycles = wait_time_in_microseconds * 2115ULL; // clock rate fixed to 2115 Mhz

        // Perform a controlled wait
        while ((clock64() - start_time) < wait_time_in_cycles);
        
    }
}
