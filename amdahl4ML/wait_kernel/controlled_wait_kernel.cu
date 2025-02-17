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
