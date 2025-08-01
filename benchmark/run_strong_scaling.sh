#!/bin/bash

######################################################################################
# Benchmark Script
#
# This script automates the benchmarking process for TrixiParticles by running the 
# benchmark script (`benchmark/benchmark.jl`) with varying thread counts. It starts 
# from a minimum number of threads (default: 1) and doubles the number of threads 
# each iteration until the maximum number of threads (based on CPU or overridden) 
# is reached.
#
# Output:
# - The results are saved in a CSV file called "benchmark_results.csv", with columns:
#     - Threads: Number of threads used during the benchmark run.
#     - Time: The time taken to complete the benchmark (in seconds).
# - Units are saved in the second row, indicating the time in seconds.
#
# Environment Variables:
# - OVERRIDE_MAX_THREADS: Optionally, the maximum number of threads can be overridden
#   by setting this environment variable. If not set, the system's CPU thread count 
#   will be used.
# - OVERRIDE_MIN_THREADS: Optionally, the minimum number of threads can be overridden
#   by setting this environment variable. If not set, the default is 1.
#
# Example Usage:
# - Run with default thread settings:
#     ./benchmark_script.sh
# - Override maximum threads:
#     OVERRIDE_MAX_THREADS=8 ./benchmark_script.sh
# - Override both minimum and maximum threads:
#     OVERRIDE_MIN_THREADS=2 OVERRIDE_MAX_THREADS=8 ./benchmark_script.sh
######################################################################################

# Get the total number of CPU threads available on the system
DEFAULT_MAX_THREADS=$(julia -e 'println(Sys.CPU_THREADS)')
DEFAULT_MIN_THREADS=1

MAX_THREADS=${OVERRIDE_MAX_THREADS:-$DEFAULT_MAX_THREADS}
MIN_THREADS=${OVERRIDE_MIN_THREADS:-$DEFAULT_MIN_THREADS}

# Initialize the output file with headers and units
OUTPUT_FILE="benchmark_results.csv"
echo "Threads,Time" > $OUTPUT_FILE
echo "[1],[s]" >> $OUTPUT_FILE 

CURRENT_THREADS=$MIN_THREADS

while [ $CURRENT_THREADS -le $MAX_THREADS ]; do
    echo "Running benchmark with $CURRENT_THREADS threads..."

    # Run the benchmark using the specified number of threads and capture the timing result
    BENCHMARK_TIME=$(julia --project=. --threads=$CURRENT_THREADS benchmark/benchmark.jl)

    # Append the number of threads and timing result to the CSV file
    echo "$CURRENT_THREADS,$BENCHMARK_TIME" >> $OUTPUT_FILE

    CURRENT_THREADS=$((CURRENT_THREADS * 2))
done

echo "Benchmarking complete. Results saved to $OUTPUT_FILE."

julia plot_benchmark.jl

echo "Plotting complete. Plot saved as benchmark_speedup.svg."