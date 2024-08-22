#!/bin/bash

# Get the maximum number of threads available on the system
MAX_THREADS=$(julia -e 'println(Sys.CPU_THREADS)')

# Initialize the output file
OUTPUT_FILE="benchmark_results.csv"
echo "Threads,Time_ns" > $OUTPUT_FILE

# Starting with 1 thread, double the number of threads each iteration until MAX_THREADS is reached
THREADS=4

while [ $THREADS -le $MAX_THREADS ]; do
    echo "Running benchmark with $THREADS threads..."

    # Capture the output of the benchmark and extract the timing
    TIME_NS=$(julia --project=. --threads=$THREADS benchmark/benchmark.jl)

    # Save the results to the CSV file
    echo "$THREADS,$TIME_NS" >> $OUTPUT_FILE

    # Double the number of threads
    THREADS=$((THREADS * 2))
done

echo "Benchmarking complete. Results saved to $OUTPUT_FILE."
