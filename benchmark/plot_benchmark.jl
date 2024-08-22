using CSV
using DataFrames
using CairoMakie

# Read the benchmark results from the CSV file
df = CSV.read("benchmark_results.csv", DataFrame)

# Convert time to seconds for better readability
df.Time_sec = df.Time_ns ./ 1e9

# Create the plot
fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1], title = "Strong Scaling Performance", xlabel = "Number of Threads", ylabel = "Time (s)", xscale = :log10, yscale = :log10)

# Plot the data
lines!(ax, df.Threads, df.Time_sec, marker = :circle)

# Display the plot
fig
