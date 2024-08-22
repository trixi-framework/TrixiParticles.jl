using CSV
using DataFrames
using CairoMakie

df = CSV.read("benchmark_results.csv", DataFrame, header=1, skipto=3)

longest_time = maximum(df.Time)
df.Speedup = longest_time ./ df.Time

x_ticks = df.Threads

fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1],
          title = "Strong Scaling Speedup",
          xlabel = "Number of Threads",
          ylabel = "Speedup",
          xticks = (x_ticks, string.(x_ticks)))  # Set ticks as powers of 2

lines!(ax, df.Threads, df.Speedup, color=:blue)
scatter!(ax, df.Threads, df.Speedup, color=:red, markersize=10)

fig
