using CSV
using DataFrames
using CairoMakie
using FileIO

csv_file = "benchmark_results.csv"

if !isfile(csv_file)
    println("Error: File '$csv_file' does not exist. Please make sure the file is available and try again.")
    return
end

df = CSV.read(csv_file, DataFrame, header=1, skipto=3)

longest_time = maximum(df.Time)
df.Speedup = longest_time ./ df.Time

x_ticks = df.Threads

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1],
          title="Strong Scaling Speedup",
          xlabel="Number of Threads",
          ylabel="Speedup",
          xticks=(x_ticks, string.(x_ticks)))

line_plot = lines!(ax, df.Threads, df.Speedup, color=:blue, label="Speedup Curve")
scatter_plot = scatter!(ax, df.Threads, df.Speedup, color=:red, markersize=10,
                        label="Data Points")

axislegend(ax, position=:rb)

save("benchmark_speedup.svg", fig)
