# ==========================================================================================
# Point and Line Interpolation Example
#
# This example demonstrates how to interpolate SPH particle data (e.g., density, pressure)
# at specific points or along a line using TrixiParticles.jl.
# A hydrostatic water column simulation is used as the base for generating particle data.
# ==========================================================================================

using TrixiParticles
# this needs to be commented out to use PythonPlot
using Plots

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"))

position_x = tank_size[1] / 2

# `interpolate_points` can be used to interpolate the properties of the `fluid_system`
# with the original kernel and `smoothing_length`.
println(interpolate_points([position_x; 0.01;;], semi, fluid_system, sol))
# Or with an increased `smoothing_length` smoothing the result
println(interpolate_points([position_x; 0.01;;], semi, fluid_system, sol,
                           smoothing_length=2.0 * smoothing_length))

# A point outside of the domain will result in properties with NaN values.
# On the boundary a result can still be obtained.
println(interpolate_points([position_x; 0.0;;], semi, fluid_system, sol))
# Slightly outside of the fluid domain the result is 0
println(interpolate_points([position_x; -0.01;;], semi, fluid_system, sol))

# Multiple points can be interpolated by providing an array
println(interpolate_points([position_x position_x position_x position_x position_x;
                            0.01 0.1 0.0 -0.01 -0.05],
                           semi, fluid_system, sol))

# It is also possible to interpolate along a line
n_interpolation_points = 10
start_point = [position_x, -fluid_particle_spacing]
end_point = [position_x, tank_size[2]]
result = interpolate_line(start_point, end_point, n_interpolation_points,
                          semi, fluid_system, sol)
result_endpoint = interpolate_line(start_point, end_point, n_interpolation_points,
                                   semi, fluid_system, sol, endpoint=false)

# Extracting wall distance for the standard and endpoint cases
walldistance = result.point_coords[2, :]
walldistance_endpoint = result_endpoint.point_coords[2, :]

# Instead of using Plots.jl one can also use PythonPlot which uses matplotlib
# using PythonPlot

# figure()
# plot(walldistance, result.density, marker="o", linestyle="-", label="With Endpoint")
# plot(walldistance_endpoint, result_endpoint.density, marker="x", linestyle="--",
#      label="Without Endpoint")

# xlabel("Wall distance")
# ylabel("Density")
# title("Density Interpolation Along a Line")
# legend()

# plotshow()

# Replace NaNs with zeros for visualization
replace!(result.density, NaN => 0.0)

p = Plots.plot(walldistance, result.density, marker=:circle, color=:blue,
               markerstrokecolor=:blue, linewidth=2, label="With Endpoint")

Plots.plot!(p, walldistance_endpoint, result_endpoint.density, marker=:xcross, linewidth=2,
            linestyle=:dash, label="Without Endpoint", color=:orange)

Plots.plot!(p, framestyle=:box, legend=:best, xlabel="Wall distance", ylabel="Density",
            title="Density Interpolation Along a Line", size=(800, 600), dpi=300)
