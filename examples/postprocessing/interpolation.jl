# Example for using interpolation
#######################################################################################
using TrixiParticles

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"))

# interpolate_point can be used to interpolate the properties of the 'fluid_system' with the original kernel and smoothing_length
println(interpolate_point([1.0, 0.01], semi, fluid_system, sol))
# or with an increased smoothing_length smoothing the result
println(interpolate_point([1.0, 0.01], semi, fluid_system, sol,
                          smoothing_length=2.0 * smoothing_length))

# a point outside of the domain will result in properties with value 0
# on the boundary a result can still be obtained
println(interpolate_point([1.0, 0.0], semi, fluid_system, sol))
# slightly befind the result is 0
println(interpolate_point([1.0, -0.01], semi, fluid_system, sol))

# multiple points can be interpolated by providing an array
println(interpolate_point([
                              [1.0, 0.01],
                              [1.0, 0.1],
                              [1.0, 0.0],
                              [1.0, -0.01],
                              [1.0, -0.05],
                          ], semi, fluid_system, sol))

using PyPlot

# it is also possible to interpolate along a line
result = interpolate_line([1.0, -0.05], [1.0, 1.0], 10, semi, fluid_system, sol)
result_endpoint = interpolate_line([1.0, -0.05], [1.0, 1.0], 10, semi, fluid_system, sol,
                                   endpoint=false)

# Extracting wall distance for the standard and endpoint cases
walldistance = [coord[2] for coord in result.coord]
walldistance_endpoint = [coord[2] for coord in result_endpoint.coord]

figure()
plot(walldistance, result.density, marker="o", linestyle="-", label="With Endpoint")
plot(walldistance_endpoint, result_endpoint.density, marker="x", linestyle="--",
     label="Without Endpoint")

# Add labels and legend
xlabel("Wall distance")
ylabel("Density")
title("Density Interpolation Along a Line")
legend()

# Display the plot
show()
