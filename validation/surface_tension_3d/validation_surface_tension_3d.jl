using JSON
using Printf

if !isdefined(@__MODULE__, :SurfaceTensionValidation)
    include(joinpath(@__DIR__, "..", "surface_tension_common.jl"))
end
using .SurfaceTensionValidation

resolution_levels = (4, 6, 8, 10)
targets_3d = round.(Int, (4pi / 3) .* resolution_levels .^ 3)
write_results = true

young_laplace_3d = young_laplace_series(3; targets_3d)

println("3D Young-Laplace operator fit")
for result in young_laplace_3d.results
    @printf("  n=%5d dx=%.4e sigma=%.6f error=%6.3f%% virial=%.6f |F|=%.3e\n",
            result.particle_count, result.particle_spacing, result.sigma_fit,
            100result.relative_error, result.sigma_virial, result.total_force)
end
@printf("  observed order: %.3f\n", young_laplace_3d.observed_order)

if write_results
    output_path = joinpath(@__DIR__, "validation_reference.json")
    output = Dict(
                  "young_laplace" => Dict(
                      "observed_order" => young_laplace_3d.observed_order,
                      "results" => [Dict(string(key) => value
                                         for (key, value) in pairs(result))
                                    for result in young_laplace_3d.results]
                  )
                  )
    open(output_path, "w") do io
        JSON.print(io, output, 2)
    end
    println("Wrote $output_path")
end
