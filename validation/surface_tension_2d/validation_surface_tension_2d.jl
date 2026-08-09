using JSON
using Printf

if !isdefined(@__MODULE__, :SurfaceTensionValidation)
    include(joinpath(@__DIR__, "..", "surface_tension_common.jl"))
end
using .SurfaceTensionValidation

targets_3d = (375, 750, 1500, 3000)
rayleigh_targets = (200, 400, 800)
write_results = true

young_laplace_2d = young_laplace_series(2; targets_3d)
rayleigh_results = [rayleigh_mode2_stiffness(target; stretch=1.04)
                    for target in rayleigh_targets]

println("2D Young-Laplace operator fit")
for result in young_laplace_2d.results
    @printf("  n=%4d dx=%.4e sigma=%.6f error=%6.3f%% virial=%.6f |F|=%.3e\n",
            result.particle_count, result.particle_spacing, result.sigma_fit,
            100result.relative_error, result.sigma_virial, result.total_force)
end
@printf("  observed order: %.3f\n", young_laplace_2d.observed_order)

println("2D Rayleigh mode-2 stiffness")
for result in rayleigh_results
    @printf("  n=%4d omega=%.5f exact=%.5f error=%6.3f%%\n",
            result.particle_count, result.omega_measured, result.omega_exact,
            100result.frequency_error)
end

if write_results
    output_path = joinpath(@__DIR__, "validation_reference.json")
    output = Dict(
        "young_laplace" => Dict(
            "observed_order" => young_laplace_2d.observed_order,
            "results" => [Dict(string(key) => value for (key, value) in pairs(result))
                          for result in young_laplace_2d.results]
        ),
        "rayleigh_mode_2" => [Dict(string(key) => value
                                   for (key, value) in pairs(result))
                              for result in rayleigh_results]
    )
    open(output_path, "w") do io
        JSON.print(io, output, 2)
    end
    println("Wrote $output_path")
end
