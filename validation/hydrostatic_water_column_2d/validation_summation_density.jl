using TrixiParticles

# Explicit four-way summation-density matrix requested in PR #1286. Keep this
# density-consistency stress test distinct from the original WCSPH continuity model.
resolutions = (3, 5)
methods = (:wcsph, :edac)
tspan = (0.0, 0.02)
sample_dt = 0.0025
reinitialization_dt = 0.005
output_root = "out_hydrostatic_summation_corrections"
save_solution = false
correction_cases = (:none, :shepard, :mixed, :combined, :continuity_reinit)

trixi_include(@__MODULE__, joinpath(@__DIR__, "validation_corrections.jl");
              resolutions, methods, tspan, sample_dt, reinitialization_dt,
              output_root, save_solution, correction_cases,
              wcsph_density_mode=:summation,
              wcsph_pressure_clipping=true,
              wcsph_artificial_viscosity=ArtificialViscosityMonaghan(alpha=0.1, beta=0.0))
