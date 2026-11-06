using TrixiParticles

if !isdefined(@__MODULE__, :HydrostaticCorrectionMethods)
    include("correction_methods.jl")
end

# Paper-length physical benchmark: the full 2×5 correction matrix in each scheme,
# plus continuity-density and declaration-order controls on four h-halving levels.
resolutions = (3, 5, 9, 17)
methods = (:wcsph, :edac)
wcsph_density_mode = :continuity
edac_density_mode = :summation
correction_cases = HydrostaticCorrectionMethods.DEFAULT_CASES
tspan = (0.0, 1.0)
averaging_interval = (0.5, 1.0)
sample_dt = 0.005
reinitialization_dt = 0.01
output_root = "out_hydrostatic_long"
save_solution = false

trixi_include(@__MODULE__, joinpath(@__DIR__, "validation_corrections.jl");
              resolutions, methods, wcsph_density_mode, edac_density_mode,
              correction_cases, tspan, averaging_interval, sample_dt,
              reinitialization_dt, output_root, save_solution)
