using TrixiParticles

include(joinpath(@__DIR__, "boundary_volume.jl"))
include(joinpath(@__DIR__, "simulate.jl"))
isdefined(@__MODULE__, :FreeSurfaceDensityDiffusionAntuono) ||
    include(joinpath(@__DIR__, "wcsph_variants.jl"))

function delta_sph_config(case_name, final_time; delta=0.1,
                          free_surface_diffusion=false,
                          akinci_support_factor=nothing,
                          curvature_factor=1.0,
                          target_particle_count=750)
    base = case_config(case_name)
    target_particle_count > 0 ||
        throw(ArgumentError("`target_particle_count` must be positive"))
    particle_spacing = cbrt(1.0e-6 / target_particle_count)
    plate_size = base.kwargs.plate_size
    n_plate = round.(Int, plate_size ./ particle_spacing)
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    plate = RectangularShape(particle_spacing, (n_plate..., 3),
                             (-plate_size[1] / 2, -plate_size[2] / 2,
                              -3particle_spacing);
                             density=1000.0)
    boundary_hydrodynamic_mass = akinci_boundary_hydrodynamic_mass(plate,
                                                                   smoothing_kernel,
                                                                   smoothing_length,
                                                                   1000.0)
    save_times = collect(range(0.0, final_time; step=0.1))
    isapprox(last(save_times), final_time) || push!(save_times, final_time)
    density_diffusion = if free_surface_diffusion
        FreeSurfaceDensityDiffusionAntuono(; delta, reference_density=1000.0)
    else
        DensityDiffusionAntuono(; delta)
    end
    surface_options = (;)
    if !isnothing(akinci_support_factor) || curvature_factor != 1
        support_factor = something(akinci_support_factor, 2.8)
        support_radius = support_factor * particle_spacing
        surface_tension = SurfaceTensionAkinciWCSPH(;
                                                    surface_tension_coefficient=base.paper.surface_tension_coefficient,
                                                    support_radius, curvature_factor)
        normal_kernel = isnothing(akinci_support_factor) ? smoothing_kernel :
                        SchoenbergCubicSplineKernel{3}()
        normal_smoothing_length = isnothing(akinci_support_factor) ? smoothing_length :
                                  support_radius / 2
        surface_normal_method = WCSPHAkinciSurfaceNormal(normal_kernel,
                                                         normal_smoothing_length)
        surface_options = (; surface_tension, surface_normal_method)
    end
    kwargs = merge(base.kwargs,
                   (; smoothing_kernel, smoothing_length, sound_speed=100.0,
                    target_particle_count,
                    fluid_density_calculator=ContinuityDensity(),
                    fluid_density_diffusion=density_diffusion,
                    boundary_hydrodynamic_mass, tspan=(0.0, final_time),
                    solution_saveat=Tuple(save_times)), surface_options)
    name = base.name * "_delta_sph_n" * string(target_particle_count)
    return merge(base, (; name, kwargs))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (3, 7) ||
        error("usage: simulate_delta_sph_wetting.jl CASE OUTPUT.jls FINAL_TIME " *
              "[DELTA FREE_SURFACE SUPPORT_FACTOR CURVATURE_FACTOR]")
    options = if length(ARGS) == 3
        (;)
    else
        support_factor = ARGS[6] == "nothing" ? nothing : parse(Float64, ARGS[6])
        (; delta=parse(Float64, ARGS[4]),
         free_surface_diffusion=parse(Bool, ARGS[5]),
         akinci_support_factor=support_factor,
         curvature_factor=parse(Float64, ARGS[7]))
    end
    config = delta_sph_config(ARGS[1], parse(Float64, ARGS[3]); options...)
    write_snapshot(config, ARGS[2])
end
