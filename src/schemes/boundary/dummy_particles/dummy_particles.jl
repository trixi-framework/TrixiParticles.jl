@doc raw"""
    BoundaryModelDummyParticles(initial_density, hydrodynamic_mass,
                                density_calculator, smoothing_kernel,
                                smoothing_length; viscosity=nothing,
                                state_equation=nothing, correction=nothing)

`boundary_model` for `BoundarySPHSystem`.

# Arguments
- `initial_density`: Vector holding the initial density of each boundary particle.
- `hydrodynamic_mass`: Vector holding the "hydrodynamic mass" of each boundary particle.
                       See description above for more information.
- `density_calculator`: Strategy to compute the hydrodynamic density of the boundary particles.
                        See description below for more information.
- `smoothing_kernel`: Smoothing kernel should be the same as for the adjacent fluid system.
- `smoothing_length`: Smoothing length should be the same as for the adjacent fluid system.

# Keywords
- `state_equation`: This should be the same as for the adjacent fluid system
                    (see e.g. [`StateEquationCole`](@ref)).
- `correction`:     Correction method of the adjacent fluid system (see [Corrections](@ref corrections)).
- `viscosity`:      Slip (default) or no-slip condition. See description below for further
                    information.

# Examples
```jldoctest; output = false, setup = :(densities = [1.0, 2.0, 3.0]; masses = [0.1, 0.2, 0.3]; smoothing_kernel = SchoenbergCubicSplineKernel{2}(); smoothing_length = 0.1)
# Free-slip condition
boundary_model = BoundaryModelDummyParticles(densities, masses, AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length)

# No-slip condition
boundary_model = BoundaryModelDummyParticles(densities, masses, AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length,
                                             viscosity=ViscosityAdami(nu=1e-6))

# output
BoundaryModelDummyParticles(AdamiPressureExtrapolation, ViscosityAdami)
```
"""
struct BoundaryModelDummyParticles{DC, ELTYPE <: Real, SE, K, V, COR, C}
    pressure           :: Vector{ELTYPE}
    hydrodynamic_mass  :: Vector{ELTYPE}
    state_equation     :: SE
    density_calculator :: DC
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    viscosity          :: V
    correction         :: COR
    cache              :: C

    function BoundaryModelDummyParticles(initial_density, hydrodynamic_mass,
                                         density_calculator, smoothing_kernel,
                                         smoothing_length; viscosity=nothing,
                                         state_equation=nothing, correction=nothing)
        pressure = initial_boundary_pressure(initial_density, density_calculator,
                                             state_equation)
        NDIMS = ndims(smoothing_kernel)

        n_particles = length(initial_density)

        cache = (; create_cache_model(viscosity, n_particles, NDIMS)...,
                 create_cache_model(initial_density, density_calculator)...)
        cache = (;
                 create_cache_model(correction, initial_density, NDIMS,
                                    n_particles)..., cache...)

        new{typeof(density_calculator), eltype(initial_density),
            typeof(state_equation), typeof(smoothing_kernel), typeof(viscosity),
            typeof(correction), typeof(cache)}(pressure, hydrodynamic_mass, state_equation,
                                               density_calculator,
                                               smoothing_kernel, smoothing_length,
                                               viscosity, correction, cache)
    end
end

@doc raw"""
    AdamiPressureExtrapolation()

`density_calculator` for `BoundaryModelDummyParticles`.
"""
struct AdamiPressureExtrapolation end

@doc raw"""
    PressureMirroring()

`density_calculator` for `BoundaryModelDummyParticles`.

!!! note
    This boundary model requires high viscosity for stability with WCSPH.
    It also produces significantly worse results than [`AdamiPressureExtrapolation`](@ref)
    and is not more efficient because smaller time steps are required due to more noise
    in the pressure.
    We added this model only for research purposes and for comparison with
    [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH).
"""
struct PressureMirroring end

@doc raw"""
    PressureZeroing()

`density_calculator` for `BoundaryModelDummyParticles`.

!!! note
    This boundary model produces significantly worse results than all other models and
    is only included for research purposes.
"""
struct PressureZeroing end

create_cache_model(correction, density, NDIMS, nparticles) = (;)

function create_cache_model(::ShepardKernelCorrection, density, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density))
end

function create_cache_model(::KernelCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{Float64}(undef, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density), dw_gamma)
end

function create_cache_model(::Union{GradientCorrection, BlendedGradientCorrection}, density,
                            NDIMS, n_particles)
    correction_matrix = Array{Float64, 3}(undef, NDIMS, NDIMS, n_particles)
    return (; correction_matrix)
end

function create_cache_model(::MixedKernelGradientCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{Float64}(undef, NDIMS, n_particles)
    correction_matrix = Array{Float64, 3}(undef, NDIMS, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density), dw_gamma, correction_matrix)
end

function create_cache_model(initial_density,
                            ::Union{SummationDensity, PressureMirroring, PressureZeroing})
    density = copy(initial_density)

    return (; density)
end

function create_cache_model(initial_density, ::ContinuityDensity)
    return (; initial_density)
end

function create_cache_model(initial_density, ::AdamiPressureExtrapolation)
    density = copy(initial_density)
    volume = similar(initial_density)

    return (; density, volume)
end

function create_cache_model(viscosity::Nothing, n_particles, n_dims)
    return (;)
end

function create_cache_model(viscosity::ViscosityAdami, n_particles, n_dims)
    ELTYPE = eltype(viscosity.nu)

    wall_velocity = zeros(ELTYPE, n_dims, n_particles)

    return (; wall_velocity)
end

@inline reset_cache!(cache, viscosity) = set_zero!(cache.volume)

function reset_cache!(cache, viscosity::ViscosityAdami)
    (; volume, wall_velocity) = cache

    set_zero!(volume)
    set_zero!(wall_velocity)

    return cache
end

function Base.show(io::IO, model::BoundaryModelDummyParticles)
    @nospecialize model # reduce precompilation time

    print(io, "BoundaryModelDummyParticles(")
    print(io, model.density_calculator |> typeof |> nameof)
    print(io, ", ")
    print(io, model.viscosity |> typeof |> nameof)
    print(io, ")")
end

# For most density calculators, the pressure is updated in every step
initial_boundary_pressure(initial_density, density_calculator, _) = similar(initial_density)
# Pressure mirroring does not use the pressure, so we set it to zero for the visualization
initial_boundary_pressure(initial_density, ::PressureMirroring, _) = zero(initial_density)

# For pressure zeroing, set the pressure to the reference pressure (zero with free surfaces)
function initial_boundary_pressure(initial_density, ::PressureZeroing, state_equation)
    return state_equation.(initial_density)
end

# With EDAC, just use zero pressure
function initial_boundary_pressure(initial_density, ::PressureZeroing, ::Nothing)
    return zero(initial_density)
end

@inline function particle_density(v, model::BoundaryModelDummyParticles, system, particle)
    return particle_density(v, model.density_calculator, model, particle)
end

# Note that the other density calculators are dispatched in `density_calculators.jl`
@inline function particle_density(v,
                                  ::Union{AdamiPressureExtrapolation, PressureMirroring,
                                          PressureZeroing},
                                  boundary_model, particle)
    (; cache) = boundary_model

    return cache.density[particle]
end

@inline function particle_pressure(v, model::BoundaryModelDummyParticles, system, particle)
    return model.pressure[particle]
end

@inline function update_density!(boundary_model::BoundaryModelDummyParticles,
                                 system, v, u, v_ode, u_ode, semi)
    (; density_calculator) = boundary_model

    compute_density!(boundary_model, density_calculator, system, v, u, v_ode, u_ode, semi)

    return boundary_model
end

function compute_density!(boundary_model,
                          ::Union{ContinuityDensity, AdamiPressureExtrapolation,
                                  PressureMirroring, PressureZeroing},
                          system, v, u, v_ode, u_ode, semi)
    # No density update for `ContinuityDensity`, `PressureMirroring` and `PressureZeroing`.
    # For `AdamiPressureExtrapolation`, the density is updated in `compute_pressure!`.
    return boundary_model
end

@inline function update_pressure!(boundary_model::BoundaryModelDummyParticles,
                                  system, v, u, v_ode, u_ode, semi)
    (; correction, density_calculator) = boundary_model

    compute_correction_values!(system, correction, u, v_ode, u_ode, semi)

    compute_gradient_correction_matrix!(correction, boundary_model, system, u, v_ode, u_ode,
                                        semi)

    # `kernel_correct_density!` only performed for `SummationDensity`
    kernel_correct_density!(boundary_model, v, u, v_ode, u_ode, semi, correction,
                            density_calculator)

    compute_pressure!(boundary_model, density_calculator, system, v, u, v_ode, u_ode, semi)

    return boundary_model
end

function kernel_correct_density!(boundary_model, v, u, v_ode, u_ode, semi,
                                 correction, density_calculator)
    return boundary_model
end

function kernel_correct_density!(boundary_model, v, u, v_ode, u_ode, semi,
                                 corr::ShepardKernelCorrection, ::SummationDensity)
    boundary_model.cache.density ./= boundary_model.cache.kernel_correction_coefficient
end

function compute_gradient_correction_matrix!(correction, boundary_model, system, u,
                                             v_ode, u_ode, semi)
    return system
end

function compute_gradient_correction_matrix!(corr::Union{GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection},
                                             boundary_model,
                                             system, u, v_ode, u_ode, semi)
    (; cache, correction, smoothing_kernel, smoothing_length) = boundary_model
    (; correction_matrix) = cache

    system_coords = current_coordinates(u, system)

    compute_gradient_correction_matrix!(correction_matrix, system, system_coords,
                                        v_ode, u_ode, semi, correction, smoothing_length,
                                        smoothing_kernel)
end

function compute_density!(boundary_model, ::SummationDensity, system, v, u, v_ode, u_ode,
                          semi)
    (; cache) = boundary_model
    (; density) = cache # Density is in the cache for SummationDensity

    summation_density!(system, semi, u, u_ode, density, particles=eachparticle(system))
end

function compute_pressure!(boundary_model, ::Union{SummationDensity, ContinuityDensity},
                           system, v, u, v_ode, u_ode, semi)

    # Limit pressure to be non-negative to avoid attractive forces between fluid and
    # boundary particles at free surfaces (sticking artifacts).
    @threaded for particle in eachparticle(system)
        apply_state_equation!(boundary_model, particle_density(v, boundary_model,
                                                               particle), particle)
    end

    return boundary_model
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function apply_state_equation!(boundary_model, density, particle)
    boundary_model.pressure[particle] = max(boundary_model.state_equation(density), 0.0)
end

function compute_pressure!(boundary_model, ::AdamiPressureExtrapolation,
                           system, v, u, v_ode, u_ode, semi)
    (; pressure, state_equation, cache, viscosity) = boundary_model
    (; volume, density) = cache

    set_zero!(pressure)

    # Set `volume` to zero. For `ViscosityAdami` the `wall_velocity` is also set to zero.
    reset_cache!(cache, viscosity)

    system_coords = current_coordinates(u, system)

    # Use all other systems for the pressure extrapolation
    @trixi_timeit timer() "compute boundary pressure" foreach_system(semi) do neighbor_system
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

        nhs = get_neighborhood_search(system, neighbor_system, semi)

        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        adami_pressure_extrapolation!(boundary_model, system, neighbor_system,
                                      system_coords, neighbor_coords,
                                      v_neighbor_system, nhs)
    end

    @trixi_timeit timer() "inverse state equation" @threaded for particle in eachparticle(system)
        compute_adami_density!(boundary_model, system, system_coords, particle)
    end
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
function compute_adami_density!(boundary_model, system, system_coords, particle)
    (; pressure, state_equation, cache, viscosity) = boundary_model
    (; volume, density) = cache

    # The summation is only over fluid particles, thus the volume stays zero when a boundary
    # particle isn't surrounded by fluid particles.
    # Check the volume to avoid NaNs in pressure and velocity.
    if volume[particle] > eps()
        pressure[particle] /= volume[particle]

        # To impose no-slip condition
        compute_wall_velocity!(viscosity, system, system_coords, particle)
    end

    # Apply inverse state equation to compute density (not used with EDAC)
    inverse_state_equation!(density, state_equation, pressure, particle)
end

function compute_pressure!(boundary_model, ::Union{PressureMirroring, PressureZeroing},
                           system, v, u, v_ode, u_ode, semi)
    # No pressure update needed with `PressureMirroring` and `PressureZeroing`.
    return boundary_model
end

@inline function adami_pressure_extrapolation!(boundary_model, system,
                                               neighbor_system::FluidSystem,
                                               system_coords, neighbor_coords,
                                               v_neighbor_system, neighborhood_search)
    (; pressure, cache, viscosity) = boundary_model

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search;
                          particles=eachparticle(system)) do particle, neighbor,
                                                             pos_diff, distance
        density_neighbor = particle_density(v_neighbor_system, neighbor_system, neighbor)

        resulting_acc = neighbor_system.acceleration -
                        current_acceleration(system, particle)

        kernel_weight = smoothing_kernel(boundary_model, distance)

        pressure[particle] += (particle_pressure(v_neighbor_system, neighbor_system,
                                                 neighbor) +
                               dot(resulting_acc, density_neighbor * pos_diff)) *
                              kernel_weight

        cache.volume[particle] += kernel_weight

        compute_smoothed_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                                   kernel_weight, particle, neighbor)
    end

    for particle in eachparticle(system)
        # Limit pressure to be non-negative to avoid attractive forces between fluid and
        # boundary particles at free surfaces (sticking artifacts).
        pressure[particle] = max(pressure[particle], 0.0)
    end
end

@inline function adami_pressure_extrapolation!(boundary_model, system, neighbor_system,
                                               system_coords, neighbor_coords,
                                               v_neighbor_system, neighborhood_search)
    return boundary_model
end

function compute_smoothed_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                                    kernel_weight, particle, neighbor)
    return cache
end

function compute_smoothed_velocity!(cache, viscosity::ViscosityAdami,
                                    neighbor_system, v_neighbor_system, kernel_weight,
                                    particle, neighbor)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    for dim in 1:ndims(neighbor_system)
        cache.wall_velocity[dim, particle] += kernel_weight * v_b[dim]
    end

    return cache
end

@inline function compute_wall_velocity!(viscosity, system, system_coords, particle)
    return viscosity
end

@inline function compute_wall_velocity!(viscosity::ViscosityAdami, system,
                                        system_coords, particle)
    (; boundary_model) = system
    (; cache) = boundary_model
    (; volume, wall_velocity) = cache

    # Prescribed velocity of the boundary particle.
    # This velocity is zero when not using moving boundaries.
    v_boundary = current_velocity(system_coords, system, particle)

    for dim in 1:ndims(system)
        # The second term is the precalculated smoothed velocity field of the fluid.
        wall_velocity[dim, particle] = 2 * v_boundary[dim] -
                                       wall_velocity[dim, particle] / volume[particle]
    end
    return viscosity
end

@inline function inverse_state_equation!(density, state_equation, pressure, particle)
    density[particle] = inverse_state_equation(state_equation, pressure[particle])
    return density
end

@inline function inverse_state_equation!(density, state_equation::Nothing, pressure,
                                         particle)
    # The density is constant when using EDAC
    return density
end

@inline function smoothing_kernel_grad(system::BoundarySystem, pos_diff,
                                       distance, particle)
    (; smoothing_kernel, smoothing_length, correction) = system.boundary_model

    return corrected_kernel_grad(smoothing_kernel, pos_diff, distance,
                                 smoothing_length, correction, system, particle)
end

@inline function correction_matrix(system::BoundarySystem, particle)
    extract_smatrix(system.boundary_model.cache.correction_matrix, system, particle)
end
