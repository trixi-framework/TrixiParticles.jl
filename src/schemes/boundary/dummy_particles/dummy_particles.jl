@doc raw"""
    BoundaryModelDummyParticles(initial_density, hydrodynamic_mass,
                                density_calculator, smoothing_kernel,
                                smoothing_length; viscosity=nothing,
                                state_equation=nothing, correction=nothing,
                                reference_particle_spacing=0.0)

Boundary model for `BoundarySPHSystem`.

# Arguments
- `initial_density`: Vector holding the initial density of each boundary particle.
- `hydrodynamic_mass`: Vector holding the "hydrodynamic mass" of each boundary particle.
                       See description above for more information.
- `density_calculator`: Strategy to compute the hydrodynamic density of the boundary particles.
                        See description below for more information.
- `smoothing_kernel`: Smoothing kernel should be the same as for the adjacent fluid system.
- `smoothing_length`: Smoothing length should be the same as for the adjacent fluid system.

# Keywords
- `state_equation`:             This should be the same as for the adjacent fluid system
                                (see e.g. [`StateEquationCole`](@ref)).
- `correction`:                 Correction method of the adjacent fluid system (see [Corrections](@ref corrections)).
- `viscosity`:                  Slip (default) or no-slip condition. See description below for further
                                information.
- `reference_particle_spacing`: The reference particle spacing used for weighting values at the boundary,
                                which currently is only needed when using surface tension.
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
struct BoundaryModelDummyParticles{DC, ELTYPE <: Real, VECTOR, SE, K, V, COR, C}
    pressure           :: VECTOR # Vector{ELTYPE}
    hydrodynamic_mass  :: VECTOR # Vector{ELTYPE}
    state_equation     :: SE
    density_calculator :: DC
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    viscosity          :: V
    correction         :: COR
    cache              :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function BoundaryModelDummyParticles(initial_density, hydrodynamic_mass,
                                     density_calculator, smoothing_kernel,
                                     smoothing_length; viscosity=nothing,
                                     state_equation=nothing, correction=nothing,
                                     reference_particle_spacing=0.0)
    pressure = initial_boundary_pressure(initial_density, density_calculator,
                                         state_equation)
    NDIMS = ndims(smoothing_kernel)
    ELTYPE = eltype(smoothing_length)
    n_particles = length(initial_density)

    cache = (; create_cache_model(viscosity, n_particles, NDIMS)...,
             create_cache_model(initial_density, density_calculator)...,
             create_cache_model(correction, initial_density, NDIMS, n_particles)...)

    # If the `reference_density_spacing` is set calculate the `ideal_neighbor_count`
    if reference_particle_spacing > 0
        # `reference_particle_spacing` has to be set for surface normals to be determined
        cache = (;
                 cache...,  # Existing cache fields
                 reference_particle_spacing=reference_particle_spacing,
                 initial_colorfield=zeros(ELTYPE, n_particles),
                 colorfield=zeros(ELTYPE, n_particles),
                 neighbor_count=zeros(ELTYPE, n_particles))
    end

    return BoundaryModelDummyParticles(pressure, hydrodynamic_mass, state_equation,
                                       density_calculator, smoothing_kernel,
                                       smoothing_length, viscosity, correction, cache)
end

@doc raw"""
    AdamiPressureExtrapolation(; pressure_offset=0, allow_loop_flipping=true)

`density_calculator` for `BoundaryModelDummyParticles`.

# Keywords
- `pressure_offset=0`: Sometimes it is necessary to artificially increase the boundary pressure
                       to prevent penetration, which is possible by increasing this value.
- `allow_loop_flipping=true`: Allow to flip the loop order for the pressure extrapolation.
                              Disable to prevent error variations between simulations with
                              different numbers of threads.
                              Usually, the first (multithreaded) loop is over the boundary
                              particles and the second loop over the fluid neighbors.
                              When the number of boundary particles is larger than
                              `ceil(0.5 * nthreads())` times the number of fluid particles,
                              it is usually more efficient to flip the loop order and loop
                              over the fluid particles first.
                              The factor depends on the number of threads, as the flipped
                              loop is not thread parallelizable.
                              This can cause error variations between simulations with
                              different numbers of threads.

"""
struct AdamiPressureExtrapolation{ELTYPE}
    pressure_offset     :: ELTYPE
    allow_loop_flipping :: Bool

    function AdamiPressureExtrapolation(; pressure_offset=0, allow_loop_flipping=true)
        return new{eltype(pressure_offset)}(pressure_offset, allow_loop_flipping)
    end
end

@doc raw"""
    BernoulliPressureExtrapolation(; pressure_offset=0, factor=1)

`density_calculator` for `BoundaryModelDummyParticles`.

# Keywords
- `pressure_offset=0`:   Sometimes it is necessary to artificially increase the boundary pressure
                         to prevent penetration, which is possible by increasing this value.
- `factor=1`         :   Setting `factor` allows to just increase the strength of the dynamic
                         pressure part.
- `allow_loop_flipping=true`: Allow to flip the loop order for the pressure extrapolation.
                              Disable to prevent error variations between simulations with
                              different numbers of threads.
                              Usually, the first (multithreaded) loop is over the boundary
                              particles and the second loop over the fluid neighbors.
                              When the number of boundary particles is larger than
                              `ceil(0.5 * nthreads())` times the number of fluid particles,
                              it is usually more efficient to flip the loop order and loop
                              over the fluid particles first.
                              The factor depends on the number of threads, as the flipped
                              loop is not thread parallelizable.
                              This can cause error variations between simulations with
                              different numbers of threads.

"""
struct BernoulliPressureExtrapolation{ELTYPE}
    pressure_offset     :: ELTYPE
    factor              :: ELTYPE
    allow_loop_flipping :: Bool

    function BernoulliPressureExtrapolation(; pressure_offset=0, factor=1,
                                            allow_loop_flipping=true)
        return new{eltype(pressure_offset)}(pressure_offset, factor, allow_loop_flipping)
    end
end

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

@inline create_cache_model(correction, density, NDIMS, nparticles) = (;)

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

@inline create_cache_model(initial_density, ::ContinuityDensity) = (; initial_density)

function create_cache_model(initial_density,
                            ::Union{AdamiPressureExtrapolation,
                                    BernoulliPressureExtrapolation})
    density = copy(initial_density)
    volume = similar(initial_density)

    return (; density, volume)
end

@inline create_cache_model(viscosity::Nothing, n_particles, n_dims) = (;)

function create_cache_model(viscosity, n_particles, n_dims)
    ELTYPE = eltype(viscosity.epsilon)

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

@inline function current_density(v, model::BoundaryModelDummyParticles, system)
    return current_density(v, model.density_calculator, model)
end

@inline function current_density(v,
                                 ::Union{SummationDensity, AdamiPressureExtrapolation,
                                         PressureMirroring, PressureZeroing,
                                         BernoulliPressureExtrapolation},
                                 model::BoundaryModelDummyParticles)
    # When using `SummationDensity`, the density is stored in the cache
    return model.cache.density
end

@inline function current_density(v, ::ContinuityDensity,
                                 model::BoundaryModelDummyParticles)
    # When using `ContinuityDensity`, the density is stored in the last row of `v`
    return view(v, size(v, 1), :)
end

@inline function current_pressure(v, model::BoundaryModelDummyParticles, system)
    return model.pressure
end

@inline function update_density!(boundary_model::BoundaryModelDummyParticles,
                                 system, v, u, v_ode, u_ode, semi)
    (; density_calculator) = boundary_model

    compute_density!(boundary_model, density_calculator, system, v, u, v_ode, u_ode, semi)

    return boundary_model
end

function compute_density!(boundary_model,
                          ::Union{ContinuityDensity, AdamiPressureExtrapolation,
                                  BernoulliPressureExtrapolation,
                                  PressureMirroring, PressureZeroing},
                          system, v, u, v_ode, u_ode, semi)
    # No density update for `ContinuityDensity`, `PressureMirroring` and `PressureZeroing`.
    # For `AdamiPressureExtrapolation` and `BernoulliPressureExtrapolation`, the density is updated in `compute_pressure!`.
    # Only SummationDensity performs a density update.
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
    (; cache, correction, smoothing_kernel) = boundary_model
    (; correction_matrix) = cache

    system_coords = current_coordinates(u, system)

    compute_gradient_correction_matrix!(correction_matrix, system, system_coords,
                                        v_ode, u_ode, semi, correction, smoothing_kernel)
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
    @threaded semi for particle in eachparticle(system)
        apply_state_equation!(boundary_model, current_density(v, system, particle),
                              particle)
    end

    return boundary_model
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function apply_state_equation!(boundary_model, density, particle)
    boundary_model.pressure[particle] = max(boundary_model.state_equation(density), 0)
end

function compute_pressure!(boundary_model,
                           ::Union{AdamiPressureExtrapolation,
                                   BernoulliPressureExtrapolation},
                           system, v, u, v_ode, u_ode, semi)
    (; pressure, cache, viscosity) = boundary_model
    (; allow_loop_flipping) = boundary_model.density_calculator

    set_zero!(pressure)

    # Set `volume` to zero. For `ViscosityAdami` the `wall_velocity` is also set to zero.
    reset_cache!(cache, viscosity)

    system_coords = current_coordinates(u, system)

    # Use all other systems for the pressure extrapolation
    @trixi_timeit timer() "compute boundary pressure" foreach_system(semi) do neighbor_system
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        # This is an optimization for simulations with large and complex boundaries.
        # Especially in 3D simulations with large and/or complex structures outside
        # of areas with permanent flow.
        # Note: The version iterating neighbors first is not thread-parallelizable
        #       and thus not GPU-compatible.
        # The factor is based on the achievable speed-up of the thread parallelizable version.
        # Use the parallel version if the number of boundary particles is not much larger
        # than the number of fluid particles.
        n_boundary_particles = nparticles(system)
        n_fluid_particles = nparticles(neighbor_system)
        speedup = ceil(Int, Threads.nthreads() / 2)
        is_gpu = system_coords isa AbstractGPUArray
        condition_boundary = n_boundary_particles < speedup * n_fluid_particles
        parallelize = is_gpu || condition_boundary || !allow_loop_flipping

        # Loop over boundary particles and then the neighboring fluid particles
        # to extrapolate fluid pressure to the boundaries.
        boundary_pressure_extrapolation!(Val(parallelize), boundary_model, system,
                                         neighbor_system, system_coords, neighbor_coords, v,
                                         v_neighbor_system, semi)
    end

    @trixi_timeit timer() "inverse state equation" @threaded semi for particle in
                                                                      eachparticle(system)
        compute_adami_density!(boundary_model, system, v, particle)
    end
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
function compute_adami_density!(boundary_model, system, v, particle)
    (; pressure, state_equation, cache, viscosity) = boundary_model
    (; volume, density) = cache

    # The summation is only over fluid particles, thus the volume stays zero when a boundary
    # particle isn't surrounded by fluid particles.
    # Check the volume to avoid NaNs in pressure and velocity.
    if @inbounds volume[particle] > eps()
        @inbounds pressure[particle] /= volume[particle]

        # To impose no-slip condition
        compute_wall_velocity!(viscosity, system, v, particle)
    end

    # Limit pressure to be non-negative to avoid attractive forces between fluid and
    # boundary particles at free surfaces (sticking artifacts).
    @inbounds pressure[particle] = max(pressure[particle], 0)

    # Apply inverse state equation to compute density (not used with EDAC)
    inverse_state_equation!(density, state_equation, pressure, particle)
end

function compute_pressure!(boundary_model, ::Union{PressureMirroring, PressureZeroing},
                           system, v, u, v_ode, u_ode, semi)
    # No pressure update needed with `PressureMirroring` and `PressureZeroing`.
    return boundary_model
end

@inline function boundary_pressure_extrapolation!(parallel, boundary_model, system,
                                                  neighbor_system, system_coords,
                                                  neighbor_coords, v, v_neighbor_system,
                                                  semi)
    return boundary_model
end

@inline function boundary_pressure_extrapolation!(parallel::Val{true}, boundary_model,
                                                  system, neighbor_system::FluidSystem,
                                                  system_coords, neighbor_coords, v,
                                                  v_neighbor_system, semi)
    (; pressure, cache, viscosity, density_calculator) = boundary_model
    (; pressure_offset) = density_calculator

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=eachparticle(system)) do particle, neighbor,
                                                           pos_diff, distance
        boundary_pressure_inner!(boundary_model, density_calculator, system,
                                 neighbor_system, v, v_neighbor_system, particle, neighbor,
                                 pos_diff, distance, viscosity, cache, pressure,
                                 pressure_offset)
    end
end

# Loop over fluid particles and then the neighboring boundary particles
# to extrapolate fluid pressure to the boundaries.
# Note that this needs to be serial, as we are writing into the same
# pressure entry from different loop iterations.
@inline function boundary_pressure_extrapolation!(parallel::Val{false}, boundary_model,
                                                  system, neighbor_system::FluidSystem,
                                                  system_coords, neighbor_coords,
                                                  v, v_neighbor_system, semi)
    (; pressure, cache, viscosity, density_calculator) = boundary_model
    (; pressure_offset) = density_calculator

    # This needs to be serial to avoid race conditions when writing into `system`
    foreach_point_neighbor(neighbor_system, system, neighbor_coords, system_coords, semi;
                           points=each_moving_particle(neighbor_system),
                           parallelization_backend=SerialBackend()) do neighbor, particle,
                                                                       pos_diff, distance
        # Since neighbor and particle are switched
        pos_diff = -pos_diff
        boundary_pressure_inner!(boundary_model, density_calculator, system,
                                 neighbor_system, v, v_neighbor_system, particle, neighbor,
                                 pos_diff, distance, viscosity, cache, pressure,
                                 pressure_offset)
    end
end

@inline function boundary_pressure_inner!(boundary_model, boundary_density_calculator,
                                          system, neighbor_system::FluidSystem, v,
                                          v_neighbor_system, particle, neighbor, pos_diff,
                                          distance, viscosity, cache, pressure,
                                          pressure_offset)
    density_neighbor = @inbounds current_density(v_neighbor_system, neighbor_system,
                                                 neighbor)

    # Fluid pressure term
    fluid_pressure = @inbounds current_pressure(v_neighbor_system, neighbor_system,
                                                neighbor)

    # Hydrostatic pressure term from fluid and boundary acceleration
    resulting_acceleration = neighbor_system.acceleration -
                             @inbounds current_acceleration(system, particle)
    hydrostatic_pressure = dot(resulting_acceleration, density_neighbor * pos_diff)

    # Additional dynamic pressure term (only with `BernoulliPressureExtrapolation`)
    dynamic_pressure_ = dynamic_pressure(boundary_density_calculator, density_neighbor,
                                         v, v_neighbor_system, pos_diff, distance,
                                         particle, neighbor, system, neighbor_system)

    sum_pressures = pressure_offset + fluid_pressure + dynamic_pressure_ +
                    hydrostatic_pressure

    kernel_weight = smoothing_kernel(boundary_model, distance, particle)

    @inbounds pressure[particle] += sum_pressures * kernel_weight
    @inbounds cache.volume[particle] += kernel_weight

    compute_smoothed_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                               kernel_weight, particle, neighbor)
end

@inline function dynamic_pressure(boundary_density_calculator, density_neighbor, v,
                                  v_neighbor_system, pos_diff, distance, particle, neighbor,
                                  system, neighbor_system)
    return zero(density_neighbor)
end

@inline function dynamic_pressure(boundary_density_calculator::BernoulliPressureExtrapolation,
                                  density_neighbor, v, v_neighbor_system, pos_diff,
                                  distance, particle, neighbor,
                                  system::BoundarySystem, neighbor_system)
    if system.ismoving[]
        relative_velocity = current_velocity(v, system, particle) .-
                            current_velocity(v_neighbor_system, neighbor_system, neighbor)
        normal_velocity = dot(relative_velocity, pos_diff)

        return boundary_density_calculator.factor * density_neighbor *
               normal_velocity^2 / distance / 2
    end
    return zero(density_neighbor)
end

@inline function dynamic_pressure(boundary_density_calculator::BernoulliPressureExtrapolation,
                                  density_neighbor, v, v_neighbor_system, pos_diff,
                                  distance, particle, neighbor,
                                  system::SolidSystem, neighbor_system)
    relative_velocity = current_velocity(v, system, particle) .-
                        current_velocity(v_neighbor_system, neighbor_system, neighbor)
    normal_velocity = dot(relative_velocity, pos_diff) / distance

    return boundary_density_calculator.factor * density_neighbor *
           dot(normal_velocity, normal_velocity) / 2
end

function compute_smoothed_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                                    kernel_weight, particle, neighbor)
    return cache
end

function compute_smoothed_velocity!(cache, viscosity::ViscosityAdami, neighbor_system,
                                    v_neighbor_system, kernel_weight, particle, neighbor)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    for dim in eachindex(v_b)
        @inbounds cache.wall_velocity[dim, particle] += kernel_weight * v_b[dim]
    end

    return cache
end

@inline function compute_wall_velocity!(viscosity::Nothing, system, system_coords, particle)
    return viscosity
end

@inline function compute_wall_velocity!(viscosity, system, v, particle)
    (; boundary_model) = system
    (; cache) = boundary_model
    (; volume, wall_velocity) = cache

    # Prescribed velocity of the boundary particle.
    # This velocity is zero when not using moving boundaries.
    v_boundary = current_velocity(v, system, particle)

    for dim in eachindex(v_boundary)
        # The second term is the precalculated smoothed velocity field of the fluid
        new_velocity = @inbounds 2 * v_boundary[dim] -
                                 wall_velocity[dim, particle] / volume[particle]
        @inbounds wall_velocity[dim, particle] = new_velocity
    end
    return viscosity
end

@inline function inverse_state_equation!(density, state_equation, pressure, particle)
    @inbounds density[particle] = inverse_state_equation(state_equation, pressure[particle])
    return density
end

@inline function inverse_state_equation!(density, state_equation::Nothing, pressure,
                                         particle)
    # The density is constant when using EDAC
    return density
end

@inline function correction_matrix(system::BoundarySystem, particle)
    extract_smatrix(system.boundary_model.cache.correction_matrix, system, particle)
end
