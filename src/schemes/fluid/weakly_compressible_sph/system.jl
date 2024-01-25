"""
    WeaklyCompressibleSPHSystem(initial_condition,
                                density_calculator, state_equation,
                                smoothing_kernel, smoothing_length;
                                viscosity=NoViscosity(),
                                acceleration=ntuple(_ -> 0.0, NDIMS))

Weakly compressible SPH introduced by (Monaghan, 1994). This formulation relies on a stiff
equation of state (see  [`StateEquationCole`](@ref)) that generates large pressure changes
for small density variations. For the choice of the appropriate `density_calculator`
see [`ContinuityDensity`](@ref) and [`SummationDensity`](@ref).

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.
- `density_calculator`: Density calculator for the SPH system. See [`ContinuityDensity`](@ref) and [`SummationDensity`](@ref).
- `state_equation`:     Equation of state for the SPH system. See [`StateEquationCole`](@ref) and [`StateEquationIdealGas`](@ref).

# Keyword Arguments
- `viscosity`:    Viscosity model for the SPH system (default: no viscosity). See [`ArtificialViscosityMonaghan`](@ref) or [`ViscosityAdami`](@ref).
- `acceleration`: Acceleration vector for the SPH system. (default: zero vector)
- `correction`:   Correction method used for this SPH system. (default: no correction)
- `surface_tension`:   Surface tension model used for this SPH system. (default: no surface tension)

## References:
- Joseph J. Monaghan. "Simulating Free Surface Flows in SPH".
  In: Journal of Computational Physics 110 (1994), pages 399-406.
  [doi: 10.1006/jcph.1994.1034](https://doi.org/10.1006/jcph.1994.1034)
"""
struct WeaklyCompressibleSPHSystem{NDIMS, ELTYPE <: Real, DC, SE, K, V, COR, SRFT, C} <:
       FluidSystem{NDIMS}
    initial_condition  :: InitialCondition{ELTYPE}
    mass               :: Array{ELTYPE, 1} # [particle]
    pressure           :: Array{ELTYPE, 1} # [particle]
    density_calculator :: DC
    state_equation     :: SE
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    viscosity          :: V
    acceleration       :: SVector{NDIMS, ELTYPE}
    correction         :: COR
    surface_tension    :: SRFT
    cache              :: C

    function WeaklyCompressibleSPHSystem(initial_condition,
                                         density_calculator, state_equation,
                                         smoothing_kernel, smoothing_length;
                                         viscosity=NoViscosity(),
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         correction=nothing, surface_tension=nothing)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)
        n_particles = nparticles(initial_condition)

        mass = copy(initial_condition.mass)
        pressure = similar(initial_condition.pressure)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        if correction isa ShepardKernelCorrection &&
           density_calculator isa ContinuityDensity
            throw(ArgumentError("`ShepardKernelCorrection` cannot be used with `ContinuityDensity`"))
        end

        if surface_tension isa SurfaceTensionAkinci && correction === nothing
            println("NOTE: Result is *probably* inaccurate when used without corrections.
                     Incorrect pressure near the boundary leads the particles near walls to
                     be too far away, which leads to surface tension being applied near walls!")
        end

        cache = create_cache(n_particles, ELTYPE, density_calculator)
        cache = (;
                 create_cache(correction, initial_condition.density, NDIMS, n_particles)...,
                 cache...)
        cache = (;
                 create_cache(surface_tension, ELTYPE, NDIMS, n_particles)...,
                 cache...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity),
                   typeof(correction), typeof(surface_tension), typeof(cache)}(initial_condition,
                                                                               mass,
                                                                               pressure,
                                                                               density_calculator,
                                                                               state_equation,
                                                                               smoothing_kernel,
                                                                               smoothing_length,
                                                                               viscosity,
                                                                               acceleration_,
                                                                               correction,
                                                                               surface_tension,
                                                                               cache)
    end
end

create_cache(correction, density, NDIMS, nparticles) = (;)

function create_cache(::ShepardKernelCorrection, density, NDIMS, n_particles)
    (; kernel_correction_coefficient=similar(density))
end

function create_cache(::KernelGradientCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{Float64}(undef, NDIMS, n_particles)
    (; kernel_correction_coefficient=similar(density), dw_gamma)
end

function create_cache(n_particles, ELTYPE, ::SummationDensity)
    density = Vector{ELTYPE}(undef, n_particles)

    return (; density)
end

function create_cache(n_particles, ELTYPE, ::ContinuityDensity)
    # Density in this case is added to the end of 'v' and allocated by modifying 'v_nvariables'.
    return (;)
end

function create_cache(::SurfaceTensionAkinci, ELTYPE, NDIMS, nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    return (; surface_normal)
end

function Base.show(io::IO, system::WeaklyCompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "WeaklyCompressibleSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
    print(io, ", ", system.correction)
    print(io, ", ", system.state_equation)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.surface_tension)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::WeaklyCompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "WeaklyCompressibleSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "density calculator",
                     system.density_calculator |> typeof |> nameof)
        summary_line(io, "correction method",
                     system.correction |> typeof |> nameof)
        summary_line(io, "state equation", system.state_equation |> typeof |> nameof)
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity)
        summary_line(io, "surface tension", system.surface_tension)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, density_calculator)
    return ndims(system)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, ::ContinuityDensity)
    return ndims(system) + 1
end

@inline function particle_pressure(v, system::WeaklyCompressibleSPHSystem, particle)
    return system.pressure[particle]
end

# Nothing to initialize for this system
initialize!(system::WeaklyCompressibleSPHSystem, neighborhood_search) = system

function update_quantities!(system::WeaklyCompressibleSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    (; density_calculator) = system

    compute_density!(system, u, u_ode, semi, density_calculator)

    return system
end

function compute_density!(system, u, u_ode, semi, ::ContinuityDensity)
    # No density update with `ContinuityDensity`
    return system
end

function compute_density!(system, u, u_ode, semi, ::SummationDensity)
    (; cache) = system
    (; density) = cache # Density is in the cache for SummationDensity

    summation_density!(system, semi, u, u_ode, density)
end

function update_pressure!(system::WeaklyCompressibleSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; density_calculator, correction, surface_tension) = system

    compute_correction_values!(system, v, u, v_ode, u_ode, semi,
                               density_calculator, correction)
    # `kernel_correct_density!` only performed for `SummationDensity`
    kernel_correct_density!(system, v, u, v_ode, u_ode, semi, correction,
                            density_calculator)
    compute_pressure!(system, v)
    compute_surface_normal!(surface_tension, v, u, system, u_ode, v_ode, semi,
                            t)

    return system
end

function kernel_correct_density!(system, v, u, v_ode, u_ode, semi,
                                 correction, density_calculator)
    return system
end

function kernel_correct_density!(system, v, u, v_ode, u_ode, semi,
                                 ::Union{ShepardKernelCorrection, KernelGradientCorrection},
                                 ::SummationDensity)
    system.cache.density ./= system.cache.kernel_correction_coefficient
end

function reinit_density!(vu_ode, semi)
    v_ode, u_ode = vu_ode.x

    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        reinit_density!(system, v, u, v_ode, u_ode, semi)
    end

    return vu_ode
end

function reinit_density!(system::WeaklyCompressibleSPHSystem, v, u,
                         v_ode, u_ode, semi)
    # Compute density with `SummationDensity` and store the result in `v`,
    # overwriting the previous integrated density.
    summation_density!(system, semi, u, u_ode, v[end, :])

    # Apply `ShepardKernelCorrection`
    kernel_correction_coefficient = zeros(size(v[end, :]))
    compute_shepard_coeff!(system, v, u, v_ode, u_ode, semi, kernel_correction_coefficient)
    v[end, :] ./= kernel_correction_coefficient

    compute_pressure!(system, v)

    return system
end

function reinit_density!(system, v, u, v_ode, u_ode, semi)
    return system
end

function compute_pressure!(system, v)
    (; state_equation, pressure) = system

    @trixi_timeit timer() "state equation" @threaded for particle in eachparticle(system)
        pressure[particle] = state_equation(particle_density(v, system, particle))
    end
end

function write_v0!(v0, system::WeaklyCompressibleSPHSystem)
    (; initial_condition, density_calculator) = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    write_v0!(v0, density_calculator, system)

    return v0
end

function write_v0!(v0, ::SummationDensity, system::WeaklyCompressibleSPHSystem)
    return v0
end

function write_v0!(v0, ::ContinuityDensity, system::WeaklyCompressibleSPHSystem)
    (; initial_condition) = system

    for particle in eachparticle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_condition.density[particle]
    end

    return v0
end

function restart_with!(system::WeaklyCompressibleSPHSystem, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
    end

    restart_with!(system, system.density_calculator, v, u)
end

function restart_with!(system, ::SummationDensity, v, u)
    return system
end

function restart_with!(system, ::ContinuityDensity, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.density[particle] = v[end, particle]
    end

    return system
end

@inline function smoothing_kernel_grad(system::WeaklyCompressibleSPHSystem, pos_diff,
                                       distance, particle)
    return corrected_kernel_grad(system.smoothing_kernel, pos_diff, distance,
                                 system.smoothing_length,
                                 system.correction, system, particle)
end

function compute_surface_normal!(surface_tension, v, u, container, container_index, u_ode,
                                 v_ode, semi, t)
end

function compute_surface_normal!(surface_tension::SurfaceTensionAkinci, v, u, system
                                 , u_ode, v_ode, semi, t)
    (; cache) = system

    # reset surface normal
    cache.surface_normal .= zero(eltype(cache.surface_normal))

    @trixi_timeit timer() "compute surface normal" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        nhs = neighborhood_searches(system, neighbor_system, semi)

        calc_normal_akinci(surface_tension, u, v_neighbor_system, u_neighbor_system, nhs,
                           system, neighbor_system)
    end
end

@inline function get_normal(particle, particle_container::FluidSystem,
                            ::SurfaceTensionAkinci)
    (; cache) = particle_container
    return extract_svector(cache.surface_normal, particle_container, particle)
end
