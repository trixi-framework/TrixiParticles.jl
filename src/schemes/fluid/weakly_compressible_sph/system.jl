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

## References:
- Joseph J. Monaghan. "Simulating Free Surface Flows in SPH".
  In: Journal of Computational Physics 110 (1994), pages 399-406.
  [doi: 10.1006/jcph.1994.1034](https://doi.org/10.1006/jcph.1994.1034)
"""
struct WeaklyCompressibleSPHSystem{NDIMS, ELTYPE <: Real, DC, SE, K, V, COR, C} <:
       System{NDIMS}
    initial_condition  :: InitialCondition{ELTYPE}
    current_coordinates:: Array{ELTYPE, 2} # [dim, particle]
    mass               :: Array{ELTYPE, 1} # [particle]
    pressure           :: Array{ELTYPE, 1} # [particle]
    density_calculator :: DC
    state_equation     :: SE
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    viscosity          :: V
    acceleration       :: SVector{NDIMS, ELTYPE}
    correction         :: COR
    cache              :: C

    function WeaklyCompressibleSPHSystem(initial_condition,
                                         density_calculator, state_equation,
                                         smoothing_kernel, smoothing_length;
                                         viscosity=NoViscosity(),
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         correction=nothing)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)
        n_particles = nparticles(initial_condition)

        current_coordinates = similar(initial_condition.coordinates)
        mass = copy(initial_condition.mass)
        pressure = Vector{ELTYPE}(undef, n_particles)

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

        cache = create_cache(n_particles, ELTYPE, density_calculator)
        cache = (;
                 create_cache(correction, initial_condition.density, NDIMS, n_particles)...,
                 cache...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity),
                   typeof(correction), typeof(cache)
                   }(initial_condition, current_coordinates, mass, pressure, density_calculator, state_equation,
                     smoothing_kernel, smoothing_length, viscosity, acceleration_,
                     correction, cache)
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
    return (;)
end

function Base.show(io::IO, system::WeaklyCompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "WeaklyCompressibleSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
    print(io, ", ", system.correction)
    print(io, ", ", system.state_equation)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
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
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

timer_name(::WeaklyCompressibleSPHSystem) = "fluid"

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, density_calculator)
    return ndims(system)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, ::ContinuityDensity)
    return ndims(system) + 1
end

@inline function hydrodynamic_mass(system::WeaklyCompressibleSPHSystem, particle)
    return system.mass[particle]
end

@inline current_coordinates(u, system::WeaklyCompressibleSPHSystem) = system.current_coordinates

# Nothing to initialize for this system
initialize!(system::WeaklyCompressibleSPHSystem, neighborhood_search) = system

function update_positions!(system::WeaklyCompressibleSPHSystem, system_index, v, u,
                           v_ode, u_ode, semi, t)
    @unpack periodic_box_size = semi

    update_positions!(system, u, periodic_box_size, semi)
end

function update_positions!(system::WeaklyCompressibleSPHSystem, u, ::Nothing, semi)
    @unpack current_coordinates = system

    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            current_coordinates[i, particle] = u[i, particle]
        end
    end

    return system
end

function update_positions!(system::WeaklyCompressibleSPHSystem, u, periodic_box_size, semi)
    @unpack current_coordinates = system
    @unpack periodic_box_min_corner = semi

    for particle in axes(u, 2)
        coords = extract_svector(u, system, particle)
        box_offset = round.((coords .- periodic_box_min_corner) ./ periodic_box_size .- 0.5)
        current_coordinates[:, particle] = coords - box_offset .* periodic_box_size
    end

    return system
end

function update_quantities!(system::WeaklyCompressibleSPHSystem, system_index, v, u,
                            v_ode, u_ode, semi, t)
    @unpack density_calculator = system

    compute_density!(system, system_index, u, u_ode, semi, density_calculator)

    return system
end

function compute_density!(system, system_index, u, u_ode, semi, ::ContinuityDensity)
    # No density update with `ContinuityDensity`
    return system
end

function compute_density!(system, system_index, u, u_ode, semi, ::SummationDensity)
    @unpack cache = system
    @unpack density = cache # Density is in the cache for SummationDensity

    summation_density!(system, system_index, semi, u, u_ode, density)
end

function update_pressure!(system::WeaklyCompressibleSPHSystem, system_index, v, u,
                          v_ode, u_ode, semi, t)
    @unpack density_calculator, correction = system

    compute_correction_values!(system, system_index, v, u, v_ode, u_ode, semi,
                               density_calculator, correction)
    # `kernel_correct_density!` only performed for `SummationDensity`
    kernel_correct_density!(system, system_index, v, u, v_ode, u_ode, semi, correction,
                            density_calculator)
    compute_pressure!(system, v)

    return system
end

function kernel_correct_density!(system, system_index, v, u, v_ode, u_ode, semi,
                                 correction, density_calculator)
    return system
end

function kernel_correct_density!(system, system_index, v, u, v_ode, u_ode, semi,
                                 ::Union{ShepardKernelCorrection, KernelGradientCorrection},
                                 ::SummationDensity)
    system.cache.density ./= system.cache.kernel_correction_coefficient
end

function compute_pressure!(system, v)
    @unpack state_equation, pressure = system

    # Note that @threaded makes this slower
    for particle in eachparticle(system)
        pressure[particle] = state_equation(particle_density(v, system, particle))
    end
end

function write_u0!(u0, system::WeaklyCompressibleSPHSystem)
    @unpack initial_condition = system

    for particle in eachparticle(system)
        # Write particle coordinates
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_condition.coordinates[dim, particle]
        end
    end

    return u0
end

function write_v0!(v0, system::WeaklyCompressibleSPHSystem)
    @unpack initial_condition, density_calculator = system

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
    @unpack initial_condition = system

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
