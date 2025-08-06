@doc raw"""
    EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                smoothing_length, sound_speed;
                                pressure_acceleration=inter_particle_averaged_pressure,
                                density_calculator=SummationDensity(),
                                transport_velocity=nothing,
                                alpha=0.5, viscosity=nothing,
                                acceleration=ntuple(_ -> 0.0, NDIMS), surface_tension=nothing,
                                surface_normal_method=nothing, buffer_size=nothing,
                                reference_particle_spacing=0.0, source_terms=nothing)

System for particles of a fluid.
As opposed to the [weakly compressible SPH scheme](@ref wcsph), which uses an equation of state,
this scheme uses a pressure evolution equation to calculate the pressure.
See [Entropically Damped Artificial Compressibility for SPH](@ref edac) for more details on the method.

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.
- `sound_speed`:        Speed of sound.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).

# Keyword Arguments
- `viscosity`:                  Viscosity model for this system (default: no viscosity).
                                Recommended: [`ViscosityAdami`](@ref).
- `acceleration`:               Acceleration vector for the system. (default: zero vector)
- `pressure_acceleration`:      Pressure acceleration formulation (default: inter-particle averaged pressure).
                                When set to `nothing`, the pressure acceleration formulation for the
                                corresponding [density calculator](@ref density_calculator) is chosen.
- `density_calculator`:         [Density calculator](@ref density_calculator) (default: [`SummationDensity`](@ref))
- `transport_velocity`:         [Transport Velocity Formulation (TVF)](@ref transport_velocity_formulation). Default is no TVF.
- `buffer_size`:                Number of buffer particles.
                                This is needed when simulating with [`OpenBoundarySPHSystem`](@ref).
- `correction`:                 Correction method used for this system. (default: no correction, see [Corrections](@ref corrections))
- `source_terms`:               Additional source terms for this system. Has to be either `nothing`
                                (by default), or a function of `(coords, velocity, density, pressure, t)`
                                (which are the quantities of a single particle), returning a `Tuple`
                                or `SVector` that is to be added to the acceleration of that particle.
                                See, for example, [`SourceTermDamping`](@ref).
                                Note that these source terms will not be used in the calculation of the
                                boundary pressure when using a boundary with
                                [`BoundaryModelDummyParticles`](@ref) and [`AdamiPressureExtrapolation`](@ref).
                                The keyword argument `acceleration` should be used instead for
                                gravity-like source terms.
- `surface_tension`:            Surface tension model used for this SPH system. (default: no surface tension)
- `surface_normal_method`:      The surface normal method to be used for this SPH system.
                                (default: no surface normal method or `ColorfieldSurfaceNormal()` if a surface_tension model is used)
- `reference_particle_spacing`: The reference particle spacing used for weighting values at the boundary,
                                which currently is only needed when using surface tension.
- `color_value`:                The value used to initialize the color of particles in the system.

"""
struct EntropicallyDampedSPHSystem{NDIMS, ELTYPE <: Real, IC, M, DC, K, V, COR, PF, TV,
                                   ST, SRFT, SRFN, B, PR, C} <: FluidSystem{NDIMS}
    initial_condition                 :: IC
    mass                              :: M # Vector{ELTYPE}: [particle]
    density_calculator                :: DC
    smoothing_kernel                  :: K
    sound_speed                       :: ELTYPE
    viscosity                         :: V
    nu_edac                           :: ELTYPE
    acceleration                      :: SVector{NDIMS, ELTYPE}
    correction                        :: COR
    pressure_acceleration_formulation :: PF
    transport_velocity                :: TV
    source_terms                      :: ST
    surface_tension                   :: SRFT
    surface_normal_method             :: SRFN
    buffer                            :: B
    particle_refinement               :: PR
    cache                             :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                     smoothing_length, sound_speed;
                                     pressure_acceleration=inter_particle_averaged_pressure,
                                     density_calculator=SummationDensity(),
                                     transport_velocity=nothing,
                                     alpha=0.5, viscosity=nothing,
                                     acceleration=ntuple(_ -> 0.0,
                                                         ndims(smoothing_kernel)),
                                     correction=nothing,
                                     source_terms=nothing, surface_tension=nothing,
                                     surface_normal_method=nothing, buffer_size=nothing,
                                     reference_particle_spacing=0.0, color_value=1)
    buffer = isnothing(buffer_size) ? nothing :
             SystemBuffer(nparticles(initial_condition), buffer_size)

    particle_refinement = nothing # TODO

    initial_condition = allocate_buffer(initial_condition, buffer)

    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)

    mass = copy(initial_condition.mass)
    n_particles = length(initial_condition.mass)

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    if surface_tension !== nothing && surface_normal_method === nothing
        surface_normal_method = ColorfieldSurfaceNormal()
    end

    if surface_normal_method !== nothing && reference_particle_spacing < eps()
        throw(ArgumentError("`reference_particle_spacing` must be set to a positive value when using `ColorfieldSurfaceNormal` or a surface tension model"))
    end

    if correction isa ShepardKernelCorrection &&
       density_calculator isa ContinuityDensity
        throw(ArgumentError("`ShepardKernelCorrection` cannot be used with `ContinuityDensity`"))
    end

    pressure_acceleration = choose_pressure_acceleration_formulation(pressure_acceleration,
                                                                     density_calculator,
                                                                     NDIMS, ELTYPE,
                                                                     correction)

    nu_edac = (alpha * smoothing_length * sound_speed) / 8

    cache = (; create_cache_density(initial_condition, density_calculator)...,
             create_cache_tvf(Val(:edac), initial_condition, transport_velocity)...,
             create_cache_surface_normal(surface_normal_method, ELTYPE, NDIMS,
                                         n_particles)...,
             create_cache_surface_tension(surface_tension, ELTYPE, NDIMS,
                                          n_particles)...,
             create_cache_refinement(initial_condition, particle_refinement,
                                     smoothing_length)...,
             create_cache_correction(correction, initial_condition.density, NDIMS,
                                     n_particles)...,
             color=Int(color_value))

    # If the `reference_density_spacing` is set calculate the `ideal_neighbor_count`
    if reference_particle_spacing > 0
        # `reference_particle_spacing` has to be set for surface normals to be determined
        cache = (;
                 cache...,  # Existing cache fields
                 reference_particle_spacing=reference_particle_spacing)
    end

    EntropicallyDampedSPHSystem{NDIMS, ELTYPE, typeof(initial_condition), typeof(mass),
                                typeof(density_calculator), typeof(smoothing_kernel),
                                typeof(viscosity), typeof(correction),
                                typeof(pressure_acceleration),
                                typeof(transport_velocity), typeof(source_terms),
                                typeof(surface_tension), typeof(surface_normal_method),
                                typeof(buffer), Nothing,
                                typeof(cache)}(initial_condition, mass, density_calculator,
                                               smoothing_kernel, sound_speed, viscosity,
                                               nu_edac, acceleration_, correction,
                                               pressure_acceleration, transport_velocity,
                                               source_terms, surface_tension,
                                               surface_normal_method, buffer,
                                               particle_refinement, cache)
end

function Base.show(io::IO, system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "EntropicallyDampedSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
    print(io, ", ", system.correction)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.surface_tension)
    print(io, ", ", system.surface_normal_method)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "EntropicallyDampedSPHSystem{$(ndims(system))}")
        if system.buffer isa SystemBuffer
            summary_line(io, "#particles", nparticles(system))
            summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        else
            summary_line(io, "#particles", nparticles(system))
        end
        summary_line(io, "density calculator",
                     system.density_calculator |> typeof |> nameof)
        summary_line(io, "correction method",
                     system.correction |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity |> typeof |> nameof)
        summary_line(io, "ν₍EDAC₎", "≈ $(round(system.nu_edac; digits=3))")
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "tansport velocity formulation",
                     system.transport_velocity |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "surface tension", system.surface_tension)
        summary_line(io, "surface normal method", system.surface_normal_method)
        summary_footer(io)
    end
end

create_cache_tvf(::Val{:edac}, initial_condition, ::Nothing) = (;)

function create_cache_tvf(::Val{:edac}, initial_condition, ::TransportVelocityAdami)
    pressure_average = copy(initial_condition.pressure)
    neighbor_counter = Vector{Int}(undef, nparticles(initial_condition))
    update_callback_used = Ref(false)

    return (; pressure_average, neighbor_counter, update_callback_used)
end

@inline function Base.eltype(::EntropicallyDampedSPHSystem{<:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, density_calculator)
    return ndims(system) * factor_tvf(system) + 1
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    return ndims(system) * factor_tvf(system) + 2
end

system_correction(system::EntropicallyDampedSPHSystem) = system.correction

@inline function current_pressure(v, system::EntropicallyDampedSPHSystem, particle)
    return v[end, particle]
end

@inline function current_velocity(v, system::EntropicallyDampedSPHSystem)
    return view(v, 1:ndims(system), :)
end

@inline system_state_equation(system::EntropicallyDampedSPHSystem) = nothing

# WARNING!
# These functions are intended to be used internally to set the pressure
# of newly activated particles in a callback.
# DO NOT use outside a callback. OrdinaryDiffEq does not allow changing `v` and `u`
# outside of callbacks.
@inline function set_particle_pressure!(v, system::EntropicallyDampedSPHSystem, particle,
                                        pressure)
    v[end, particle] = pressure

    return v
end

@inline system_sound_speed(system::EntropicallyDampedSPHSystem) = system.sound_speed

@inline average_pressure(system, particle) = zero(eltype(system))

@inline function average_pressure(system::EntropicallyDampedSPHSystem, particle)
    average_pressure(system, system.transport_velocity, particle)
end

@inline function average_pressure(system, ::TransportVelocityAdami, particle)
    return system.cache.pressure_average[particle]
end

@inline average_pressure(system, ::Nothing, particle) = zero(eltype(system))

@inline function current_density(v, system::EntropicallyDampedSPHSystem)
    return current_density(v, system.density_calculator, system)
end

@inline function current_density(v, ::SummationDensity,
                                 system::EntropicallyDampedSPHSystem)
    # When using `SummationDensity`, the density is stored in the cache
    return system.cache.density
end

@inline function current_density(v, ::ContinuityDensity,
                                 system::EntropicallyDampedSPHSystem)
    # When using `ContinuityDensity`, the density is stored in the second to last row of `v`
    return view(v, size(v, 1) - 1, :)
end

@inline function current_pressure(v, ::EntropicallyDampedSPHSystem)
    return view(v, size(v, 1), :)
end

function update_quantities!(system::EntropicallyDampedSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    compute_density!(system, u, u_ode, semi, system.density_calculator)
end

function update_pressure!(system::EntropicallyDampedSPHSystem, v, u, v_ode, u_ode, semi, t)
    compute_surface_normal!(system, system.surface_normal_method, v, u, v_ode, u_ode, semi,
                            t)
    compute_surface_delta_function!(system, system.surface_tension, semi)
end

function update_final!(system::EntropicallyDampedSPHSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    (; surface_tension) = system

    # Surface normal of neighbor and boundary needs to have been calculated already
    compute_curvature!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    compute_stress_tensors!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    update_average_pressure!(system, system.transport_velocity, v_ode, u_ode, semi)

    # Check that TVF is only used together with `UpdateCallback`
    check_tvf_configuration(system, system.transport_velocity, v, u, v_ode, u_ode, semi, t;
                            update_from_callback)
end

function update_average_pressure!(system, ::Nothing, v_ode, u_ode, semi)
    return system
end

# This technique is for a more robust `pressure_acceleration` but only with TVF.
# It results only in significant improvement for EDAC and not for WCSPH.
# See Ramachandran (2019) p. 582.
function update_average_pressure!(system, ::TransportVelocityAdami, v_ode, u_ode, semi)
    (; cache) = system
    (; pressure_average, neighbor_counter) = cache

    set_zero!(pressure_average)
    set_zero!(neighbor_counter)

    u = wrap_u(u_ode, system, semi)

    # Use all other systems for the average pressure
    @trixi_timeit timer() "compute average pressure" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            pressure_average[particle] += current_pressure(v_neighbor_system,
                                                           neighbor_system,
                                                           neighbor)
            neighbor_counter[particle] += 1
        end
    end

    # We do not need to check for zero division here, as `neighbor_counter = 1`
    # for zero neighbors. That is, the `particle` itself is also taken into account.
    pressure_average ./= neighbor_counter

    return system
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::SummationDensity)
    # Note that `.=` is very slightly faster, but not GPU-compatible
    v0[end, :] = system.initial_condition.pressure

    return v0
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    # Note that `.=` is very slightly faster, but not GPU-compatible
    v0[end - 1, :] = system.initial_condition.density
    v0[end, :] = system.initial_condition.pressure

    return v0
end

function restart_with!(system::EntropicallyDampedSPHSystem, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
        system.initial_condition.pressure[particle] = v[end, particle]
    end
end
