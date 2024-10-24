@doc raw"""
    EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                smoothing_length, sound_speed;
                                pressure_acceleration=inter_particle_averaged_pressure,
                                density_calculator=SummationDensity(),
                                alpha=0.5, viscosity=nothing,
                                acceleration=ntuple(_ -> 0.0, NDIMS), buffer_size=nothing,
                                source_terms=nothing)

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
- `viscosity`:      Viscosity model for this system (default: no viscosity).
                    Recommended: [`ViscosityAdami`](@ref).
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `pressure_acceleration`: Pressure acceleration formulation (default: inter-particle averaged pressure).
                        When set to `nothing`, the pressure acceleration formulation for the
                        corresponding [density calculator](@ref density_calculator) is chosen.
- `density_calculator`: [Density calculator](@ref density_calculator) (default: [`SummationDensity`](@ref))
- `buffer_size`:    Number of buffer particles.
                    This is needed when simulating with [`OpenBoundarySPHSystem`](@ref).
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure, t)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).
                    Note that these source terms will not be used in the calculation of the
                    boundary pressure when using a boundary with
                    [`BoundaryModelDummyParticles`](@ref) and [`AdamiPressureExtrapolation`](@ref).
                    The keyword argument `acceleration` should be used instead for
                    gravity-like source terms.
"""
struct EntropicallyDampedSPHSystem{NDIMS, ELTYPE <: Real, IC, M, DC, K, V,
                                   PF, ST, SRFT, SRFN, B, C} <: FluidSystem{NDIMS, IC}
    initial_condition                 :: IC
    mass                              :: M # Vector{ELTYPE}: [particle]
    density_calculator                :: DC
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
    number_density                    :: Int64
    color                             :: Int64
    sound_speed                       :: ELTYPE
    viscosity                         :: V
    nu_edac                           :: ELTYPE
    acceleration                      :: SVector{NDIMS, ELTYPE}
    correction                        :: Nothing
    pressure_acceleration_formulation :: PF
    source_terms                      :: ST
    surface_tension                   :: SRFT
    surface_normal_method             :: SRFN
    buffer                            :: B
    cache                             :: C

    function EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                         smoothing_length, sound_speed;
                                         pressure_acceleration=inter_particle_averaged_pressure,
                                         density_calculator=SummationDensity(),
                                         alpha=0.5, viscosity=nothing,
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         source_terms=nothing, surface_tension=nothing,
                                         surface_normal_method=nothing, buffer_size=nothing,
                                         reference_particle_spacing=0.0, color_value=1)
        buffer = isnothing(buffer_size) ? nothing :
                 SystemBuffer(nparticles(initial_condition), buffer_size)

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
            surface_normal_method = ColorfieldSurfaceNormal(smoothing_kernel,
                                                            smoothing_length)
        end

        if surface_normal_method !== nothing && reference_particle_spacing < eps()
            throw(ArgumentError("`reference_particle_spacing` must be set to a positive value when using `ColorfieldSurfaceNormal` or a surface tension model"))
        end

        number_density_ = 0
        if reference_particle_spacing > 0.0
            number_density_ = number_density(Val(NDIMS), reference_particle_spacing,
                                             compact_support(smoothing_kernel,
                                                             smoothing_length))
        end

        pressure_acceleration = choose_pressure_acceleration_formulation(pressure_acceleration,
                                                                         density_calculator,
                                                                         NDIMS, ELTYPE,
                                                                         nothing)

        nu_edac = (alpha * smoothing_length * sound_speed) / 8

        cache = create_cache_density(initial_condition, density_calculator)
        cache = (;
                 create_cache_surface_normal(surface_normal_method, ELTYPE, NDIMS,
                                             n_particles)...,
                 create_cache_surface_tension(surface_tension, ELTYPE, NDIMS,
                                              n_particles)...,
                 cache...)

        new{NDIMS, ELTYPE, typeof(initial_condition), typeof(mass),
            typeof(density_calculator), typeof(smoothing_kernel),
            typeof(viscosity), typeof(pressure_acceleration), typeof(source_terms),
            typeof(surface_tension), typeof(surface_normal_method),
            typeof(buffer),
            typeof(cache)}(initial_condition, mass, density_calculator, smoothing_kernel,
                           smoothing_length, number_density_, color_value, sound_speed,
                           viscosity, nu_edac,
                           acceleration_, nothing, pressure_acceleration, source_terms,
                           surface_tension, surface_normal_method,
                           buffer, cache)
    end
end

function Base.show(io::IO, system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "EntropicallyDampedSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
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
        summary_line(io, "viscosity", system.viscosity |> typeof |> nameof)
        summary_line(io, "ν₍EDAC₎", "≈ $(round(system.nu_edac; digits=3))")
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "surface tension", system.surface_tension)
        summary_line(io, "surface normal method", system.surface_normal_method)
        summary_footer(io)
    end
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, density_calculator)
    return ndims(system) + 1
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    return ndims(system) + 2
end

@inline function particle_density(v, ::ContinuityDensity,
                                  system::EntropicallyDampedSPHSystem, particle)
    return v[end - 1, particle]
end

@inline function particle_pressure(v, system::EntropicallyDampedSPHSystem, particle)
    return v[end, particle]
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

function update_quantities!(system::EntropicallyDampedSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    compute_density!(system, u, u_ode, semi, system.density_calculator)
end

function update_pressure!(system::EntropicallyDampedSPHSystem, v, u, v_ode, u_ode, semi, t)
    compute_surface_normal!(system, system.surface_normal_method, v, u, v_ode, u_ode, semi,
                            t)
    compute_surface_delta_function!(system, system.surface_tension)
end

function update_final!(system::EntropicallyDampedSPHSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    (; surface_tension) = system

    # Surface normal of neighbor and boundary needs to have been calculated already
    compute_curvature!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    compute_stress_tensors!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, density_calculator)
    for particle in eachparticle(system)
        v0[end, particle] = system.initial_condition.pressure[particle]
    end

    return v0
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    for particle in eachparticle(system)
        v0[end - 1, particle] = system.initial_condition.density[particle]
        v0[end, particle] = system.initial_condition.pressure[particle]
    end

    return v0
end

function restart_with!(system::EntropicallyDampedSPHSystem, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
        system.initial_condition.pressure[particle] = v[end, particle]
    end
end
