@doc raw"""
    EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                smoothing_length, sound_speed;
                                alpha=0.5, viscosity=nothing,
                                acceleration=ntuple(_ -> 0.0, NDIMS),
                                source_terms=nothing)

Entropically damped artiﬁcial compressibility (EDAC) for SPH introduced by Ramachandran (2019).

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.
- `sound_speed`:        Speed of sound.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [`SmoothingKernel`](@ref).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [`SmoothingKernel`](@ref).

# Keyword Arguments
- `viscosity`:      Viscosity model for this system (default: no viscosity).
                    Recommended: [`ViscosityAdami`](@ref).
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).
                    Note that these source terms will not be used in the calculation of the
                    boundary pressure when using a boundary with
                    [`BoundaryModelDummyParticles`](@ref) and [`AdamiPressureExtrapolation`](@ref).
                    The keyword argument `acceleration` should be used instead for
                    gravity-like source terms.
"""
struct EntropicallyDampedSPHSystem{NDIMS, ELTYPE <: Real, DC, K, V, ST} <:
       FluidSystem{NDIMS}
    initial_condition  :: InitialCondition{ELTYPE}
    mass               :: Array{ELTYPE, 1} # [particle]
    density            :: Array{ELTYPE, 1} # [particle]
    density_calculator :: DC
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    sound_speed        :: ELTYPE
    viscosity          :: V
    nu_edac            :: ELTYPE
    acceleration       :: SVector{NDIMS, ELTYPE}
    source_terms       :: ST

    function EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                         smoothing_length, sound_speed;
                                         alpha=0.5, viscosity=nothing,
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         source_terms=nothing)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)
        density = copy(initial_condition.density)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        nu_edac = (alpha * smoothing_length * sound_speed) / 8

        density_calculator = SummationDensity()

        new{NDIMS, ELTYPE, typeof(density_calculator),
            typeof(smoothing_kernel), typeof(viscosity),
            typeof(source_terms)}(initial_condition, mass, density, density_calculator,
                                  smoothing_kernel, smoothing_length, sound_speed,
                                  viscosity, nu_edac, acceleration_, source_terms)
    end
end

function Base.show(io::IO, system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "EntropicallyDampedSPHSystem{", ndims(system), "}(")
    print(io, system.viscosity)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "EntropicallyDampedSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "viscosity", system.viscosity |> typeof |> nameof)
        summary_line(io, "ν₍EDAC₎", "≈ $(round(system.nu_edac; digits=3))")
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline function particle_density(v, system::EntropicallyDampedSPHSystem, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::EntropicallyDampedSPHSystem, particle)
    return v[end, particle]
end

@inline v_nvariables(system::EntropicallyDampedSPHSystem) = ndims(system) + 1

function update_quantities!(system::EntropicallyDampedSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    summation_density!(system, semi, u, u_ode, system.density)
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem)
    (; initial_condition) = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
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
