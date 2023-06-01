
# TODO: Put this in docs somehow

# Ramachandran p. 580: "Clausen shows that the artiﬁcial compressibility equation results
# when isentropic ﬂow is assumed and this implies that the ﬂuid is inviscid and therefore
# the viscous dissipation is ignored."
#
# refs:
# - Clausen 10.1103/physreve.87.013309
# - Ramachandran 2019
#
# !important! (Ramachandran 2019)
#
# To summarize the schemes:
#
# -   for external ﬂow problems, Eqs. (4), (6), and (10) are used.
#     The particles move with the ﬂuid velocity u and are advected according to (11).
#
# -   for internal ﬂows, Eqs. (4), (12), (16), (17) and (10) are used.
#     Eq. (14) is used to advect the particles. The transport velocity is
#     found from Eq. (15).
#
# For each of the schemes, the value of ν used in the Eq. (10) is found using Eq. (18).
# The value of ν used in the momentum equation is the ﬂuid viscosity.

struct EntropicallyDampedSPH{NDIMS, ELTYPE <: Real, DC, K, V} <: FluidSystem{NDIMS}
    initial_condition  :: InitialCondition{ELTYPE}
    mass               :: Array{ELTYPE, 1} # [particle]
    density            :: Array{ELTYPE, 1} # [particle]
    density_calculator :: DC
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    sound_speed        :: ELTYPE
    viscosity          :: V
    # (Ramachandran 2019) "viscosity is used to diffuse the pressure
    # The original formulation assumes that the value of ν is the same as
    # the fluid viscosity. [...] if the viscosity is too small, the pressure builds up too
    # fast and eventually blows up. If the viscosity is too large it diffuses too fast
    # resulting in a non-physical simulation.
    #
    # "[...] it is found that α=0.5 is a good choice for a wide range of Reynolds numbers."
    nu           :: ELTYPE
    acceleration :: SVector{NDIMS, ELTYPE}

    function EntropicallyDampedSPH(initial_condition, smoothing_kernel, smoothing_length,
                                   sound_speed; alpha=0.5, viscosity=NoViscosity(),
                                   acceleration=ntuple(_ -> 0.0, ndims(smoothing_kernel)))
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)
        n_particles = nparticles(initial_condition)

        mass = copy(initial_condition.mass)
        density = copy(initial_condition.density)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        nu = (alpha * smoothing_length * sound_speed) / 8

        density_calculator = SummationDensity()

        new{NDIMS, ELTYPE, typeof(density_calculator), typeof(smoothing_kernel),
            typeof(viscosity)}(initial_condition, mass, density, density_calculator,
                               smoothing_kernel, smoothing_length, sound_speed, viscosity,
                               nu, acceleration_)
    end
end

function Base.show(io::IO, system::EntropicallyDampedSPH)
    @nospecialize system # reduce precompilation time

    print(io, "EntropicallyDampedSPH{ ", ndims(system), "}(")
    print(io, system.viscosity)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::EntropicallyDampedSPH)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "EntropicallyDampedSPH{ $(ndims(system)) }")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "viscosity",
                     system.viscosity |> typeof |> nameof)
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline function particle_density(v, system::EntropicallyDampedSPH, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::EntropicallyDampedSPH, particle)
    return v[end, particle]
end

@inline function v_nvariables(system::EntropicallyDampedSPH)
    ndims(system) + 1
end

function update_quantities!(system::EntropicallyDampedSPH, system_index, v, u,
                            v_ode, u_ode, semi, t)
    summation_density!(system, system_index, semi, u, u_ode, system.density)
end

function write_v0!(v0, system::EntropicallyDampedSPH)
    @unpack initial_condition = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
        v0[end, particle] = initial_condition.pressure[particle]
    end

    return v0
end

function restart_with!(system::EntropicallyDampedSPH, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
        system.initial_condition.pressure[particle] = v[end, particle]
    end
end
