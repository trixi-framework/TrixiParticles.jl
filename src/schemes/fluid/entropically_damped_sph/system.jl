
# Ramachandran p. 580: "Clausen shows that the artiﬁcial compressibility equation results
# when isentropic ﬂow is assumed and this implies that the ﬂuid is inviscid and therefore
# the viscous dissipation is ignored."
#
# refs:
# - Clausen 10.1103/physreve.87.013309
#
# To summarize the schemes (Ramachandran p. 582):
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
#
@doc raw"""
    EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel, smoothing_length,
                          sound_speed; alpha=0.5, viscosity=NoViscosity(),
                          acceleration=ntuple(_ -> 0.0, NDIMS))

Entropically damped artiﬁcial compressibility (EDAC) for SPH introduced by (Ramachandran 2019).
As opposed to the weakly compressible SPH scheme which uses a equation of state
(see [`WeaklyCompressibleSPHSystem`](@ref)) this scheme uses a pressure evolution equation
(PEE) to calculate the pressure. This equation is similar to the continuity equation (see
[`ContinuityDensity`](@ref)) but also contains a pressure damping term which reduces
oscillations and is discretized as
```math
\frac{\mathrm{d} p_a}{\mathrm{d}t} = \sum_{b} m_b \frac{\rho_a}{\rho_b} c_s^2 v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h) +
\frac{V_a^2 + V_b^2}{m_a} \tilde{\eta}_{ab} \frac{p_{ab}}{\Vert r_{ab}^2 \Vert + \eta h_{ab}^2} \nabla_{r_a}
W(\Vert r_a - r_b \Vert, h) \cdot r_{ab},
```
where ``\rho_a``, ``\rho_b``,  ``r_a``, ``r_b``, ``V_a`` and
``V_b`` denote the density, coordinates and volume of particles ``a`` and ``b`` respectively, ``c_s``
is the speed of sound and ``v_{ab} = v_a - v_b`` and ``p_{ab}= p_a -p_b`` is the difference
of the velocities and pressure of particles ``a`` and ``b`` respectively.

The viscosity parameter ``\eta_a`` for a particle ``a`` is given as
```math
\eta_a = \rho_a \frac{\alpha h c_s}{8}
```
where it is found in the numerical experiments of (Ramachandran 2019) that ``\alpha = 0.5``
is a good choice for a wide range of Reynolds numbers (0.0125 to 10000).

## References:
- Prabhu Ramachandran "Entropically damped artiﬁcial compressibility for SPH".
  In: Computers and Fluids 179 (2019), pages 579-594.
  [doi: 10.1016/j.compfluid.2018.11.023](https://doi.org/10.1016/j.compfluid.2018.11.023)
"""
struct EntropicallyDampedSPHSystem{NDIMS, ELTYPE <: Real, DC, K, V, PF, TV, C} <:
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
    pressure_function  :: PF
    acceleration       :: SVector{NDIMS, ELTYPE}
    transport_velocity :: TV
    cache              :: C

    function EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                         smoothing_length, sound_speed;
                                         alpha=0.5, viscosity=NoViscosity(),
                                         pressure_function=nothing,
                                         transport_velocity=nothing,
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)))
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

        nu_edac = (alpha * smoothing_length * sound_speed) / 8

        density_calculator = SummationDensity()

        cache = create_cache(n_particles, ELTYPE, NDIMS, transport_velocity)

        new{NDIMS, ELTYPE, typeof(density_calculator), typeof(smoothing_kernel),
            typeof(viscosity), typeof(pressure_function), typeof(transport_velocity),
            typeof(cache)}(initial_condition, mass, density, density_calculator,
                           smoothing_kernel, smoothing_length, sound_speed,
                           viscosity, nu_edac, pressure_function, acceleration_,
                           transport_velocity, cache)
    end
end

function Base.show(io::IO, system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "EntropicallyDampedSPHSystem{ ", ndims(system), "}(")
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
        summary_header(io, "EntropicallyDampedSPHSystem{ $(ndims(system)) }")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "viscosity",
                     system.viscosity |> typeof |> nameof)
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

create_cache(n_particles, ELTYPE, NDIMS, ::Nothing) = (;)

function create_cache(n_particles, ELTYPE, NDIMS, ::TransportVelocityAdami)
    pressure_average = zeros(ELTYPE, n_particles)
    neighbor_counter = Vector{Int}(undef, n_particles)
    advection_velocity = zeros(ELTYPE, NDIMS, n_particles)

    return (; pressure_average, neighbor_counter, advection_velocity)
end

@inline function particle_density(v, system::EntropicallyDampedSPHSystem, particle)
    return system.density[particle]
end
@inline function set_particle_density(particle, v, system::EntropicallyDampedSPHSystem,
                                      density)
    system.density[particle] = density
end

@inline function particle_pressure(v, system::EntropicallyDampedSPHSystem, particle)
    return v[end, particle]
end

@inline function set_particle_pressure(particle, v, system::EntropicallyDampedSPHSystem,
                                       pressure)
    return v[end, particle] = pressure
end

@inline function average_pressure(system::EntropicallyDampedSPHSystem, particle)
    average_pressure(system, system.transport_velocity, particle)
end

@inline function average_pressure(system, ::TransportVelocityAdami, particle)
    return system.cache.pressure_average[particle]
end

@inline average_pressure(system, ::Nothing, particle) = 0.0

@inline function v_nvariables(system::EntropicallyDampedSPHSystem)
    v_nvariables(system, system.transport_velocity)
end

@inline v_nvariables(system, ::Nothing) = ndims(system) + 1
@inline v_nvariables(system, ::TransportVelocityAdami) = ndims(system) * 2 + 1

@inline function add_velocity!(du, v, particle, system::EntropicallyDampedSPHSystem)
    add_velocity!(du, v, particle, system, system.transport_velocity)
end

@inline function add_velocity!(du, v, particle, system, ::Nothing)
    for i in 1:ndims(system)
        du[i, particle] = v[i, particle]
    end

    return du
end

@inline function add_velocity!(du, v, particle, system, ::TransportVelocityAdami)
    @unpack cache = system

    for i in 1:ndims(system)
        du[i, particle] = v[ndims(system) + i, particle]
        cache.advection_velocity[i, particle] = v[ndims(system) + i, particle]
    end

    return du
end

@inline function advection_velocity(v, system::EntropicallyDampedSPHSystem, particle)
    @unpack cache = system
    extract_svector(cache.advection_velocity, system, particle)
end

function update_quantities!(system::EntropicallyDampedSPHSystem, system_index, v, u,
                            v_ode, u_ode, semi, t)
    summation_density!(system, system_index, semi, u, u_ode, system.density)
end

function update_average_pressure!(system::EntropicallyDampedSPHSystem, system_index,
                                  v_ode, u_ode, semi)
    update_average_pressure!(system, system.transport_velocity, system_index,
                             v_ode, u_ode, semi)
end

function update_average_pressure!(system, ::Nothing, system_index, v_ode, u_ode, semi)
end

function update_average_pressure!(system, ::TransportVelocityAdami, system_index,
                                  v_ode, u_ode, semi)
    @unpack systems, neighborhood_searches = semi
    @unpack cache = system
    @unpack pressure_average, neighbor_counter = cache

    set_zero!(pressure_average)
    set_zero!(neighbor_counter)

    u = wrap_u(u_ode, system_index, system, semi)

    # Use all other systems for the average pressure
    @trixi_timeit timer() "compute average pressure" foreach_enumerate(systems) do (neighbor_system_index,
                                                                                    neighbor_system)
        u_neighbor_system = wrap_u(u_ode, neighbor_system_index, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system_index, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        neighborhood_search = neighborhood_searches[system_index][neighbor_system_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        for_particle_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                              neighborhood_search) do particle, neighbor, pos_diff, distance
            ## Only consider particles with a distance > 0.
            #distance < sqrt(eps()) && return

            pressure_average[particle] += particle_pressure(v_neighbor_system,
                                                            neighbor_system,
                                                            neighbor)
            neighbor_counter[particle] += 1
        end
    end

    for particle in eachparticle(system)
        if neighbor_counter[particle] > 0
            pressure_average[particle] /= neighbor_counter[particle]
        end
    end
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem)
    write_v0!(v0, system, system.transport_velocity)
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::Nothing)
    @unpack initial_condition = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
        v0[end, particle] = initial_pressure(system, particle)
    end

    return v0
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::TransportVelocityAdami)
    @unpack initial_condition = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
            v0[ndims(system) + dim, particle] = initial_condition.velocity[dim, particle]
        end
        v0[end, particle] = initial_pressure(system, particle)
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

@inline function initial_pressure(system, particle)
    initial_pressure(system, particle, system.pressure_function)
end

@inline function initial_pressure(system, particle, ::Nothing)
    return system.initial_condition.pressure[particle]
end

@inline function initial_pressure(system, particle, pressure_function)
    particle_position = initial_coords(system, particle)
    return pressure_function(particle_position)
end
