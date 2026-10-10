using TrixiParticles
using LinearAlgebra

struct NBodySystem{NDIMS, ELTYPE <: Real, IC, GR} <: TrixiParticles.AbstractSystem{NDIMS}
    initial_condition :: IC
    mass              :: Array{ELTYPE, 1} # [particle]
    # Kept for compatibility with n-body benchmark code that reads `system.G`.
    G       :: ELTYPE
    gravity :: GR
    buffer  :: Nothing

    function NBodySystem(initial_condition, gravity::NewtonianGravity)
        mass = copy(initial_condition.mass)
        mass_eltype = eltype(mass)
        gravitational_constant = convert(mass_eltype, gravity.gravitational_constant)
        softening_length = convert(mass_eltype, gravity.softening_length)
        cutoff_radius = convert(mass_eltype, gravity.cutoff_radius)
        gravity_ = NewtonianGravity(; gravitational_constant, softening_length,
                                    cutoff_radius)
        gravitational_constant = gravity_.gravitational_constant

        new{size(initial_condition.coordinates, 1),
            eltype(mass), typeof(initial_condition), typeof(gravity_)}(initial_condition,
                                                                       mass,
                                                                       gravitational_constant,
                                                                       gravity_,
                                                                       nothing)
    end
end

function NBodySystem(initial_condition, gravitational_constant)
    gravity = NewtonianGravity(; gravitational_constant)

    return NBodySystem(initial_condition, gravity)
end

TrixiParticles.timer_name(::NBodySystem) = "nbody"

@inline Base.eltype(system::NBodySystem{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

function TrixiParticles.compact_support(system::NBodySystem, ::NBodySystem)
    system.gravity.cutoff_radius
end

function TrixiParticles.write_u0!(u0, system::NBodySystem)
    u0 .= system.initial_condition.coordinates

    return u0
end

function TrixiParticles.write_v0!(v0, system::NBodySystem)
    v0 .= system.initial_condition.velocity

    return v0
end

# NHS update
function TrixiParticles.update_nhs!(neighborhood_search,
                                    system::NBodySystem, neighbor::NBodySystem,
                                    u_system, u_neighbor, semi)
    TrixiParticles.PointNeighbors.update!(neighborhood_search,
                                          u_system, u_neighbor,
                                          points_moving=(true, true))
end

function TrixiParticles.interact!(dv, v_particle_system, u_particle_system,
                                  v_neighbor_system, u_neighbor_system,
                                  particle_system::NBodySystem,
                                  neighbor_system::NBodySystem, semi)
    (; mass) = neighbor_system
    gravity = particle_system.gravity

    # Different parameters in the two ordered interactions would violate pairwise symmetry.
    if particle_system !== neighbor_system
        neighbor_gravity = neighbor_system.gravity
        if gravity.gravitational_constant != neighbor_gravity.gravitational_constant ||
           gravity.softening_length != neighbor_gravity.softening_length ||
           gravity.cutoff_radius != neighbor_gravity.cutoff_radius
            throw(ArgumentError("interacting `NBodySystem`s must use identical gravity " *
                                "parameters to preserve pairwise force symmetry"))
        end
    end

    system_coords = TrixiParticles.current_coordinates(u_particle_system, particle_system)
    neighbor_coords = TrixiParticles.current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    TrixiParticles.foreach_point_neighbor(particle_system, neighbor_system,
                                          system_coords, neighbor_coords,
                                          semi) do particle, neighbor, pos_diff, distance
        # No interaction of a particle with itself
        particle_system === neighbor_system && particle === neighbor && return

        if iszero(distance) && gravity isa NewtonianGravity{<:Real, false}
            throw(DomainError(distance,
                              "unsoftened Newtonian gravity is singular for " *
                              "distinct particles at the same position"))
        end

        factor = TrixiParticles.gravity_acceleration_factor(gravity, distance,
                                                            mass[neighbor])

        @inbounds for i in 1:ndims(particle_system)
            dv[i, particle] += factor * pos_diff[i]
        end
    end

    return dv
end

function energy(v_ode, u_ode, system, semi)
    (; mass) = system
    (; gravitational_constant, softening_length, cutoff_radius) = system.gravity

    # Shift a finite-cutoff potential to zero at the cutoff without changing its force.
    inverse_cutoff_distance = if isinf(cutoff_radius)
        zero(eltype(system))
    else
        inv(sqrt(cutoff_radius^2 + softening_length^2))
    end

    e = zero(eltype(system))

    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)

    for particle in TrixiParticles.eachparticle(system)
        e += 0.5 * mass[particle] *
             sum(TrixiParticles.current_velocity(v, system, particle) .^ 2)

        particle_coords = TrixiParticles.current_coords(u, system, particle)
        for neighbor in (particle + 1):TrixiParticles.nparticles(system)
            neighbor_coords = TrixiParticles.current_coords(u, system, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if distance <= cutoff_radius
                softened_distance = sqrt(distance^2 + softening_length^2)
                e -= gravitational_constant * mass[particle] * mass[neighbor] *
                     (inv(softened_distance) - inverse_cutoff_distance)
            end
        end
    end

    return e
end

TrixiParticles.vtkname(system::NBodySystem) = "n-body"

function TrixiParticles.write2vtk!(vtk, v, u, t, system::NBodySystem)
    (; mass) = system

    vtk["velocity"] = v
    vtk["mass"] = mass

    return vtk
end

function TrixiParticles.add_system_data!(system_data, system::NBodySystem)
    return system_data
end

function Base.show(io::IO, system::NBodySystem)
    print(io, "NBodySystem{", ndims(system), "}() with ")
    print(io, TrixiParticles.nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::NBodySystem)
    if get(io, :compact, false)
        show(io, system)
    else
        TrixiParticles.summary_header(io, "NBodySystem{$(ndims(system))}")
        TrixiParticles.summary_line(io, "#particles", TrixiParticles.nparticles(system))
        TrixiParticles.summary_footer(io)
    end
end
