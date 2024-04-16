@doc raw"""
    RigidSPHSystem(initial_condition;
                             boundary_model=nothing,
                             acceleration=ntuple(_ -> 0.0, NDIMS))

System for a SPH particles of a rigid structure.

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.

# Keyword Arguments
- `boundary_model`: Boundary model to compute the hydrodynamic density and pressure for
                    fluid-structure interaction (see [Boundary Models](@ref boundary_models)).
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
"""
struct RigidSPHSystem{BM, NDIMS, ELTYPE <: Real} <: SolidSystem{NDIMS}
    initial_condition :: InitialCondition{ELTYPE}
    local_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    mass              :: Array{ELTYPE, 1} # [particle]
    material_density  :: Array{ELTYPE, 1} # [particle]
    acceleration      :: SVector{NDIMS, ELTYPE}
    center_of_gravity :: Array{ELTYPE, 1} # [dimension]
    collision_impulse :: Array{ELTYPE, 1} # [dimension]
    collision_u       :: Array{ELTYPE, 1} # [dimension]
    particle_spacing  :: ELTYPE
    has_collided      :: MutableBool
    boundary_model    :: BM

    function RigidSPHSystem(initial_condition; boundary_model=nothing,
                            acceleration=ntuple(_ -> 0.0, ndims(initial_condition)),
                            particle_spacing=NaN)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        local_coordinates = copy(initial_condition.coordinates)
        mass = copy(initial_condition.mass)
        material_density = copy(initial_condition.density)

        # TODO: calculate center of gravity
        cog = zeros(SVector{NDIMS, ELTYPE})
        collision_impulse = zeros(SVector{NDIMS, ELTYPE})
        collision_u = zeros(SVector{NDIMS, ELTYPE})

        return new{typeof(boundary_model), NDIMS, ELTYPE}(initial_condition,
                                                          local_coordinates,
                                                          mass, material_density,
                                                          acceleration_, cog,
                                                          collision_impulse,
                                                          collision_u,
                                                          particle_spacing,
                                                          MutableBool(false),
                                                          boundary_model)
    end
end

# Info output functions

function Base.show(io::IO, system::RigidSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "RigidSPHSystem{", ndims(system), "}(")
    print(io, ", ", system.acceleration)
    print(io, ", ", system.boundary_model)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::RigidSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "RigidSPHSystem{$(ndims(system))}")
        summary_line(io, "total #particles", nparticles(system))
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "boundary model", system.boundary_model)
        summary_footer(io)
    end
end

# Memory allocation accessors

@inline function v_nvariables(system::RigidSPHSystem)
    return ndims(system)
end

@inline function v_nvariables(system::RigidSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return ndims(system) + 1
end

# Variable accessors

@inline function n_moving_particles(system::RigidSPHSystem)
    return nparticles(system)
end

@inline local_coordinates(system::RigidSPHSystem) = system.local_coordinates

# @inline function current_coordinates(u, system::RigidSPHSystem)
#     return system.coordinates
# end

@inline function current_velocity(v, system::RigidSPHSystem, particle)
    return extract_svector(v, system, particle)
end

@inline function viscous_velocity(v, system::RigidSPHSystem, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@inline function particle_density(v, system::RigidSPHSystem, particle)
    return particle_density(v, system.boundary_model, system, particle)
end

# In fluid-solid interaction, use the "hydrodynamic pressure" of the solid particles
# corresponding to the chosen boundary model.
@inline function particle_pressure(v, system::RigidSPHSystem, particle)
    return particle_pressure(v, system.boundary_model, system, particle)
end

@inline function hydrodynamic_mass(system::RigidSPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

function initialize!(system::RigidSPHSystem, neighborhood_search)
end

function write_u0!(u0, system::RigidSPHSystem)
    (; initial_condition) = system

    for particle in each_moving_particle(system)
        # Write particle coordinates
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_condition.coordinates[dim, particle]
        end
    end

    return u0
end

function write_v0!(v0, system::RigidSPHSystem)
    (; initial_condition, boundary_model) = system

    for particle in each_moving_particle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    write_v0!(v0, boundary_model, system)

    return v0
end

function write_v0!(v0, model, system::RigidSPHSystem)
    return v0
end

function write_v0!(v0, ::BoundaryModelDummyParticles{ContinuityDensity},
                   system::RigidSPHSystem)
    (; cache) = system.boundary_model
    (; initial_density) = cache

    for particle in each_moving_particle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_density[particle]
    end

    return v0
end

# Update functions
# function update_positions!(system::RigidSPHSystem, v, u, v_ode, u_ode, semi, t)
#     # (; movement) = system

#     # movement(system, t)
# end

# function update_quantities!(system::RigidSPHSystem, v, u, v_ode, u_ode, semi, t)
#     return system
# end

function update_final!(system::RigidSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; boundary_model) = system

    # Only update boundary model
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)
end

function viscosity_model(system::RigidSPHSystem)
    return system.boundary_model.viscosity
end
