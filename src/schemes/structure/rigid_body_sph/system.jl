@doc raw"""
    RigidSPHSystem(initial_condition;
                   boundary_model=nothing,
                   boundary_contact_model=nothing,
                   acceleration=ntuple(_ -> 0.0, ndims(initial_condition)),
                   angular_velocity=zero(eltype(initial_condition)),
                   particle_spacing=initial_condition.particle_spacing,
                   source_terms=nothing, color_value=0)

System for particles of a rigid structure.

The rigid body is represented by particles and advanced with rigid-body translation
and rotation. Fluid-structure interaction forces are reduced to resultant force and
torque and applied consistently to all rigid particles.

# Arguments
- `initial_condition`: Initial condition representing the rigid particles.

# Keywords
- `boundary_model`: Boundary model for fluid-structure interaction
                    (see [Boundary Models](@ref boundary_models)).
- `boundary_contact_model`: Optional rigid-wall contact model.
                            If specified, rigid-wall collisions with Coulomb friction are enabled.
- `acceleration`: Global acceleration vector applied to all rigid particles.
- `angular_velocity`: Initial angular velocity of the rigid body.
  For 2D, pass a scalar. For 3D, pass a vector of length 3.
- `particle_spacing`: Reference particle spacing used for time-step estimation.
- `source_terms`: Optional source terms of the form
                  `(coords, velocity, density, pressure, t) -> source`.
- `color_value`: The value used to initialize the color of particles in the system.
"""
struct RigidSPHSystem{BM, BCM, NDIMS, ELTYPE <: Real, IC, ARRAY1D, ARRAY2D,
                      AV, ST, C} <: AbstractStructureSystem{NDIMS}
    initial_condition::IC
    initial_velocity::ARRAY2D # [dimension, particle]
    local_coordinates::ARRAY2D # [dimension, particle]
    mass::ARRAY1D # [particle]
    material_density::ARRAY1D # [particle]
    acceleration::SVector{NDIMS, ELTYPE}
    initial_angular_velocity::AV
    particle_spacing::ELTYPE
    boundary_model::BM
    boundary_contact_model::BCM
    source_terms::ST
    cache::C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function RigidSPHSystem(initial_condition; boundary_model=nothing,
                        boundary_contact_model=nothing,
                        acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                            ndims(initial_condition)),
                        angular_velocity=zero(eltype(initial_condition)),
                        particle_spacing=initial_condition.particle_spacing,
                        source_terms=nothing, color_value=0)
    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)

    if NDIMS != 2 && NDIMS != 3
        throw(ArgumentError("`RigidSPHSystem` currently supports only 2D and 3D, got $(NDIMS)D"))
    end

    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    initial_angular_velocity = convert_angular_velocity(angular_velocity, Val(NDIMS),
                                                        ELTYPE)

    particle_spacing_ = convert(ELTYPE, particle_spacing)
    boundary_contact_model_ = convert_boundary_contact_model(boundary_contact_model,
                                                             particle_spacing_, ELTYPE)

    initial_velocity = copy(initial_condition.velocity)
    local_coordinates = copy(initial_condition.coordinates)
    mass = copy(initial_condition.mass)
    material_density = copy(initial_condition.density)

    center_of_mass,
    total_mass = center_of_mass_and_total_mass(local_coordinates, mass,
                                               Val(NDIMS), ELTYPE)
    update_relative_coordinates!(local_coordinates, local_coordinates,
                                 center_of_mass, Val(NDIMS))

    add_initial_rotation!(initial_velocity, local_coordinates,
                          initial_angular_velocity, Val(NDIMS))

    cache = create_cache_rigid(Val(NDIMS), ELTYPE, nparticles(initial_condition),
                               total_mass, center_of_mass, local_coordinates,
                               initial_angular_velocity, boundary_contact_model_,
                               color_value)

    return RigidSPHSystem(initial_condition, initial_velocity, local_coordinates,
                          mass, material_density, acceleration_,
                          initial_angular_velocity, particle_spacing_,
                          boundary_model, boundary_contact_model_, source_terms, cache)
end

function convert_angular_velocity(angular_velocity, ::Val{2}, ELTYPE)
    if angular_velocity isa Number
        return convert(ELTYPE, angular_velocity)
    end

    if angular_velocity isa Union{Tuple, AbstractArray} && length(angular_velocity) == 1
        return convert(ELTYPE, first(angular_velocity))
    end

    throw(ArgumentError("`angular_velocity` must be a scalar for a 2D problem"))
end

function convert_angular_velocity(angular_velocity, ::Val{3}, ELTYPE)
    if !(angular_velocity isa Union{Tuple, AbstractArray})
        throw(ArgumentError("`angular_velocity` must be of length 3 for a 3D problem"))
    end

    angular_velocity_ = SVector(angular_velocity...)
    if length(angular_velocity_) != 3
        throw(ArgumentError("`angular_velocity` must be of length 3 for a 3D problem"))
    end

    return SVector{3, ELTYPE}(angular_velocity_)
end

function center_of_mass_and_total_mass(coordinates, mass, ::Val{NDIMS},
                                       ELTYPE) where {NDIMS}
    center_of_mass = zero(SVector{NDIMS, ELTYPE})
    total_mass = zero(ELTYPE)

    for particle in eachindex(mass)
        particle_mass = convert(ELTYPE, mass[particle])
        total_mass += particle_mass
        center_of_mass += particle_mass * extract_svector(coordinates, Val(NDIMS), particle)
    end

    if total_mass <= eps(ELTYPE)
        throw(ArgumentError("`RigidSPHSystem` requires a positive total mass"))
    end

    return center_of_mass / total_mass, total_mass
end

function create_cache_rigid(::Val{2}, ELTYPE, n_particles, total_mass,
                            center_of_mass, local_coordinates,
                            initial_angular_velocity, boundary_contact_model, color_value)
    force_per_particle = zeros(ELTYPE, 2, n_particles)
    relative_coordinates = copy(local_coordinates)
    tangential_displacement = create_contact_tangential_displacement(boundary_contact_model,
                                                                     ELTYPE, Val(2))
    manifold_cache = create_contact_manifold_cache(Val(2), ELTYPE, n_particles)

    return (; color=Int(color_value), total_mass, force_per_particle,
            relative_coordinates, center_of_mass=Ref(center_of_mass),
            center_of_mass_velocity=Ref(zero(SVector{2, ELTYPE})),
            inertia=Ref(zero(ELTYPE)), inverse_inertia=Ref(zero(ELTYPE)),
            angular_velocity=Ref(initial_angular_velocity),
            resultant_force=Ref(zero(SVector{2, ELTYPE})),
            resultant_torque=Ref(zero(ELTYPE)),
            angular_acceleration_force=Ref(zero(ELTYPE)),
            gyroscopic_acceleration=Ref(zero(ELTYPE)),
            contact_tangential_displacement=tangential_displacement,
            boundary_contact_count=Ref(0),
            max_boundary_penetration=Ref(zero(ELTYPE)),
            manifold_cache...)
end

function create_cache_rigid(::Val{3}, ELTYPE, n_particles, total_mass,
                            center_of_mass, local_coordinates,
                            initial_angular_velocity, boundary_contact_model, color_value)
    force_per_particle = zeros(ELTYPE, 3, n_particles)
    relative_coordinates = copy(local_coordinates)
    tangential_displacement = create_contact_tangential_displacement(boundary_contact_model,
                                                                     ELTYPE, Val(3))
    manifold_cache = create_contact_manifold_cache(Val(3), ELTYPE, n_particles)

    return (; color=Int(color_value), total_mass, force_per_particle,
            relative_coordinates, center_of_mass=Ref(center_of_mass),
            center_of_mass_velocity=Ref(zero(SVector{3, ELTYPE})),
            inertia=Ref(zero(SMatrix{3, 3, ELTYPE, 9})),
            inverse_inertia=Ref(zero(SMatrix{3, 3, ELTYPE, 9})),
            angular_velocity=Ref(initial_angular_velocity),
            resultant_force=Ref(zero(SVector{3, ELTYPE})),
            resultant_torque=Ref(zero(SVector{3, ELTYPE})),
            angular_acceleration_force=Ref(zero(SVector{3, ELTYPE})),
            gyroscopic_acceleration=Ref(zero(SVector{3, ELTYPE})),
            contact_tangential_displacement=tangential_displacement,
            boundary_contact_count=Ref(0),
            max_boundary_penetration=Ref(zero(ELTYPE)),
            manifold_cache...)
end

function create_contact_manifold_cache(::Val{NDIMS}, ELTYPE,
                                       n_particles) where {NDIMS}
    max_manifolds = 8

    return (; contact_manifold_count=zeros(Int, n_particles),
            contact_manifold_weight_sum=zeros(ELTYPE, max_manifolds, n_particles),
            contact_manifold_penetration_sum=zeros(ELTYPE, max_manifolds, n_particles),
            contact_manifold_normal_sum=zeros(ELTYPE, NDIMS, max_manifolds, n_particles),
            contact_manifold_wall_velocity_sum=zeros(ELTYPE, NDIMS, max_manifolds,
                                                     n_particles),
            contact_manifold_tangential_displacement_sum=zeros(ELTYPE, NDIMS,
                                                               max_manifolds, n_particles))
end

function update_relative_coordinates!(relative_coordinates, coordinates, center_of_mass,
                                      ::Val{NDIMS}) where {NDIMS}
    for particle in axes(relative_coordinates, 2)
        relative_position = extract_svector(coordinates, Val(NDIMS), particle) -
                            center_of_mass
        for i in 1:NDIMS
            relative_coordinates[i, particle] = relative_position[i]
        end
    end

    return relative_coordinates
end

function add_initial_rotation!(initial_velocity, local_coordinates, angular_velocity,
                               ::Val{2})
    iszero(angular_velocity) && return initial_velocity

    for particle in axes(initial_velocity, 2)
        x = local_coordinates[1, particle]
        y = local_coordinates[2, particle]
        initial_velocity[1, particle] += -angular_velocity * y
        initial_velocity[2, particle] += angular_velocity * x
    end

    return initial_velocity
end

function add_initial_rotation!(initial_velocity, local_coordinates, angular_velocity,
                               ::Val{3})
    iszero(angular_velocity) && return initial_velocity

    for particle in axes(initial_velocity, 2)
        relative_position = extract_svector(local_coordinates, Val(3), particle)
        rotational_velocity = cross(angular_velocity, relative_position)

        for i in 1:3
            initial_velocity[i, particle] += rotational_velocity[i]
        end
    end

    return initial_velocity
end

@inline function Base.eltype(::RigidSPHSystem{<:Any, <:Any, <:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline function v_nvariables(system::RigidSPHSystem)
    return ndims(system)
end

@inline function v_nvariables(system::RigidSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return ndims(system) + 1
end

@inline function local_coordinates(system::RigidSPHSystem)
    return system.local_coordinates
end

@inline function particle_spacing(system::RigidSPHSystem, particle)
    return system.particle_spacing
end

@propagate_inbounds function current_velocity(v, system::RigidSPHSystem)
    # For `ContinuityDensity`, the density is stored in the last row of `v`.
    # Return only the velocity components for rigid systems.
    return view(v, 1:ndims(system), :)
end

@inline function current_density(v, system::RigidSPHSystem)
    return current_density(v, system.boundary_model, system)
end

@inline function current_density(v, ::Nothing, system::RigidSPHSystem)
    return system.material_density
end

# In fluid-structure interaction, use the hydrodynamic pressure corresponding to the
# configured boundary model.
@inline function current_pressure(v, system::RigidSPHSystem)
    return current_pressure(v, system.boundary_model, system)
end

@inline function current_pressure(v, ::Nothing, system::RigidSPHSystem)
    return zero(system.material_density)
end

@inline function hydrodynamic_mass(system::RigidSPHSystem, particle)
    return hydrodynamic_mass(system, system.boundary_model, particle)
end

@inline function hydrodynamic_mass(system::RigidSPHSystem, ::Nothing, particle)
    return system.mass[particle]
end

@inline function hydrodynamic_mass(system::RigidSPHSystem, boundary_model, particle)
    if hasproperty(boundary_model, :hydrodynamic_mass)
        return boundary_model.hydrodynamic_mass[particle]
    end

    return system.mass[particle]
end

@inline function viscous_velocity(v, system::RigidSPHSystem, particle)
    boundary_model = system.boundary_model

    if isnothing(boundary_model) || isnothing(boundary_model.viscosity)
        return current_velocity(v, system, particle)
    end

    return extract_svector(boundary_model.cache.wall_velocity, system, particle)
end

@inline function smoothing_length(system::RigidSPHSystem, particle)
    return smoothing_length(system.boundary_model, particle)
end

@inline function smoothing_length(system::RigidSPHSystem{Nothing}, particle)
    return system.particle_spacing
end

@inline function system_smoothing_kernel(system::RigidSPHSystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.smoothing_kernel
end

@inline function system_correction(system::RigidSPHSystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.correction
end

initialize!(system::RigidSPHSystem, semi) = system

function write_u0!(u0, system::RigidSPHSystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_condition.coordinates)
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

function write_v0!(v0, system::RigidSPHSystem)
    (; initial_velocity, boundary_model) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_velocity)
    copyto!(v0, indices, initial_velocity, indices)

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

    for particle in each_integrated_particle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::RigidSPHSystem, v, u)
    indices_u = CartesianIndices(system.initial_condition.coordinates)
    copyto!(system.initial_condition.coordinates, indices_u, u, indices_u)

    indices_v = CartesianIndices(system.initial_condition.velocity)
    copyto!(system.initial_condition.velocity, indices_v,
            view(v, 1:ndims(system), :), indices_v)
    copyto!(system.initial_velocity, indices_v,
            view(v, 1:ndims(system), :), indices_v)

    center_of_mass,
    _ = center_of_mass_and_total_mass(system.initial_condition.coordinates,
                                      system.mass, Val(ndims(system)),
                                      eltype(system))
    update_relative_coordinates!(system.local_coordinates,
                                 system.initial_condition.coordinates,
                                 center_of_mass, Val(ndims(system)))
    copyto!(system.cache.relative_coordinates, system.local_coordinates)
    system.cache.center_of_mass[] = center_of_mass
    if !isnothing(system.cache.contact_tangential_displacement)
        empty!(system.cache.contact_tangential_displacement)
    end
    system.cache.boundary_contact_count[] = 0
    system.cache.max_boundary_penetration[] = zero(eltype(system))

    return system
end

function update_boundary_interpolation!(system::RigidSPHSystem, v, u, v_ode, u_ode,
                                        semi, t)
    (; boundary_model) = system

    isnothing(boundary_model) && return system

    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

function update_final!(system::RigidSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; cache) = system
    total_mass = cache.total_mass
    total_mass <= eps(eltype(system)) && return system

    system_coords = current_coordinates(u, system)
    system_velocity = current_velocity(v, system)

    center_of_mass = zero(SVector{ndims(system), eltype(system)})
    center_of_mass_velocity = zero(SVector{ndims(system), eltype(system)})

    for particle in each_integrated_particle(system)
        particle_mass = system.mass[particle]
        center_of_mass += particle_mass * extract_svector(system_coords, system, particle)
        center_of_mass_velocity += particle_mass *
                                   extract_svector(system_velocity, system, particle)
    end

    center_of_mass /= total_mass
    center_of_mass_velocity /= total_mass

    cache.center_of_mass[] = center_of_mass
    cache.center_of_mass_velocity[] = center_of_mass_velocity

    update_relative_coordinates!(cache.relative_coordinates, system_coords,
                                 center_of_mass, Val(ndims(system)))
    update_rotational_kinematics!(system, system_velocity, center_of_mass_velocity,
                                  Val(ndims(system)))

    return system
end

function update_rotational_kinematics!(system::RigidSPHSystem, system_velocity,
                                       center_of_mass_velocity, ::Val{2})
    (; cache) = system
    inertia = zero(eltype(system))
    angular_momentum = zero(eltype(system))

    for particle in each_integrated_particle(system)
        particle_mass = system.mass[particle]
        relative_position = extract_svector(cache.relative_coordinates, system, particle)
        relative_velocity = extract_svector(system_velocity, system, particle) -
                            center_of_mass_velocity
        inertia += particle_mass * dot(relative_position, relative_position)
        angular_momentum += particle_mass * cross(relative_position, relative_velocity)
    end

    inverse_inertia = inertia > eps(eltype(system)) ? inv(inertia) : zero(inertia)
    angular_velocity = inverse_inertia * angular_momentum

    cache.inertia[] = inertia
    cache.inverse_inertia[] = inverse_inertia
    cache.angular_velocity[] = angular_velocity
    cache.gyroscopic_acceleration[] = zero(eltype(system))

    return system
end

function update_rotational_kinematics!(system::RigidSPHSystem, system_velocity,
                                       center_of_mass_velocity, ::Val{3})
    (; cache) = system
    inertia = zero(SMatrix{3, 3, eltype(system), 9})
    angular_momentum = zero(SVector{3, eltype(system)})

    for particle in each_integrated_particle(system)
        particle_mass = system.mass[particle]
        relative_position = extract_svector(cache.relative_coordinates, system, particle)
        relative_velocity = extract_svector(system_velocity, system, particle) -
                            center_of_mass_velocity
        inertia += particle_mass * inertia_tensor(relative_position)
        angular_momentum += particle_mass * cross(relative_position, relative_velocity)
    end

    inverse_inertia = inverse_inertia_tensor(inertia)
    angular_velocity = inverse_inertia * angular_momentum
    gyroscopic_acceleration = inverse_inertia * cross(angular_velocity,
                                    inertia * angular_velocity)

    cache.inertia[] = inertia
    cache.inverse_inertia[] = inverse_inertia
    cache.angular_velocity[] = angular_velocity
    cache.gyroscopic_acceleration[] = gyroscopic_acceleration

    return system
end

@inline function inertia_tensor(relative_position)
    # Built-in expression of r^2 * I - r*r^T for improved readability.
    return dot(relative_position, relative_position) * I -
           relative_position * transpose(relative_position)
end

function inverse_inertia_tensor(inertia::SMatrix{3, 3, ELTYPE, 9}) where {ELTYPE}
    inertia_determinant = det(inertia)
    inertia_scale = max(one(ELTYPE), norm(inertia, Inf))
    determinant_tolerance = eps(ELTYPE) * inertia_scale^3

    if !isfinite(inertia_determinant) || abs(inertia_determinant) <= determinant_tolerance
        return zero(inertia)
    end

    return inv(inertia)
end

function calculate_dt(v_ode, u_ode, cfl_number, system::RigidSPHSystem, semi)
    acceleration_norm = norm(system.acceleration)
    spacing = particle_spacing(system, first(eachparticle(system)))
    contact_dt = contact_time_step(system)

    if acceleration_norm <= eps(eltype(system)) || !isfinite(spacing) || spacing <= 0
        return cfl_number * contact_dt
    end

    acceleration_dt = cfl_number * spacing / acceleration_norm

    return min(acceleration_dt, cfl_number * contact_dt)
end

# To account for boundary effects in the viscosity term of fluid-structure interactions,
# use the viscosity model of the neighboring system.
@inline function viscosity_model(system::RigidSPHSystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::Union{AbstractFluidSystem, OpenBoundarySystem},
                                 neighbor_system::RigidSPHSystem)
    if isnothing(neighbor_system.boundary_model)
        return nothing
    end

    return neighbor_system.boundary_model.viscosity
end

@inline acceleration_source(system::RigidSPHSystem) = system.acceleration

@inline function add_acceleration!(dv, particle, system::RigidSPHSystem)
    relative_position = extract_svector(system.cache.relative_coordinates, system, particle)
    rotational_acceleration = rigid_kinematic_acceleration(system.cache, relative_position,
                                                           Val(ndims(system)))

    for i in 1:ndims(system)
        dv[i, particle] += system.acceleration[i] + rotational_acceleration[i]
    end

    return dv
end

@inline function rigid_kinematic_acceleration(cache, relative_position, ::Val{2})
    angular_velocity = cache.angular_velocity[]

    return -(angular_velocity^2) * relative_position
end

@inline function rigid_kinematic_acceleration(cache, relative_position, ::Val{3})
    angular_velocity = cache.angular_velocity[]
    gyroscopic_acceleration = cache.gyroscopic_acceleration[]

    centripetal_acceleration = cross(angular_velocity,
                                     cross(angular_velocity, relative_position))
    gyroscopic_correction = cross(gyroscopic_acceleration, relative_position)

    return centripetal_acceleration - gyroscopic_correction
end

function system_data(system::RigidSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    acceleration = current_velocity(dv, system)
    density = current_density(v, system)
    pressure = current_pressure(v, system)
    center_of_mass = system.cache.center_of_mass[]
    center_of_mass_velocity = system.cache.center_of_mass_velocity[]
    angular_velocity = system.cache.angular_velocity[]
    resultant_force = system.cache.resultant_force[]
    resultant_torque = system.cache.resultant_torque[]
    angular_acceleration_force = system.cache.angular_acceleration_force[]
    gyroscopic_acceleration = system.cache.gyroscopic_acceleration[]
    boundary_contact_count = system.cache.boundary_contact_count[]
    max_boundary_penetration = system.cache.max_boundary_penetration[]
    relative_coordinates = system.cache.relative_coordinates

    return (; coordinates, velocity, mass=system.mass,
            material_density=system.material_density,
            local_coordinates=system.local_coordinates,
            relative_coordinates,
            center_of_mass, center_of_mass_velocity,
            angular_velocity,
            resultant_force, resultant_torque,
            angular_acceleration_force, gyroscopic_acceleration,
            boundary_contact_count, max_boundary_penetration,
            density, pressure, acceleration)
end

function available_data(::RigidSPHSystem)
    return (:coordinates, :velocity, :mass, :material_density,
            :local_coordinates, :relative_coordinates,
            :center_of_mass, :center_of_mass_velocity,
            :angular_velocity, :resultant_force, :resultant_torque,
            :angular_acceleration_force, :gyroscopic_acceleration,
            :boundary_contact_count, :max_boundary_penetration,
            :density, :pressure, :acceleration)
end

function Base.show(io::IO, system::RigidSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "RigidSPHSystem{", ndims(system), "}(")
    print(io, system.acceleration)
    print(io, ", ", system.boundary_model)
    print(io, ", ", system.boundary_contact_model)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::RigidSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "RigidSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "initial angular velocity", system.initial_angular_velocity)
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "boundary contact model", system.boundary_contact_model)
        summary_footer(io)
    end
end
