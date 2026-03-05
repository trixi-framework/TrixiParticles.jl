@doc raw"""
    RigidSPHSystem(initial_condition;
                   boundary_model=nothing,
                   acceleration=ntuple(_ -> 0.0, ndims(initial_condition)),
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
- `acceleration`: Global acceleration vector applied to all rigid particles.
- `initial_condition.angular_velocity`: Initial angular velocity `ω` of the rigid body
  (not angular momentum). In 2D, pass a scalar angular speed in rad/s.
  In 3D, pass a vector of length 3: direction gives the rotation axis
  (right-hand rule), and `|ω|` gives angular speed in rad/s.
- `particle_spacing`: Reference particle spacing used for time-step estimation.
- `source_terms`: Optional source terms of the form
                  `(coords, velocity, density, pressure, t) -> source`.
- `color_value`: The value used to initialize the color of particles in the system.
"""
struct RigidSPHSystem{BM, NDIMS, ELTYPE <: Real, IC, ARRAY1D, ARRAY2D,
                      ST, CM, CMV, I, II, AV, RF, RT, AAF, GA, C} <:
       AbstractStructureSystem{NDIMS}
    initial_condition          :: IC
    initial_velocity           :: ARRAY2D # [dimension, particle]
    local_coordinates          :: ARRAY2D # [dimension, particle]
    mass                       :: ARRAY1D # [particle]
    material_density           :: ARRAY1D # [particle]
    acceleration               :: SVector{NDIMS, ELTYPE}
    particle_spacing           :: ELTYPE
    total_mass                 :: ELTYPE
    force_per_particle         :: ARRAY2D
    relative_coordinates       :: ARRAY2D
    center_of_mass             :: CM
    center_of_mass_velocity    :: CMV
    inertia                    :: I
    inverse_inertia            :: II
    angular_velocity           :: AV
    resultant_force            :: RF
    resultant_torque           :: RT
    angular_acceleration_force :: AAF
    gyroscopic_acceleration    :: GA
    boundary_model             :: BM
    source_terms               :: ST
    cache                      :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function RigidSPHSystem(initial_condition; boundary_model=nothing,
                        acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                            ndims(initial_condition)),
                        particle_spacing=initial_condition.particle_spacing,
                        source_terms=nothing, color_value=0)
    NDIMS = ndims(initial_condition)

    if NDIMS == 2
        return RigidSPHSystem(initial_condition, Val(2);
                              boundary_model=boundary_model,
                              acceleration=acceleration,
                              particle_spacing=particle_spacing,
                              source_terms=source_terms,
                              color_value=color_value)
    elseif NDIMS == 3
        return RigidSPHSystem(initial_condition, Val(3);
                              boundary_model=boundary_model,
                              acceleration=acceleration,
                              particle_spacing=particle_spacing,
                              source_terms=source_terms,
                              color_value=color_value)
    end

    throw(ArgumentError("`RigidSPHSystem` currently supports only 2D and 3D, got $(NDIMS)D"))
end

function RigidSPHSystem(initial_condition, ::Val{2}; boundary_model=nothing,
                        acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                            ndims(initial_condition)),
                        particle_spacing=initial_condition.particle_spacing,
                        source_terms=nothing, color_value=0)
    ELTYPE = eltype(initial_condition)
    init = init_rigid_system(initial_condition, acceleration, particle_spacing, Val(2))

    force_per_particle = zeros(ELTYPE, 2, nparticles(initial_condition))
    relative_coordinates = copy(init.local_coordinates)

    return RigidSPHSystem(initial_condition, init.initial_velocity, init.local_coordinates,
                          init.mass, init.material_density, init.acceleration,
                          init.particle_spacing, init.total_mass, force_per_particle,
                          relative_coordinates, Ref(init.center_of_mass),
                          Ref(zero(SVector{2, ELTYPE})),
                          Ref(zero(ELTYPE)), Ref(zero(ELTYPE)),
                          Ref(initial_condition.angular_velocity),
                          Ref(zero(SVector{2, ELTYPE})),
                          Ref(zero(ELTYPE)), Ref(zero(ELTYPE)), Ref(zero(ELTYPE)),
                          boundary_model, source_terms, create_cache_rigid(color_value))
end

function RigidSPHSystem(initial_condition, ::Val{3}; boundary_model=nothing,
                        acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                            ndims(initial_condition)),
                        particle_spacing=initial_condition.particle_spacing,
                        source_terms=nothing, color_value=0)
    ELTYPE = eltype(initial_condition)
    init = init_rigid_system(initial_condition, acceleration, particle_spacing, Val(3))

    force_per_particle = zeros(ELTYPE, 3, nparticles(initial_condition))
    relative_coordinates = copy(init.local_coordinates)

    return RigidSPHSystem(initial_condition, init.initial_velocity, init.local_coordinates,
                          init.mass, init.material_density, init.acceleration,
                          init.particle_spacing, init.total_mass, force_per_particle,
                          relative_coordinates, Ref(init.center_of_mass),
                          Ref(zero(SVector{3, ELTYPE})),
                          Ref(zero(SMatrix{3, 3, ELTYPE, 9})),
                          Ref(zero(SMatrix{3, 3, ELTYPE, 9})),
                          Ref(initial_condition.angular_velocity),
                          Ref(zero(SVector{3, ELTYPE})),
                          Ref(zero(SVector{3, ELTYPE})),
                          Ref(zero(SVector{3, ELTYPE})),
                          Ref(zero(SVector{3, ELTYPE})),
                          boundary_model, source_terms, create_cache_rigid(color_value))
end

function init_rigid_system(initial_condition, acceleration, particle_spacing,
                           ::Val{NDIMS}) where {NDIMS}
    if ndims(initial_condition) != NDIMS
        throw(ArgumentError("`initial_condition` dimensionality must be $NDIMS for this constructor"))
    end

    ELTYPE = eltype(initial_condition)
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    particle_spacing_ = convert(ELTYPE, particle_spacing)
    initial_velocity = copy(initial_condition.velocity)
    local_coordinates = copy(initial_condition.coordinates)
    mass = copy(initial_condition.mass)
    material_density = copy(initial_condition.density)

    center_of_mass,
    total_mass = center_of_mass_and_total_mass(local_coordinates, mass,
                                               Val(NDIMS), ELTYPE)
    update_relative_coordinates!(local_coordinates, local_coordinates,
                                 center_of_mass, Val(NDIMS))

    return (; acceleration=acceleration_,
            particle_spacing=particle_spacing_,
            initial_velocity, local_coordinates, mass, material_density,
            center_of_mass, total_mass)
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

create_cache_rigid(color_value) = (; color=Int(color_value))

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

@inline function Base.eltype(::RigidSPHSystem{<:Any, <:Any, ELTYPE}) where {ELTYPE}
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
    return zero(eltype(system))
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
    return viscous_velocity(v, system.boundary_model, system, particle)
end

@inline function viscous_velocity(v, ::Nothing, system::RigidSPHSystem, particle)
    return current_velocity(v, system, particle)
end

@inline function viscous_velocity(v, boundary_model, system::RigidSPHSystem, particle)
    return viscous_velocity(v, boundary_model.viscosity, boundary_model, system, particle)
end

@inline function viscous_velocity(v, ::Nothing, boundary_model,
                                  system::RigidSPHSystem, particle)
    return current_velocity(v, system, particle)
end

@inline function viscous_velocity(v, viscosity, boundary_model,
                                  system::RigidSPHSystem, particle)
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
    copyto!(system.relative_coordinates, system.local_coordinates)
    system.center_of_mass[] = center_of_mass

    return system
end

function update_boundary_interpolation!(system::RigidSPHSystem, v, u, v_ode, u_ode,
                                        semi, t)
    return update_boundary_interpolation!(system.boundary_model, system, v, u, v_ode,
                                          u_ode, semi, t)
end

function update_boundary_interpolation!(::Nothing, system::RigidSPHSystem, v, u, v_ode,
                                        u_ode, semi, t)
    return system
end

function update_boundary_interpolation!(boundary_model, system::RigidSPHSystem, v, u,
                                        v_ode, u_ode, semi, t)
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)
    return system
end

function update_final!(system::RigidSPHSystem, v, u, v_ode, u_ode, semi, t)
    total_mass = system.total_mass
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

    system.center_of_mass[] = center_of_mass
    system.center_of_mass_velocity[] = center_of_mass_velocity

    update_relative_coordinates!(system.relative_coordinates, system_coords,
                                 center_of_mass, Val(ndims(system)))
    update_rotational_kinematics!(system, system_velocity, center_of_mass_velocity,
                                  Val(ndims(system)))

    return system
end

function update_rotational_kinematics!(system::RigidSPHSystem, system_velocity,
                                       center_of_mass_velocity, ::Val{NDIMS}) where {NDIMS}
    inertia = zero(system.inertia[])
    angular_momentum = zero(system.angular_velocity[])

    for particle in each_integrated_particle(system)
        particle_mass = system.mass[particle]
        relative_position = extract_svector(system.relative_coordinates, system, particle)
        relative_velocity = extract_svector(system_velocity, system, particle) -
                            center_of_mass_velocity
        inertia += particle_mass * inertia_contribution(relative_position)
        angular_momentum += particle_mass * cross(relative_position, relative_velocity)
    end

    inverse_inertia = inverse_inertia_tensor(inertia)
    angular_velocity = inverse_inertia * angular_momentum
    gyroscopic_acceleration = gyroscopic_acceleration_term(inertia, inverse_inertia,
                                                           angular_velocity)

    system.inertia[] = inertia
    system.inverse_inertia[] = inverse_inertia
    system.angular_velocity[] = angular_velocity
    system.gyroscopic_acceleration[] = gyroscopic_acceleration

    return system
end

@inline function inertia_contribution(relative_position::SVector{NDIMS, ELTYPE}) where {NDIMS,
                                                                                        ELTYPE}
    return dot(relative_position, relative_position)
end

@inline function inertia_contribution(relative_position::SVector{3, ELTYPE}) where {ELTYPE}
    return inertia_tensor(relative_position)
end

@inline function inertia_tensor(relative_position)
    # Built-in expression of r^2 * I - r*r^T for improved readability.
    return dot(relative_position, relative_position) * I -
           relative_position * transpose(relative_position)
end

@inline function inverse_inertia_tensor(inertia::ELTYPE) where {ELTYPE <: Real}
    return inertia > eps(ELTYPE) ? inv(inertia) : zero(inertia)
end

function inverse_inertia_tensor(inertia::SMatrix{3, 3, ELTYPE, 9}) where {ELTYPE}
    inertia_determinant = det(inertia)
    inertia_scale = max(one(ELTYPE), norm(inertia, Inf))
    determinant_tolerance = eps(inertia_scale^3)

    if !isfinite(inertia_determinant) || abs(inertia_determinant) <= determinant_tolerance
        return zero(inertia)
    end

    return inv(inertia)
end

@inline function gyroscopic_acceleration_term(inertia::ELTYPE, inverse_inertia::ELTYPE,
                                              angular_velocity::ELTYPE) where {ELTYPE}
    return zero(angular_velocity)
end

@inline function gyroscopic_acceleration_term(inertia::SMatrix{3, 3, ELTYPE, 9},
                                              inverse_inertia::SMatrix{3, 3, ELTYPE, 9},
                                              angular_velocity::SVector{3, ELTYPE}) where {ELTYPE}
    return inverse_inertia * cross(angular_velocity, inertia * angular_velocity)
end

function calculate_dt(v_ode, u_ode, cfl_number, system::RigidSPHSystem, semi)
    spacing = particle_spacing(system, first(eachparticle(system)))

    radius = maximum_particle_radius(system)
    angular_speed = norm(system.angular_velocity[])
    angular_acceleration = norm(system.angular_acceleration_force[])

    total_mass = system.total_mass
    translational_acceleration = system.acceleration
    if total_mass > eps(eltype(system))
        translational_acceleration += system.resultant_force[] / total_mass
    end

    # Simple rigid-body scales:
    # acceleration ~ a_trans + r*(ω² + α), velocity ~ v_com + r*ω.
    acceleration_scale = norm(translational_acceleration) +
                         radius * (angular_speed^2 + angular_acceleration)
    dt_acceleration = acceleration_scale <= eps(eltype(system)) ? Inf :
                      0.25 * sqrt(spacing / acceleration_scale)

    speed_scale = norm(system.center_of_mass_velocity[]) + radius * angular_speed
    dt_velocity = speed_scale <= eps(eltype(system)) ? Inf :
                  cfl_number * spacing / speed_scale

    return min(dt_acceleration, dt_velocity)
end

function maximum_particle_radius(system::RigidSPHSystem)
    max_radius = zero(eltype(system))

    for particle in each_integrated_particle(system)
        relative_position = extract_svector(system.relative_coordinates, system, particle)
        max_radius = max(max_radius, norm(relative_position))
    end

    return max_radius
end

# To account for boundary effects in the viscosity term of fluid-structure interactions,
# use the viscosity model of the neighboring system.
@inline function viscosity_model(system::RigidSPHSystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::Union{AbstractFluidSystem, OpenBoundarySystem},
                                 neighbor_system::RigidSPHSystem{Nothing})
    return nothing
end

@inline function viscosity_model(system::Union{AbstractFluidSystem, OpenBoundarySystem},
                                 neighbor_system::RigidSPHSystem)
    return neighbor_system.boundary_model.viscosity
end

@inline acceleration_source(system::RigidSPHSystem) = system.acceleration

@inline function add_acceleration!(dv, particle, system::RigidSPHSystem)
    relative_position = extract_svector(system.relative_coordinates, system, particle)
    rotational_acceleration = rigid_kinematic_acceleration(system, relative_position,
                                                           Val(ndims(system)))

    for i in 1:ndims(system)
        dv[i, particle] += system.acceleration[i] + rotational_acceleration[i]
    end

    return dv
end

@inline function rigid_kinematic_acceleration(system::RigidSPHSystem, relative_position,
                                              ::Val{2})
    angular_velocity = system.angular_velocity[]

    return -(angular_velocity^2) * relative_position
end

@inline function rigid_kinematic_acceleration(system::RigidSPHSystem, relative_position,
                                              ::Val{3})
    angular_velocity = system.angular_velocity[]
    gyroscopic_acceleration = system.gyroscopic_acceleration[]

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
    center_of_mass = system.center_of_mass[]
    center_of_mass_velocity = system.center_of_mass_velocity[]
    angular_velocity = system.angular_velocity[]
    resultant_force = system.resultant_force[]
    resultant_torque = system.resultant_torque[]
    angular_acceleration_force = system.angular_acceleration_force[]
    gyroscopic_acceleration = system.gyroscopic_acceleration[]
    relative_coordinates = system.relative_coordinates

    return (; coordinates, velocity, mass=system.mass,
            material_density=system.material_density,
            local_coordinates=system.local_coordinates,
            relative_coordinates,
            center_of_mass, center_of_mass_velocity,
            angular_velocity,
            resultant_force, resultant_torque,
            angular_acceleration_force, gyroscopic_acceleration,
            density, pressure, acceleration)
end

function available_data(::RigidSPHSystem)
    return (:coordinates, :velocity, :mass, :material_density,
            :local_coordinates, :relative_coordinates,
            :center_of_mass, :center_of_mass_velocity,
            :angular_velocity, :resultant_force, :resultant_torque,
            :angular_acceleration_force, :gyroscopic_acceleration,
            :density, :pressure, :acceleration)
end

function Base.show(io::IO, system::RigidSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "RigidSPHSystem{", ndims(system), "}(")
    print(io, system.acceleration)
    print(io, ", ", system.boundary_model)
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
        summary_line(io, "initial angular velocity",
                     system.initial_condition.angular_velocity)
        summary_line(io, "boundary model", system.boundary_model)
        summary_footer(io)
    end
end
