@doc raw"""
    RigidBodySystem(initial_condition;
                   boundary_model=nothing,
                   acceleration=ntuple(_ -> 0.0, ndims(initial_condition)),
                   particle_spacing=initial_condition.particle_spacing,
                   source_terms=nothing, adhesion_coefficient=0.0,
                   color_value=0)

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
- `particle_spacing`: Reference particle spacing used for time-step estimation.
- `source_terms`: Optional source terms of the form
                  `(coords, velocity, density, pressure, t) -> source`.
- `adhesion_coefficient`: Wall-adhesion strength used by Akinci-type surface tension
                          models when fluids interact with this rigid body. This is
                          only evaluated for fluid-structure interaction with
                          surface-tension-enabled fluid systems.
- `color_value`: Integer label stored as `system.cache.color`.
                 Currently this is used with `BoundaryModelDummyParticles` during
                 colorfield initialization so fluids using
                 [`ColorfieldSurfaceNormal`](@ref) can detect contact with rigid
                 bodies, it participates in the multi-system color sanity check for
                 surface-tension setups, and it is written to VTK output as `"color"`.
"""
struct RigidBodySystem{BM, NDIMS, ELTYPE <: Real, IC, ARRAY1D, ARRAY2D,
                      ST, CM, CMV, I, II, AV, RF, RT, AAF, GA, C} <:
       AbstractStructureSystem{NDIMS}
    initial_condition          :: IC
    initial_velocity           :: ARRAY2D # [dimension, particle]
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
    adhesion_coefficient       :: ELTYPE
    cache                      :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function RigidBodySystem(initial_condition; boundary_model=nothing,
                        acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                            ndims(initial_condition)),
                        particle_spacing=initial_condition.particle_spacing,
                        source_terms=nothing, adhesion_coefficient=0.0,
                        color_value=0)
    NDIMS = ndims(initial_condition)
    if NDIMS != 2 && NDIMS != 3
        throw(ArgumentError("`RigidBodySystem` currently supports only 2D and 3D, got $(NDIMS)D"))
    end

    ELTYPE = eltype(initial_condition)
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    particle_spacing_ = convert(ELTYPE, particle_spacing)
    initial_velocity = copy(initial_condition.velocity)
    relative_coordinates = copy(initial_condition.coordinates)
    mass = copy(initial_condition.mass)
    material_density = copy(initial_condition.density)

    center_of_mass,
    total_mass = center_of_mass_and_total_mass(relative_coordinates, mass,
                                               Val(NDIMS), ELTYPE)
    update_relative_coordinates!(relative_coordinates, relative_coordinates,
                                 center_of_mass, Val(NDIMS))

    force_per_particle = zeros(ELTYPE, NDIMS, nparticles(initial_condition))
    zero_rotational_quantity = NDIMS == 2 ? zero(ELTYPE) : zero(SVector{3, ELTYPE})
    if NDIMS == 2
        inertia = Ref(zero(ELTYPE))
        inverse_inertia = Ref(zero(ELTYPE))
    else # NDIMS == 3
        inertia = Ref(zero(SMatrix{3, 3, ELTYPE, 9}))
        inverse_inertia = Ref(zero(SMatrix{3, 3, ELTYPE, 9}))
    end

    center_of_mass_velocity = zero(SVector{NDIMS, ELTYPE})
    for particle in eachindex(mass)
        particle_mass = convert(ELTYPE, mass[particle])
        center_of_mass_velocity += particle_mass *
                                   extract_svector(initial_velocity, Val(NDIMS), particle)
    end
    center_of_mass_velocity /= total_mass

    system = RigidBodySystem(initial_condition, initial_velocity, mass,
                            material_density, acceleration_,
                            particle_spacing_, total_mass, force_per_particle,
                            relative_coordinates, Ref(center_of_mass),
                            Ref(center_of_mass_velocity),
                            inertia, inverse_inertia,
                            Ref(zero_rotational_quantity),
                            Ref(zero(SVector{NDIMS, ELTYPE})),
                            Ref(zero_rotational_quantity),
                            Ref(zero_rotational_quantity),
                            Ref(zero_rotational_quantity),
                            boundary_model, source_terms,
                            convert(ELTYPE, adhesion_coefficient),
                            create_cache_rigid(color_value))

    # Initialize rotational kinematics consistently with the initial velocity field.
    update_rotational_kinematics!(system, initial_velocity, center_of_mass_velocity,
                                  Val(NDIMS))

    return system
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
        throw(ArgumentError("`RigidBodySystem` requires a positive total mass"))
    end

    return center_of_mass / total_mass, total_mass
end

# Per-system color tag for colorfield surface-normal logic and VTK output.
create_cache_rigid(color_value) = (; color=Int(color_value))

function update_relative_coordinates!(relative_coordinates, coordinates, center_of_mass,
                                      ::Val{NDIMS}) where {NDIMS}
    for particle in axes(relative_coordinates, 2)
        update_relative_coordinate!(relative_coordinates, coordinates, center_of_mass,
                                    particle, Val(NDIMS))
    end

    return relative_coordinates
end

function update_relative_coordinates!(relative_coordinates, coordinates, center_of_mass,
                                      ::Val{NDIMS}, semi) where {NDIMS}
    @threaded semi for particle in axes(relative_coordinates, 2)
        update_relative_coordinate!(relative_coordinates, coordinates, center_of_mass,
                                    particle, Val(NDIMS))
    end

    return relative_coordinates
end

@inline function update_relative_coordinate!(relative_coordinates, coordinates,
                                             center_of_mass, particle,
                                             ::Val{NDIMS}) where {NDIMS}
    relative_position = extract_svector(coordinates, Val(NDIMS), particle) -
                        center_of_mass
    for i in 1:NDIMS
        relative_coordinates[i, particle] = relative_position[i]
    end

    return relative_coordinates
end

# Refresh the rigid-body kinematic cache from a consistent particle state after the
# ODE state has changed. Force/torque resultants are updated separately during RHS assembly.
function update_kinematic_cache!(system::RigidBodySystem, coordinates, velocity,
                                 semi=nothing)
    total_mass = system.total_mass
    total_mass <= eps(eltype(system)) && return system

    center_of_mass = zero(SVector{ndims(system), eltype(system)})
    center_of_mass_velocity = zero(SVector{ndims(system), eltype(system)})

    for particle in each_integrated_particle(system)
        particle_mass = system.mass[particle]
        center_of_mass += particle_mass * extract_svector(coordinates, system, particle)
        center_of_mass_velocity += particle_mass *
                                   extract_svector(velocity, system, particle)
    end

    center_of_mass /= total_mass
    center_of_mass_velocity /= total_mass

    system.center_of_mass[] = center_of_mass
    system.center_of_mass_velocity[] = center_of_mass_velocity

    if isnothing(semi)
        update_relative_coordinates!(system.relative_coordinates, coordinates,
                                     center_of_mass, Val(ndims(system)))
    else
        update_relative_coordinates!(system.relative_coordinates, coordinates,
                                     center_of_mass, Val(ndims(system)), semi)
    end
    update_rotational_kinematics!(system, velocity, center_of_mass_velocity,
                                  Val(ndims(system)))

    return system
end

@inline function Base.eltype(::RigidBodySystem{<:Any, <:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline function v_nvariables(system::RigidBodySystem)
    return ndims(system)
end

@inline function v_nvariables(system::RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return ndims(system) + 1
end

@inline function particle_spacing(system::RigidBodySystem, particle)
    return system.particle_spacing
end

@propagate_inbounds function current_velocity(v, system::RigidBodySystem)
    # For `ContinuityDensity`, the density is stored in the last row of `v`.
    # Return only the velocity components for rigid systems.
    return view(v, 1:ndims(system), :)
end

@inline function current_density(v, system::RigidBodySystem)
    # `current_density` for rigid systems means hydrodynamic density (FSI coupling density),
    # not the physical solid material density.
    return current_density(v, system.boundary_model, system)
end

# In fluid-structure interaction, use the hydrodynamic pressure corresponding to the
# configured boundary model.
@inline function current_pressure(v, system::RigidBodySystem)
    return current_pressure(v, system.boundary_model, system)
end

@inline function current_pressure(v, ::Nothing, system::RigidBodySystem)
    return zero(eltype(system))
end

@inline function hydrodynamic_mass(system::RigidBodySystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end



@inline function smoothing_length(system::RigidBodySystem{<:BoundaryModelDummyParticles},
                                  particle)
    return smoothing_length(system.boundary_model, particle)
end

@inline function system_smoothing_kernel(system::RigidBodySystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.smoothing_kernel
end

@inline function system_correction(system::RigidBodySystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.correction
end

function initialize!(system::RigidBodySystem, semi)
    initialize_colorfield!(system, system.boundary_model, semi)
    return system
end

function calc_normal!(system::AbstractFluidSystem,
                      neighbor_system::RigidBodySystem{<:BoundaryModelDummyParticles},
                      u_system, v, v_neighbor_system, u_neighbor_system, semi,
                      surface_normal_method, neighbor_surface_normal_method)
    haskey(neighbor_system.boundary_model.cache, :initial_colorfield) || return system

    return calc_boundary_normal!(system, neighbor_system, u_system, v, u_neighbor_system,
                                 semi, surface_normal_method)
end

@inline function adhesion_force(surface_tension::AkinciTypeSurfaceTension,
                                particle_system::AbstractFluidSystem,
                                neighbor_system::RigidBodySystem,
                                particle, neighbor, pos_diff, distance)
    (; adhesion_coefficient) = neighbor_system

    # No adhesion with oneself. See `src/general/smoothing_kernels.jl` for more details.
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return zero(pos_diff)

    abs(adhesion_coefficient) < eps() && return zero(pos_diff)

    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    support_radius = compact_support(system_smoothing_kernel(particle_system),
                                     smoothing_length(particle_system, particle))

    return adhesion_force_akinci(surface_tension, support_radius, m_b, pos_diff, distance,
                                 adhesion_coefficient)
end

function write_u0!(u0, system::RigidBodySystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_condition.coordinates)
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

function write_v0!(v0, system::RigidBodySystem)
    (; initial_velocity, boundary_model) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_velocity)
    copyto!(v0, indices, initial_velocity, indices)

    write_v0!(v0, boundary_model, system)

    return v0
end

function write_v0!(v0, model, system::RigidBodySystem)
    return v0
end

function write_v0!(v0, ::BoundaryModelDummyParticles{ContinuityDensity},
                   system::RigidBodySystem)
    (; cache) = system.boundary_model
    (; initial_density) = cache

    for particle in each_integrated_particle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::RigidBodySystem, v, u)
    indices_u = CartesianIndices(system.initial_condition.coordinates)
    copyto!(system.initial_condition.coordinates, indices_u, u, indices_u)

    indices_v = CartesianIndices(system.initial_condition.velocity)
    copyto!(system.initial_condition.velocity, indices_v,
            view(v, 1:ndims(system), :), indices_v)
    copyto!(system.initial_velocity, indices_v,
            view(v, 1:ndims(system), :), indices_v)

    return update_kinematic_cache!(system, system.initial_condition.coordinates,
                                   system.initial_velocity)
end

function update_boundary_interpolation!(system::RigidBodySystem, v, u, v_ode, u_ode,
                                        semi, t)
    return update_boundary_interpolation!(system.boundary_model, system, v, u, v_ode,
                                          u_ode, semi, t)
end

function update_boundary_interpolation!(::Nothing, system::RigidBodySystem, v, u, v_ode,
                                        u_ode, semi, t)
    return system
end

function update_boundary_interpolation!(boundary_model, system::RigidBodySystem, v, u,
                                        v_ode, u_ode, semi, t)
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)
    return system
end

function update_final!(system::RigidBodySystem, v, u, v_ode, u_ode, semi, t)
    system_coords = current_coordinates(u, system)
    system_velocity = current_velocity(v, system)

    return update_kinematic_cache!(system, system_coords, system_velocity, semi)
end

function update_rotational_kinematics!(system::RigidBodySystem, system_velocity,
                                       center_of_mass_velocity, ::Val{NDIMS}) where {NDIMS}
    inertia = zero(system.inertia[])
    angular_momentum = zero(system.angular_velocity[])

    for particle in each_integrated_particle(system)
        particle_mass = system.mass[particle]
        relative_position = extract_svector(system.relative_coordinates, system, particle)
        relative_velocity = extract_svector(system_velocity, system, particle) -
                            center_of_mass_velocity
        inertia += particle_mass * inertia_contribution(relative_position)
        angular_momentum += particle_mass * cross_product(relative_position,
                                          relative_velocity)
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
        # Embedded 1D/2D particle layouts in 3D lead to singular inertia tensors.
        # Use the Moore-Penrose pseudoinverse so angular velocity can still be
        # reconstructed in the resolvable rotational subspace.
        return SMatrix{3, 3, ELTYPE, 9}(pinv(Matrix(inertia)))
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

function calculate_dt(v_ode, u_ode, cfl_number, system::RigidBodySystem, semi)
    spacing = particle_spacing(system, first(eachparticle(system)))

    radius = maximum_particle_radius(system)
    angular_speed = norm(system.angular_velocity[])
    # In 3D, the total angular acceleration contains both external torque and
    # gyroscopic contributions.
    angular_acceleration = norm(system.angular_acceleration_force[] -
                                system.gyroscopic_acceleration[])

    # Use only interaction-induced translational acceleration for the time-step
    # estimate. Uniform source acceleration (e.g. gravity applied to all systems)
    # is a frame shift and does not change relative rigid-body dynamics.
    translational_acceleration = zero(SVector{ndims(system), eltype(system)})
    if system.total_mass > eps(eltype(system))
        translational_acceleration += system.resultant_force[] / system.total_mass
    end

    # Rigid-body scales:
    # acceleration ~ a_trans,relative + r*(ω² + |α|), velocity ~ v_com + r*ω.
    acceleration_scale = norm(translational_acceleration) +
                         radius * (angular_speed^2 + angular_acceleration)
    dt_acceleration = acceleration_scale <= eps(eltype(system)) ? Inf :
                      cfl_number * sqrt(spacing / acceleration_scale)

    speed_scale = norm(system.center_of_mass_velocity[]) + radius * angular_speed
    dt_velocity = speed_scale <= eps(eltype(system)) ? Inf :
                  cfl_number * spacing / speed_scale

    return min(dt_acceleration, dt_velocity)
end

function maximum_particle_radius(system::RigidBodySystem)
    # `relative_coordinates` are particle positions relative to the center of mass.
    # Their norm is the particle's radial distance `r`; the maximum is the rigid body's
    # characteristic radius used as lever arm in the `calculate_dt` estimates (`r*ω`, `r*α`).
    max_radius = zero(eltype(system))

    for particle in each_integrated_particle(system)
        relative_position = extract_svector(system.relative_coordinates, system, particle)
        max_radius = max(max_radius, norm(relative_position))
    end

    return max_radius
end

# To account for boundary effects in the viscosity term of fluid-structure interactions,
# use the viscosity model of the neighboring system.
@inline function viscosity_model(system::RigidBodySystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::Union{AbstractFluidSystem, OpenBoundarySystem},
                                 neighbor_system::RigidBodySystem{Nothing})
    return nothing
end

@inline function viscosity_model(system::Union{AbstractFluidSystem, OpenBoundarySystem},
                                 neighbor_system::RigidBodySystem)
    return neighbor_system.boundary_model.viscosity
end

@inline acceleration_source(system::RigidBodySystem) = system.acceleration

@inline function add_acceleration!(dv, particle, system::RigidBodySystem)
    relative_position = extract_svector(system.relative_coordinates, system, particle)
    rotational_acceleration = rigid_kinematic_acceleration(system, relative_position,
                                                           Val(ndims(system)))

    for i in 1:ndims(system)
        dv[i, particle] += system.acceleration[i] + rotational_acceleration[i]
    end

    return dv
end

@inline function rigid_kinematic_acceleration(system::RigidBodySystem, relative_position,
                                              ::Val{2})
    angular_velocity = system.angular_velocity[]

    return -(angular_velocity^2) * relative_position
end

@inline function rigid_kinematic_acceleration(system::RigidBodySystem, relative_position,
                                              ::Val{3})
    angular_velocity = system.angular_velocity[]
    gyroscopic_acceleration = system.gyroscopic_acceleration[]

    centripetal_acceleration = cross(angular_velocity,
                                     cross(angular_velocity, relative_position))
    gyroscopic_correction = cross(gyroscopic_acceleration, relative_position)

    return centripetal_acceleration - gyroscopic_correction
end

function system_data(system::RigidBodySystem, dv_ode, du_ode, v_ode, u_ode, semi)
    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    acceleration = current_velocity(dv, system)
    density = system.material_density
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
            relative_coordinates,
            center_of_mass, center_of_mass_velocity,
            angular_velocity,
            resultant_force, resultant_torque,
            angular_acceleration_force, gyroscopic_acceleration,
            density, pressure, acceleration)
end

function available_data(::RigidBodySystem)
    return (:coordinates, :velocity, :mass, :material_density,
            :relative_coordinates,
            :center_of_mass, :center_of_mass_velocity,
            :angular_velocity, :resultant_force, :resultant_torque,
            :angular_acceleration_force, :gyroscopic_acceleration,
            :density, :pressure, :acceleration)
end

function Base.show(io::IO, system::RigidBodySystem)
    @nospecialize system # reduce precompilation time

    print(io, "RigidBodySystem{", ndims(system), "}(")
    print(io, system.acceleration)
    print(io, ", ", system.boundary_model)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::RigidBodySystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "RigidBodySystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "initial angular velocity", system.angular_velocity[])
        summary_line(io, "boundary model", system.boundary_model)
        summary_footer(io)
    end
end

function check_configuration(system::RigidBodySystem, systems, nhs)
    (; boundary_model) = system

    if !isnothing(boundary_model)
        n_particles_model = length(boundary_model.hydrodynamic_mass)
        if n_particles_model != nparticles(system)
            throw(ArgumentError("the boundary model was initialized with $n_particles_model " *
                                "particles, but the `RigidBodySystem` has " *
                                "$(nparticles(system)) particles."))
        end
    end

    foreach_system(systems) do neighbor
        if neighbor isa AbstractFluidSystem && boundary_model === nothing
            throw(ArgumentError("a boundary model for `RigidBodySystem` must be specified " *
                                "when simulating a fluid-structure interaction."))
        end

        if neighbor isa AbstractFluidSystem &&
           neighbor.surface_normal_method isa ColorfieldSurfaceNormal
            if !(boundary_model isa BoundaryModelDummyParticles)
                throw(ArgumentError("`RigidBodySystem` is only compatible with " *
                                    "`ColorfieldSurfaceNormal` when using " *
                                    "`BoundaryModelDummyParticles`."))
            end

            if !haskey(boundary_model.cache, :initial_colorfield)
                throw(ArgumentError("`RigidBodySystem` with `BoundaryModelDummyParticles` " *
                                    "requires `reference_particle_spacing` to be set on " *
                                    "the boundary model when used together with " *
                                    "`ColorfieldSurfaceNormal` or a surface tension model."))
            end
        end
    end
end
