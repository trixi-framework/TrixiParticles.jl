@doc raw"""
    RigidSPHSystem(initial_condition;
                   boundary_model=nothing,
                   acceleration=ntuple(_ -> 0.0, ndims(initial_condition)),
                   particle_spacing=initial_condition.particle_spacing,
                   source_terms=nothing, color_value=0)

System for particles of a rigid structure.

This system currently advances all rigid particles as Lagrangian particles while
applying one body-averaged fluid-structure interaction force to all particles.

# Arguments
- `initial_condition`: Initial condition representing the rigid particles.

# Keywords
- `boundary_model`: Boundary model for fluid-structure interaction
                    (see [Boundary Models](@ref boundary_models)).
- `acceleration`: Global acceleration vector applied to all rigid particles.
- `particle_spacing`: Reference particle spacing used for time-step estimation.
- `source_terms`: Optional source terms of the form
                  `(coords, velocity, density, pressure, t) -> source`.
- `color_value`: The value used to initialize the color of particles in the system.
"""
struct RigidSPHSystem{BM, NDIMS, ELTYPE <: Real, IC, ARRAY1D, ARRAY2D,
                      ST, C} <: AbstractStructureSystem{NDIMS}
    initial_condition :: IC
    local_coordinates :: ARRAY2D # [dimension, particle]
    mass              :: ARRAY1D # [particle]
    material_density  :: ARRAY1D # [particle]
    acceleration      :: SVector{NDIMS, ELTYPE}
    particle_spacing  :: ELTYPE
    boundary_model    :: BM
    source_terms      :: ST
    cache             :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function RigidSPHSystem(initial_condition; boundary_model=nothing,
                        acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                            ndims(initial_condition)),
                        particle_spacing=initial_condition.particle_spacing,
                        source_terms=nothing, color_value=0)
    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)

    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    particle_spacing_ = convert(ELTYPE, particle_spacing)

    local_coordinates = copy(initial_condition.coordinates)
    mass = copy(initial_condition.mass)
    material_density = copy(initial_condition.density)

    force_per_particle = zeros(ELTYPE, NDIMS, nparticles(initial_condition))
    cache = (; color=Int(color_value), force_per_particle)

    return RigidSPHSystem(initial_condition, local_coordinates, mass,
                          material_density, acceleration_, particle_spacing_,
                          boundary_model, source_terms, cache)
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
    (; initial_condition, boundary_model) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(initial_condition.velocity)
    copyto!(v0, indices, initial_condition.velocity, indices)

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

    return system
end

function update_boundary_interpolation!(system::RigidSPHSystem, v, u, v_ode, u_ode,
                                        semi, t)
    (; boundary_model) = system

    isnothing(boundary_model) && return system

    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

function calculate_dt(v_ode, u_ode, cfl_number, system::RigidSPHSystem, semi)
    acceleration_norm = norm(system.acceleration)
    spacing = particle_spacing(system, first(eachparticle(system)))

    if acceleration_norm <= eps(eltype(system)) || !isfinite(spacing) || spacing <= 0
        return Inf
    end

    return cfl_number * spacing / acceleration_norm
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

function system_data(system::RigidSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    acceleration = current_velocity(dv, system)
    density = current_density(v, system)
    pressure = current_pressure(v, system)

    return (; coordinates, velocity, mass=system.mass,
            material_density=system.material_density,
            density, pressure, acceleration)
end

function available_data(::RigidSPHSystem)
    return (:coordinates, :velocity, :mass, :material_density,
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
        summary_line(io, "boundary model", system.boundary_model)
        summary_footer(io)
    end
end
