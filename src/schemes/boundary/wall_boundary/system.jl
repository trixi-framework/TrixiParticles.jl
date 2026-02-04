"""
    WallBoundarySystem(initial_condition, boundary_model;
                       prescribed_motion=nothing, adhesion_coefficient=0.0)

System for boundaries modeled by boundary particles.
The interaction between fluid and boundary particles is specified by the boundary model.

# Arguments
- `initial_condition`: Initial condition (see [`InitialCondition`](@ref))
- `boundary_model`: Boundary model (see [Boundary Models](@ref boundary_models))

# Keywords
- `prescribed_motion`: For moving boundaries, a [`PrescribedMotion`](@ref) can be passed.
- `adhesion_coefficient`: Coefficient specifying the adhesion of a fluid to the surface.
   Note: currently it is assumed that all fluids have the same adhesion coefficient.
"""
struct WallBoundarySystem{BM, ELTYPE <: Real, NDIMS, IC, CO, M, IM,
                          CA} <: AbstractBoundarySystem{NDIMS}
    initial_condition    :: IC
    coordinates          :: CO # Array{coordinates_eltype, 2}
    boundary_model       :: BM
    prescribed_motion    :: M
    ismoving             :: IM # Ref{Bool} (to make a mutable field compatible with GPUs)
    adhesion_coefficient :: ELTYPE
    cache                :: CA

    # This constructor is necessary for Adapt.jl to work with this struct.
    # See the comments in general/gpu.jl for more details.
    function WallBoundarySystem(initial_condition, coordinates, boundary_model,
                                prescribed_motion, ismoving, adhesion_coefficient, cache)
        ELTYPE = eltype(initial_condition)

        new{typeof(boundary_model), ELTYPE, size(coordinates, 1), typeof(initial_condition),
            typeof(coordinates), typeof(prescribed_motion), typeof(ismoving),
            typeof(cache)}(initial_condition, coordinates, boundary_model,
                           prescribed_motion, ismoving, adhesion_coefficient, cache)
    end
end

function WallBoundarySystem(initial_condition, model; prescribed_motion=nothing,
                            adhesion_coefficient=0.0, color_value=0)
    coordinates = copy(initial_condition.coordinates)

    ismoving = Ref(!isnothing(prescribed_motion))
    initialize_prescribed_motion!(prescribed_motion, initial_condition)

    cache = create_cache_boundary(prescribed_motion, initial_condition)
    cache = (cache..., color=Int(color_value))

    return WallBoundarySystem(initial_condition, coordinates, model, prescribed_motion,
                              ismoving, adhesion_coefficient, cache)
end

create_cache_boundary(::Nothing, initial_condition) = (;)

function create_cache_boundary(prescribed_motion::PrescribedMotion, initial_condition)
    initial_coordinates = copy(initial_condition.coordinates)
    velocity = zero(initial_condition.velocity)
    acceleration = zero(initial_condition.velocity)

    return (; velocity, acceleration, initial_coordinates)
end

@inline Base.eltype(::WallBoundarySystem{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

@inline function nparticles(system::WallBoundarySystem)
    size(system.coordinates, 2)
end

# No particle positions are advanced for wall boundary systems,
# except when using `BoundaryModelDummyParticles` with `ContinuityDensity`.
@inline function n_integrated_particles(system::WallBoundarySystem)
    return 0
end

@inline function n_integrated_particles(system::WallBoundarySystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return nparticles(system)
end

@inline u_nvariables(system::WallBoundarySystem) = 0

# For BoundaryModelDummyParticles with ContinuityDensity, this needs to be 1.
# For all other models and density calculators, it's irrelevant.
@inline v_nvariables(system::WallBoundarySystem) = 1

@inline function initial_coordinates(system::WallBoundarySystem)
    initial_coordinates(system::WallBoundarySystem, system.prescribed_motion)
end

@inline initial_coordinates(system::WallBoundarySystem, ::Nothing) = system.coordinates

# We need static initial coordinates as reference when system is moving
@inline function initial_coordinates(system::WallBoundarySystem, prescribed_motion)
    return system.cache.initial_coordinates
end

@inline function current_coordinates(u, system::WallBoundarySystem)
    return system.coordinates
end

@inline function current_velocity(v, system::WallBoundarySystem, particle)
    return current_velocity(v, system, system.prescribed_motion, particle)
end

@inline function current_velocity(v, system, prescribed_motion, particle)
    (; cache, ismoving) = system

    if ismoving[]
        return extract_svector(cache.velocity, system, particle)
    end

    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_velocity(v, system, prescribed_motion::Nothing, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_velocity(v, system::WallBoundarySystem)
    error("`current_velocity(v, system)` is not implemented for `WallBoundarySystem`")
end

@inline function current_acceleration(system::WallBoundarySystem, particle)
    return current_acceleration(system, system.prescribed_motion, particle)
end

@inline function current_acceleration(system::WallBoundarySystem, ::PrescribedMotion,
                                      particle)
    (; cache, ismoving) = system

    if ismoving[]
        return extract_svector(cache.acceleration, system, particle)
    end

    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_acceleration(system::WallBoundarySystem, ::Nothing, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function viscous_velocity(v, system::WallBoundarySystem, particle)
    return viscous_velocity(v, system.boundary_model.viscosity, system, particle)
end

@inline function viscous_velocity(v, viscosity, system::WallBoundarySystem, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@inline function viscous_velocity(v, ::Nothing, system::WallBoundarySystem, particle)
    return current_velocity(v, system, particle)
end

@inline function current_density(v, system::WallBoundarySystem)
    return current_density(v, system.boundary_model, system)
end

@inline function current_pressure(v, system::WallBoundarySystem)
    return current_pressure(v, system.boundary_model, system)
end

@inline function hydrodynamic_mass(system::WallBoundarySystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@inline function smoothing_kernel(system::WallBoundarySystem, distance, particle)
    (; smoothing_kernel, smoothing_length) = system.boundary_model
    return kernel(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_length(system::WallBoundarySystem, particle)
    return smoothing_length(system.boundary_model, particle)
end

function update_positions!(system::WallBoundarySystem, v, u, v_ode, u_ode, semi, t)
    (; prescribed_motion) = system

    apply_prescribed_motion!(system, prescribed_motion, semi, t)
end

function apply_prescribed_motion!(system::WallBoundarySystem,
                                  prescribed_motion::PrescribedMotion, semi, t)
    (; ismoving, coordinates, cache) = system
    (; acceleration, velocity) = cache

    @trixi_timeit timer() "apply prescribed motion" begin
        prescribed_motion(coordinates, velocity, acceleration, ismoving, system, semi, t)
    end

    return system
end

function apply_prescribed_motion!(system::WallBoundarySystem, ::Nothing, semi, t)
    return system
end

function update_quantities!(system::WallBoundarySystem, v, u, v_ode, u_ode, semi, t)
    (; boundary_model) = system

    update_density!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

# This update depends on the computed quantities of the fluid system and therefore
# has to be in `update_boundary_interpolation!` after `update_quantities!`.
function update_boundary_interpolation!(system::WallBoundarySystem, v, u, v_ode, u_ode,
                                        semi, t)
    (; boundary_model) = system

    # Note that `update_pressure!(::WallBoundarySystem, ...)` is empty,
    # so no pressure is updated in the previous update steps.
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

function write_u0!(u0, ::WallBoundarySystem)
    return u0
end

function write_v0!(v0, ::WallBoundarySystem)
    return v0
end

function write_v0!(v0,
                   system::WallBoundarySystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    (; cache) = system.boundary_model
    (; initial_density) = cache

    v0[1, :] = initial_density

    return v0
end

function restart_with!(system::WallBoundarySystem, v, u)
    return system
end

function restart_with!(system::WallBoundarySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                       v, u)
    (; initial_density) = model.cache

    for particle in eachparticle(system)
        initial_density[particle] = v[1, particle]
    end

    return system
end

# To incorporate the effect at boundaries in the viscosity term of the RHS, the neighbor
# viscosity model has to be used.
@inline function viscosity_model(system::WallBoundarySystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

function calculate_dt(v_ode, u_ode, cfl_number, system::AbstractBoundarySystem, semi)
    return Inf
end

function initialize!(system::WallBoundarySystem, semi)
    initialize_colorfield!(system, system.boundary_model, semi)
    return system
end

function initialize_colorfield!(system, boundary_model, semi)
    return system
end

function initialize_colorfield!(system, ::BoundaryModelDummyParticles, semi)
    system_coords = system.coordinates
    (; smoothing_kernel, smoothing_length, cache) = system.boundary_model

    if haskey(cache, :initial_colorfield)
        foreach_point_neighbor(system, system, system_coords, system_coords, semi,
                               points=eachparticle(system)) do particle, neighbor,
                                                               pos_diff, distance
            cache.initial_colorfield[particle] += system.initial_condition.mass[particle] /
                                                  system.initial_condition.density[particle] *
                                                  system.cache.color *
                                                  kernel(smoothing_kernel,
                                                         distance,
                                                         smoothing_length)
            cache.neighbor_count[particle] += 1
        end
    end
    return system
end

function system_smoothing_kernel(system::WallBoundarySystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.smoothing_kernel
end

function system_correction(system::WallBoundarySystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.correction
end

function system_data(system::WallBoundarySystem, dv_ode, du_ode, v_ode, u_ode, semi)
    dv = [current_acceleration(system, particle) for particle in eachparticle(system)]
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = [current_velocity(v, system, particle) for particle in eachparticle(system)]
    density = current_density(v, system)
    pressure = current_pressure(v, system)

    return (; coordinates, velocity, density, pressure, acceleration=dv)
end

function available_data(::WallBoundarySystem)
    return (:coordinates, :velocity, :density, :pressure, :acceleration)
end

function Base.show(io::IO, system::WallBoundarySystem)
    @nospecialize system # reduce precompilation time

    print(io, "WallBoundarySystem{", ndims(system), "}(")
    print(io, system.boundary_model)
    print(io, ", ", system.prescribed_motion)
    print(io, ", ", system.adhesion_coefficient)
    print(io, ", ", system.cache.color)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::WallBoundarySystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "WallBoundarySystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "movement function",
                     isnothing(system.prescribed_motion) ? "nothing" :
                     string(system.prescribed_motion.movement_function))
        summary_line(io, "adhesion coefficient", system.adhesion_coefficient)
        summary_line(io, "color", system.cache.color)
        summary_footer(io)
    end
end
