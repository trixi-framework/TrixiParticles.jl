"""
    BoundaryDEMSystem(initial_condition, normal_stiffness)

System for boundaries modeled by boundary particles.
The interaction between fluid and boundary particles is specified by the boundary model.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in a future releases.

"""
struct BoundaryDEMSystem{NDIMS, ELTYPE <: Real, IC,
                         ARRAY1D, ARRAY2D} <: AbstractBoundarySystem{NDIMS}
    initial_condition :: IC
    coordinates       :: ARRAY2D # [dimension, particle]
    radius            :: ARRAY1D # [particle]
    normal_stiffness  :: ELTYPE
    buffer            :: Nothing

    function BoundaryDEMSystem(initial_condition, coordinates, radius,
                               normal_stiffness, buffer)
        NDIMS = ndims(initial_condition)

        return new{NDIMS, eltype(initial_condition), typeof(initial_condition),
                   typeof(radius),
                   typeof(coordinates)}(initial_condition, coordinates, radius,
                                        normal_stiffness, buffer)
    end
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function BoundaryDEMSystem(initial_condition, normal_stiffness)
    ELTYPE = eltype(initial_condition)
    coordinates = initial_condition.coordinates
    radius = initial_condition.particle_spacing *
             ones(ELTYPE, length(initial_condition.mass)) / 2

    return BoundaryDEMSystem(initial_condition, coordinates, radius,
                             normal_stiffness, nothing)
end

@inline Base.eltype(::BoundaryDEMSystem{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

@inline function nparticles(system::BoundaryDEMSystem)
    size(system.coordinates, 2)
end

# No particle positions are advanced for DEM boundary systems
@inline function n_integrated_particles(system::BoundaryDEMSystem)
    return 0
end

@inline v_nvariables(system::BoundaryDEMSystem) = 0
@inline u_nvariables(system::BoundaryDEMSystem) = 0

@inline initial_coordinates(system::BoundaryDEMSystem) = system.coordinates

@inline function current_coordinates(u, system::BoundaryDEMSystem)
    return system.coordinates
end

@inline function current_velocity(v, system::BoundaryDEMSystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

function write_u0!(u0, ::BoundaryDEMSystem)
    return u0
end

function write_v0!(v0, ::BoundaryDEMSystem)
    return v0
end

function system_data(system::BoundaryDEMSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    (; coordinates, radius, normal_stiffness) = system

    return (; coordinates, radius, normal_stiffness)
end

function available_data(::BoundaryDEMSystem)
    return (:coordinates, :radius, :normal_stiffness)
end

function Base.show(io::IO, system::BoundaryDEMSystem)
    @nospecialize system # reduce precompilation time

    print(io, "BoundaryDEMSystem{", ndims(system), "}(")
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::BoundaryDEMSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "BoundaryDEMSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_footer(io)
    end
end
