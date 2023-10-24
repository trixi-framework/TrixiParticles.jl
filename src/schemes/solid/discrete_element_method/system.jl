struct DEMSystem{NDIMS, ELTYPE <: Real} <: System{NDIMS}
    initial_condition :: InitialCondition{ELTYPE}
    mass              :: Array{ELTYPE, 1}     # [particle]
    radius            :: Array{ELTYPE, 1}     # [particle]
    kn                :: ELTYPE               # Normal stiffness
    acceleration      :: SVector{NDIMS, ELTYPE}

    function DEMSystem(initial_condition, kn; acceleration=ntuple(_ -> 0.0,
        ndims(initial_condition)))
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)
        radius = copy(initial_condition.radius)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end


        return new{NDIMS, ELTYPE}(initial_condition, mass, radius, kn, acceleration_)
    end
end

function Base.show(io::IO, system::DEMSystem)
    @nospecialize system # reduce precompilation time

    print(io, "DEMSystem{", ndims(system), "}(")
    print(io, system.initial_condition)
    print(io, ", ", system.kn)
    print(io, ") with ", TrixiParticles.nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::DEMSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        TrixiParticles.summary_header(io, "DEMSystem{$(ndims(system))}")
        TrixiParticles.summary_line(io, "#particles", TrixiParticles.nparticles(system))
        TrixiParticles.summary_line(io, "kn", system.kn)
        TrixiParticles.summary_footer(io)
    end
end

timer_name(::DEMSystem) = "solid"

function TrixiParticles.write_u0!(u0, system::DEMSystem)
    u0 .= system.initial_condition.coordinates
    return u0
end

function TrixiParticles.write_v0!(v0, system::DEMSystem)
    v0 .= system.initial_condition.velocity
    return v0
end

# Nothing to initialize for this system
initialize!(system::DEMSystem, neighborhood_search) = system

function compact_support(system::DEMSystem, neighbor::DEMSystem)
    # we for now assume that the compact support is 3 * radius
    # todo: needs to be changed for more complex simulations
    return 3 * max(maximum(system.radius), maximum(neighbor.radius))
end
