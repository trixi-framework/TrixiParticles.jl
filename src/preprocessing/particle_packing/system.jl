"""
    ParticlePackingSystem(shape::InitialCondition;
                          signed_distance_field::Union{SignedDistanceField, Nothing},
                          smoothing_kernel=SchoenbergQuinticSplineKernel{ndims(shape)}(),
                          smoothing_length=shape.particle_spacing,
                          smoothing_length_interpolation=smoothing_length,
                          is_boundary=false, boundary_compress_factor=1,
                          neighborhood_search=GridNeighborhoodSearch{ndims(shape)}(),
                          background_pressure, tlsph=false, fixed_system=false)

System to generate body-fitted particles for complex shapes.
For more information on the methods, see [particle packing](@ref particle_packing).

# Arguments
- `shape`: [`InitialCondition`](@ref) to be packed.

# Keywords
- `background_pressure`:   Constant background pressure to physically pack the particles.
                           A large `background_pressure` can cause high accelerations
                           which requires a properly adjusted time step.
- `tlsph`:                 With the [`TotalLagrangianSPHSystem`](@ref), particles need to be placed
                           on the boundary of the shape and not half a particle spacing away,
                           as for fluids. When `tlsph=true`, particles will be placed
                           on the boundary of the shape.
- `is_boundary`:           When `shape` is inside the geometry that was used to create
                           `signed_distance_field`, set `is_boundary=false`.
                           Otherwise (`shape` is the sampled boundary), set `is_boundary=true`.
                           The thickness of the boundary is specified by creating
                           `signed_distance_field` with:
                              - `use_for_boundary_packing=true`
                              - `max_signed_distance=boundary_thickness`
                           See [`SignedDistanceField`](@ref).
- `fixed_system`:          When set to `true`, the system remains static, meaning particles
                           will not move and the `InitialCondition` will stay unchanged.
                           This is useful when the system is packed together with another
                           (non-fixed) `ParticlePackingSystem`.
                           In this case, no `SignedDistanceField` is required for both
                           the fixed and non-fixed system (use `nothing` as signed distance field).
- `signed_distance_field`: To constrain particles onto the surface, the information about
                           the signed distance from a particle to a face is required.
                           The precalculated signed distances will be interpolated
                           to each particle during the packing procedure.
                           Set `signed_distance_field=nothing` when packing with a fixed system
                           (see `fixed_system` description above).
- `smoothing_kernel`:      Smoothing kernel to be used for this system.
                           See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:      Smoothing length to be used for the gradient estimation.
                           See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length_interpolation`: Smoothing length to be used for interpolating the `SignedDistanceField` information.
                                    The default is `smoothing_length_interpolation = smoothing_length`.
- `boundary_compress_factor`: Factor to compress the boundary particles by reducing the boundary thickness by a factor of `boundary_compress_factor`.
                              The default value is `1`, which means no compression.
                              Compression can be useful for highly convex geometries,
                              where the boundary volume increases significantly while the mass of the boundary particles remains constant.
                              Recommended values are `0.8` or `0.9`.
"""
struct ParticlePackingSystem{S, F, NDIMS, ELTYPE <: Real, PR, C, AV,
                             IC, M, D, K, N, SD, UCU} <: FluidSystem{NDIMS}
    initial_condition              :: IC
    advection_velocity             :: AV
    mass                           :: M
    density                        :: D
    particle_spacing               :: ELTYPE
    smoothing_kernel               :: K
    smoothing_length_interpolation :: ELTYPE
    background_pressure            :: ELTYPE
    tlsph                          :: Bool
    signed_distance_field          :: S
    is_boundary                    :: Bool
    shift_length                   :: ELTYPE
    neighborhood_search            :: N
    signed_distances               :: SD # Only for visualization
    particle_refinement            :: PR
    buffer                         :: Nothing
    update_callback_used           :: UCU
    cache                          :: C

    # This constructor is necessary for Adapt.jl to work with this struct.
    # See the comments in general/gpu.jl for more details.
    function ParticlePackingSystem(initial_condition, mass, density, particle_spacing,
                                   smoothing_kernel, smoothing_length_interpolation,
                                   background_pressure, tlsph, signed_distance_field,
                                   is_boundary, shift_length, neighborhood_search,
                                   signed_distances, particle_refinement, buffer,
                                   update_callback_used, fixed_system, cache,
                                   advection_velocity)
        return new{typeof(signed_distance_field), fixed_system, ndims(smoothing_kernel),
                   eltype(density), typeof(particle_refinement), typeof(cache),
                   typeof(advection_velocity), typeof(initial_condition), typeof(mass),
                   typeof(density), typeof(smoothing_kernel), typeof(neighborhood_search),
                   typeof(signed_distances),
                   typeof(update_callback_used)}(initial_condition, advection_velocity,
                                                 mass, density, particle_spacing,
                                                 smoothing_kernel,
                                                 smoothing_length_interpolation,
                                                 background_pressure, tlsph,
                                                 signed_distance_field, is_boundary,
                                                 shift_length, neighborhood_search,
                                                 signed_distances, particle_refinement,
                                                 buffer, update_callback_used, cache)
    end
end

function ParticlePackingSystem(shape::InitialCondition;
                               signed_distance_field::Union{SignedDistanceField, Nothing},
                               smoothing_kernel=SchoenbergQuinticSplineKernel{ndims(shape)}(),
                               smoothing_length=shape.particle_spacing,
                               smoothing_length_interpolation=smoothing_length,
                               is_boundary=false, boundary_compress_factor=1,
                               neighborhood_search=GridNeighborhoodSearch{ndims(shape)}(),
                               background_pressure, tlsph=false, fixed_system=false)
    NDIMS = ndims(shape)
    ELTYPE = eltype(shape)
    mass = copy(shape.mass)
    density = copy(shape.density)

    particle_refinement = nothing

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    # Create neighborhood search for `ParticlePackingSystem` with the positions
    # of the `SignedDistanceField` as neighbors.
    # This is an intern NHS and is thus not organized by `Semidiscretization`.
    if isnothing(signed_distance_field)
        nhs = nothing
        @info "No `SignedDistanceField` provided. Particles will not be constraint onto a geometric surface."
    else
        nhs_ = isnothing(neighborhood_search) ? TrivialNeighborhoodSearch{NDIMS}() :
               neighborhood_search
        nhs = copy_neighborhood_search(nhs_,
                                       compact_support(smoothing_kernel,
                                                       smoothing_length_interpolation),
                                       length(signed_distance_field.positions))

        # Initialize neighborhood search with signed distances
        PointNeighbors.initialize_grid!(nhs, stack(signed_distance_field.positions))
    end

    # If `distance_signed >= -shift_length`, the particle position is modified
    # by a surface bounding:
    # `particle_position -= (distance_signed + shift_length) * normal_vector`,
    # where `normal_vector` is the normal vector to the surface of the geometry
    # and `distance_signed` is the level-set value at the particle position,
    # which means the signed distance to the surface.
    # Its value is negative if the particle is inside the geometry.
    # Otherwise (if outside), the value is positive.
    if is_boundary
        offset = tlsph ? shape.particle_spacing : shape.particle_spacing / 2

        shift_length = -boundary_compress_factor *
                       signed_distance_field.max_signed_distance - offset
    else
        shift_length = tlsph ? zero(ELTYPE) : shape.particle_spacing / 2
    end

    cache = (; create_cache_refinement(shape, particle_refinement, smoothing_length)...)

    advection_velocity = copy(shape.velocity)

    return ParticlePackingSystem(shape, mass, density, shape.particle_spacing,
                                 smoothing_kernel, smoothing_length_interpolation,
                                 background_pressure, tlsph, signed_distance_field,
                                 is_boundary, shift_length, nhs,
                                 fill(zero(ELTYPE), nparticles(shape)), particle_refinement,
                                 nothing, Ref(false), fixed_system, cache,
                                 advection_velocity)
end

function Base.show(io::IO, system::ParticlePackingSystem)
    @nospecialize system # reduce precompilation time

    print(io, "ParticlePackingSystem{", ndims(system), "}(")
    print(io, "", system.smoothing_kernel)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::ParticlePackingSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "ParticlePackingSystem{$(ndims(system))}")
        summary_line(io, "neighborhood search",
                     system.neighborhood_search |> typeof |> nameof)
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "tlsph", system.tlsph ? "yes" : "no")
        summary_line(io, "boundary", system.is_boundary ? "yes" : "no")
        summary_footer(io)
    end
end

timer_name(::ParticlePackingSystem) = "packing"

@inline fixed_packing_system(::ParticlePackingSystem{<:Any, F}) where {F} = F

@inline function Base.eltype(::ParticlePackingSystem{<:Any, <:Any, <:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline function v_nvariables(system::ParticlePackingSystem)
    # Don't integrate fixed systems
    fixed_packing_system(system) && return 0

    return ndims(system)
end

@inline function u_nvariables(system::ParticlePackingSystem)
    # Don't integrate fixed systems
    fixed_packing_system(system) && return 0

    return ndims(system)
end

function reset_callback_flag!(system::ParticlePackingSystem)
    system.update_callback_used[] = false

    return system
end

update_callback_used!(system::ParticlePackingSystem) = system.update_callback_used[] = true

function write2vtk!(vtk, v, u, t, system::ParticlePackingSystem; write_meta_data=true)
    vtk["velocity"] = [advection_velocity(v, system, particle)
                       for particle in active_particles(system)]
    if write_meta_data
        vtk["signed_distances"] = system.signed_distances
    end
end

# Skip for fixed systems
write_u0!(u0, system::ParticlePackingSystem{<:Any, true}) = u0

# Skip for fixed systems
write_v0!(v0, system::ParticlePackingSystem{<:Any, true}) = v0

# Skip for fixed systems
@inline function current_coordinates(u, system::ParticlePackingSystem{<:Any, true})
    return system.initial_condition.coordinates
end

@inline function advection_velocity(v, system::ParticlePackingSystem, particle)
    return extract_svector(system.advection_velocity, system, particle)
end

write_v0!(v0, system::ParticlePackingSystem) = (v0 .= zero(eltype(system)))

# Zero for fixed systems
function kinetic_energy(system::ParticlePackingSystem{<:Any, true}, v_ode, u_ode, semi, t)
    return zero(eltype(system))
end

function kinetic_energy(system::ParticlePackingSystem, v_ode, u_ode, semi, t)
    (; initial_condition, is_boundary) = system

    v = wrap_v(v_ode, system, semi)

    # Exclude boundary packing system
    is_boundary && return zero(eltype(system))

    # If `each_moving_particle` is empty (no moving particles), return zero
    return sum(each_moving_particle(system), init=zero(eltype(system))) do particle
        velocity = advection_velocity(v, system, particle)
        return initial_condition.mass[particle] * dot(velocity, velocity) / 2
    end
end

@inline source_terms(system::ParticlePackingSystem) = nothing
@inline add_acceleration!(dv, particle, system::ParticlePackingSystem) = dv

# Update from `UpdateCallback` (between time steps)
update_particle_packing(system, v_ode, u_ode, semi, integrator) = system

# Update from `UpdateCallback` (between time steps)
function update_particle_packing(system::ParticlePackingSystem, v_ode, u_ode,
                                 semi, integrator)
    u = wrap_u(u_ode, system, semi)

    update_position!(u, system, semi)
end

function update_position!(u, system::ParticlePackingSystem, semi)
    func_name = "constrain outside particles onto surface"
    @trixi_timeit timer() func_name constrain_particles_onto_surface!(u, system, semi)

    return u
end

function update_final!(system::ParticlePackingSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    if !update_from_callback && !(system.update_callback_used[])
        throw(ArgumentError("`UpdateCallback` is required when using `ParticlePackingSystem`"))
    end

    return system
end

# Skip for systems without `SignedDistanceField`
constrain_particles_onto_surface!(u, system::ParticlePackingSystem{Nothing}, semi) = u

function constrain_particles_onto_surface!(u, system::ParticlePackingSystem, semi)
    (; neighborhood_search, signed_distance_field, smoothing_length_interpolation) = system
    (; positions, distances, normals) = signed_distance_field

    search_radius2 = compact_support(system, system)^2

    @threaded semi for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        volume = zero(eltype(system))
        distance_signed = zero(eltype(system))
        normal_vector = fill(volume, SVector{ndims(system), eltype(system)})

        # Interpolate signed distances and normals
        for neighbor in PointNeighbors.eachneighbor(particle_position, neighborhood_search)
            pos_diff = positions[neighbor] - particle_position
            distance2 = dot(pos_diff, pos_diff)
            distance2 > search_radius2 && continue

            distance = sqrt(distance2)
            kernel_weight = kernel(system.smoothing_kernel, distance,
                                   smoothing_length_interpolation)

            distance_signed += distances[neighbor] * kernel_weight

            normal_vector += normals[neighbor] * kernel_weight

            volume += kernel_weight
        end

        if volume > eps()
            distance_signed /= volume
            normal_vector /= volume

            # Store signed distance for visualization
            system.signed_distances[particle] = distance_signed

            constrain_particle!(u, system, particle, distance_signed, normal_vector)
        end
    end

    return u
end

function constrain_particle!(u, system, particle, distance_signed, normal_vector)
    (; shift_length) = system

    # For fluid particles:
    # - `tlsph = true`: `shift_length = 0`
    # - `tlsph = false`: `shift_length = particle_spacing / 2`
    # For boundary particles:
    # `shift_length` is the thickness of the boundary.
    if distance_signed >= -shift_length
        # Constrain outside particles onto surface
        shift = (distance_signed + shift_length) * normal_vector

        for dim in 1:ndims(system)
            u[dim, particle] -= shift[dim]
        end
    end

    system.is_boundary || return u

    particle_spacing = system.initial_condition.particle_spacing
    shift_length_inner = system.tlsph ? particle_spacing : particle_spacing / 2

    if distance_signed < shift_length_inner
        shift = (distance_signed - shift_length_inner) * normal_vector

        for dim in 1:ndims(system)
            u[dim, particle] -= shift[dim]
        end
    end

    return u
end

# Skip for fixed systems
@inline update_transport_velocity!(system::ParticlePackingSystem{<:Any, true}, v_ode,
                                   semi) = system

# Update from `UpdateCallback` (between time steps)
@inline function update_transport_velocity!(system::ParticlePackingSystem, v_ode, semi)
    v = wrap_v(v_ode, system, semi)
    @threaded semi for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            system.advection_velocity[i, particle] = v[i, particle]

            # The particle velocity is set to zero at the beginning of each time step to
            # achieve a fully stationary state.
            v[i, particle] = zero(eltype(system))
        end
    end

    return system
end

# Skip for fixed systems
@inline add_velocity!(du, v, particle, system::ParticlePackingSystem{<:Any, true}) = du

# Add advection velocity.
@inline function add_velocity!(du, v, particle, system::ParticlePackingSystem)
    for i in 1:ndims(system)
        du[i, particle] = system.advection_velocity[i, particle]
    end

    return du
end
