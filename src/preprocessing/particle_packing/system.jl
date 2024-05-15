struct ParticlePackingSystem{NDIMS, ELTYPE <: Real, B, K} <: FluidSystem{NDIMS}
    initial_condition   :: InitialCondition{ELTYPE}
    boundary            :: B
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    background_pressure :: ELTYPE
    tlsph               :: Bool
    sd_positions        :: Vector{SVector{NDIMS, ELTYPE}}
    normals             :: Vector{SVector{NDIMS, ELTYPE}}
    distances           :: Vector{ELTYPE}
    signed_distances    :: Vector{ELTYPE} # TODO: Remove this (only for debugging)
    precalculate_sdf    :: Bool
    nhs_faces           :: Union{TrivialNeighborhoodSearch, FaceNeighborhoodSearch}

    function ParticlePackingSystem(initial_condition;
                                   smoothing_kernel=SchoenbergCubicSplineKernel{ndims(initial_condition)}(),
                                   smoothing_length=1.2initial_condition.particle_spacing,
                                   precalculate_sdf=false,
                                   neighborhood_search=true,
                                   background_pressure, boundary, tlsph=false)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        sd_positions = Vector{SVector{NDIMS, ELTYPE}}()
        normals = Vector{SVector{NDIMS, ELTYPE}}()
        distances = Vector{ELTYPE}()

        signed_distances = fill(NaN, nparticles(initial_condition))

        if neighborhood_search
            nhs_faces = FaceNeighborhoodSearch{NDIMS}(compact_support(smoothing_kernel,
                                                                      smoothing_length))
        else
            nhs_faces = TrivialNeighborhoodSearch{NDIMS}(1.0, eachface(boundary))
        end

        return new{NDIMS, ELTYPE, typeof(boundary),
                   typeof(smoothing_kernel)}(initial_condition, boundary, smoothing_kernel,
                                             smoothing_length, background_pressure, tlsph,
                                             sd_positions, normals, distances,
                                             signed_distances, precalculate_sdf,
                                             nhs_faces)
    end
end

function Base.show(io::IO, system::ParticlePackingSystem)
    @nospecialize system # reduce precompilation time

    print(io, "ParticlePackingSystem{", ndims(system), "}(")
    print(io, ", ", system.smoothing_kernel)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::ParticlePackingSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "ParticlePackingSystem{$(ndims(system))}")
        summary_line(io, "neighborhood search", system.nhs_faces |> typeof |> nameof)
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "#faces", nfaces(system.boundary))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "mesh", system.boundary |> typeof |> nameof)
        summary_line(io, "tlsph", system.tlsph ? "yes" : "no")
        summary_line(io, "precalculate sdf", system.precalculate_sdf ? "yes" : "no")
        summary_footer(io)
    end
end

function write2vtk!(vtk, v, u, t, system::ParticlePackingSystem; write_meta_data=true)
    if write_meta_data
        vtk["n_neigboring_faces"] = [length(eachneighborface(current_coords(u, system,
                                                                            particle),
                                                             system.nhs_faces))
                                     for particle in eachparticle(system)]
        vtk["signed_distances"] = system.signed_distances
    end
end

write_v0!(v0, system::ParticlePackingSystem) = v0 .= zero(eltype(system))

@inline source_terms(system::ParticlePackingSystem) = nothing
@inline add_acceleration!(dv, particle, system::ParticlePackingSystem) = dv

# Number of particles in the system
@inline nparticles(system::ParticlePackingSystem) = length(system.initial_condition.mass)

function update_particle_packing(system::ParticlePackingSystem, v_ode, u_ode,
                                 semi, integrator)
    u = wrap_u(u_ode, system, semi)

    @trixi_timeit timer() "bounding method" update_position!(u, system)

    update_transport_velocity!(system, v_ode, semi)
end

function update_position!(u, system::ParticlePackingSystem)
    system.precalculate_sdf && return update_poisition_2!(u, system)

    return update_position_1!(u, system)
end

# TODO: Naming not really creative
function update_position_1!(u, system::ParticlePackingSystem)
    (; boundary, initial_condition, nhs_faces, signed_distances) = system
    (; particle_spacing) = initial_condition

    shift_condition = system.tlsph ? zero(eltype(system)) : 0.5particle_spacing

    @threaded for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        distance2 = Inf
        distance_sign = true
        normal_vector = fill(0.0, SVector{ndims(system), eltype(system)})

        # Calculate minimum unsigned distance to boundary
        for face in eachneighborface(particle_position, nhs_faces)
            new_distance_sign, new_distance2, n = signed_point_face_distance(particle_position,
                                                                             boundary, face)

            if new_distance2 < distance2
                distance2 = new_distance2
                distance_sign = new_distance_sign
                normal_vector = n
            end
        end

        distance = distance_sign ? -sqrt(distance2) : sqrt(distance2)

        if distance >= -shift_condition
            signed_distances[particle] = distance
            # Constrain outside particles onto surface
            shift = (distance + shift_condition) * normal_vector

            for dim in 1:ndims(system)
                u[dim, particle] -= shift[dim]
            end
        end
    end
end

function update_poisition_2!(u, system::ParticlePackingSystem)
    (; initial_condition, nhs_faces, distances, normals, sd_positions) = system
    (; particle_spacing) = initial_condition

    search_radius2 = compact_support(system, system)^2

    shift_condition = system.tlsph ? zero(eltype(system)) : 0.5particle_spacing

    @threaded for particle in eachparticle(system)
        particle_position = current_coords(u, system, particle)

        volume = zero(eltype(system))
        distance_signed = zero(eltype(system))
        normal_vector = fill(volume, SVector{ndims(system), eltype(system)})

        for neighbor in eachneighbor(particle_position, nhs_faces)
            pos_diff = sd_positions[neighbor] - particle_position
            distance2 = dot(pos_diff, pos_diff)
            distance2 > search_radius2 && continue

            distance = sqrt(distance2)
            kernel_weight = smoothing_kernel(system, distance)

            distance_signed += distances[neighbor] * kernel_weight

            normal_vector += normals[neighbor] * kernel_weight

            volume += kernel_weight
        end

        if volume > eps()
            distance_signed /= volume
            normal_vector /= volume
            system.signed_distances[particle] = distance_signed
            if distance_signed >= -shift_condition
                # Constrain outside particles onto surface
                shift = (distance_signed + shift_condition) * normal_vector

                for dim in 1:ndims(system)
                    u[dim, particle] -= shift[dim]
                end
            end
        end
    end

    return u
end

@inline function update_transport_velocity!(system::ParticlePackingSystem, v_ode, semi)
    v = wrap_v(v_ode, system, semi)
    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            # Use the initial condition to store the evolved velocity
            system.initial_condition.velocity[i, particle] = v[i, particle]

            # The particle velocity is set to zero at the beginning of each time step to
            # achieve a fully stationary state.
            v[i, particle] = zero(eltype(system))
        end
    end

    return system
end

# Evolved velocity is stored in `initial_condition`
@inline function add_velocity!(du, v, particle, system::ParticlePackingSystem)
    for i in 1:ndims(system)
        du[i, particle] = system.initial_condition.velocity[i, particle]
    end

    return du
end

function calculate_signed_distance!(system::ParticlePackingSystem)
    (; boundary, initial_condition, nhs_faces,
    sd_positions, normals, distances) = system
    (; particle_spacing, coordinates) = initial_condition
    (; hashtable_sdf, cell_size) = nhs_faces

    empty!(hashtable_sdf)

    min_corner = (minimum(coordinates[i, :]) - 5particle_spacing for i in 1:ndims(system))
    max_corner = (maximum(coordinates[i, :]) + 5particle_spacing for i in 1:ndims(system))

    point_grid = meshgrid(min_corner, max_corner; increment=particle_spacing)

    resize!(normals, length(point_grid))
    resize!(distances, length(point_grid))
    resize!(sd_positions, length(point_grid))

    fill!(distances, Inf)
    fill!(normals, SVector(ntuple(dim -> Inf, ndims(system)...)))
    fill!(sd_positions, SVector(ntuple(dim -> Inf, ndims(system)...)))

    for (point, point_coords) in enumerate(point_grid)
        point_coords_ = SVector(point_coords...)

        for face in eachneighborface(point_coords_, nhs_faces)
            sdf = signed_point_face_distance(point_coords_, boundary, face)

            if sdf[2] <= (4particle_spacing)^2 && sdf[2] < distances[point]^2
                sd_positions[point] = point_coords_
                distances[point] = sdf[1] ? -sqrt(sdf[2]) : sqrt(sdf[2])
                # TODO: Calculate normals with n = ∇ϕ/|∇ϕ|
                # TODO: Is it possible to calulate n with distances * gradW ?
                normals[point] = sdf[3]
            end
        end
    end

    filter!(isfinite ∘ sum, sd_positions)
    filter!(isfinite ∘ sum, distances)
    filter!(isfinite ∘ sum, normals)

    for point in eachindex(sd_positions)
        cell = cell_coords(sd_positions[point], nothing, cell_size)

        # Add particle to corresponding cell or create cell if it does not exist
        if haskey(hashtable_sdf, cell)
            append!(hashtable_sdf[cell], point)
        else
            hashtable_sdf[cell] = [point]
        end
    end

    file = "out/rigid_system"
    points = stack(sd_positions)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in axes(points, 2)]
    vtk_grid(file, points, cells) do vtk
        # Store particle index
        vtk["index"] = 1:length(distances)
        vtk["signed_distances"] = distances
        vtk["normals"] = stack(normals)
    end

    return system
end
