@doc raw"""
    interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system, sol;
                         smoothing_length=initial_smoothing_length(ref_system), cut_off_bnd=true,
                         clip_negative_pressure=false, include_wall_velocity=false)

Interpolates properties along a plane in a TrixiParticles simulation.
The region for interpolation is defined by its lower left and top right corners,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points within the defined region,
spaced uniformly according to the given resolution.

See also: [`interpolate_plane_2d_vtk`](@ref), [`interpolate_plane_3d`](@ref),
          [`interpolate_line`](@ref), [`interpolate_points`](@ref).

# Arguments
- `min_corner`: The lower left corner of the interpolation region.
- `max_corner`: The top right corner of the interpolation region.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `smoothing_length=initial_smoothing_length(ref_system)`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.
- `include_wall_velocity=false`: If set to `true`, the wall velocity at the boundary is included
                                 in the interpolation. This is particularly useful for simulations
                                 with no-slip boundary conditions, where it is necessary
                                 to obtain a correct boundary layer in the interpolation.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point within the plane.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used, which is not as
      accurate as a real surface reconstruction.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing); ref_system = fluid_system)
# Interpolating across a plane from [0.0, 0.0] to [1.0, 1.0] with a resolution of 0.2
results = interpolate_plane_2d([0.0, 0.0], [1.0, 1.0], 0.2, semi, ref_system, sol)

# output
(computed_density = ...)
```
"""
function interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system,
                              sol::ODESolution; include_wall_velocity=false,
                              smoothing_length=initial_smoothing_length(ref_system),
                              cut_off_bnd=true, clip_negative_pressure=false)
    # Filter out particles without neighbors
    filter_no_neighbors = true
    v_ode, u_ode = sol.u[end].x

    results, _,
    _ = interpolate_plane_2d(min_corner, max_corner, resolution,
                             semi, ref_system, v_ode, u_ode,
                             filter_no_neighbors, smoothing_length, cut_off_bnd,
                             clip_negative_pressure, include_wall_velocity)

    return results
end

function interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system,
                              v_ode, u_ode; include_wall_velocity=false,
                              smoothing_length=initial_smoothing_length(ref_system),
                              cut_off_bnd=true, clip_negative_pressure=false)
    # Filter out particles without neighbors
    filter_no_neighbors = true

    results, _,
    _ = interpolate_plane_2d(min_corner, max_corner, resolution,
                             semi, ref_system, v_ode, u_ode,
                             filter_no_neighbors, smoothing_length, cut_off_bnd,
                             clip_negative_pressure, include_wall_velocity)

    return results
end

@doc raw"""
    interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system, sol;
                             smoothing_length=initial_smoothing_length(ref_system), cut_off_bnd=true,
                             clip_negative_pressure=false, output_directory="out", filename="plane",
                             include_wall_velocity=false)

Interpolates properties along a plane in a TrixiParticles simulation and exports the result
as a VTI file.
The region for interpolation is defined by its lower left and top right corners,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points within the defined region,
spaced uniformly according to the given resolution.

See also: [`interpolate_plane_2d`](@ref), [`interpolate_plane_3d`](@ref),
          [`interpolate_line`](@ref), [`interpolate_points`](@ref).

# Arguments
- `min_corner`: The lower left corner of the interpolation region.
- `max_corner`: The top right corner of the interpolation region.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `smoothing_length=initial_smoothing_length(ref_system)`: The smoothing length used in the interpolation.
- `output_directory="out"`: Directory to save the VTI file.
- `filename="plane"`:       Name of the VTI file.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.
- `include_wall_velocity=false`: If set to `true`, the wall velocity at the boundary is included
                                 in the interpolation. This is particularly useful for simulations
                                 with no-slip boundary conditions, where it is necessary
                                 to obtain a correct boundary layer in the interpolation.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used, which is not as
      accurate as a real surface reconstruction.

# Examples
```julia
# Interpolating across a plane from [0.0, 0.0] to [1.0, 1.0] with a resolution of 0.2
results = interpolate_plane_2d([0.0, 0.0], [1.0, 1.0], 0.2, semi, ref_system, sol)
```
"""
function interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system,
                                  sol::ODESolution; clip_negative_pressure=false,
                                  smoothing_length=initial_smoothing_length(ref_system),
                                  cut_off_bnd=true, include_wall_velocity=false,
                                  output_directory="out", filename="plane")
    v_ode, u_ode = sol.u[end].x

    interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system,
                             v_ode, u_ode; clip_negative_pressure, include_wall_velocity,
                             smoothing_length, cut_off_bnd, output_directory, filename)
end

function interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system,
                                  v_ode, u_ode; include_wall_velocity=false,
                                  smoothing_length=initial_smoothing_length(ref_system),
                                  cut_off_bnd=true, clip_negative_pressure=false,
                                  output_directory="out", filename="plane")
    # Don't filter out particles without neighbors to keep 2D grid structure
    filter_no_neighbors = false
    @trixi_timeit timer() "interpolate plane" begin
        results, x_range,
        y_range = interpolate_plane_2d(min_corner, max_corner, resolution,
                                       semi, ref_system, v_ode, u_ode,
                                       filter_no_neighbors,
                                       smoothing_length, cut_off_bnd,
                                       clip_negative_pressure, include_wall_velocity)
    end

    density = reshape(results.density, length(x_range), length(y_range))
    velocity = reshape(results.velocity, ndims(ref_system), length(x_range),
                       length(y_range))
    pressure = reshape(results.pressure, length(x_range), length(y_range))

    @trixi_timeit timer() "write to vtk" vtk_grid(joinpath(output_directory, filename),
                                                  x_range, y_range) do vtk
        vtk["density"] = density
        vtk["velocity"] = velocity
        vtk["pressure"] = pressure
    end
end

function interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system,
                              v_ode, u_ode, filter_no_neighbors, smoothing_length,
                              cut_off_bnd, clip_negative_pressure, include_wall_velocity)
    dims = length(min_corner)
    if dims != 2 || length(max_corner) != 2
        throw(ArgumentError("function is intended for 2D coordinates only"))
    end

    if any(min_corner .> max_corner)
        throw(ArgumentError("`min_corner` should be smaller than `max_corner` in every dimension"))
    end

    # Calculate the number of points in each dimension based on the resolution
    n_points_per_dimension = Tuple(ceil.(Int,
                                         (max_corner .- min_corner) ./ resolution) .+ 1)
    x_range = range(min_corner[1], max_corner[1], length=n_points_per_dimension[1])
    y_range = range(min_corner[2], max_corner[2], length=n_points_per_dimension[2])

    # Generate points within the plane. Use `place_on_shell=true` to generate points
    # on the shell of the geometry.
    point_coords = rectangular_shape_coords(resolution, n_points_per_dimension, min_corner,
                                            place_on_shell=true)

    results = interpolate_points(point_coords, semi, ref_system, v_ode, u_ode;
                                 smoothing_length, cut_off_bnd, include_wall_velocity,
                                 clip_negative_pressure)

    if filter_no_neighbors
        # Find indices where neighbor_count > 0
        indices = findall(x -> x > 0, results.neighbor_count)

        # Filter all arrays in the named tuple using these indices
        results = map(results) do x
            if isa(x, AbstractVector)
                return x[indices]
            else
                return x[:, indices]
            end
        end
    end

    return results, x_range, y_range
end

@doc raw"""
    interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system, sol;
                         smoothing_length=initial_smoothing_length(ref_system), cut_off_bnd=true,
                         clip_negative_pressure=false, include_wall_velocity=false)

Interpolates properties along a plane in a 3D space in a TrixiParticles simulation.
The plane for interpolation is defined by three points in 3D space,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points on a parallelogram within the plane defined by the
three points, spaced uniformly according to the given resolution.

See also: [`interpolate_plane_2d`](@ref), [`interpolate_plane_2d_vtk`](@ref),
          [`interpolate_line`](@ref), [`interpolate_points`](@ref).

# Arguments
- `point1`:     The first point defining the plane.
- `point2`:     The second point defining the plane.
- `point3`:     The third point defining the plane. The points must not be collinear.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `smoothing_length=initial_smoothing_length(ref_system)`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.
- `include_wall_velocity=false`: If set to `true`, the wall velocity at the boundary is included
                                 in the interpolation. This is particularly useful for simulations
                                 with no-slip boundary conditions, where it is necessary
                                 to obtain a correct boundary layer in the interpolation.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point within the plane.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
      accurate as a real surface reconstruction.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_3d.jl"), tspan=(0.0, 0.01)); ref_system = fluid_system)
# Interpolating across a plane defined by points [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], and [0.0, 1.0, 0.0]
# with a resolution of 0.1
results = interpolate_plane_3d([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1, semi, ref_system, sol)

# output
(computed_density = ...)
```
"""
function interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system,
                              sol::ODESolution; include_wall_velocity=false,
                              smoothing_length=initial_smoothing_length(ref_system),
                              cut_off_bnd=true, clip_negative_pressure=false)
    v_ode, u_ode = sol.u[end].x

    interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system,
                         v_ode, u_ode; smoothing_length, cut_off_bnd, include_wall_velocity,
                         clip_negative_pressure)
end

function interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system,
                              v_ode, u_ode; include_wall_velocity=false,
                              smoothing_length=initial_smoothing_length(ref_system),
                              cut_off_bnd=true, clip_negative_pressure=false)
    if ndims(ref_system) != 3
        throw(ArgumentError("`interpolate_plane_3d` requires a 3D simulation"))
    end

    points_coords = sample_face_with_fixed_size((point1, point2, point3), resolution)

    # Interpolate using the generated points
    results = interpolate_points(points_coords, semi, ref_system, v_ode, u_ode;
                                 smoothing_length, cut_off_bnd, include_wall_velocity,
                                 clip_negative_pressure)

    # Filter results
    indices = findall(x -> x > 0, results.neighbor_count)
    filtered_results = map(results) do x
        if isa(x, AbstractVector)
            return x[indices]
        else
            return x[:, indices]
        end
    end

    return filtered_results
end

@doc raw"""
    interpolate_line(start, end_, n_points, semi, ref_system, sol; endpoint=true,
                     smoothing_length=initial_smoothing_length(ref_system), cut_off_bnd=true,
                     clip_negative_pressure=false, include_wall_velocity=false)


Interpolates properties along a line in a TrixiParticles simulation.
The line interpolation is accomplished by generating a series of
evenly spaced points between `start` and `end_`.
If `endpoint` is `false`, the line is interpolated between the start and end points,
but does not include these points.

See also: [`interpolate_points`](@ref), [`interpolate_plane_2d`](@ref),
          [`interpolate_plane_2d_vtk`](@ref), [`interpolate_plane_3d`](@ref).

# Arguments
- `start`:      The starting point of the line.
- `end_`:       The ending point of the line.
- `n_points`:   The number of points to interpolate along the line.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `endpoint=true`: A boolean to include (`true`) or exclude (`false`) the end point in the interpolation.
- `smoothing_length=initial_smoothing_length(ref_system)`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.
- `include_wall_velocity=false`: If set to `true`, the wall velocity at the boundary is included
                                 in the interpolation. This is particularly useful for simulations
                                 with no-slip boundary conditions, where it is necessary
                                 to obtain a correct boundary layer in the interpolation.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point along the line.

!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations
      along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
      accurate as a real surface reconstruction.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing); ref_system = fluid_system)
# Interpolating along a line from [1.0, 0.0] to [1.0, 1.0] with 5 points
results = interpolate_line([1.0, 0.0], [1.0, 1.0], 5, semi, ref_system, sol)

# output
(computed_density = ...)
```
"""
function interpolate_line(start, end_, n_points, semi, ref_system, sol::ODESolution;
                          endpoint=true, include_wall_velocity=false,
                          smoothing_length=initial_smoothing_length(ref_system),
                          cut_off_bnd=true, clip_negative_pressure=false)
    v_ode, u_ode = sol.u[end].x

    interpolate_line(start, end_, n_points, semi, ref_system, v_ode, u_ode; endpoint,
                     smoothing_length, cut_off_bnd, clip_negative_pressure,
                     include_wall_velocity)
end

function interpolate_line(start, end_, n_points, semi, ref_system, v_ode, u_ode;
                          endpoint=true, include_wall_velocity=false,
                          smoothing_length=initial_smoothing_length(ref_system),
                          cut_off_bnd=true, clip_negative_pressure=false)
    start_svector = SVector{ndims(ref_system)}(start)
    end_svector = SVector{ndims(ref_system)}(end_)
    points_coords = range(start_svector, end_svector, length=n_points)

    if !endpoint
        points_coords = points_coords[2:(end - 1)]
    end

    # Convert to coordinate matrix
    points_coords_ = collect(reinterpret(reshape, eltype(start_svector), points_coords))

    return interpolate_points(points_coords_, semi, ref_system, v_ode, u_ode;
                              smoothing_length=smoothing_length, include_wall_velocity,
                              cut_off_bnd=cut_off_bnd, clip_negative_pressure)
end

@doc raw"""
    interpolate_points(point_coords::AbstractMatrix, semi, ref_system, sol;
                       smoothing_length=initial_smoothing_length(ref_system),
                       cut_off_bnd=true, clip_negative_pressure=false,
                       include_wall_velocity=false)

Performs interpolation of properties at specified points in a TrixiParticles simulation.
The interpolation utilizes the same kernel function of the SPH simulation to weigh
contributions from nearby particles.

See also: [`interpolate_line`](@ref), [`interpolate_plane_2d`](@ref),
          [`interpolate_plane_2d_vtk`](@ref), [`interpolate_plane_3d`](@ref), .

# Arguments
- `point_coords`:   A matrix of point coordinates, where the $i$-th column holds the
                    coordinates of particle $i$.
- `semi`:           The semidiscretization used in the SPH simulation.
- `ref_system`:     The reference system defining the properties of the SPH particles.
- `sol`:            The current solution state from which properties are interpolated.

# Keywords
- `smoothing_length=initial_smoothing_length(ref_system)`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.
- `include_wall_velocity=false`: If set to `true`, the wall velocity at the boundary is included
                                 in the interpolation. This is particularly useful for simulations
                                 with no-slip boundary conditions, where it is necessary
                                 to obtain a correct boundary layer in the interpolation.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing); ref_system = fluid_system)
# For a single point create a 2x1 matrix
result = interpolate_points([1.0; 0.5;;], semi, ref_system, sol)

# For multiple points
points = [1.0 1.0 1.0; 0.5 0.6 0.7]
results = interpolate_points(points, semi, ref_system, sol)

# output
(computed_density = ...)
```
!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations
      along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
    accurate as a real surface reconstruction.
"""
@inline function interpolate_points(point_coords, semi, ref_system, sol::ODESolution;
                                    smoothing_length=initial_smoothing_length(ref_system),
                                    cut_off_bnd=true, clip_negative_pressure=false,
                                    include_wall_velocity=false)
    v_ode, u_ode = sol.u[end].x

    interpolate_points(point_coords, semi, ref_system, v_ode, u_ode; include_wall_velocity,
                       smoothing_length, cut_off_bnd, clip_negative_pressure)
end

# Create neighborhood searches and then interpolate points
function interpolate_points(point_coords, semi, ref_system, v_ode, u_ode;
                            smoothing_length=initial_smoothing_length(ref_system),
                            cut_off_bnd=true, clip_negative_pressure=false,
                            include_wall_velocity=false)
    neighborhood_searches = process_neighborhood_searches(semi, u_ode, ref_system,
                                                          smoothing_length, point_coords)

    return interpolate_points(point_coords, semi, ref_system,
                              v_ode, u_ode, neighborhood_searches; include_wall_velocity,
                              smoothing_length, cut_off_bnd, clip_negative_pressure)
end

function process_neighborhood_searches(semi, u_ode, ref_system, smoothing_length,
                                       point_coords)
    if isapprox(smoothing_length, initial_smoothing_length(ref_system))
        # Check if the neighborhood searches can be used with different points
        # than it was initialized with.
        f(system) = PointNeighbors.requires_update(get_neighborhood_search(ref_system,
                                                                           system, semi))[1]
        if !any(f, semi.systems)
            # We can use the existing neighborhood searches.
            # Update existing NHS with the current coordinates.
            update_nhs!(semi, u_ode)
            return semi.neighborhood_searches[system_indices(ref_system, semi)]
        end
    end

    # Copy neighborhood searches with new smoothing length
    ref_smoothing_kernel = ref_system.smoothing_kernel
    search_radius = compact_support(ref_smoothing_kernel, smoothing_length)
    neighborhood_searches = map(semi.systems) do system
        u = wrap_u(u_ode, system, semi)
        system_coords = current_coordinates(u, system)
        old_nhs = get_neighborhood_search(ref_system, system, semi)
        nhs = PointNeighbors.copy_neighborhood_search(old_nhs, search_radius,
                                                      nparticles(system))
        PointNeighbors.initialize!(nhs, point_coords, system_coords)

        return nhs
    end

    return neighborhood_searches
end

# Interpolate points with given neighborhood searches
@inline function interpolate_points(point_coords_, semi, ref_system, v_ode, u_ode,
                                    neighborhood_searches; include_wall_velocity=false,
                                    smoothing_length=initial_smoothing_length(ref_system),
                                    cut_off_bnd=true, clip_negative_pressure=false)
    (; parallelization_backend) = semi

    if parallelization_backend isa KernelAbstractions.Backend
        point_coords = Adapt.adapt(parallelization_backend, point_coords_)
    else
        point_coords = point_coords_
    end

    n_points = size(point_coords, 2)
    ELTYPE = eltype(point_coords)
    computed_density = allocate(semi.parallelization_backend, ELTYPE, n_points)
    other_density = allocate(semi.parallelization_backend, ELTYPE, n_points)
    shepard_coefficient = allocate(semi.parallelization_backend, ELTYPE, n_points)
    neighbor_count = allocate(semi.parallelization_backend, Int, n_points)
    # The wall velocity considers more neighbors, so we need to use
    # a different Shepard coefficient.
    shepard_coefficient_wall = allocate(semi.parallelization_backend, ELTYPE, n_points)
    set_zero!(shepard_coefficient_wall)

    set_zero!(computed_density)
    set_zero!(other_density)
    set_zero!(shepard_coefficient)
    set_zero!(neighbor_count)

    cache = create_cache_interpolation(ref_system, n_points, semi)

    ref_id = system_indices(ref_system, semi)
    ref_smoothing_kernel = ref_system.smoothing_kernel

    # If we don't cut at the boundary, we only need to iterate over the reference system
    systems = cut_off_bnd ? semi : (ref_system,)

    foreach_system(systems) do neighbor_system
        system_id = system_indices(neighbor_system, semi)
        nhs = neighborhood_searches[system_id]

        v = wrap_v(v_ode, neighbor_system, semi)
        u = wrap_u(u_ode, neighbor_system, semi)

        neighbor_coords = current_coordinates(u, neighbor_system)

        foreach_point_neighbor(point_coords, neighbor_coords, nhs;
                               parallelization_backend) do point, neighbor, pos_diff,
                                                           distance
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            volume_b = m_b / current_density(v, neighbor_system, neighbor)
            W_ab = kernel(ref_smoothing_kernel, distance, smoothing_length)

            if include_wall_velocity
                # The wall velocity considers more neighbors, so we need to use
                # a different Shepard coefficient.
                shepard_coefficient_wall[point] += volume_b * W_ab
            end

            if system_id == ref_id
                computed_density[point] += m_b * W_ab
                shepard_coefficient[point] += volume_b * W_ab

                # According to:
                # u(r_a) = (∑_b u(r_b) ⋅ V_b ⋅ W(r_a-r_b)) / (∑_b V_b ⋅ W(r_a-r_b)),
                # where V_b = m_b / ρ_b.
                interpolate_system!(cache, v, neighbor_system,
                                    point, neighbor, volume_b, W_ab, clip_negative_pressure)
            else
                other_density[point] += m_b * W_ab

                if include_wall_velocity
                    velocity_neighbor = viscous_velocity(v, neighbor_system, neighbor)
                    for i in axes(velocity_neighbor, 1)
                        cache.velocity[i, point] += velocity_neighbor[i] * volume_b * W_ab
                    end
                end
            end

            neighbor_count[point] += 1
        end
    end

    @threaded parallelization_backend for point in axes(point_coords, 2)
        if other_density[point] > computed_density[point] ||
           computed_density[point] < eps()
            # Return NaN values that can be filtered out in ParaView
            computed_density[point] = NaN
            neighbor_count[point] = 0

            # We need to convert the `NamedTuple` to a `Tuple` for GPU compatibility
            foreach(Tuple(cache)) do field
                set_nan!(field, point)
            end
        else
            # Normalize all quantities by the shepard coefficient.
            # We need to convert the `NamedTuple` to a `Tuple` for GPU compatibility.
            foreach(Tuple(cache)) do field
                divide_by_shepard_coefficient!(field, shepard_coefficient, point)
            end
            if include_wall_velocity
                # The wall velocity considers more neighbors, so we need to use
                # a different Shepard coefficient.
                # The coefficient `shepard_coefficient_wall / shepard_coefficient`
                # reverts the incorrect division above for this field.
                new_coefficient = shepard_coefficient_wall[point] /
                                  shepard_coefficient[point]
                for dim in axes(cache.velocity, 1)
                    cache.velocity[dim, point] /= new_coefficient
                end
            end
        end
    end

    return (; computed_density=computed_density, point_coords=point_coords,
            neighbor_count=neighbor_count, cache...)
end

@inline function create_cache_interpolation(ref_system::AbstractFluidSystem, n_points, semi)
    (; parallelization_backend) = semi

    velocity = allocate(parallelization_backend, eltype(ref_system),
                        (ndims(ref_system), n_points))
    pressure = allocate(parallelization_backend, eltype(ref_system), n_points)
    density = allocate(parallelization_backend, eltype(ref_system), n_points)

    set_zero!(velocity)
    set_zero!(pressure)
    set_zero!(density)

    return (; velocity, pressure, density)
end

@inline function create_cache_interpolation(ref_system::AbstractStructureSystem,
                                            n_points, semi)
    (; parallelization_backend) = semi

    velocity = allocate(parallelization_backend, eltype(ref_system),
                        (ndims(ref_system), n_points))
    jacobian = allocate(parallelization_backend, eltype(ref_system), n_points)
    von_mises_stress = allocate(parallelization_backend, eltype(ref_system), n_points)
    cauchy_stress = allocate(parallelization_backend, eltype(ref_system),
                             (ndims(ref_system), ndims(ref_system), n_points))

    set_zero!(velocity)
    set_zero!(jacobian)
    set_zero!(von_mises_stress)
    set_zero!(cauchy_stress)

    return (; velocity, jacobian, von_mises_stress, cauchy_stress)
end

function interpolate_system!(cache, v, neighbor_system,
                             point, neighbor, volume_b, W_ab, clip_negative_pressure)
    return cache
end

@inline function interpolate_system!(cache, v, system::AbstractFluidSystem,
                                     point, neighbor, volume_b, W_ab,
                                     clip_negative_pressure)
    velocity = current_velocity(v, system, neighbor)
    for i in axes(cache.velocity, 1)
        cache.velocity[i, point] += velocity[i] * volume_b * W_ab
    end

    pressure = current_pressure(v, system, neighbor)
    if clip_negative_pressure
        pressure = max(zero(eltype(pressure)), pressure)
    end
    cache.pressure[point] += pressure * volume_b * W_ab

    density = current_density(v, system, neighbor)
    cache.density[point] += density * volume_b * W_ab

    return cache
end

@inline function interpolate_system!(cache, v, system::AbstractStructureSystem,
                                     point, neighbor, volume_b, W_ab,
                                     clip_negative_pressure)
    velocity = current_velocity(v, system, neighbor)
    for i in axes(cache.velocity, 1)
        cache.velocity[i, point] += velocity[i] * volume_b * W_ab
    end

    cache.jacobian[point] += det(deformation_gradient(system, neighbor)) * volume_b * W_ab
    cache.von_mises_stress[point] += von_mises_stress(system, neighbor) * volume_b * W_ab

    sigma = cauchy_stress(system)
    for j in axes(cache.cauchy_stress, 2), i in axes(cache.cauchy_stress, 1)
        cache.cauchy_stress[i, j, point] += sigma[i, j, neighbor] * volume_b * W_ab
    end

    return cache
end

function set_nan!(field::AbstractVector, point)
    field[point] = NaN

    return field
end

function set_nan!(field::AbstractMatrix, point)
    for dim in axes(field, 1)
        field[dim, point] = NaN
    end

    return field
end

function set_nan!(field::AbstractArray{T, 3}, point) where {T}
    for j in axes(field, 2), i in axes(field, 1)
        field[i, j, point] = NaN
    end

    return field
end

function divide_by_shepard_coefficient!(field::AbstractVector, shepard_coefficient, point)
    field[point] /= shepard_coefficient[point]

    return field
end

function divide_by_shepard_coefficient!(field::AbstractMatrix, shepard_coefficient, point)
    for dim in axes(field, 1)
        field[dim, point] /= shepard_coefficient[point]
    end

    return field
end

function divide_by_shepard_coefficient!(field::AbstractArray{T, 3}, shepard_coefficient,
                                        point) where {T}
    for j in axes(field, 2), i in axes(field, 1)
        field[i, j, point] /= shepard_coefficient[point]
    end

    return field
end

function sample_face_with_fixed_size(face_points::NTuple{3}, resolution)
    # Verify that points are in 3D space
    if any(length.(face_points) .!= 3)
        throw(ArgumentError("all points must be 3D coordinates"))
    end

    point1_ = SVector{3}(face_points[1])
    point2_ = SVector{3}(face_points[2])
    point3_ = SVector{3}(face_points[3])

    # Vectors defining the edges of the parallelogram
    edge1 = point2_ - point1_
    edge2 = point3_ - point1_

    # Check if the points are collinear
    if isapprox(norm(cross(edge1, edge2)), 0; atol=eps())
        throw(ArgumentError("the vectors `AB` and `AC` must not be collinear"))
    end

    # Determine the number of points along each edge
    num_points_edge1 = ceil(Int, norm(edge1) / resolution)
    num_points_edge2 = ceil(Int, norm(edge2) / resolution)

    coords = zeros(3, (num_points_edge1 + 1) * (num_points_edge2 + 1))

    index = 1
    for i in 0:num_points_edge1
        for j in 0:num_points_edge2
            point_on_plane = point1_ + (i / num_points_edge1) * edge1 +
                             (j / num_points_edge2) * edge2
            coords[:, index] = point_on_plane
            index += 1
        end
    end

    resolution_new = min(norm(edge1 / num_points_edge1),
                         norm(edge2 / num_points_edge2))

    if !isapprox(resolution, resolution_new, rtol=sqrt(eps(resolution)))
        @info "The desired face size is not a multiple of the resolution $resolution." *
              "\nNew resolution is set to $resolution_new."
    end

    return coords
end
