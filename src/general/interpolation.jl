using LinearAlgebra

@doc raw"""
    interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system, sol;
                         smoothing_length=ref_system.smoothing_length, cut_off_bnd=true)

Interpolates properties along a plane in a TrixiParticles simulation.
The region for interpolation is defined by its lower left and top right corners,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points within the defined region,
spaced uniformly according to the given resolution.

# Arguments
- `min_corner`: The lower left corner of the interpolation region.
- `max_corner`: The top right corner of the interpolation region.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`: The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`: The solution state from which the properties are interpolated.

# Keywords
- `cut_off_bnd`: `cut_off_bnd`: Boolean to indicate if quantities should be set to zero when a
                  point is "closer" to a boundary than to the fluid system
                  (see an explanation for "closer" below). Defaults to `true`.
- `smoothing_length`: The smoothing length used in the interpolation. Default is `ref_system.smoothing_length`.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point within the plane.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used, which is not as
      accurate as a real surface reconstruction.

# Examples
```@meta
DocTestSetup = quote
    using TrixiParticles
    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing)
    ref_system = fluid_system
end
```

```jldoctest; output = false
# Interpolating across a plane from [0.0, 0.0] to [1.0, 1.0] with a resolution of 0.2
results = interpolate_plane_2d([0.0, 0.0], [1.0, 1.0], 0.2, semi, ref_system, sol)

# output
(density = [1070.7435921185397, 1070.8842613465367, 1070.884261346536, 1070.7435921185408, 1058.6792713899165, 1058.647116710229, 1058.80276188339, 1058.80276188339, 1058.6471167102286, 1058.6792713899163  …  1027.3958134838153, 1027.3958134838151, 1027.3649667675872, 1027.3446543930306, 1009.5794867019508, 1009.5376041572841, 1009.5410892933542, 1009.5410892933543, 1009.5376041572842, 1009.5794867019508], neighbor_count = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16  …  16, 16, 16, 16, 16, 16, 16, 16, 16, 16], coord = SVector{2, Float64}[[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [0.0, 0.2], [0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2], [1.0, 0.2]  …  [0.4, 0.6], [0.6, 0.6], [0.8, 0.6], [1.0, 0.6], [0.0, 0.8], [0.2, 0.8], [0.4, 0.8], [0.6, 0.8], [0.8, 0.8], [1.0, 0.8]], velocity = SVector{2, Float64}[[-0.001839000309344367, -0.07726459491325653], [-6.485216827564495e-6, -0.07728480183984775], [6.485216824490022e-6, -0.07728480183984798], [0.0018390003093481644, -0.07726459491325452], [-0.048572649853344495, -0.006654655474058412], [-0.00168204451542095, -0.004085927899958911], [-5.708002023611255e-6, -0.003932804866269535], [5.708002024095139e-6, -0.003932804866269714], [0.0016820445154198482, -0.004085927899958546], [0.04857264985336423, -0.006654655474058595]  …  [-8.970488320934665e-7, -0.0011005164274695424], [8.970488310562758e-7, -0.0011005164274695857], [0.0003194129696719864, -0.001177340741919144], [0.010958436149271052, -0.002774677873911282], [-0.0011494135004953357, -0.0017215313858909198], [-2.9786389661093735e-5, -0.0005635794240165437], [-1.151251817222545e-7, -0.0005314364911879212], [1.1512518126262214e-7, -0.0005314364911875398], [2.978638966110058e-5, -0.0005635794240161471], [0.0011494135004991072, -0.001721531385892317]], pressure = [8768.179397867027, 8789.363347603688, 8789.363347603616, 8768.179397867163, 7010.5765787106675, 7005.670267047256, 7027.607602746006, 7027.607602745984, 7005.670267047249, 7010.576578710651  …  2978.1844430792635, 2978.184443079225, 2974.543905692346, 2972.1691248378456, 988.8567285108618, 984.4672719971668, 984.8414277183937, 984.8414277183853, 984.467271997168, 988.8567285108505])
```
"""
function interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system, sol;
                              smoothing_length=ref_system.smoothing_length,
                              cut_off_bnd=true)
    dims = length(min_corner)
    if dims != 2 || length(max_corner) != 2
        throw(ArgumentError("function is intended for 2D coordinates only"))
    end

    if any(min_corner .> max_corner)
        throw(ArgumentError("`min_corner` should be smaller than `max_corner` in every dimension"))
    end

    # Calculate the number of points in each dimension based on the resolution
    no_points_x = ceil(Int, (max_corner[1] - min_corner[1]) / resolution) + 1
    no_points_y = ceil(Int, (max_corner[2] - min_corner[2]) / resolution) + 1

    x_range = range(min_corner[1], max_corner[1], length=no_points_x)
    y_range = range(min_corner[2], max_corner[2], length=no_points_y)

    # Generate points within the plane
    points_coords = [SVector(x, y) for x in x_range, y in y_range]

    results = interpolate_point(points_coords, semi, ref_system, sol,
                                smoothing_length=smoothing_length,
                                cut_off_bnd=cut_off_bnd)

    # Find indices where neighbor_count > 0
    indices = findall(x -> x > 0, results.neighbor_count)

    # Filter all arrays in the named tuple using these indices
    filtered_results = map(x -> x[indices], results)

    return filtered_results
end

@doc raw"""
    interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system, sol;
                         smoothing_length=ref_system.smoothing_length, cut_off_bnd=true)

Interpolates properties along a plane in a 3D space in a TrixiParticles simulation.
The plane for interpolation is defined by three points in 3D space,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points on a parallelogram within the plane defined by the three points, spaced uniformly according to the given resolution.

# Arguments
- `point1`: The first point defining the plane.
- `point2`: The second point defining the plane.
- `point3`: The third point defining the plane. The points must not be collinear.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`: The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`: The solution state from which the properties are interpolated.

# Keywords
- `cut_off_bnd`: Boolean to indicate if quantities should be set to zero when a
                  point is "closer" to a boundary than to the fluid system
                  (see an explanation for "closer" below). Defaults to `true`.
- `smoothing_length`: The smoothing length used in the interpolation. Default is `ref_system.smoothing_length`.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point within the plane.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
      accurate as a real surface reconstruction.

# Examples
```@meta
DocTestSetup = quote
    using TrixiParticles
    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_3d.jl"), tspan=(0.0, 0.01))
    ref_system = fluid_system
end
```

```jldoctest; output = false
# Interpolating across a plane defined by points [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], and [0.0, 1.0, 0.0]
# with a resolution of 0.1
results = interpolate_plane_3d([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1, semi, ref_system, sol)

# output
(density = [1068.4800174062125, 1069.5387406396876, 1069.6500988593036, 1069.6572192535955, 1069.6575176306608, 1069.6572192535957, 1069.6500988593032, 1069.5387406396874, 1068.480017406212, 1069.5387406396876  …  1069.5387406396871, 1068.480017406212, 1069.5387406396876, 1069.650098859303, 1069.657219253595, 1069.65751763066, 1069.6572192535955, 1069.6500988593032, 1069.5387406396871, 1068.4800174062118], neighbor_count = [56, 56, 56, 56, 56, 56, 56, 56, 56, 56  …  56, 56, 56, 56, 56, 56, 56, 56, 56, 56], coord = SVector{3, Float64}[[0.1, 0.1, 0.0], [0.1, 0.2, 0.0], [0.1, 0.3, 0.0], [0.1, 0.4, 0.0], [0.1, 0.5, 0.0], [0.1, 0.6, 0.0], [0.1, 0.7, 0.0], [0.1, 0.8, 0.0], [0.1, 0.9, 0.0], [0.2, 0.1, 0.0]  …  [0.8, 0.9, 0.0], [0.9, 0.1, 0.0], [0.9, 0.2, 0.0], [0.9, 0.3, 0.0], [0.9, 0.4, 0.0], [0.9, 0.5, 0.0], [0.9, 0.6, 0.0], [0.9, 0.7, 0.0], [0.9, 0.8, 0.0], [0.9, 0.9, 0.0]], velocity = SVector{3, Float64}[[-0.008198022498887462, -0.008198022498887446, -0.0740661893013865], [-0.009338596786690821, -0.0014372716580729275, -0.0762199212190053], [-0.009443315502499517, -0.00010076861971446308, -0.07622005585130646], [-0.009449451245755103, -4.077469080372514e-6, -0.07621888519053732], [-0.009449664930060442, 9.320362567215287e-16, -0.07621867614709486], [-0.009449451245754071, 4.0774690800038896e-6, -0.0762188851905383], [-0.009443315502499357, 0.00010076861971040222, -0.07622005585130633], [-0.00933859678669284, 0.0014372716580721985, -0.07621992121900546], [-0.008198022498886346, 0.008198022498892886, -0.07406618930138634], [-0.001437271658072922, -0.00933859678669082, -0.07621992121900525]  …  [0.0014372716580736892, 0.009338596786696064, -0.07621992121900374], [0.008198022498892928, -0.008198022498886206, -0.07406618930138625], [0.009338596786693368, -0.0014372716580744406, -0.0762199212190044], [0.009443315502504209, -0.00010076861971414696, -0.07622005585130617], [0.009449451245762616, -4.077469079684132e-6, -0.07621888519053682], [0.009449664930063452, 1.0300568990152858e-15, -0.07621867614709442], [0.009449451245761029, 4.077469079020453e-6, -0.07621888519053699], [0.009443315502503663, 0.00010076861971035304, -0.07622005585130456], [0.009338596786696044, 0.0014372716580736918, -0.07621992121900374], [0.0081980224988911, 0.008198022498891096, -0.0740661893013845]], pressure = [8430.375392490492, 8587.74895584953, 8604.405233498728, 8605.47155488239, 8605.516250263034, 8605.471554882351, 8604.405233498728, 8587.748955849534, 8430.375392490452, 8587.748955849533  …  8587.748955849473, 8430.37539249045, 8587.748955849493, 8604.40523349866, 8605.471554882355, 8605.516250262977, 8605.471554882319, 8604.405233498677, 8587.748955849473, 8430.375392490423])
```
"""
function interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system, sol;
                              smoothing_length=ref_system.smoothing_length,
                              cut_off_bnd=true)
    # Verify that points are in 3D space
    if length(point1) != 3 || length(point2) != 3 || length(point3) != 3
        throw(ArgumentError("all points must be 3D coordinates"))
    end

    point1_ = SVector{3}(point1)
    point2_ = SVector{3}(point2)
    point3_ = SVector{3}(point3)

    # Vectors defining the edges of the parallelogram
    edge1 = point2_ - point1_
    edge2 = point3_ - point1_

    # Check if the points are collinear
    if norm(cross(edge1, edge2)) == 0
        throw(ArgumentError("the points must not be collinear"))
    end

    # Determine the number of points along each edge
    num_points_edge1 = ceil(Int, norm(edge1) / resolution)
    num_points_edge2 = ceil(Int, norm(edge2) / resolution)

    # Create a set of points on the plane
    points_coords = Vector{SVector{3, Float64}}(undef,
                                                (num_points_edge1 + 1) *
                                                (num_points_edge2 + 1))
    index = 1
    for i in 0:num_points_edge1
        for j in 0:num_points_edge2
            point_on_plane = point1 + (i / num_points_edge1) * edge1 +
                             (j / num_points_edge2) * edge2
            points_coords[index] = point_on_plane
            index += 1
        end
    end

    # Interpolate using the generated points
    results = interpolate_point(points_coords, semi, ref_system, sol,
                                smoothing_length=smoothing_length,
                                cut_off_bnd=cut_off_bnd)

    # Filter results
    indices = findall(x -> x > 0, results.neighbor_count)
    filtered_results = map(x -> x[indices], results)

    return filtered_results
end

@doc raw"""
    interpolate_line(start, end_, no_points, semi, ref_system, sol; endpoint=true,
                     smoothing_length=ref_system.smoothing_length, cut_of_bnd=true)

Interpolates properties along a line in a TrixiParticles simulation.
The line interpolation is accomplished by generating a series of
evenly spaced points between `start` and `end_`.
If `endpoint` is `false`, the line is interpolated between the start and end points,
but does not include these points.

# Arguments
- `start`: The starting point of the line.
- `end_`: The ending point of the line.
- `n_points`: The number of points to interpolate along the line.
- `semi`: The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`: The solution state from which the properties are interpolated.

# Keywords
- `cut_off_bnd`: `cut_off_bnd`: Boolean to indicate if quantities should be set to zero when a
                  point is "closer" to a boundary than to the fluid system
                  (see an explanation for "closer" below). Defaults to `true`.
- `endpoint`: A boolean to include (`true`) or exclude (`false`) the end point in the interpolation. Default is `true`.
- `smoothing_length`: The smoothing length used in the interpolation. Default is `ref_system.smoothing_length`.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point along the line.

!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations
      along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
      accurate as a real surface reconstruction.

# Examples
```@meta
DocTestSetup = quote
    using TrixiParticles
    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing)
    ref_system = fluid_system
end
```

```jldoctest
# Interpolating along a line from [1.0, 0.0] to [1.0, 1.0] with 5 points
results = interpolate_line([1.0, 0.0], [1.0, 1.0], 5, semi, ref_system, sol)

# output
(density = [0.0, 1055.119085384885, 1035.6965900503906, 1014.1106862437452, 0.0], neighbor_count = [0, 16, 16, 16, 0], coord = SVector{2, Float64}[[1.0, 0.0], [1.0, 0.25], [1.0, 0.5], [1.0, 0.75], [1.0, 1.0]], velocity = SVector{2, Float64}[[0.0, 0.0], [0.04291716380076006, -0.0049235044574082575], [0.018401567437275865, -0.0032529808336723186], [0.0030293088936451554, -0.0021403468030886524], [0.0, 0.0]], pressure = [0.0, 6514.440958976941, 3978.2850396284425, 1475.2896923295314, 0.0])
```
"""
function interpolate_line(start, end_, n_points, semi, ref_system, sol; endpoint=true,
                          smoothing_length=ref_system.smoothing_length,
                          cut_off_bnd=true)
    start_svector = SVector{ndims(ref_system)}(start)
    end_svector = SVector{ndims(ref_system)}(end_)
    points_coords = range(start_svector, end_svector, length=n_points)

    if !endpoint
        points_coords = points_coords[2:(end - 1)]
    end

    return interpolate_point(points_coords, semi, ref_system, sol,
                             smoothing_length=smoothing_length,
                             cut_off_bnd=cut_off_bnd)
end

@doc raw"""
    interpolate_point(points_coords::Array{Array{Float64,1},1}, semi, ref_system, sol;
                      smoothing_length=ref_system.smoothing_length, cut_of_bnd=true)

    interpolate_point(point_coords, semi, ref_system, sol;
                      smoothing_length=ref_system.smoothing_length, cut_of_bnd=true)

Performs interpolation of properties at specified points or an array of points in a TrixiParticles simulation.

When given an array of points (`points_coords`), it iterates over each point and applies interpolation individually.
For a single point (`point_coords`), it performs the interpolation at that specific location.
The interpolation utilizes the same kernel function of the SPH simulation to weigh contributions from nearby particles.

# Arguments
- `points_coords`: An array of point coordinates, for which to interpolate properties.
- `point_coords`: The coordinates of a single point for interpolation.
- `semi`: The semidiscretization used in the SPH simulation.
- `ref_system`: The reference system defining the properties of the SPH particles.
- `sol`: The current solution state from which properties are interpolated.

# Keywords
- `cut_off_bnd`: Cut-off at the boundary.
- `smoothing_length`: The smoothing length used in the kernel function. Defaults to `ref_system.smoothing_length`.

# Returns
- For multiple points:  A `NamedTuple` of arrays containing interpolated properties at each point.
- For a single point: A `NamedTuple` of interpolated properties at the point.

# Examples
```@meta
DocTestSetup = quote
    using TrixiParticles
    trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing)
    ref_system = fluid_system
end
```

```jldoctest
# For a single point
result = interpolate_point([1.0, 0.5], semi, ref_system, sol)

# For multiple points
points = [[1.0, 0.5], [1.0, 0.6], [1.0, 0.7]]
results = interpolate_point(points, semi, ref_system, sol)

# output
(density = [1035.6965900503906, 1027.3446543930306, 1018.6272718984667], neighbor_count = [16, 16, 16], coord = [[1.0, 0.5], [1.0, 0.6], [1.0, 0.7]], velocity = [[0.018401567437275865, -0.0032529808336723186], [0.010958436149271052, -0.002774677873911282], [0.0052172774600182945, -0.002239521414344099]], pressure = [3978.2850396284425, 2972.1691248378456, 1973.1157186797843])
```
!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations
      along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
    accurate as a real surface reconstruction.
"""
function interpolate_point(points_coords::AbstractArray{<:AbstractArray}, semi, ref_system,
                           sol; smoothing_length=ref_system.smoothing_length,
                           cut_off_bnd=true)
    num_points = length(points_coords)
    coords = similar(points_coords)
    velocities = similar(points_coords)
    densities = Vector{Float64}(undef, num_points)
    pressures = Vector{Float64}(undef, num_points)
    neighbor_counts = Vector{Int}(undef, num_points)

    neighborhood_searches = process_neighborhood_searches(semi, sol, ref_system,
                                                          smoothing_length)

    for (i, point) in enumerate(points_coords)
        result = interpolate_point(SVector{ndims(ref_system)}(point), semi, ref_system, sol,
                                   neighborhood_searches, smoothing_length=smoothing_length,
                                   cut_off_bnd=cut_off_bnd)
        densities[i] = result.density
        neighbor_counts[i] = result.neighbor_count
        coords[i] = result.coord
        velocities[i] = result.velocity
        pressures[i] = result.pressure
    end

    return (density=densities, neighbor_count=neighbor_counts, coord=coords,
            velocity=velocities, pressure=pressures)
end

function interpolate_point(point_coords, semi, ref_system, sol;
                           smoothing_length=ref_system.smoothing_length,
                           cut_off_bnd=true)
    neighborhood_searches = process_neighborhood_searches(semi, sol, ref_system,
                                                          smoothing_length)

    return interpolate_point(SVector{ndims(ref_system)}(point_coords), semi, ref_system,
                             sol, neighborhood_searches, smoothing_length=smoothing_length,
                             cut_off_bnd=cut_off_bnd)
end

function process_neighborhood_searches(semi, sol, ref_system, smoothing_length)
    if isapprox(smoothing_length, ref_system.smoothing_length)
        # Update existing NHS
        update_nhs(sol.u[end].x[2], semi)
        neighborhood_searches = semi.neighborhood_searches[system_indices(ref_system, semi)]
    else
        ref_smoothing_kernel = ref_system.smoothing_kernel
        search_radius = compact_support(ref_smoothing_kernel, smoothing_length)
        neighborhood_searches = map(semi.systems) do system
            u = wrap_u(sol.u[end].x[2], system, semi)
            system_coords = current_coordinates(u, system)
            old_nhs = get_neighborhood_search(ref_system, system, semi)
            nhs = copy_neighborhood_search(old_nhs, search_radius, system_coords)
            return nhs
        end
    end

    return neighborhood_searches
end

@inline function interpolate_point(point_coords, semi, ref_system, sol,
                                   neighborhood_searches;
                                   smoothing_length=ref_system.smoothing_length,
                                   cut_off_bnd=true)
    interpolated_density = 0.0
    interpolated_velocity = zero(SVector{ndims(ref_system)})
    interpolated_pressure = 0.0

    shepard_coefficient = 0.0
    ref_id = system_indices(ref_system, semi)
    neighbor_count = 0
    other_density = 0.0
    ref_smoothing_kernel = ref_system.smoothing_kernel

    if cut_off_bnd
        systems = semi
    else
        # Don't loop over other systems
        systems = (ref_system,)
    end

    foreach_system(systems) do system
        system_id = system_indices(system, semi)
        nhs = neighborhood_searches[system_id]
        (; search_radius, periodic_box) = nhs

        v = wrap_v(sol.u[end].x[1], system, semi)
        u = wrap_u(sol.u[end].x[2], system, semi)

        system_coords = current_coordinates(u, system)

        # This is basically `for_particle_neighbor` unrolled
        for particle in eachneighbor(point_coords, nhs)
            coords = extract_svector(system_coords, Val(ndims(system)), particle)

            pos_diff = point_coords - coords
            distance2 = dot(pos_diff, pos_diff)
            pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2,
                                                            search_radius, periodic_box)
            if distance2 > search_radius^2
                continue
            end

            distance = sqrt(distance2)
            mass = hydrodynamic_mass(system, particle)
            kernel_value = kernel(ref_smoothing_kernel, distance, smoothing_length)
            m_W = mass * kernel_value

            if system_id == ref_id
                interpolated_density += m_W

                volume = mass / particle_density(v, system, particle)
                particle_velocity = current_velocity(v, system, particle)
                interpolated_velocity += particle_velocity * (volume * kernel_value)

                pressure = particle_pressure(v, system, particle)
                interpolated_pressure += pressure * (volume * kernel_value)
                shepard_coefficient += volume * kernel_value
            else
                other_density += m_W
            end

            neighbor_count += 1
        end
    end

    # Point is not within the ref_system
    if other_density > interpolated_density || shepard_coefficient < eps()
        return (density=0.0, neighbor_count=0, coord=point_coords,
                velocity=zero(SVector{ndims(ref_system)}), pressure=0.0)
    end

    return (density=interpolated_density / shepard_coefficient,
            neighbor_count=neighbor_count,
            coord=point_coords, velocity=interpolated_velocity / shepard_coefficient,
            pressure=interpolated_pressure / shepard_coefficient)
end
