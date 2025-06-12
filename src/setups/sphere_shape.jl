"""
    SphereShape(particle_spacing, radius, center_position, density;
                sphere_type=VoxelSphere(), n_layers=-1, layer_outwards=false,
                cutout_min=(0.0, 0.0), cutout_max=(0.0, 0.0), tlsph=false,
                velocity=zeros(length(center_position)), mass=nothing, pressure=0.0)

Generate a sphere that is either completely filled (by default)
or hollow (by passing `n_layers`).

With the sphere type [`VoxelSphere`](@ref), a sphere of voxels (where particles are placed
in the voxel center) with a regular inner structure but corners on the surface is created.
Essentially, a grid of particles is generated and all particles outside the sphere are removed.
With the sphere type [`RoundSphere`](@ref), a perfectly round sphere with an imperfect inner
structure is created.

A cuboid can be cut out of the sphere by specifying the two corners in negative and positive
coordinate directions as `cutout_min` and `cutout_max`.

# Arguments
- `particle_spacing`:   Spacing between the particles.
- `radius`:             Radius of the sphere.
- `center_position`:    The coordinates of the center of the sphere.
- `density`:            Either a function mapping each particle's coordinates to its density,
                        or a scalar for a constant density over all particles.

# Keywords
- `sphere_type`:    Either [`VoxelSphere`](@ref) or [`RoundSphere`](@ref) (see
                    explanation above).
- `n_layers`:       Set to an integer greater than zero to generate a hollow sphere,
                    where the shell consists of `n_layers` layers.
- `layer_outwards`: When set to `false` (by default), `radius` is the outer radius
                    of the sphere. When set to `true`, `radius` is the inner radius
                    of the sphere. This is only used when `n_layers > 0`.
- `cutout_min`:     Corner in negative coordinate directions of a cuboid that is to be
                    cut out of the sphere.
- `cutout_max`:     Corner in positive coordinate directions of a cuboid that is to be
                    cut out of the sphere.
- `tlsph`:          With the [`TotalLagrangianSPHSystem`](@ref), particles need to be placed
                    on the boundary of the shape and not one particle radius away, as for fluids.
                    When `tlsph=true`, particles will be placed on the boundary of the shape.
- `velocity`:   Either a function mapping each particle's coordinates to its velocity,
                or, for a constant fluid velocity, a vector holding this velocity.
                Velocity is constant zero by default.
- `mass`:       Either `nothing` (default) to automatically compute particle mass from particle
                density and spacing, or a function mapping each particle's coordinates to its mass,
                or a scalar for a constant mass over all particles.
- `pressure`:   Either a function mapping each particle's coordinates to its pressure,
                or a scalar for a constant pressure over all particles. This is optional and
                only needed when using the [`EntropicallyDampedSPHSystem`](@ref).

# Examples
```jldoctest; output = false
# Filled circle with radius 0.5, center in (0.2, 0.4) and a particle spacing of 0.1
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0)

# Same as before, but perfectly round
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, sphere_type=RoundSphere())

# Hollow circle with ~3 layers, outer radius 0.5, center in (0.2, 0.4) and a particle
# spacing of 0.1.
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3)

# Same as before, but perfectly round
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3, sphere_type=RoundSphere())

# Hollow circle with 3 layers, inner radius 0.5, center in (0.2, 0.4) and a particle spacing
# of 0.1.
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3, layer_outwards=true)

# Filled circle with radius 0.1, center in (0.0, 0.0), particle spacing 0.1, but the
# rectangle [0, 1] x [-0.2, 0.2] is cut out.
SphereShape(0.1, 1.0, (0.0, 0.0), 1000.0, cutout_min=(0.0, -0.2), cutout_max=(1.0, 0.2))

# Filled 3D sphere with radius 0.5, center in (0.2, 0.4, 0.3) and a particle spacing of 0.1
SphereShape(0.1, 0.5, (0.2, 0.4, 0.3), 1000.0)

# Same as before, but perfectly round
SphereShape(0.1, 0.5, (0.2, 0.4, 0.3), 1000.0, sphere_type=RoundSphere())

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ InitialCondition{Float64}                                                                        │
│ ═════════════════════════                                                                        │
│ #dimensions: ……………………………………………… 3                                                                │
│ #particles: ………………………………………………… 518                                                              │
│ particle spacing: ………………………………… 0.1                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
function SphereShape(particle_spacing, radius, center_position, density;
                     sphere_type=VoxelSphere(), n_layers=-1, layer_outwards=false,
                     cutout_min=(0.0, 0.0), cutout_max=(0.0, 0.0), tlsph=false,
                     velocity=zeros(length(center_position)), mass=nothing, pressure=0)
    if particle_spacing < eps()
        throw(ArgumentError("`particle_spacing` needs to be positive and larger than $(eps())"))
    end

    NDIMS = length(center_position)

    coordinates = sphere_shape_coords(sphere_type, particle_spacing, radius,
                                      SVector{NDIMS}(center_position),
                                      n_layers, layer_outwards, tlsph)

    # Convert tuples to vectors
    cutout_min_ = collect(cutout_min)
    cutout_max_ = collect(cutout_max)

    # Remove particles in cutout
    # TODO This should consider the particle radius as well
    has_cutout = norm(cutout_max_ - cutout_min_) > eps()
    function in_cutout(particle)
        return has_cutout &&
               all(cutout_min_ .<= view(coordinates, :, particle) .<= cutout_max_)
    end

    particles_not_in_cutout = map(!in_cutout, axes(coordinates, 2))
    coordinates = coordinates[:, particles_not_in_cutout]

    return InitialCondition(; coordinates, velocity, mass, density, pressure,
                            particle_spacing)
end

"""
    VoxelSphere()

Construct a sphere of voxels (where particles are placed in the voxel center)
with a regular inner structure but corners on the surface.
Essentially, a grid of particles is generated and all particles outside the sphere are removed.
The resulting sphere will have a perfect inner structure, but is not perfectly round,
as it will have corners (like a sphere in Minecraft).

!!! note "Usage"
    See [`SphereShape`](@ref) on how to use this.
"""
struct VoxelSphere end

"""
    RoundSphere(; start_angle=0.0, end_angle=2π)

Construct a sphere (or sphere segment) by nesting perfectly round concentric spheres.
The resulting ball will be perfectly round, but will not have a regular inner structure.

# Keywords
- `start_angle`: The starting angle of the sphere segment in radians. It determines the
                 beginning point of the segment. The default is set to `0.0` representing
                 the positive x-axis.
- `end_angle`: The ending angle of the sphere segment in radians. It defines the termination
               point of the segment. The default is set to `2pi`, completing a full sphere.

!!! note "Usage"
    See [`SphereShape`](@ref) on how to use this.

!!! warning "Warning"
    The sphere segment is intended for 2D geometries and hollow spheres. If used for filled
    spheres or in a 3D context, results may not be accurate.
"""
struct RoundSphere{AR}
    angle_range::AR

    function RoundSphere(; start_angle=0.0, end_angle=2pi)
        if start_angle > end_angle
            throw(ArgumentError("`end_angle` should be greater than `start_angle`"))
        end

        angle_range = (start_angle, end_angle)

        new{typeof(angle_range)}(angle_range)
    end
end

function sphere_shape_coords(::VoxelSphere, particle_spacing, radius, center_position,
                             n_layers, layer_outwards, tlsph)
    if n_layers > 0
        if layer_outwards
            inner_radius = radius
            outer_radius = radius + n_layers * particle_spacing

            if !tlsph
                # Put first layer of particles half a particle spacing outside of `radius`
                inner_radius += particle_spacing / 2
                outer_radius += particle_spacing / 2
            end
        else
            inner_radius = radius - n_layers * particle_spacing
            outer_radius = radius

            if !tlsph
                # Put first layer of particles half a particle spacing inside of `radius`
                inner_radius -= particle_spacing / 2
                outer_radius -= particle_spacing / 2
            end
        end
    else
        outer_radius = radius
        inner_radius = -1

        if !tlsph
            # Put first layer of particles half a particle spacing inside of `radius`
            outer_radius -= particle_spacing / 2
        end
    end

    NDIMS = length(center_position)
    ELTYPE = typeof(particle_spacing)
    coords = SVector{NDIMS, ELTYPE}[]

    n_particles_cube = round(Int, outer_radius / particle_spacing)

    # Loop over all indices in [-n_particles_cube, n_particles_cube]^NDIMS
    for i in CartesianIndices(ntuple(_ -> (-n_particles_cube):n_particles_cube, NDIMS))
        x = center_position + particle_spacing * SVector(Tuple(i))

        # Add a small tolerance of to make sure that spheres where the radius is
        # a multiple of the particle spacing are symmetric.
        # Otherwise, we have `norm(x - center) == radius`, which yields non-deterministic
        # results due to machine rounding errors.
        # Add the tolerance to the inner radius as well to avoid duplicate particles when
        # multiple concentric spheres are generated.
        if inner_radius + 10eps() < norm(x - center_position) <= outer_radius + 10eps()
            push!(coords, x)
        end
    end

    return reinterpret(reshape, ELTYPE, coords)
end

function sphere_shape_coords(sphere::RoundSphere, particle_spacing, radius, center,
                             n_layers, layer_outwards, tlsph)
    if n_layers > 0
        if layer_outwards
            inner_radius = radius
        else
            inner_radius = radius - n_layers * particle_spacing
        end

        if !tlsph
            # Put first layer of particles half a particle spacing outside of inner radius
            inner_radius += particle_spacing / 2
        end
    else
        if tlsph
            # Just create a sphere that is 0.5 particle spacing larger
            radius += particle_spacing / 2
        end

        # Each layer has thickness `particle_spacing`
        n_layers = round(Int, radius / particle_spacing)

        if n_layers < 1
            # Just return one particle at the center
            return collect(reshape(center, (length(center), 1)))
        end

        # Same as above, which puts the inner radius between 0 and `particle_spacing`
        inner_radius = max(0, radius - n_layers * particle_spacing + particle_spacing / 2)
    end

    coords = zeros(eltype(particle_spacing), length(center), 0)

    for layer in 0:(n_layers - 1)
        sphere_coords = round_sphere(sphere, particle_spacing,
                                     inner_radius + layer * particle_spacing, center)
        coords = hcat(coords, convert(Array{eltype(coords)}, sphere_coords))
    end

    return coords
end

function round_sphere(sphere, particle_spacing, radius, center::SVector{2})
    (; angle_range) = sphere

    theta = angle_range[2] - angle_range[1]
    n_particles = round(Int, theta * radius / particle_spacing)

    if n_particles <= 2
        # 2 or less particles produce weird, asymmetric results.
        # Just return one particle at the center.
        return collect(reshape(center, (2, 1)))
    end

    if !isapprox(theta, 2pi)
        t = LinRange(angle_range[1], angle_range[2], n_particles + 1)
    else
        # Remove the last particle at 2pi, which overlaps with the first at 0
        t = LinRange(angle_range[1], angle_range[2], n_particles + 1)[1:(end - 1)]
    end

    particle_coords = Array{Float64, 2}(undef, 2, length(t))

    for i in axes(particle_coords, 2)
        particle_coords[:, i] = center + radius * SVector(cos(t[i]), sin(t[i]))
    end

    return particle_coords
end

function round_sphere(sphere, particle_spacing, radius, center::SVector{3})
    # The number of particles can either be calculated in 2D or in 3D.
    # Let δ be the particle spacing and r the sphere radius.
    #
    # The volume of a particle is δ^3 and the volume of the sphere shell with
    # inner radius r - δ/2 and outer radius r + δ/2 is 4pi/3 * ((r + δ/2)^3 - (r - δ/2)^3).
    # The number of particles is then
    # n = 4pi / (3 δ^3) * ((r + δ/2)^3 - (r - δ/2)^3) = 4pi r^2 / δ^2 + pi/3.
    #
    # For small numbers of particles, we get better results without the term pi/3.
    # Omitting the term for the inner layers yields results with only ~5 particles less than
    # the theoretically optimal number of particles for the target density.
    n_particles = round(Int, 4pi * radius^2 / particle_spacing^2 + pi / 3)
    if n_particles < 300
        n_particles = round(Int, 4pi * radius^2 / particle_spacing^2)
    end

    # With fewer than 5 particles, this doesn't work properly
    if n_particles < 5
        if n_particles == 4
            # Return tetrahedron
            return [+1 -1 -1 +1;
                    +1 -1 +1 -1;
                    +1 +1 -1 -1] * radius / sqrt(3) .+ center
        elseif n_particles == 3
            # Return 2D triangle
            y = sin(2pi / 3)
            return [1 -1/2 -1/2;
                    0 y -y;
                    0 0 0] * radius .+ center
        elseif n_particles == 2
            # Return two particles
            return [-1 1;
                    0 0;
                    0 0] * radius .+ center
        else
            return collect(reshape(center, (3, 1)))
        end
    end

    # The following is a slightly adapted version of the "recursive zonal equal area
    # partition" of the sphere as explained by Leopardi (2006).
    #
    # With the equal area partition, the density at the poles is too high.
    # Instead, we slightly increase the area of the poles and modify the algorithm
    # accordingly.
    #
    # References:
    # - Paul Leopardi.
    #   "A partition of the unit sphere into regions of equal area and small diameter".
    #   In: Electronic Transactions on Numerical Analysis 25 (2006), pages 309-327.
    #   [http://eudml.org/doc/129860](http://eudml.org/doc/129860).

    # This is the Θ function, which is defined by Leopardi only as the inverse of V, without
    # giving a closed formula.
    theta(v) = acos(1 - v / 2pi)

    # Ideal area of the equal area partition
    ideal_area = 4pi / n_particles

    # Increase polar area to avoid higher density at the poles
    polar_area = 1.23ideal_area

    polar_radius = theta(polar_area)

    # Divide the remaining surface area equally
    collar_cell_area = (4pi - 2polar_area) / (n_particles - 2)

    # Strictly following Leopardi here. The collars should have equiangular spacing.
    collar_angle = sqrt(collar_cell_area)
    n_collars = max(1, round(Int, (pi - 2polar_radius) / collar_angle))
    fitting_collar_angle = (pi - 2polar_radius) / n_collars

    collar_area = [2pi * (cos(polar_radius + (j - 2) * fitting_collar_angle) -
                    cos(polar_radius + (j - 1) * fitting_collar_angle))
                   for j in 2:(n_collars + 1)]

    # Here, we count the poles as well
    ideal_number_cells = collar_area / collar_cell_area
    pushfirst!(ideal_number_cells, 1)
    push!(ideal_number_cells, 1)

    # Cumulative rounding to maintain the total number of cells
    actual_number_cells = ones(Int, length(ideal_number_cells))
    a = zeros(length(ideal_number_cells))
    for j in 2:(n_collars + 1)
        actual_number_cells[j] = round(Int, ideal_number_cells[j] + a[j - 1])

        a[j] = a[j - 1] + ideal_number_cells[j] - actual_number_cells[j]
    end

    collar_start_latitude = [theta(polar_area +
                                   sum(actual_number_cells[2:(j - 1)]) * collar_cell_area)
                             for j in 2:(n_collars + 2)]

    # Put particles in the center of each collar
    collar_latitude = [0.5 * (collar_start_latitude[i] + collar_start_latitude[i + 1])
                       for i in 1:n_collars]

    # Put the first and last particle on the pole
    pushfirst!(collar_latitude, 0.0)
    push!(collar_latitude, pi)

    # To compute the particle positions in each collar, we use the 2D `round_sphere`
    # function to generate a circle.
    particle_coords = zeros(3, 0)

    for circle in 1:(n_collars + 2)
        z = radius * cos(collar_latitude[circle])
        circle_radius = radius * sin(collar_latitude[circle])

        circle_spacing = 2pi * circle_radius / actual_number_cells[circle]

        # At the poles, `circle_radius` is zero, so we can pass any positive spacing
        if circle_spacing < eps()
            circle_spacing = 1.0
        end

        circle_coords_2d = round_sphere(sphere, circle_spacing, circle_radius,
                                        SVector(center[1], center[2]))
        circle_coords_3d = vcat(circle_coords_2d,
                                center[3] .+ z * ones(1, size(circle_coords_2d, 2)))

        particle_coords = hcat(particle_coords, circle_coords_3d)
    end

    return particle_coords
end
