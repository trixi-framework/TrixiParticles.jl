"""
    SphereShape(particle_spacing, radius, center_position, density;
                sphere_type=VoxelSphere(), n_layers=-1, layer_outwards=false,
                cutout_min=(0.0, 0.0), cutout_max=(0.0, 0.0),
                init_velocity=zeros(length(center_position)))

Generate a sphere that is either completely filled (by default)
or hollow (by passing `n_layers`).

With the sphere type [`VoxelSphere`](@ref), a sphere of voxels (where particles are placed
in the voxel center) with a regular inner structure but small corners on the surface
is created.
With the sphere type [`RoundSphere`](@ref), a perfectly round sphere with an imperfect inner
structure is created.

A cuboid can be cut out of the sphere by specifying the two corners in negative and positive
coordinate directions as `cutoff_min` and `cutoff_max`.

# Arguments
- `particle_spacing`:   Spacing between the particles.
- `radius`:             Radius of the sphere.
- `center_position`:    The coordinates of the center of the sphere.
- `density`:            Density of the sphere.

# Keywords
- `sphere_type`:        Either [`VoxelSphere`](@ref) or [`RoundSphere`](@ref) (see
                        explanation above).
- `n_layers`:           Set to an integer greater than zero to generate a hollow sphere,
                        where the shell consists of `n_layers` layers.
- `layer_outwards`:     When set to `false` (by default), `radius` is the outer radius
                        of the sphere. When set to `true`, `radius` is the inner radius
                        of the sphere. This is only used when `n_layers > 0`.
- `cutout_min`:         Corner in negative coordinate directions of a cuboid that is to be
                        cut out of the sphere.
- `cutout_max`:         Corner in positive coordinate directions of a cuboid that is to be
                        cut out of the sphere.
- `init_velocity`:      Initial velocity vector to be assigned to each particle.

# Examples
```julia
# Filled circle with radius 0.5, center in (0.2, 0.4) and a particle spacing of 0.1
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0)

# Same as before, but perfectly round
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, sphere_type=RoundSphere())

# Hollow circle with ~3 layers, outer radius 0.5, center in (0.2, 0.4) and a particle
# spacing of 0.1.
```julia
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3)
```

# Same as before, but perfectly round
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3, sphere_type=RoundSphere())

# Hollow circle with 3 layers, inner radius 0.5, center in (0.2, 0.4) and a particle spacing
# of 0.1.
```julia
SphereShape(0.1, 0.5, (0.2, 0.4), 1000.0, n_layers=3, layer_outwards=true)

# Filled circle with radius 0.1, center in (0.0, 0.0), particle spacing 0.1, but the
# rectangle [0, 1] x [-0.2, 0.2] is cut out.
SphereShape(0.1, 1.0, (0.0, 0.0), 1000.0, cutout_min=(0.0, -0.2, cutout_max=(1.0, 0.2)))

# Filled 3D sphere with radius 0.5, center in (0.2, 0.4, 0.3) and a particle spacing of 0.1
SphereShape(0.1, 0.5, (0.2, 0.4, 0.3), 1000.0)

# Same as before, but perfectly round
SphereShape(0.1, 0.5, (0.2, 0.4, 0.3), 1000.0, sphere_type=RoundSphere())
```
````
"""
function SphereShape(particle_spacing, radius, center_position, density;
                     sphere_type=VoxelSphere(), n_layers=-1, layer_outwards=false,
                     cutout_min=(0.0, 0.0), cutout_max=(0.0, 0.0),
                     init_velocity=zeros(length(center_position)))
    if particle_spacing < eps()
        throw(ArgumentError("`particle_spacing` needs to be positive and larger than $(eps())"))
    end

    if density < eps()
        throw(ArgumentError("`density` needs to be positive and larger than $(eps())"))
    end

    NDIMS = length(center_position)
    ELTYPE = eltype(particle_spacing)

    coordinates = sphere_shape_coords(sphere_type, particle_spacing, radius,
                                      SVector{NDIMS}(center_position),
                                      n_layers, layer_outwards)

    # Convert tuples to vectors
    cutout_min_ = collect(cutout_min)
    cutout_max_ = collect(cutout_max)

    # Remove particles in cutout
    has_cutout = norm(cutout_max_ - cutout_min_) > eps()
    function in_cutout(particle)
        return has_cutout &&
               all(cutout_min_ .<= view(coordinates, :, particle) .<= cutout_max_)
    end

    particles_not_in_cutout = map(!in_cutout, axes(coordinates, 2))
    coordinates = coordinates[:, particles_not_in_cutout]

    n_particles = size(coordinates, 2)
    densities = density * ones(ELTYPE, n_particles)
    masses = density * particle_spacing^NDIMS * ones(ELTYPE, n_particles)
    velocities = init_velocity .* ones(ELTYPE, size(coordinates))

    return InitialCondition(coordinates, velocities, masses, densities)
end

"""
    VoxelSphere()

Construct a sphere of voxels (where particles are placed
    in the voxel center) with a regular inner structure but small corners on the surface
    is created.
Construct a sphere by generating particles on a Cartesian grid and removing those with
a distance from the sphere's center greater than the sphere's radius.
The resulting sphere will have a perfect inner structure, but is not perfectly round,
as it will have small corners (like a sphere in Minecraft).
"""
struct VoxelSphere end

"""
    RoundSphere()

Construct a sphere by nesting perfectly round concentric spheres with a thickness
of one particle.
The resulting ball will be perfectly round, but will not have a regular inner structure.
"""
struct RoundSphere end

function sphere_shape_coords(::VoxelSphere, particle_spacing, radius, center_position,
                             n_layers, layer_outwards)
    if n_layers > 0
        if layer_outwards
            inner_radius = radius
            outer_radius = radius + n_layers * particle_spacing
        else
            inner_radius = radius - n_layers * particle_spacing
            outer_radius = radius
        end
    else
        inner_radius = -1.0
        outer_radius = radius
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

function sphere_shape_coords(::RoundSphere, particle_spacing, radius, center,
                             n_layers, layer_outwards)
    if n_layers > 0
        layer_increment = if layer_outwards
            particle_spacing
        else
            -particle_spacing
        end
    else
        # Choose `n_layers` and `layer_increment` exactly such that the last layer is
        # one particle in the center.
        n_layers = round(Int, radius / particle_spacing + 1)
        layer_increment = if n_layers > 1
            -radius / (n_layers - 1)
        else
            0.0
        end
    end

    coords = zeros(length(center), 0)

    for layer in 0:(n_layers - 1)
        sphere_coords = hollow_sphere(particle_spacing, radius + layer_increment * layer,
                                      center)
        coords = hcat(coords, sphere_coords)
    end

    return coords
end

function hollow_sphere(particle_spacing, radius, center::SVector{2})
    n_particles = round(Int, 2pi * radius / particle_spacing)

    if n_particles <= 2
        # 2 or less particles produce weird, asymmetric results.
        # Just return one particle at the center.
        return collect(reshape(center, (2, 1)))
    end

    # Remove the last particle at 2pi, which overlaps with the first at 0
    t = LinRange(0, 2pi, n_particles + 1)[1:(end - 1)]

    particle_coords = Array{Float64, 2}(undef, 2, length(t))

    for i in axes(particle_coords, 2)
        particle_coords[:, i] = center + radius * SVector(cos(t[i]), sin(t[i]))
    end

    return particle_coords
end

function hollow_sphere(particle_spacing, radius, center::SVector{3})
    # Number of circles from North Pole to South Pole (including the poles)
    n_circles = round(Int, pi * radius / particle_spacing + 1)

    if n_circles <= 2
        # 2 or less circles produce weird, asymmetric results.
        # Just return one particle at the center.
        return collect(reshape(center, (3, 1)))
    end

    polar_angle_increment = pi / (n_circles - 1)

    particle_coords = zeros(3, 0)

    for circle in 1:n_circles
        polar_angle = (circle - 1) * polar_angle_increment

        z = radius * cos(polar_angle)
        circle_radius = sqrt(radius^2 - z^2)

        circle_coords_2d = hollow_sphere(particle_spacing, circle_radius,
                                         SVector(center[1], center[2]))
        circle_coords_3d = vcat(circle_coords_2d,
                                center[3] .+ z * ones(1, size(circle_coords_2d, 2)))

        particle_coords = hcat(particle_coords, circle_coords_3d)
    end

    return particle_coords
end
