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
coordinate directions as `cutout_min` and `cutout_max`.

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
        # Each layer has thickness `particle_spacing`
        n_layers = round(Int, radius / particle_spacing)

        if n_layers < 1
            # Just return one particle at the center.
            return collect(reshape(center, (length(center), 1)))
        end

        layer_increment = -radius / n_layers
    end

    coords = zeros(length(center), 0)

    for layer in 0:(n_layers - 1)
        sphere_coords = round_sphere(particle_spacing, radius + layer_increment * layer,
                                     center, layer=layer)
        coords = hcat(coords, sphere_coords)
    end

    return coords
end

function round_sphere(particle_spacing, radius, center::SVector{2}; layer=0)
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

# This is an implementation of the Offset Fibonacci Sphere from
# https://web.archive.org/web/20230610125433/https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
function round_sphere(particle_spacing, radius, center::SVector{3}; layer=0)
    # The number of particles can either be calculated in 2D or in 3D.
    # Let δ be the particle spacing and r the sphere radius.
    #
    # In 2D, the volume (surface area) of a particle is δ^2 and the surface area of
    # the sphere with radius r is 4pi r^2. Therefore, we need to have
    # n = 4pi r^2 / δ^2 particles on the sphere surface.
    #
    # In 3D, the volume of a particle is δ^3 and the volume of the sphere shell with
    # inner radius r - δ/2 and outer radius r + δ/2 is 4pi/3 * ((r + δ/2)^3 - (r - δ/2)^3).
    # The number of particles is then
    # n = 4pi / (3 δ^3) * ((r + δ/2)^3 - (r - δ/2)^3) = 4pi r^2 / δ^2 + pi/3,
    # which is the same as in the 2D calculations, plus the additional term pi/3.
    #
    # Note that for large numbers of particles, this term becomes insignificant.
    # For small numbers of particles, my experiments showed much better results with
    # the 2D version without the additional term.
    n_particles = round(Int, 4pi * radius^2 / particle_spacing^2)

    if n_particles <= 2
        # Just return one particle at the center.
        return collect(reshape(center, (3, 1)))
    end

    coords = Array{Float64}(undef, 3, n_particles)

    golden_ratio = (1 + sqrt(5)) / 2

    # Epsilon for best average point distance
    epsilon = 0.36

    # Rotate sphere around x-axis and then z-axis to avoid all layers from having
    # singularities (North and South Pole) at the same point.
    # With the singularities separated like this, we get slightly better results.
    rotate_x = layer * 3pi / 7
    rotate_z = layer * 2pi / 3

    for i in 1:n_particles
        theta = 2pi * (i - 1) / golden_ratio
        phi = acos(1 - 2 * (i - 1 + epsilon) / (n_particles - 1 + 2epsilon))

        x = radius * cos(theta) * sin(phi)
        y = radius * sin(theta) * sin(phi)
        z = radius * cos(phi)

        # Rotate around x-axis
        x2 = x
        y2 = cos(rotate_x) * y - sin(rotate_x) * z
        z2 = sin(rotate_x) * y + cos(rotate_x) * z

        # Rotate around z-axis
        coords[1, i] = cos(rotate_z) * x2 - sin(rotate_z) * y2 + center[1]
        coords[2, i] = sin(rotate_z) * x2 + cos(rotate_z) * y2 + center[2]
        coords[3, i] = z2 + center[3]
    end

    return coords
end
