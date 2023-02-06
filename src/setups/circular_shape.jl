"""
    CircularShape(R, x_center, y_center, particle_spacing;
                  shape_type=FillCircle(), density=0.0)

Either a circle filled with particles or a circumference drawn by particles.

# Arguments
- `R`:                      Radius of the circle
- `x_center`, `y_center`:   Center of the circle in x and y direction, respectively.
- `particle_spacing`:       Spacing betweeen the particles.

# Keywords
- `shape_type`:    `Type` to specify the circular shape (see [`FillCircle`](@ref) and [`DrawCircle`](@ref))
- `density`:       Specify the density if the `densities` or `masses` fields will be used

# Fields
- `coordinates::Matrix`: Coordinates of the particles
- `masses::Vector`: Masses of the particles
- `densities::Vector`: Densities of the particles

For adding a recess in the particle filled circle or for only drawing the circumference
see [`FillCircle`](@ref) and [`DrawCircle`](@ref) respectively.
"""
struct CircularShape{NDIMS, ELTYPE <: Real}
    coordinates      :: Array{ELTYPE, 2}
    velocities       :: Array{ELTYPE, 2}
    masses           :: Vector{ELTYPE}
    densities        :: Vector{ELTYPE}
    particle_spacing :: ELTYPE
    n_particles      :: Int

    function CircularShape(R, x_center, y_center, particle_spacing;
                           shape_type=FillCircle(), density=0.0)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        n_particles = size(coordinates, 2)
        coordinates = generate_particles(shape_type, R, x_center, y_center,
                                         particle_spacing)

        densities = density * ones(ELTYPE, n_particles)
        masses = density * particle_spacing^2 * ones(ELTYPE, n_particles)

        return new{NDIMS, ELTYPE}(coordinates, masses, densities,
                                  particle_spacing, n_particles)
    end
end

"""
    FillCircle(; x_recess = (typemax(Int), typemax(Int)),
                 y_recess = (typemax(Int), typemax(Int)))

Particle filled circle (required by [`CircularShape`](@ref)).

The particles are arranged in an equidistiant grid
where the distance between the points is determined by the `particle_spacing`.
For adding a recess see example below.

# Keywords
- `x_recess`: Tuple for recess start and end coordinates in x direction
- `y_recess`: Tuple for recess start and end coordinates in y direction

# Example

Particle filled circle with recess:
```julia
FillCircle(x_recess=(0.5, recess_length), y_recess=(0.0, recess_height))
```

"""
struct FillCircle{ELTYPE <: Real}
    x_recess::NTuple{2, ELTYPE}
    y_recess::NTuple{2, ELTYPE}

    function FillCircle(; x_recess=(typemax(Int), typemax(Int)),
                        y_recess=(typemax(Int), typemax(Int)))
        return new{eltype(x_recess)}(x_recess, y_recess)
    end
end

"""
    DrawCircle(; n_layers=1, layer_inwards=false)

Circumference drawn by particles (required by [`CircularShape`](@ref)).

Unlike in [`FillCircle`](@ref), the particles are parametrized in a way
that the distance between neighboring particles is the `particle_spacing`.

Multiple layers are generated by calling the function additionally with the number of layers (see example).

# Keywords
- `n_layers`: Number of layers
- `layer_inwards`: Boolean to extend layers inwards.

# Example

Circumference with one layer:
```julia
DrawCircle()
```

Circumference with multiple layers extending outwards:
```julia
DrawCircle(n_layers=3)
```

Circumference with multiple layers extending inwards:
```julia
DrawCircle(n_layers=3, layer_inwards=true)
```
"""
struct DrawCircle{}
    n_layers      :: Int
    layer_inwards :: Bool

    function DrawCircle(; n_layers=1, layer_inwards=false)
        return new{}(n_layers, layer_inwards)
    end
end

function generate_particles(shape::FillCircle, R, x_center, y_center, particle_spacing)
    @unpack x_recess, y_recess = shape

    x_vec = Vector{Float64}(undef, 0)
    y_vec = Vector{Float64}(undef, 0)

    r(x, y) = sqrt((x - x_center)^2 + (y - y_center)^2)

    # recess condition
    recess(x, y) = (y_recess[2] >= y >= y_recess[1] &&
                    x_recess[2] >= x >= x_recess[1])

    n_particles = round(Int, R / particle_spacing)

    for j in (-n_particles):n_particles,
        i in (-n_particles):n_particles

        x = x_center + i * particle_spacing
        y = y_center + j * particle_spacing

        if r(x, y) < R && !recess(x, y)
            append!(x_vec, x)
            append!(y_vec, y)
        end
    end

    particle_coords = Array{Float64, 2}(undef, 2, length(x_vec))
    particle_coords[1, :] = x_vec
    particle_coords[2, :] = y_vec

    return particle_coords
end

function generate_particles(shape::DrawCircle, R, x_center, y_center, particle_spacing)
    @unpack n_layers, layer_inwards = shape

    x_vec = Vector{Float64}(undef, 0)
    y_vec = Vector{Float64}(undef, 0)

    layers = if layer_inwards
        (-n_layers + 1):0
    else
        0:(n_layers - 1)
    end

    for layer in layers
        coords = draw_circle(R + particle_spacing * layer, x_center, y_center,
                             particle_spacing)
        append!(x_vec, coords[1, :])
        append!(y_vec, coords[2, :])
    end

    particle_coords = Array{Float64, 2}(undef, 2, size(x_vec, 1))
    particle_coords[1, :] = x_vec
    particle_coords[2, :] = y_vec

    return particle_coords
end

function draw_circle(R, x_center, y_center, particle_spacing)
    n_particles = round(Int, 2pi * R / particle_spacing)

    # Remove the last particle at 2pi, which overlaps with the first at 0
    t = LinRange(0, 2pi, n_particles + 1)[1:(end - 1)]

    particle_coords = Array{Float64, 2}(undef, 2, length(t))

    for i in eachindex(t)
        particle_coords[1, i] = R * cos(t[i]) + x_center
        particle_coords[2, i] = R * sin(t[i]) + y_center
    end

    return particle_coords
end
