@doc raw"""
    RectangularTank(particle_spacing, fluid_size, tank_size, fluid_density;
                    velocity=zeros(length(fluid_size)), fluid_mass=nothing,
                    pressure=0, acceleration=nothing, state_equation=nothing,
                    boundary_density=fluid_density, n_layers=1, spacing_ratio=1,
                    min_coordinates=zeros(length(fluid_size)),
                    faces=Tuple(trues(2 * length(fluid_size))),
                    coordinates_eltype=Float64)

Rectangular tank filled with a fluid to set up dam-break-style simulations.

# Arguments
- `particle_spacing`:   Spacing between the fluid particles. The type of this argument
                        determines the eltype of the initial condition.
- `fluid_size`:         The dimensions of the fluid as `(x, y)` (or `(x, y, z)` in 3D).
- `tank_size`:          The dimensions of the tank as `(x, y)` (or `(x, y, z)` in 3D).
- `fluid_density`:      The rest density of the fluid. Will only be used as default for
                        `boundary_density` when using a state equation.

# Keywords
- `velocity`:       Either a function mapping each particle's coordinates to its velocity,
                    or, for a constant fluid velocity, a vector holding this velocity.
                    Velocity is constant zero by default.
- `fluid_mass`:     By default, automatically compute particle mass from particle
                    density and spacing. Can also be a function mapping each particle's
                    coordinates to its mass, or a scalar for a constant mass over all particles.
- `pressure`:       Scalar to set the pressure of all particles to this value.
                    This is only used by the [`EntropicallyDampedSPHSystem`](@ref) and
                    will be overwritten when using an initial pressure function in the system.
                    Cannot be used together with hydrostatic pressure gradient.
- `acceleration`:   In order to initialize particles with a hydrostatic pressure gradient,
                    an acceleration vector can be passed. Note that only accelerations
                    in one coordinate direction and no diagonal accelerations are supported.
                    This will only change the pressure of the particles. When using the
                    [`WeaklyCompressibleSPHSystem`](@ref), pass a `state_equation` as well
                    to initialize the particles with the corresponding density and mass.
                    When using the [`EntropicallyDampedSPHSystem`](@ref), the pressure
                    will be overwritten when using an initial pressure function in the system.
                    This cannot be used together with the `pressure` keyword argument.
- `state_equation`: When calculating a hydrostatic pressure gradient by setting `acceleration`,
                    the `state_equation` will be used to set the corresponding density.
                    Cannot be used together with `density`.
- `boundary_density = fluid_density`: Density of each boundary particle.
- `n_layers = 1`:   Number of boundary layers.
- `spacing_ratio = 1`: Ratio of `particle_spacing` to boundary particle spacing.
                    A value of 2 means that the boundary particle spacing will be
                    half the fluid particle spacing.
- `min_coordinates`: Coordinates of the corner in negative coordinate directions.
                    By default this is set to the origin.
- `faces`:          By default all faces are generated. Set faces by passing a
                    bit-array of length 4 (2D) or 6 (3D) to generate the faces in the
                    normal direction: -x,+x,-y,+y,-z,+z.
- `coordinates_eltype = Float64`: Eltype of the particle coordinates.
                    See [the docs on GPU support](@ref gpu_support) for more information.

# Fields
- `fluid::InitialCondition`:    [`InitialCondition`](@ref) for the fluid.
- `boundary::InitialCondition`: [`InitialCondition`](@ref) for the boundary.
- `fluid_size::Tuple`:          Tuple containing the size of the fluid in each dimension after rounding.
- `tank_size::Tuple`:           Tuple containing the size of the tank in each dimension after rounding.

# Examples
```jldoctest; output = false, filter = r"RectangularTank.*", setup = :(particle_spacing = 0.1; water_width = water_depth = container_width = container_height = container_depth = 1.0; water_height = 0.5; fluid_density = 1000.0)
# 2D
setup = RectangularTank(particle_spacing, (water_width, water_height),
                        (container_width, container_height), fluid_density,
                        n_layers=2, spacing_ratio=3)

# 2D with hydrostatic pressure gradient.
# `state_equation` has to be the same as for the WCSPH system.
state_equation = StateEquationCole(sound_speed=10.0, exponent=1, reference_density=1000.0)
setup = RectangularTank(particle_spacing, (water_width, water_height),
                        (container_width, container_height), fluid_density;
                        acceleration=(0.0, -9.81), state_equation)

# 3D
setup = RectangularTank(particle_spacing, (water_width, water_height, water_depth),
                        (container_width, container_height, container_depth), fluid_density,
                        n_layers=2)

# output
RectangularTank{3, 6, Float64}(...) *the rest of this line is ignored by filter*
```

See also: [`reset_wall!`](@ref).
"""
struct RectangularTank{NDIMS, NDIMSt2, ELTYPE <: Real, F, B}
    fluid                     :: F
    boundary                  :: B
    fluid_size                :: NTuple{NDIMS, ELTYPE}
    tank_size                 :: NTuple{NDIMS, ELTYPE}
    faces_                    :: NTuple{NDIMSt2, Bool} # Store if face in dir exists (-x +x -y +y -z +z)
    face_indices              :: NTuple{NDIMSt2, Array{Int, 2}} # see `reset_wall!`
    particle_spacing          :: ELTYPE
    spacing_ratio             :: ELTYPE
    n_layers                  :: Int
    n_particles_per_dimension :: NTuple{NDIMS, Int}

    function RectangularTank(particle_spacing, fluid_size, tank_size, fluid_density;
                             velocity=zeros(length(fluid_size)), fluid_mass=nothing,
                             pressure=0, acceleration=nothing, state_equation=nothing,
                             boundary_density=fluid_density,
                             n_layers=1, spacing_ratio=1,
                             min_coordinates=zeros(length(fluid_size)),
                             faces=Tuple(trues(2 * length(fluid_size))),
                             coordinates_eltype=Float64)
        NDIMS = length(fluid_size)
        ELTYPE = eltype(particle_spacing)
        fluid_size_ = Tuple(convert.(ELTYPE, fluid_size))
        tank_size_ = Tuple(convert.(ELTYPE, tank_size))

        if particle_spacing < eps()
            throw(ArgumentError("`particle_spacing` needs to be positive and larger than $(eps())."))
        end

        if fluid_density < eps()
            throw(ArgumentError("`fluid_density` needs to be positive and larger than $(eps())."))
        end

        if length(tank_size) != NDIMS
            throw(ArgumentError("`tank_size` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        # Fluid particle data
        n_particles_per_dim,
        fluid_size_ = fluid_particles_per_dimension(fluid_size_, particle_spacing)

        # If sizes were equal before rounding, make sure they're equal after rounding as well
        for dim in 1:NDIMS
            if isapprox(fluid_size[dim], tank_size[dim])
                tank_size_ = setindex(tank_size_, fluid_size_[dim], dim)
            end
        end

        # Boundary particle data
        n_boundaries_per_dim,
        tank_size_ = boundary_particles_per_dimension(tank_size_, particle_spacing,
                                                      spacing_ratio)

        boundary_spacing = particle_spacing / spacing_ratio

        # The type of the particle spacing determines the eltype of the coordinates
        boundary_coordinates,
        face_indices,
        boundary_indices... = initialize_boundaries(convert.(coordinates_eltype,
                                                             boundary_spacing),
                                                    tank_size_, n_boundaries_per_dim,
                                                    n_layers, faces)

        boundary_masses = boundary_density * boundary_spacing^NDIMS *
                          ones(ELTYPE, size(boundary_coordinates, 2))
        boundary_densities = boundary_density * ones(ELTYPE, size(boundary_coordinates, 2))
        boundary_velocities = zeros(ELTYPE, size(boundary_coordinates))

        n_particles_per_dim,
        fluid_size_ = check_tank_overlap(fluid_size_, tank_size_,
                                         particle_spacing, n_particles_per_dim)

        normals = zeros(ELTYPE, size(boundary_coordinates))
        calculate_normals!(normals, boundary_coordinates, boundary_spacing,
                           face_indices, boundary_indices, faces, Val(NDIMS))

        boundary = InitialCondition(coordinates=boundary_coordinates,
                                    velocity=boundary_velocities,
                                    mass=boundary_masses, density=boundary_densities,
                                    particle_spacing=boundary_spacing, normals=normals)

        # Move the tank corner in the negative coordinate directions to the desired position
        boundary.coordinates .+= min_coordinates

        if norm(fluid_size) > eps()
            if state_equation !== nothing
                # Use hydrostatic pressure gradient and calculate density from inverse state
                # equation, so don't pass fluid density.
                fluid = RectangularShape(particle_spacing, n_particles_per_dim,
                                         zeros(NDIMS);
                                         velocity, pressure, acceleration, state_equation,
                                         mass=fluid_mass, coordinates_eltype)
            else
                fluid = RectangularShape(particle_spacing, n_particles_per_dim,
                                         zeros(NDIMS);
                                         density=fluid_density, velocity, pressure,
                                         acceleration, state_equation, mass=fluid_mass,
                                         coordinates_eltype)
            end
            # Move the tank corner in the negative coordinate directions to the desired position
            fluid.coordinates .+= min_coordinates
        else
            # Fluid is empty
            fluid = InitialCondition(coordinates=zeros(coordinates_eltype, NDIMS, 0),
                                     density=1.0,
                                     particle_spacing=convert(ELTYPE, particle_spacing))
        end

        return new{NDIMS, 2 * NDIMS, ELTYPE, typeof(fluid),
                   typeof(boundary)}(fluid, boundary, fluid_size_, tank_size_,
                                     faces, face_indices, particle_spacing, spacing_ratio,
                                     n_layers, n_particles_per_dim)
    end
end

# Return the geometric position of the wall for a given face and the indices
# of the boundary particles associated with this face.
function wall_position(face_id, dimension, indices, particle_coords, offset)
    if iseven(face_id) # Positive coordinate direction: right, top, or back
        return minimum(view(particle_coords, dimension, indices)) - offset
    else # Negative coordinate direction: left, bottom, or front
        return maximum(view(particle_coords, dimension, indices)) + offset
    end
end

# Compute the normals for boundary particles located at the intersection of one or multiple boundary 
# faces (i.e. for faces, edges and corners) by geometrically combining the normals of the intersecting `face_ids`.
function set_intersection_normals!(indices, face_ids, face_exists, particle_coords, offset,
                                   normals)
    # Do nothing if no particles belong to these faces
    # or if the intersection doesn't exist (not all faces of it activated).
    isempty(indices) && return
    all(face_exists[face_id] for face_id in face_ids) || return
    for face_id in face_ids
        dimension = div(face_id, 2, RoundUp)
        wall = wall_position(face_id, dimension, indices, particle_coords, offset)
        for particle in indices
            normals[dimension, particle] = particle_coords[dimension, particle] - wall
        end
    end
end

# 2D
function calculate_normals!(normals, boundary_coordinates, boundary_spacing,
                            face_indices, boundary_indices, faces, ::Val{2})
    corner_indices, = boundary_indices
    offset = boundary_spacing / 2

    for (face_id, indices) in enumerate(face_indices)
        set_intersection_normals!(indices, (face_id,), faces, boundary_coordinates, offset,
                                  normals)
    end

    # The order matches `edge_indices` from `initialize_boundaries` (important for `zip`).
    corner_faces = ((1, 3), (1, 4),
                    (2, 3), (2, 4))

    for (indices, face_ids) in zip(corner_indices, corner_faces)
        set_intersection_normals!(indices, face_ids, faces, boundary_coordinates, offset,
                                  normals)
    end
end

# 3D
function calculate_normals!(normals, boundary_coordinates, boundary_spacing,
                            face_indices, boundary_indices, faces, ::Val{3})
    corner_indices, edge_indices = boundary_indices
    offset = boundary_spacing / 2

    # Compute normals for the faces
    for (face_id, indices) in enumerate(face_indices)
        set_intersection_normals!(indices, (face_id,), faces, boundary_coordinates, offset,
                                  normals)
    end

    # The order matches `edge_indices` from `initialize_boundaries` (important for `zip`).
    edge_faces = ((1, 3), (1, 4), (1, 5), (1, 6),
                  (2, 3), (2, 4), (2, 5), (2, 6),
                  (3, 5), (3, 6), (4, 5), (4, 6))

    for (indices, face_ids) in zip(edge_indices, edge_faces)
        set_intersection_normals!(indices, face_ids, faces, boundary_coordinates, offset,
                                  normals)
    end

    # The order matches `corner_indices` from `initialize_boundaries` (important for `zip`).
    corner_faces = ((1, 3, 5), (1, 4, 5),
                    (1, 3, 6), (1, 4, 6),
                    (2, 3, 5), (2, 4, 5),
                    (2, 3, 6), (2, 4, 6))

    for (indices, face_ids) in zip(corner_indices, corner_faces)
        set_intersection_normals!(indices, face_ids, faces, boundary_coordinates, offset,
                                  normals)
    end
end

function round_n_particles(size, spacing, type)
    n_particles = round(Int, size / spacing)

    new_size = n_particles * spacing
    if round(new_size, digits=4) != round(size, digits=4)
        @info "The desired $type $size is not a multiple of the particle spacing " *
              "$spacing.\nNew $type is set to $new_size."
    end

    return n_particles, new_size
end

function fluid_particles_per_dimension(size::NTuple{2}, particle_spacing)
    n_particles_x,
    new_width = round_n_particles(size[1], particle_spacing,
                                  "fluid length in x-direction")
    n_particles_y,
    new_height = round_n_particles(size[2], particle_spacing,
                                   "fluid length in y-direction")

    return (n_particles_x, n_particles_y), (new_width, new_height)
end

function fluid_particles_per_dimension(size::NTuple{3}, particle_spacing)
    n_particles_x,
    new_x_size = round_n_particles(size[1], particle_spacing,
                                   "fluid length in x-direction")
    n_particles_y,
    new_y_size = round_n_particles(size[2], particle_spacing,
                                   "fluid length in y-direction")
    n_particles_z,
    new_z_size = round_n_particles(size[3], particle_spacing,
                                   "fluid length in z-direction")

    return (n_particles_x, n_particles_y, n_particles_z),
           (new_x_size, new_y_size, new_z_size)
end

function boundary_particles_per_dimension(tank_size::NTuple{2}, particle_spacing,
                                          spacing_ratio)
    n_particles_x,
    new_width = round_n_particles(tank_size[1],
                                  particle_spacing / spacing_ratio,
                                  "tank length in x-direction")
    n_particles_y,
    new_height = round_n_particles(tank_size[2],
                                   particle_spacing / spacing_ratio,
                                   "tank length in y-direction")

    return (n_particles_x, n_particles_y), (new_width, new_height)
end

function boundary_particles_per_dimension(tank_size::NTuple{3}, particle_spacing,
                                          spacing_ratio)
    n_particles_x,
    new_x_size = round_n_particles(tank_size[1],
                                   particle_spacing / spacing_ratio,
                                   "tank length in x-direction")
    n_particles_y,
    new_y_size = round_n_particles(tank_size[2],
                                   particle_spacing / spacing_ratio,
                                   "tank length in y-direction")
    n_particles_z,
    new_z_size = round_n_particles(tank_size[3],
                                   particle_spacing / spacing_ratio,
                                   "tank length in z-direction")

    return (n_particles_x, n_particles_y, n_particles_z),
           (new_x_size, new_y_size, new_z_size)
end

function check_tank_overlap(fluid_size::NTuple{2}, tank_size, particle_spacing,
                            n_particles_per_dim)
    n_particles_x, n_particles_y = n_particles_per_dim
    fluid_size_x, fluid_size_y = fluid_size

    if tank_size[1] < fluid_size[1] - 1e-5 * particle_spacing
        n_particles_x -= 1
        fluid_size_x = n_particles_x * particle_spacing

        @info "The fluid was overlapping.\n New fluid length in x-direction is set to $fluid_size_x."
    end

    if tank_size[2] < fluid_size[2] - 1e-5 * particle_spacing
        n_particles_y -= 1
        fluid_size_y = n_particles_y * particle_spacing

        @info "The fluid was overlapping.\n New fluid length in y-direction is set to $fluid_size_y."
    end

    return (n_particles_x, n_particles_y), (fluid_size_x, fluid_size_y)
end

function check_tank_overlap(fluid_size::NTuple{3}, tank_size, particle_spacing,
                            n_particles_per_dim)
    n_particles_x, n_particles_y, n_particles_z = n_particles_per_dim
    fluid_size_x, fluid_size_y, fluid_size_z = fluid_size

    if tank_size[1] < fluid_size[1] - 1e-5 * particle_spacing
        n_particles_x -= 1
        fluid_size_x = n_particles_x * particle_spacing

        @info "The fluid was overlapping.\n New fluid length in x-direction is set to $fluid_size_x."
    end

    if tank_size[2] < fluid_size[2] - 1e-5 * particle_spacing
        n_particles_y -= 1
        fluid_size_y = n_particles_y * particle_spacing

        @info "The fluid was overlapping.\n New fluid length in y-direction is set to $fluid_size_y."
    end

    if tank_size[3] < fluid_size[3] - 1e-5 * particle_spacing
        n_particles_z -= 1
        fluid_size_z = n_particles_z * particle_spacing

        @info "The fluid was overlapping.\n New fluid length in z-direction is set to $fluid_size_z."
    end

    return (n_particles_x, n_particles_y, n_particles_z),
           (fluid_size_x, fluid_size_y, fluid_size_z)
end

# 2D
function initialize_boundaries(particle_spacing, tank_size::NTuple{2},
                               n_boundaries_per_dim, n_layers, faces)
    n_particles_x, n_particles_y = n_boundaries_per_dim

    # Store each particle index
    face_indices_1 = Array{Int, 2}(undef, n_layers, n_particles_y) # Left
    face_indices_2 = Array{Int, 2}(undef, n_layers, n_particles_y) # Right
    face_indices_3 = Array{Int, 2}(undef, n_layers, n_particles_x) # Bottom
    face_indices_4 = Array{Int, 2}(undef, n_layers, n_particles_x) # Top
    corner_indices_13 = Array{Int, 2}(undef, n_layers, n_layers)
    corner_indices_14 = Array{Int, 2}(undef, n_layers, n_layers)
    corner_indices_23 = Array{Int, 2}(undef, n_layers, n_layers)
    corner_indices_24 = Array{Int, 2}(undef, n_layers, n_layers)

    # Create empty array to extend later depending on faces and corners to build
    boundary_coordinates = Array{typeof(particle_spacing), 2}(undef, 2, 0)

    # Counts the global index of the particles
    index = 0

    # For odd faces we need to shift the face outwards if we have multiple layers
    layer_offset = -n_layers * particle_spacing

    #### Left boundary
    if faces[1]
        left_boundary = rectangular_shape_coords(particle_spacing,
                                                 (n_layers, n_particles_y),
                                                 (layer_offset, 0.0),
                                                 loop_order=:x_first)

        # Store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, left_boundary)

        # Store the indices of each particle
        particles_per_layer = n_particles_y
        for i in 1:n_layers
            face_indices_1[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Right boundary
    if faces[2]
        right_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_layers, n_particles_y),
                                                  (tank_size[1], 0.0),
                                                  loop_order=:x_first)

        # Store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, right_boundary)

        # Store the indices of each particle
        particles_per_layer = n_particles_y
        for i in 1:n_layers
            face_indices_2[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Bottom boundary
    if faces[3]
        bottom_boundary = rectangular_shape_coords(particle_spacing,
                                                   (n_particles_x, n_layers),
                                                   (0.0, layer_offset),
                                                   loop_order=:y_first)

        # Store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, bottom_boundary)

        # Store the indices of each particle
        particles_per_layer = n_particles_x
        for i in 1:n_layers
            face_indices_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Top boundary
    if faces[4]
        top_boundary = rectangular_shape_coords(particle_spacing,
                                                (n_particles_x, n_layers),
                                                (0.0, tank_size[2]),
                                                loop_order=:y_first)

        # Store coordinates of left boundary
        boundary_coordinates = hcat(boundary_coordinates, top_boundary)

        # Store the indices of each particle
        particles_per_layer = n_particles_x
        for i in 1:n_layers
            face_indices_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### Add corners
    # Bottom left
    if faces[1] && faces[3]
        bottom_left_corner = rectangular_shape_coords(particle_spacing,
                                                      (n_layers, n_layers),
                                                      (layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, bottom_left_corner)

        # Store the indices of each particle
        particles_per_layer = n_layers
        for i in 1:n_layers
            corner_indices_13[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # Top left
    if faces[1] && faces[4]
        top_left_corner = rectangular_shape_coords(particle_spacing,
                                                   (n_layers, n_layers),
                                                   (layer_offset, tank_size[2]))
        boundary_coordinates = hcat(boundary_coordinates, top_left_corner)

        # Store the indices of each particle
        particles_per_layer = n_layers
        for i in 1:n_layers
            corner_indices_14[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # Bottom right
    if faces[2] && faces[3]
        bottom_right_corner = rectangular_shape_coords(particle_spacing,
                                                       (n_layers, n_layers),
                                                       (tank_size[1], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, bottom_right_corner)

        # Store the indices of each particle
        particles_per_layer = n_layers
        for i in 1:n_layers
            corner_indices_23[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # Top right
    if faces[2] && faces[4]
        top_right_corner = rectangular_shape_coords(particle_spacing,
                                                    (n_layers, n_layers),
                                                    (tank_size[1], tank_size[2]))
        boundary_coordinates = hcat(boundary_coordinates, top_right_corner)

        # Store the indices of each particle
        particles_per_layer = n_layers
        for i in 1:n_layers
            corner_indices_24[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    return boundary_coordinates,
           (face_indices_1, face_indices_2, face_indices_3, face_indices_4),
           (corner_indices_13, corner_indices_14, corner_indices_23, corner_indices_24)
end

# 3D
function initialize_boundaries(particle_spacing, tank_size::NTuple{3},
                               n_boundaries_per_dim, n_layers, faces)
    n_particles_x, n_particles_y, n_particles_z = n_boundaries_per_dim

    # Faces
    face_indices_1 = Array{Int, 2}(undef, n_layers, n_particles_y * n_particles_z) # Left
    face_indices_2 = Array{Int, 2}(undef, n_layers, n_particles_y * n_particles_z) # Right
    face_indices_3 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_z) # Bottom
    face_indices_4 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_z) # Top
    face_indices_5 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_y) # Front
    face_indices_6 = Array{Int, 2}(undef, n_layers, n_particles_x * n_particles_y) # Back

    # Corners
    corner_indices_1_3_5 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)
    corner_indices_1_4_5 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)
    corner_indices_1_3_6 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)
    corner_indices_1_4_6 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)
    corner_indices_2_3_5 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)
    corner_indices_2_4_5 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)
    corner_indices_2_3_6 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)
    corner_indices_2_4_6 = Array{Int, 2}(undef, n_layers, n_layers * n_layers)

    # Edges
    edge_indices_1_3 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_z)
    edge_indices_1_4 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_z)
    edge_indices_2_3 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_z)
    edge_indices_2_4 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_z)

    edge_indices_3_5 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_x)
    edge_indices_3_6 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_x)
    edge_indices_4_5 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_x)
    edge_indices_4_6 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_x)

    edge_indices_1_5 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_y)
    edge_indices_1_6 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_y)
    edge_indices_2_5 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_y)
    edge_indices_2_6 = Array{Int, 2}(undef, n_layers, n_layers * n_particles_y)

    # Create empty array to extend later depending on faces and corners to build
    boundary_coordinates = Array{typeof(particle_spacing), 2}(undef, 3, 0)

    # Counts the global index of the particles
    index = 0

    # For odd faces we need to shift the face outwards if we have multiple layers
    layer_offset = -n_layers * particle_spacing

    # --- Faces ---

    particles_per_layer = prod((n_particles_y, n_particles_z))
    #### -x boundary (y-z-plane)
    if faces[1]
        x_neg_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_layers, n_particles_y, n_particles_z),
                                                  (layer_offset, 0.0, 0.0),
                                                  loop_order=:x_first)

        # Store coordinates of -x boundary
        boundary_coordinates = hcat(boundary_coordinates, x_neg_boundary)

        # Store the indices of each particle
        for i in 1:n_layers
            face_indices_1[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +x boundary (y-z-plane)
    if faces[2]
        x_pos_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_layers, n_particles_y, n_particles_z),
                                                  (tank_size[1], 0.0, 0.0),
                                                  loop_order=:x_first)
        boundary_coordinates = hcat(boundary_coordinates, x_pos_boundary)

        for i in 1:n_layers
            face_indices_2[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    particles_per_layer = prod((n_particles_x, n_particles_z))
    #### -y boundary (x-z-plane)
    if faces[3]
        y_neg_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_layers, n_particles_z),
                                                  (0.0, layer_offset, 0.0),
                                                  loop_order=:y_first)
        boundary_coordinates = hcat(boundary_coordinates, y_neg_boundary)

        for i in 1:n_layers
            face_indices_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +y boundary (x-z-plane)
    if faces[4]
        y_pos_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_layers, n_particles_z),
                                                  (0.0, tank_size[2], 0.0),
                                                  loop_order=:y_first)
        boundary_coordinates = hcat(boundary_coordinates, y_pos_boundary)

        for i in 1:n_layers
            face_indices_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    particles_per_layer = prod((n_particles_x, n_particles_y))
    #### -z boundary (x-y-plane)
    if faces[5]
        z_neg_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_particles_y, n_layers),
                                                  (0.0, 0.0, layer_offset),
                                                  loop_order=:z_first)
        boundary_coordinates = hcat(boundary_coordinates, z_neg_boundary)

        for i in 1:n_layers
            face_indices_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### +z boundary (x-y-plane)
    if faces[6]
        z_pos_boundary = rectangular_shape_coords(particle_spacing,
                                                  (n_particles_x, n_particles_y, n_layers),
                                                  (0.0, 0.0, tank_size[3]),
                                                  loop_order=:z_first)
        boundary_coordinates = hcat(boundary_coordinates, z_pos_boundary)

        for i in 1:n_layers
            face_indices_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # --- Edges ---

    particles_per_layer = prod((n_layers, n_particles_z))
    # -x / -y edge (z-aligned)
    if faces[1] && faces[3]
        edge_1_3 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (layer_offset, layer_offset, 0.0))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_3)

        for i in 1:n_layers
            edge_indices_1_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # -x / +y edge (z-aligned)
    if faces[1] && faces[4]
        edge_1_4 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (layer_offset, tank_size[2], 0.0))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_4)

        for i in 1:n_layers
            edge_indices_1_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / -y edge (z-aligned)
    if faces[2] && faces[3]
        edge_2_3 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (tank_size[1], layer_offset, 0.0))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_3)

        for i in 1:n_layers
            edge_indices_2_3[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / +y edge (z-aligned)
    if faces[2] && faces[4]
        edge_2_4 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_layers, n_particles_z),
                                            (tank_size[1], tank_size[2], 0.0))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_4)

        for i in 1:n_layers
            edge_indices_2_4[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    particles_per_layer = prod((n_layers, n_particles_x))
    # -y / -z edge (x-aligned)
    if faces[5] && faces[3]
        edge_3_5 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (0.0, layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_3_5)

        for i in 1:n_layers
            edge_indices_3_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +y / -z edge (x-aligned)
    if faces[5] && faces[4]
        edge_4_5 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (0.0, tank_size[2], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_4_5)

        for i in 1:n_layers
            edge_indices_4_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # -y / +z edge (x-aligned)
    if faces[6] && faces[3]
        edge_3_6 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (0.0, layer_offset, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_3_6)

        for i in 1:n_layers
            edge_indices_3_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +y / +z edge (x-aligned)
    if faces[6] && faces[4]
        edge_4_6 = rectangular_shape_coords(particle_spacing,
                                            (n_particles_x, n_layers, n_layers),
                                            (0.0, tank_size[2], tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_4_6)

        for i in 1:n_layers
            edge_indices_4_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    particles_per_layer = prod((n_layers, n_particles_y))
    # -x / -z edge (y-aligned)
    if faces[1] && faces[5]
        edge_1_5 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (layer_offset, 0.0, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_5)

        for i in 1:n_layers
            edge_indices_1_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # -x / +z edge (y-aligned)
    if faces[1] && faces[6]
        edge_1_6 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (layer_offset, 0.0, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_1_6)

        for i in 1:n_layers
            edge_indices_1_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / -z edge (y-aligned)
    if faces[2] && faces[5]
        edge_2_5 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (tank_size[1], 0.0, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_5)

        for i in 1:n_layers
            edge_indices_2_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / +z edge (y-aligned)
    if faces[2] && faces[6]
        edge_2_6 = rectangular_shape_coords(particle_spacing,
                                            (n_layers, n_particles_y, n_layers),
                                            (tank_size[1], 0.0, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, edge_2_6)

        for i in 1:n_layers
            edge_indices_2_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    #### --- Corners ---
    particles_per_layer = prod((n_layers, n_layers))

    # -x / -y / -z corner
    if faces[1] && faces[3] && faces[5]
        corner_1_3_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_3_5)

        for i in 1:n_layers
            corner_indices_1_3_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # -x / +y / -z corner
    if faces[1] && faces[4] && faces[5]
        corner_1_4_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, tank_size[2], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_4_5)

        for i in 1:n_layers
            corner_indices_1_4_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # -x / -y / +z corner
    if faces[1] && faces[3] && faces[6]
        corner_1_3_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, layer_offset, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_3_6)

        for i in 1:n_layers
            corner_indices_1_3_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # -x / +y / +z corner
    if faces[1] && faces[4] && faces[6]
        corner_1_4_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (layer_offset, tank_size[2], tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_1_4_6)

        for i in 1:n_layers
            corner_indices_1_4_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / -y / -z corner
    if faces[2] && faces[3] && faces[5]
        corner_2_3_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], layer_offset, layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_3_5)

        for i in 1:n_layers
            corner_indices_2_3_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / +y / -z corner
    if faces[2] && faces[4] && faces[5]
        corner_2_4_5 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], tank_size[2], layer_offset))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_4_5)

        for i in 1:n_layers
            corner_indices_2_4_5[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / -y / +z corner
    if faces[2] && faces[3] && faces[6]
        corner_2_3_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], layer_offset, tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_3_6)

        for i in 1:n_layers
            corner_indices_2_3_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    # +x / +y / +z corner
    if faces[2] && faces[4] && faces[6]
        corner_2_4_6 = rectangular_shape_coords(particle_spacing,
                                                (n_layers, n_layers, n_layers),
                                                (tank_size[1], tank_size[2],
                                                 tank_size[3]))
        boundary_coordinates = hcat(boundary_coordinates, corner_2_4_6)

        for i in 1:n_layers
            corner_indices_2_4_6[i, :] = collect((index + 1):(particles_per_layer + index))
            index += particles_per_layer
        end
    end

    return boundary_coordinates,
           (face_indices_1, face_indices_2, face_indices_3, face_indices_4,
            face_indices_5, face_indices_6),
           (corner_indices_1_3_5, corner_indices_1_4_5,
            corner_indices_1_3_6, corner_indices_1_4_6,
            corner_indices_2_3_5, corner_indices_2_4_5,
            corner_indices_2_3_6, corner_indices_2_4_6),
           (edge_indices_1_3, edge_indices_1_4, edge_indices_1_5, edge_indices_1_6,
            edge_indices_2_3, edge_indices_2_4, edge_indices_2_5, edge_indices_2_6,
            edge_indices_3_5, edge_indices_3_6, edge_indices_4_5, edge_indices_4_6)
end

@doc raw"""
    reset_wall!(rectangular_tank::RectangularTank, reset_faces, positions)

The selected walls of the tank will be placed at the new positions.

# Arguments
- `reset_faces`: Boolean tuple of 4 (in 2D) or 6 (in 3D) dimensions, similar to `faces` in [`RectangularTank`](@ref).
- `positions`: Tuple of new positions

!!! warning "Warning"
    There are overlapping particles when adjacent walls are moved inwards simultaneously.
"""
function reset_wall!(rectangular_tank, reset_faces, positions)
    (; boundary, particle_spacing, spacing_ratio, n_layers, face_indices) = rectangular_tank
    boundary_spacing = particle_spacing / spacing_ratio

    for face in eachindex(reset_faces)
        dim = div(face - 1, 2) + 1

        reset_faces[face] && for layer in 1:n_layers

            # `face_indices` contains the associated particle indices for each face.
            for particle in view(face_indices[face], layer, :)

                # For "odd" faces the layer direction is outwards
                # and for "even" faces inwards.
                layer_shift = if iseven(face)
                    (layer - 1) * boundary_spacing
                else
                    # Odd faces need to be shifted outwards by `boundary_spacing`
                    # to be outside of the fluid.
                    -(layer - 1) * boundary_spacing - boundary_spacing
                end

                # Set position
                boundary.coordinates[dim,
                                     particle] = positions[face] + layer_shift +
                                                 0.5boundary_spacing
            end
        end
    end

    return rectangular_tank
end
