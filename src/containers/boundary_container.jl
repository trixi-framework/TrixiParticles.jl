"""
    BoundaryParticleContainer(coordinates, mass, model;
                              movement_function=nothing)

Container for boundaries modeled by boundary particles.
The container is initialized with the coordinates of the particles and their masses.
The interaction between fluid and boundary particles is specified by the boundary model.

The `movement_function` is to define in which way the boundary particles move over time. Its
boolean return value is mandatory to determine in each timestep if the particles are moving or not.
This determines if the neighborhood search will be updated.
In the example below the `movement_function` only returns `true` (container is moving)
if the simulation time is lower than `0.1`.


# Examples
```julia
function movement_function(coordinates, t)

    if t < 0.1
        f(t) = 0.5*t^2 + t
        pos_1 = coordinates[2,1]
        pos_2 = f(t)
        diff_pos = pos_2 - pos_1
        coordinates[2,:] .+= diff_pos

        return true
    end

    return false
end
```
"""
struct BoundaryParticleContainer{NDIMS, ELTYPE<:Real, BM, MF} <: ParticleContainer{NDIMS}
    initial_coordinates ::Array{ELTYPE, 2}
    mass                ::Vector{ELTYPE}
    boundary_model      ::BM
    movement_function   ::MF
    ismoving            ::Vector{Bool}

    function BoundaryParticleContainer(coordinates, mass, model;
                                       movement_function=nothing)
        NDIMS = size(coordinates, 1)
        ismoving = zeros(Bool, 1)

        return new{NDIMS, eltype(coordinates), typeof(model), typeof(movement_function)}(
            coordinates, mass, model, movement_function, ismoving)
    end
end


# TODO
@doc raw"""
    BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing)

Boundaries modeled as boundary particles which exert forces on the fluid particles (Monaghan, Kajtar, 2009).
The force on fluid particle ``a`` is given by
```math
f_a = m_a \left(\sum_{b \in B} f_{ab} - m_b \Pi_{ab} \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)\right)
```
with
```math
f_{ab} = \frac{K}{\beta} \frac{r_{ab}}{\Vert r_{ab} \Vert (\Vert r_{ab} \Vert - d)} \Phi(\Vert r_{ab} \Vert, h)
\frac{2 m_b}{m_a + m_b},
```
where ``B`` denotes the set of boundary particles, ``m_a`` and ``m_b`` are the masses of
fluid particle ``a`` and boundary particle ``b`` respectively,
``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``,
and ``d`` denotes the boundary particle spacing
(see (Monaghan, Kajtar, 2009, Equation (3.1)) and (Valizadeh, Monaghan, 2015)).
Here, ``\Phi`` denotes the 1D Wendland C4 kernel, normalized to ``1.77`` for ``q=0``
(Monaghan, Kajtar, 2009, Section 4), with ``\Phi(r, h) = w(r/h)`` and
```math
w(q) =
\begin{cases}
  (1.77/32) (1 + (5/2)q + 2q^2)(2 - q)^5  & \text{if } 0 \leq q < 2 \\
  0                                       & \text{if } q \geq 2.
\end{cases}
```

The boundary particles are assumed to have uniform spacing by the factor ``\beta`` smaller
than the expected fluid particle spacing.
For example, if the fluid particles have an expected spacing of ``0.3`` and the boundary particles
have a uniform spacing of ``0.1``, then this parameter should be set to ``\beta = 3``.
According to (Monaghan, Kajtar, 2009), a value of ``\beta = 3`` for the Wendland C4 that
we use here is reasonable for most computing purposes.

The parameter ``K`` is used to scale the force exerted by the boundary particles.
In (Monaghan, Kajtar, 2009), a value of ``gD`` is used for static tank simulations,
where ``g`` is the gravitational acceleration and ``D`` is the depth of the fluid.

The viscosity ``\Pi_{ab}`` is calculated according to the viscosity used in the
simulation, where the density of the boundary particle if needed is assumed to be
identical to the density of the fluid particle.

References:
- Joseph J. Monaghan, Jules B. Kajtar. "SPH particle boundary forces for arbitrary boundaries".
  In: Computer Physics Communications 180.10 (2009), pages 1811–1820.
  [doi: 10.1016/j.cpc.2009.05.008](https://doi.org/10.1016/j.cpc.2009.05.008)
- Alireza Valizadeh, Joseph J. Monaghan. "A study of solid wall models for weakly compressible SPH."
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)
"""
struct BoundaryModelMonaghanKajtar{ELTYPE<:Real}
    K                           ::ELTYPE
    beta                        ::ELTYPE
    boundary_particle_spacing   ::ELTYPE

    function BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing)
        new{typeof(K)}(K, beta, boundary_particle_spacing)
    end
end


"""
TODO
"""
struct BoundaryModelDummyParticles{ELTYPE<:Real, SE, DC, K, C}
    pressure            ::Vector{ELTYPE}
    state_equation      ::SE
    density_calculator  ::DC
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    cache               ::C

    function BoundaryModelDummyParticles(initial_density, state_equation, density_calculator,
                                         smoothing_kernel, smoothing_length)
        pressure = similar(initial_density)

        cache = create_cache(initial_density, density_calculator)

        new{eltype(initial_density), typeof(state_equation),
            typeof(density_calculator), typeof(smoothing_kernel),
            typeof(cache)}(pressure, state_equation, density_calculator,
                           smoothing_kernel, smoothing_length, cache)
    end
end


"""
TODO
"""
struct AdamiPressureExtrapolation end


function create_cache(initial_density, ::SummationDensity)
    density = similar(initial_density)

    return (; density)
end

function create_cache(initial_density, ::ContinuityDensity)
    return (; initial_density)
end

function create_cache(initial_density, ::AdamiPressureExtrapolation)
    density = similar(initial_density)
    volume = similar(initial_density)

    return (; density, volume)
end


@inline function boundary_particle_impact(particle, boundary_particle,
                                          u_particle_container, u_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b)
    @unpack boundary_model = boundary_container

    boundary_particle_impact(particle, boundary_particle,
                             u_particle_container, u_boundary_container,
                             particle_container, boundary_container,
                             pos_diff, distance, m_b, boundary_model)
end


@inline function boundary_particle_impact(particle, boundary_particle,
                                          u_particle_container, u_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b,
                                          boundary_model::BoundaryModelMonaghanKajtar)
    @unpack smoothing_length = particle_container
    @unpack K, beta, boundary_particle_spacing = boundary_model

    NDIMS = ndims(particle_container)
    return K / beta^(NDIMS-1) * pos_diff / (distance * (distance - boundary_particle_spacing)) *
        boundary_kernel(distance, smoothing_length)
end

@inline function boundary_kernel(r, h)
    q = r / h

    # TODO The neighborhood search fluid->boundary should use this search distance
    if q >= 2
        return 0.0
    end

    # (Monaghan, Kajtar, 2009, Section 4): The kernel should be normalized to 1.77 for q=0
    return 1.77/32 * (1 + 5/2 * q + 2 * q^2) * (2 - q)^5
end


@inline function boundary_particle_impact(particle, boundary_particle,
                                          u_particle_container, u_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b,
                                          boundary_model::BoundaryModelDummyParticles)
    @unpack smoothing_kernel, smoothing_length = particle_container

    density_particle          = get_particle_density(particle, u_particle_container, particle_container)
    density_boundary_particle = get_particle_density(boundary_particle, u_boundary_container, boundary_container)

    grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

    return -m_b * (particle_container.pressure[particle] / density_particle^2 +
                   boundary_model.pressure[boundary_particle] / density_boundary_particle^2) * grad_kernel
end


# No particle positions are advanced for boundary containers,
# except when using BoundaryModelDummyParticles with ContinuityDensity.
@inline n_moving_particles(container::BoundaryParticleContainer) = n_moving_particles(container, container.boundary_model)
@inline n_moving_particles(container::BoundaryParticleContainer, model) = 0
@inline n_moving_particles(container::BoundaryParticleContainer, model::BoundaryModelDummyParticles) = n_moving_particles(container, model.density_calculator)
@inline n_moving_particles(container::BoundaryParticleContainer, ::ContinuityDensity) = nparticles(container)

# For BoundaryModelDummyParticles with ContinuityDensity, this needs to be 1.
# For all other models and density calculators, it's irrelevant.
@inline nvariables(container::BoundaryParticleContainer) = 1

@inline nvariables(container::SolidParticleContainer, model::BoundaryModelDummyParticles) = nvariables(container, model.density_calculator)
@inline nvariables(container::SolidParticleContainer, ::ContinuityDensity) = 2 * ndims(container) + 1


@inline function get_current_coords(particle, u, container::BoundaryParticleContainer)
    @unpack initial_coordinates = container

    return get_particle_coords(particle, initial_coordinates, container)
end


@inline function get_particle_vel(particle, u, container::BoundaryParticleContainer)
    # TODO moving boundaries
    return SVector(ntuple(_ -> 0.0, Val(ndims(container))))
end


# This will only be called for BoundaryModelDummyParticles
@inline function get_particle_density(particle, u, container::BoundaryParticleContainer)
    @unpack boundary_model = container
    @unpack density_calculator = boundary_model

    get_particle_density(particle, u, density_calculator, boundary_model)
end

@inline function get_particle_density(particle, u, ::AdamiPressureExtrapolation, boundary_model)
    @unpack cache = boundary_model

    return cache.density[particle]
end


function update!(container::BoundaryParticleContainer, container_index, u, u_ode, semi, t)
    @unpack initial_coordinates, movement_function, boundary_model = container

    container.ismoving[1] = move_boundary_particles!(movement_function, initial_coordinates, t)

    update!(boundary_model, container, container_index, u, u_ode, semi)

    return container
end

move_boundary_particles!(movement_function, coordinates, t) = movement_function(coordinates, t)
move_boundary_particles!(movement_function::Nothing, coordinates, t) = false

@inline function update!(boundary_model::BoundaryModelMonaghanKajtar, container, container_index, u, u_ode, semi)
    # Nothing to do in the update step
    return boundary_model
end

@inline function update!(boundary_model::BoundaryModelDummyParticles,
                         container, container_index, u, u_ode, semi)
    @unpack pressure, density_calculator = boundary_model
    @unpack particle_containers, neighborhood_searches = semi

    pressure .= zero(eltype(pressure))

    compute_quantities!(boundary_model, density_calculator,
                        container, container_index, u, u_ode, semi)

    return boundary_model
end


function compute_quantities!(boundary_model, ::SummationDensity,
                             container, container_index, u, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack state_equation, pressure, cache = boundary_model
    @unpack density = cache # Density is in the cache for SummationDensity

    density .= zero(eltype(density))

    # Use all other containers for the density summation
    @pixie_timeit timer() "compute density" foreach_enumerate(particle_containers) do (neighbor_container_index, neighbor_container)
        u_neighbor_container = wrap_array(u_ode, neighbor_container_index, neighbor_container, semi)

        @threaded for particle in eachparticle(container)
            compute_density_per_particle(particle, u, u_neighbor_container,
                                         container, neighbor_container,
                                         neighborhood_searches[container_index][neighbor_container_index])
        end
    end

    for particle in eachparticle(container)
        pressure[particle] = state_equation(get_particle_density(particle, u, boundary_model))
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_density_per_particle(particle, u_particle_container, u_neighbor_container,
                                              particle_container::BoundaryParticleContainer,
                                              neighbor_container, neighborhood_search)
    @unpack boundary_model = particle_container
    @unpack pressure, smoothing_kernel, smoothing_length, cache = boundary_model
    @unpack density = cache # Density is in the cache for SummationDensity
    @unpack mass = neighbor_container

    particle_coords = get_current_coords(particle, u_particle_container, particle_container)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        distance = norm(particle_coords - get_current_coords(neighbor, u_neighbor_container, neighbor_container))

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density[particle] += mass[neighbor] * kernel(smoothing_kernel, distance, smoothing_length)
        end
    end
end


function compute_quantities!(boundary_model, ::ContinuityDensity,
                             container, container_index, u, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack pressure, state_equation = boundary_model

    for particle in eachparticle(container)
        pressure[particle] = state_equation(get_particle_density(particle, u, boundary_model))
    end
end


function compute_quantities!(boundary_model, ::AdamiPressureExtrapolation,
                             container, container_index, u, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack pressure, state_equation, cache = boundary_model
    @unpack density, volume = cache

    density .= zero(eltype(density))
    volume  .= zero(eltype(volume))

    # Use all other containers for the pressure summation
    @pixie_timeit timer() "compute boundary pressure" foreach_enumerate(particle_containers) do (neighbor_container_index, neighbor_container)
        u_neighbor_container = wrap_array(u_ode, neighbor_container_index, neighbor_container, semi)

        @threaded for particle in eachparticle(container)
            compute_pressure_per_particle(particle, u, u_neighbor_container,
                                          container, neighbor_container,
                                          neighborhood_searches[container_index][neighbor_container_index],
                                          boundary_model)
        end
    end

    pressure ./= volume

    for particle in eachparticle(container)
        density[particle] = inverse_state_equation(state_equation, pressure[particle])
    end
end


# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_pressure_per_particle(particle, u_particle_container, u_neighbor_container,
                                               particle_container,
                                               neighbor_container::FluidParticleContainer,
                                               neighborhood_search, boundary_model)
    @unpack pressure, smoothing_kernel, smoothing_length, cache = boundary_model
    @unpack volume = cache

    particle_coords = get_current_coords(particle, u_particle_container, particle_container)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        pos_diff = particle_coords - get_current_coords(neighbor, u_neighbor_container, neighbor_container)
        distance = norm(pos_diff)

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density_neighbor = get_particle_density(neighbor, u_neighbor_container, neighbor_container)

            # TODO moving boundaries
            pressure[particle] += (neighbor_container.pressure[neighbor] +
                                   dot(neighbor_container.acceleration, density_neighbor * pos_diff)
                                  ) * kernel(smoothing_kernel, distance, smoothing_length)
            volume[particle] += kernel(smoothing_kernel, distance, smoothing_length)
        end
    end
end

@inline function compute_pressure_per_particle(particle, u_particle_container, u_neighbor_container,
                                               particle_container, neighbor_container, neighborhood_search,
                                               boundary_model)
    return nothing
end


function write_variables!(u0, container::BoundaryParticleContainer)
    @unpack boundary_model = container

    write_variables!(u0, boundary_model, container)
end

function write_variables!(u0, model, container::BoundaryParticleContainer)
    return u0
end

function write_variables!(u0, boundary_model::BoundaryModelDummyParticles, container::SolidParticleContainer)
    @unpack density_calculator = boundary_model

    write_variables!(u0, boundary_model, density_calculator, container)
end

function write_variables!(u0, model::BoundaryModelDummyParticles, container::BoundaryParticleContainer)
    @unpack density_calculator = model

    write_variables!(u0, density_calculator, container)
end

function write_variables!(u0, ::ContinuityDensity, container::BoundaryParticleContainer)
    @unpack cache = container.boundary_model
    @unpack initial_density = cache

    for particle in eachparticle(container)
        # Set particle densities
        u0[1, particle] = initial_density[particle]
    end

    return u0
end
