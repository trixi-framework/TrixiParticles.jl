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
struct BoundaryParticleContainer{NDIMS, ELTYPE <: Real, BM, MF} <: ParticleContainer{NDIMS}
    initial_coordinates :: Array{ELTYPE, 2}
    boundary_model      :: BM
    movement_function   :: MF
    ismoving            :: Vector{Bool}

    function BoundaryParticleContainer(coordinates, model; movement_function=nothing)
        NDIMS = size(coordinates, 1)
        ismoving = zeros(Bool, 1)

        return new{NDIMS, eltype(coordinates), typeof(model), typeof(movement_function)}(coordinates,
                                                                                         model,
                                                                                         movement_function,
                                                                                         ismoving)
    end
end

function Base.show(io::IO, container::BoundaryParticleContainer)
    @nospecialize container # reduce precompilation time

    print(io, "BoundaryParticleContainer{", ndims(container), "}(")
    print(io, container.boundary_model)
    print(io, ", ", container.movement_function)
    print(io, ") with ", nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::BoundaryParticleContainer)
    @nospecialize container # reduce precompilation time

    if get(io, :compact, false)
        show(io, container)
    else
        summary_header(io, "BoundaryParticleContainer{$(ndims(container))}")
        summary_line(io, "#particles", nparticles(container))
        summary_line(io, "boundary model", container.boundary_model)
        summary_line(io, "movement function", container.movement_function)
        summary_footer(io)
    end
end

@inline function boundary_particle_impact(particle, boundary_particle,
                                          v_particle_container, v_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b)
    @unpack boundary_model = boundary_container

    boundary_particle_impact(particle, boundary_particle,
                             v_particle_container, v_boundary_container,
                             particle_container, boundary_container,
                             pos_diff, distance, m_b, boundary_model)
end

function Base.show(io::IO, model::BoundaryModelMonaghanKajtar)
    @nospecialize model # reduce precompilation time

    print(io, "BoundaryModelMonaghanKajtar")
end

@inline function boundary_particle_impact(particle, boundary_particle,
                                          v_particle_container, v_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b,
                                          boundary_model::BoundaryModelMonaghanKajtar)
    @unpack smoothing_length = particle_container
    @unpack K, beta, boundary_particle_spacing = boundary_model

    NDIMS = ndims(particle_container)
    return K / beta^(NDIMS - 1) * pos_diff /
           (distance * (distance - boundary_particle_spacing)) *
           boundary_kernel(distance, smoothing_length)
end

@inline function boundary_kernel(r, h)
    q = r / h

    # TODO The neighborhood search fluid->boundary should use this search distance
    if q >= 2
        return 0.0
    end

    # (Monaghan, Kajtar, 2009, Section 4): The kernel should be normalized to 1.77 for q=0
    return 1.77 / 32 * (1 + 5 / 2 * q + 2 * q^2) * (2 - q)^5
end

function Base.show(io::IO, model::BoundaryModelDummyParticles)
    @nospecialize model # reduce precompilation time

    print(io, "BoundaryModelDummyParticles(")
    print(io, model.density_calculator |> typeof |> nameof)
    print(io, ")")
end

@inline function boundary_particle_impact(particle, boundary_particle,
                                          v_particle_container, v_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b,
                                          boundary_model::BoundaryModelDummyParticles)
    rho_a = particle_density(v_particle_container, particle_container, particle)
    rho_b = particle_density(v_boundary_container, boundary_container, boundary_particle)

    grad_kernel = smoothing_kernel_grad(particle_container, pos_diff, distance)

    return -m_b *
           (particle_container.pressure[particle] / rho_a^2 +
            boundary_model.pressure[boundary_particle] / rho_b^2) *
           grad_kernel
end

@doc raw"""
    AdamiPressureExtrapolation()

The pressure of the boundary particles is obtained by extrapolating the pressure of the fluid
according to (Adami et al., 2012).
The pressure of a boundary particle ``b`` is given by
```math
p_b = \frac{\sum_f (p_f + \rho_f (\bm{g} - \bm{a}_b) \cdot \bm{r}_{bf}) W(\Vert r_{bf} \Vert, h)}{\sum_f W(\Vert r_{bf} \Vert, h)},
```
where the sum is over all fluid particles, ``\rho_f`` and ``p_f`` denote the density and pressure of fluid particle ``f``, respectively,
``r_{bf} = r_b - r_f`` denotes the difference of the coordinates of particles ``b`` and ``f``,
``\bm{g}`` denotes the gravitational acceleration acting on the fluid, and ``\bm{a}_b`` denotes the acceleration of the boundary particle ``b``.

## References:
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)
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

@inline function nparticles(container::BoundaryParticleContainer)
    length(container.boundary_model.hydrodynamic_mass)
end

# No particle positions are advanced for boundary containers,
# except when using BoundaryModelDummyParticles with ContinuityDensity.
@inline function n_moving_particles(container::BoundaryParticleContainer)
    n_moving_particles(container, container.boundary_model)
end
@inline n_moving_particles(container::BoundaryParticleContainer, model) = 0
@inline function n_moving_particles(container::BoundaryParticleContainer,
                                    model::BoundaryModelDummyParticles)
    n_moving_particles(container, model.density_calculator)
end
@inline function n_moving_particles(container::BoundaryParticleContainer,
                                    ::ContinuityDensity)
    nparticles(container)
end

@inline u_nvariables(container::BoundaryParticleContainer) = 0

# For BoundaryModelDummyParticles with ContinuityDensity, this needs to be 1.
# For all other models and density calculators, it's irrelevant.
@inline v_nvariables(container::BoundaryParticleContainer) = 1

@inline function v_nvariables(container::SolidParticleContainer,
                              model::BoundaryModelDummyParticles)
    v_nvariables(container, model.density_calculator)
end
@inline function v_nvariables(container::SolidParticleContainer, ::ContinuityDensity)
    2 * ndims(container) + 1
end

@inline function current_coordinates(u, container::BoundaryParticleContainer)
    return container.initial_coordinates
end

@inline function current_velocity(v, container::BoundaryParticleContainer, particle)
    # TODO moving boundaries
    return SVector(ntuple(_ -> 0.0, Val(ndims(container))))
end

@inline function particle_density(v,
                                  container::Union{BoundaryParticleContainer,
                                                   SolidParticleContainer},
                                  particle)
    @unpack boundary_model = container

    particle_density(v, boundary_model, container, particle)
end

@inline function particle_density(v, boundary_model::BoundaryModelDummyParticles, container,
                                  particle)
    @unpack boundary_model = container
    @unpack density_calculator = boundary_model

    particle_density(v, density_calculator, boundary_model, container, particle)
end

@inline function particle_density(v, ::AdamiPressureExtrapolation, boundary_model,
                                  container, particle)
    @unpack cache = boundary_model

    return cache.density[particle]
end

@inline function particle_density(v, ::ContinuityDensity, boundary_model, container,
                                  particle)
    return v[end, particle]
end

@inline function particle_density(v, boundary_model::BoundaryModelMonaghanKajtar,
                                  container, particle)
    @unpack hydrodynamic_mass, boundary_particle_spacing = boundary_model

    # This model does not use any particle density. However, a mean density is used for
    # `ArtificialViscosityMonaghan` in the fluid interaction.
    return hydrodynamic_mass[particle] / boundary_particle_spacing^ndims(container)
end

@inline function hydrodynamic_mass(container, boundary_model, particle)
    return boundary_model.hydrodynamic_mass[particle]
end

function update!(container::BoundaryParticleContainer, container_index,
                 v, u, v_ode, u_ode, semi, t)
    @unpack initial_coordinates, movement_function, boundary_model = container

    container.ismoving[1] = move_boundary_particles!(movement_function, initial_coordinates,
                                                     t)

    update!(boundary_model, container, container_index, v, u, v_ode, u_ode, semi)

    return container
end

function move_boundary_particles!(movement_function, coordinates, t)
    movement_function(coordinates, t)
end
move_boundary_particles!(movement_function::Nothing, coordinates, t) = false

@inline function update!(boundary_model::BoundaryModelMonaghanKajtar, container,
                         container_index, v, u, v_ode, u_ode, semi)
    # Nothing to do in the update step
    return boundary_model
end

@inline function update!(boundary_model::BoundaryModelDummyParticles,
                         container, container_index, v, u, v_ode, u_ode, semi)
    @unpack pressure, density_calculator = boundary_model
    @unpack particle_containers, neighborhood_searches = semi

    pressure .= zero(eltype(pressure))

    compute_quantities!(boundary_model, density_calculator,
                        container, container_index, v, u, v_ode, u_ode, semi)

    return boundary_model
end

function compute_quantities!(boundary_model, ::SummationDensity,
                             container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack state_equation, pressure, cache = boundary_model
    @unpack density = cache # Density is in the cache for SummationDensity

    density .= zero(eltype(density))

    # Use all other containers for the density summation
    @trixi_timeit timer() "compute density" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                       neighbor_container)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index,
                                      neighbor_container, semi)

        container_coords = current_coordinates(u, container)
        neighbor_coords = current_coordinates(u_neighbor_container, neighbor_container)

        neighborhood_search = neighborhood_searches[container_index][neighbor_container_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        for_particle_neighbor(container, neighbor_container,
                              container_coords, neighbor_coords,
                              neighborhood_search;
                              particles=eachparticle(container)) do particle, neighbor,
                                                                    pos_diff, distance
            mass = hydrodynamic_mass(neighbor_container, neighbor)
            density[particle] += mass * smoothing_kernel(boundary_model, distance)
        end
    end

    for particle in eachparticle(container)
        pressure[particle] = state_equation(particle_density(v, boundary_model, particle))
    end
end

function compute_quantities!(boundary_model, ::ContinuityDensity,
                             container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack pressure, state_equation = boundary_model

    for particle in eachparticle(container)
        pressure[particle] = state_equation(particle_density(v, boundary_model, particle))
    end
end

function compute_quantities!(boundary_model, ::AdamiPressureExtrapolation,
                             container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack pressure, state_equation, cache = boundary_model
    @unpack density, volume = cache

    density .= zero(eltype(density))
    volume .= zero(eltype(volume))

    # Use all other containers for the pressure extrapolation
    @trixi_timeit timer() "compute boundary pressure" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                                 neighbor_container)
        v_neighbor_container = wrap_v(v_ode, neighbor_container_index,
                                      neighbor_container, semi)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index,
                                      neighbor_container, semi)

        neighborhood_search = neighborhood_searches[container_index][neighbor_container_index]

        container_coords = current_coordinates(u, container)
        neighbor_coords = current_coordinates(u_neighbor_container, neighbor_container)

        adami_pressure_extrapolation!(boundary_model, container, neighbor_container,
                                      container_coords, neighbor_coords,
                                      v_neighbor_container, neighborhood_search)
    end

    pressure ./= volume

    for particle in eachparticle(container)
        density[particle] = inverse_state_equation(state_equation, pressure[particle])
    end
end

@inline function adami_pressure_extrapolation!(boundary_model, container,
                                               neighbor_container::FluidParticleContainer,
                                               container_coords, neighbor_coords,
                                               v_neighbor_container, neighborhood_search)
    @unpack pressure, cache = boundary_model
    @unpack volume = cache

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(container, neighbor_container,
                          container_coords, neighbor_coords,
                          neighborhood_search;
                          particles=eachparticle(container)) do particle, neighbor,
                                                                pos_diff, distance
        density_neighbor = particle_density(v_neighbor_container, neighbor_container,
                                            neighbor)

        # TODO moving boundaries
        pressure[particle] += (neighbor_container.pressure[neighbor] +
                               dot(neighbor_container.acceleration,
                                   density_neighbor * pos_diff)) *
                              smoothing_kernel(boundary_model, distance)
        volume[particle] += smoothing_kernel(boundary_model, distance)
    end

    # Limit pressure to be non-negative to avoid negative pressures at free surfaces
    for particle in eachparticle(container)
        pressure[particle] = max(pressure[particle], 0.0)
    end
end

@inline function adami_pressure_extrapolation!(boundary_model, container,
                                               neighbor_container,
                                               container_coords, neighbor_coords,
                                               v_neighbor_container, neighborhood_search)
    return boundary_model
end

function write_u0!(u0, container::BoundaryParticleContainer)
    return u0
end

function write_v0!(v0, container::BoundaryParticleContainer)
    @unpack boundary_model = container

    write_v0!(v0, boundary_model, container)
end

function write_v0!(v0, model, container::BoundaryParticleContainer)
    return v0
end

function write_v0!(v0, boundary_model::BoundaryModelDummyParticles,
                   container::SolidParticleContainer)
    @unpack density_calculator = boundary_model

    write_v0!(v0, boundary_model, density_calculator, container)
end

function write_v0!(v0, model::BoundaryModelDummyParticles,
                   container::BoundaryParticleContainer)
    @unpack density_calculator = model

    write_v0!(v0, density_calculator, container)
end

function write_v0!(v0, ::ContinuityDensity, container::BoundaryParticleContainer)
    @unpack cache = container.boundary_model
    @unpack initial_density = cache

    for particle in eachparticle(container)
        # Set particle densities
        v0[1, particle] = initial_density[particle]
    end

    return v0
end
