# ==========================================================================================
# 3D Wave Through Flexible Structures (FSI), C01 v38 GPU Configuration
#
# This standalone example reproduces the accepted v38 production setup in single precision
# on an NVIDIA GPU. At the default resolution, it contains approximately 3.17 million
# particles and writes 601 VTK states over two simulated seconds.
# ==========================================================================================

using CUDA
using LinearAlgebra: norm
using OrdinaryDiffEqLowStorageRK
using TrixiParticles

CUDA.allowscalar(false)

# ==========================================================================================
# ==== Resolution and experiment setup
particle_spacing = 0.006885f0
structure_boundary_spacing = particle_spacing / 2
neighborhood_padding = 10 * particle_spacing
parallelization_backend = CUDABackend()

gravity = 9.81f0
acceleration = (0.0f0, -gravity, 0.0f0)
tspan = (0.0f0, 2.0f0)

tank_size = (2.0f0, 4.0f0, 1.0f0)
initial_fluid_size = (0.6f0, 0.6f0, 1.0f0)
fluid_density = 1000.0f0
structure_density = 1100.0f0

reference_velocity = sqrt(2 * gravity * initial_fluid_size[2])
sound_speed = 30 * reference_velocity
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7.0f0,
                                   clip_negative_pressure=false,
                                   minimum_pressure=-100_000.0f0,
                                   minimum_pressure_transition_width=10_000.0f0)

tank = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density;
                       n_layers=3,
                       faces=(true, true, true, true, true, true),
                       acceleration, state_equation,
                       coordinates_eltype=Float32)

# ==========================================================================================
# ==== Flexible blades
blade_thickness = 0.1f0
blade_height = 0.52f0
blade_width = 0.16f0
blade_chamfer_size = 0.01f0
structure_boundary_layers = 3

# Match the clamp height used by the C01 experiment at its original 0.0185 m resolution.
clamp_height = 4 * 0.0185f0
clamp_layers = round(Int, clamp_height / particle_spacing)
blade_centers = ((0.88f0, 0.18f0), (0.88f0, 0.50f0), (0.88f0, 0.82f0),
                 (1.25f0, 0.34f0), (1.25f0, 0.66f0))

function particles_for_extent(extent; minimum=2)
    max(minimum, round(Int, extent / particle_spacing) + 1)
end

n_x = particles_for_extent(blade_thickness; minimum=3)
n_y = particles_for_extent(blade_height; minimum=clamp_layers + 2)
n_z = particles_for_extent(blade_width; minimum=3)

function add_blade_boundary_normals(initial_condition, min_x, min_z)
    normals = zeros(eltype(initial_condition), 3, nparticles(initial_condition))

    for particle in eachparticle(initial_condition)
        x = initial_condition.coordinates[1, particle]
        y = initial_condition.coordinates[2, particle]
        z = initial_condition.coordinates[3, particle]
        i = round(Int, (x - min_x) / particle_spacing)
        j = round(Int, y / particle_spacing)
        k = round(Int, (z - min_z) / particle_spacing)
        distances = (i, n_x - 1 - i, j, n_y - 1 - j, k, n_z - 1 - k)
        minimum_distance = minimum(distances)

        minimum_distance < structure_boundary_layers || continue
        distances[1] == minimum_distance && (normals[1, particle] -= 1)
        distances[2] == minimum_distance && (normals[1, particle] += 1)
        distances[3] == minimum_distance && (normals[2, particle] -= 1)
        distances[4] == minimum_distance && (normals[2, particle] += 1)
        distances[5] == minimum_distance && (normals[3, particle] -= 1)
        distances[6] == minimum_distance && (normals[3, particle] += 1)
        normal_norm = norm(normals[:, particle])
        iszero(normal_norm) || (normals[:, particle] ./= normal_norm)
    end

    return InitialCondition(; coordinates=initial_condition.coordinates,
                            velocity=initial_condition.velocity,
                            mass=initial_condition.mass,
                            density=initial_condition.density,
                            pressure=initial_condition.pressure,
                            particle_spacing=initial_condition.particle_spacing,
                            normals)
end

blade_parts = map(blade_centers) do (center_x, center_z)
    min_x = center_x - (n_x - 1) * particle_spacing / 2
    min_z = center_z - (n_z - 1) * particle_spacing / 2

    clamp = RectangularShape(particle_spacing, (n_x, clamp_layers, n_z),
                             (min_x, 0.0f0, min_z); density=structure_density,
                             place_on_shell=true, coordinates_eltype=Float32)
    flexible = RectangularShape(particle_spacing, (n_x, n_y - clamp_layers, n_z),
                                (min_x, clamp_layers * particle_spacing, min_z);
                                density=structure_density, place_on_shell=true,
                                coordinates_eltype=Float32)
    structure = add_blade_boundary_normals(union(clamp, flexible), min_x, min_z)

    (; structure, n_clamped=nparticles(clamp), min_x, min_y=zero(min_x), min_z,
     n_x, n_y, n_z)
end

# Construct a half-spacing shell whose motion is interpolated from the TLSPH particles. The
# same interpolation weights transfer the fluid reaction forces back to their parent system.
function make_attached_blade_boundary(structure_system, blade_parts, parent_system_index,
                                      parent_spacing, boundary_spacing, fluid_density,
                                      smoothing_kernel, smoothing_length, state_equation,
                                      pressure_extrapolation, boundary_state,
                                      clip_negative_pressure, chamfer_size)
    ratio = parent_spacing / boundary_spacing
    subdivisions = round(Int, ratio)
    isapprox(ratio, subdivisions) ||
        error("the attached boundary spacing must divide the structure spacing")
    chamfer_layers = iszero(chamfer_size) ? 0 :
                     ceil(Int, chamfer_size / boundary_spacing)

    function is_boundary_particle(i, j, k, dimensions)
        distances = (i, dimensions[1] - 1 - i,
                     j, dimensions[2] - 1 - j,
                     k, dimensions[3] - 1 - k)
        iszero(chamfer_layers) && return iszero(minimum(distances))

        chamfer_distances = (i + k - chamfer_layers,
                             i + dimensions[3] - 1 - k - chamfer_layers,
                             dimensions[1] - 1 - i + k - chamfer_layers,
                             dimensions[1] + dimensions[3] - 2 - i - k -
                             chamfer_layers)
        minimum(chamfer_distances) >= 0 || return false
        return iszero(min(minimum(distances), minimum(chamfer_distances)))
    end

    n_ghosts = sum(blade_parts; init=0) do blade
        dimensions = subdivisions .* ((blade.n_x, blade.n_y, blade.n_z) .- 1) .+ 1
        2 * chamfer_layers < min(dimensions[1], dimensions[3]) ||
            error("the blade chamfer removes the complete attached boundary")
        sum(Iterators.product(0:(dimensions[1] - 1),
                              0:(dimensions[2] - 1),
                              0:(dimensions[3] - 1)); init=0) do (i, j, k)
            is_boundary_particle(i, j, k, dimensions)
        end
    end

    coordinates = zeros(eltype(structure_system), 3, n_ghosts)
    normals = zeros(eltype(structure_system), 3, n_ghosts)
    parent_particles = zeros(Int, 8, n_ghosts)
    parent_weights = zeros(eltype(structure_system), 8, n_ghosts)
    boundary_volumes = fill(boundary_spacing^2 * parent_spacing, n_ghosts)

    initial_coordinates = structure_system.initial_coordinates
    key_spacing = parent_spacing / 4
    coordinate_key(x, y,
                   z) = (round(Int, x / key_spacing),
                         round(Int, y / key_spacing),
                         round(Int, z / key_spacing))
    parent_lookup = Dict{NTuple{3, Int}, Int}()
    for parent in axes(initial_coordinates, 2)
        key = coordinate_key(initial_coordinates[1, parent],
                             initial_coordinates[2, parent],
                             initial_coordinates[3, parent])
        haskey(parent_lookup, key) && error("duplicate structural particle coordinate")
        parent_lookup[key] = parent
    end

    ghost = 0
    for blade in blade_parts
        dimensions = subdivisions .* ((blade.n_x, blade.n_y, blade.n_z) .- 1) .+ 1
        for k in 0:(dimensions[3] - 1), j in 0:(dimensions[2] - 1),
            i in 0:(dimensions[1] - 1)
            is_boundary_particle(i, j, k, dimensions) || continue
            ghost += 1

            lower_i, remainder_i = divrem(i, subdivisions)
            lower_j, remainder_j = divrem(j, subdivisions)
            lower_k, remainder_k = divrem(k, subdivisions)
            fractions = (remainder_i / subdivisions, remainder_j / subdivisions,
                         remainder_k / subdivisions)

            support = 0
            for upper_k in 0:1, upper_j in 0:1, upper_i in 0:1
                weight = (iszero(upper_i) ? 1 - fractions[1] : fractions[1]) *
                         (iszero(upper_j) ? 1 - fractions[2] : fractions[2]) *
                         (iszero(upper_k) ? 1 - fractions[3] : fractions[3])
                iszero(weight) && continue

                x = blade.min_x + (lower_i + upper_i) * parent_spacing
                y = blade.min_y + (lower_j + upper_j) * parent_spacing
                z = blade.min_z + (lower_k + upper_k) * parent_spacing
                key = coordinate_key(x, y, z)
                parent = get(parent_lookup, key, 0)
                iszero(parent) && error("attached boundary support particle not found")

                support += 1
                parent_particles[support, ghost] = parent
                parent_weights[support, ghost] = weight
                for dim in 1:3
                    coordinates[dim, ghost] += weight * initial_coordinates[dim, parent]
                end
            end

            distances = (i, dimensions[1] - 1 - i,
                         j, dimensions[2] - 1 - j,
                         k, dimensions[3] - 1 - k)
            chamfer_distances = (i + k - chamfer_layers,
                                 i + dimensions[3] - 1 - k - chamfer_layers,
                                 dimensions[1] - 1 - i + k - chamfer_layers,
                                 dimensions[1] + dimensions[3] - 2 - i - k -
                                 chamfer_layers)
            minimum_distance = min(minimum(distances), minimum(chamfer_distances))
            distances[1] == minimum_distance && (normals[1, ghost] -= 1)
            distances[2] == minimum_distance && (normals[1, ghost] += 1)
            distances[3] == minimum_distance && (normals[2, ghost] -= 1)
            distances[4] == minimum_distance && (normals[2, ghost] += 1)
            distances[5] == minimum_distance && (normals[3, ghost] -= 1)
            distances[6] == minimum_distance && (normals[3, ghost] += 1)
            chamfer_distances[1] == minimum_distance &&
                (normals[:, ghost] .+= (-1, 0, -1))
            chamfer_distances[2] == minimum_distance &&
                (normals[:, ghost] .+= (-1, 0, 1))
            chamfer_distances[3] == minimum_distance &&
                (normals[:, ghost] .+= (1, 0, -1))
            chamfer_distances[4] == minimum_distance &&
                (normals[:, ghost] .+= (1, 0, 1))
            on_chamfer = any(==(minimum_distance), chamfer_distances)
            on_other_face = any(==(minimum_distance), distances)
            on_chamfer && !on_other_face && (boundary_volumes[ghost] *= sqrt(2))
            normals[:, ghost] ./= norm(normals[:, ghost])
        end
    end

    initial_condition = InitialCondition(; coordinates, density=fluid_density,
                                         mass=fluid_density * boundary_volumes,
                                         particle_spacing=boundary_spacing, normals)
    boundary_model = BoundaryModelDummyParticles(initial_condition.density,
                                                 initial_condition.mass,
                                                 pressure_extrapolation,
                                                 smoothing_kernel, smoothing_length;
                                                 state_equation, boundary_state,
                                                 clip_negative_pressure)
    attachment = BoundaryAttachment(parent_system_index, parent_particles, parent_weights,
                                    nparticles(structure_system))

    return WallBoundarySystem(initial_condition, boundary_model;
                              prescribed_motion=attachment, color_value=3)
end

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.5f0 * particle_spacing
fluid_kernel = WendlandC2Kernel{3}()
fluid_viscosity = ArtificialViscosityMonaghan(alpha=0.02f0, beta=0.0f0)
density_diffusion = DensityDiffusionAntuono(delta=0.1f0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid;
                                           smoothing_kernel=fluid_kernel,
                                           smoothing_length=fluid_smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           state_equation, viscosity=fluid_viscosity,
                                           density_diffusion, acceleration)

# ==========================================================================================
# ==== Tank boundary
tank_pressure_extrapolation = AdamiPressureExtrapolation(pressure_offset=0.0f0,
                                                         anti_sticking_threshold=0.1f0)
tank_boundary_state = BoundaryStateWallRiemann(contact_distance_ratio=0.5f0)
tank_boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                  tank.boundary.mass,
                                                  tank_pressure_extrapolation,
                                                  fluid_kernel, fluid_smoothing_length;
                                                  state_equation,
                                                  boundary_state=tank_boundary_state,
                                                  clip_negative_pressure=true)
tank_boundary_system = WallBoundarySystem(tank.boundary, tank_boundary_model)

# ==========================================================================================
# ==== TLSPH structure and attached boundary
structure_kernel = WendlandC2Kernel{3}()
combined_structure = union((blade.structure for blade in blade_parts)...)

blade_particle_counts = map(blade -> nparticles(blade.structure), blade_parts)
blade_offsets = map(i -> sum(blade_particle_counts[1:(i - 1)]; init=0),
                    eachindex(blade_parts))
clamped_particles = vcat((offset .+ (1:blade.n_clamped)
                          for (blade, offset) in zip(blade_parts, blade_offsets))...)
sum(blade_particle_counts) == nparticles(combined_structure) ||
    error("combining blades unexpectedly removed overlapping particles")

hydrodynamic_density = fill(fluid_density, nparticles(combined_structure))
hydrodynamic_mass = fill(fluid_density * particle_spacing^3,
                         nparticles(combined_structure))
structure_pressure_extrapolation = AdamiPressureExtrapolation(pressure_offset=0.0f0,
                                                              anti_sticking_threshold=0.1f0)
structure_boundary_state = BoundaryStateWallRiemann()
structure_boundary_model = BoundaryModelDummyParticles(hydrodynamic_density,
                                                       hydrodynamic_mass,
                                                       structure_pressure_extrapolation,
                                                       fluid_kernel,
                                                       fluid_smoothing_length;
                                                       state_equation,
                                                       boundary_state=structure_boundary_state,
                                                       clip_negative_pressure=true)

structure_viscosity = ArtificialViscosityMonaghan(alpha=0.1f0, beta=0.0f0)
structure_system = TotalLagrangianSPHSystem(combined_structure;
                                            smoothing_kernel=structure_kernel,
                                            smoothing_length=sqrt(3.0f0) * particle_spacing,
                                            young_modulus=500_000.0f0,
                                            poisson_ratio=0.35f0,
                                            boundary_model=structure_boundary_model,
                                            hydrodynamic_boundary_particles=Int[],
                                            clamped_particles, acceleration,
                                            penalty_force=PenaltyForceGanzenmueller(alpha=1.0f0),
                                            viscosity=structure_viscosity)

# The parent structure is system 3 in `systems` below.
attached_structure_boundary_system = make_attached_blade_boundary(structure_system,
                                                                  blade_parts, 3,
                                                                  particle_spacing,
                                                                  structure_boundary_spacing,
                                                                  fluid_density,
                                                                  fluid_kernel,
                                                                  fluid_smoothing_length,
                                                                  state_equation,
                                                                  structure_pressure_extrapolation,
                                                                  structure_boundary_state,
                                                                  true,
                                                                  blade_chamfer_size)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(tank.boundary.coordinates, dims=2) .- neighborhood_padding
max_corner = maximum(tank.boundary.coordinates, dims=2) .+ neighborhood_padding
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{3}(; cell_list,
                                                update_strategy=ParallelUpdate())
systems = (fluid_system, tank_boundary_system, structure_system,
           attached_structure_boundary_system)
semi = Semidiscretization(systems...; neighborhood_search, parallelization_backend)
ode = semidiscretize(semi, tspan)

output_dt = 1.0f0 / 300.0f0
output_directory = "out"
save_times = collect(first(tspan):output_dt:last(tspan))
if isempty(save_times) || !isapprox(last(save_times), last(tspan))
    push!(save_times, last(tspan))
else
    save_times[end] = last(tspan)
end

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(; save_times, prefix="",
                                         output_directory,
                                         compress=false,
                                         parallel_compression=false)
sorting_callback = SortingCallback(dt=0.1f0)
callbacks = CallbackSet(info_callback, saving_callback, sorting_callback)

fluid_algorithm = RDPK3SpFSAL49()
fluid_solve_kwargs = (; abstol=1.0f-6, reltol=1.0f-4,
                      dt=1.0f-6, dtmax=0.01f0 * particle_spacing)

CUDA.synchronize()
solve_elapsed_seconds = @elapsed begin
    sol = solve(ode, fluid_algorithm; fluid_solve_kwargs..., save_everystep=false,
                callback=callbacks)
    CUDA.synchronize()
end
