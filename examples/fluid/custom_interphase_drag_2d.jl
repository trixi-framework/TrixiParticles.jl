# ==========================================================================================
# 2D Interphase Drag Example
#
# This example demonstrates a custom ordered-pair interaction in the `interaction_matrix`.
# Two interpenetrating weakly-compressible phases start with opposite horizontal velocities.
# The custom drag interaction exchanges momentum between phases and relaxes the velocity slip.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.05
n_particles = (12, 4)

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 0.2)

smoothing_length = 1.5 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
sound_speed = 20.0

water_density = 1000.0
oil_density = 850.0

water = RectangularShape(particle_spacing, n_particles, (0.0, 0.0);
                         density=water_density, velocity=(0.25, 0.0))

# Stagger the second phase by half a particle spacing. This represents an interpenetrating
# two-phase discretization without placing particles exactly on top of each other.
oil = RectangularShape(particle_spacing, n_particles,
                       (0.5 * particle_spacing, 0.5 * particle_spacing);
                       density=oil_density, velocity=(-0.25, 0.0))

water_state_equation = StateEquationCole(; sound_speed,
                                         reference_density=water_density,
                                         exponent=7)
oil_state_equation = StateEquationCole(; sound_speed,
                                       reference_density=oil_density,
                                       exponent=7)

water_system = WeaklyCompressibleSPHSystem(water;
                                           smoothing_kernel, smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           state_equation=water_state_equation)

oil_system = WeaklyCompressibleSPHSystem(oil;
                                         smoothing_kernel, smoothing_length,
                                         density_calculator=ContinuityDensity(),
                                         state_equation=oil_state_equation)

# ==========================================================================================
# ==== Custom Interaction

struct InterphaseDrag{T}
    coefficient::T
end

function (drag::InterphaseDrag)(dv, v_system, u_system,
                                v_neighbor, u_neighbor,
                                system, neighbor, semi)
    system_coords = TrixiParticles.current_coordinates(u_system, system)
    neighbor_coords = TrixiParticles.current_coordinates(u_neighbor, neighbor)

    TrixiParticles.foreach_point_neighbor(system, neighbor, system_coords,
                                          neighbor_coords, semi) do particle,
                                                                    neighbor_particle,
                                                                    pos_diff,
                                                                    distance
        v_a = TrixiParticles.current_velocity(v_system, system, particle)
        v_b = TrixiParticles.current_velocity(v_neighbor, neighbor, neighbor_particle)

        rho_a = TrixiParticles.current_density(v_system, system, particle)
        rho_b = TrixiParticles.current_density(v_neighbor, neighbor, neighbor_particle)
        m_b = TrixiParticles.hydrodynamic_mass(neighbor, neighbor_particle)
        kernel = TrixiParticles.smoothing_kernel(system, distance, particle)

        # Symmetric SPH interphase drag. When the same callable is used for both ordered
        # pairs, the pairwise momentum exchange is equal and opposite.
        acceleration = drag.coefficient * m_b / (rho_a * rho_b) * kernel * (v_b - v_a)

        for i in 1:ndims(system)
            dv[i, particle] += acceleration[i]
        end
    end

    return dv
end

drag = InterphaseDrag(1.0e3)

# `true` keeps the default interaction, while a callable replaces the default interaction
# for that ordered pair. Since the matrix is ordered, reciprocal coupling needs both entries.
interaction_matrix = Matrix{Any}(trues(2, 2))
interaction_matrix[1, 2] = drag
interaction_matrix[2, 1] = drag

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(water_system, oil_system;
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          interaction_matrix=interaction_matrix)
ode = semidiscretize(semi, tspan)

sol = solve(ode, RDPK3SpFSAL35();
            abstol=1e-6,
            reltol=1e-4,
            dtmax=1e-3,
            save_everystep=false)
