# ==========================================================================================
# 2D Interphase Drag Example
#
# This example demonstrates a custom ordered-pair interaction in the `interaction_matrix`.
# A water layer and an oil layer start with opposite horizontal velocities. The custom
# interaction keeps the default fluid coupling and adds tangential drag across the interface.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.05
n_particles_per_layer = (12, 4)

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 0.2)

smoothing_length = 1.5 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
sound_speed = 20.0

water_density = 1000.0
oil_density = 850.0

layer_height = n_particles_per_layer[2] * particle_spacing

water = RectangularShape(particle_spacing, n_particles_per_layer, (0.0, 0.0);
                         density=water_density, velocity=(0.25, 0.0))

# Place the lighter oil directly above the water. The layers touch at the interface, so
# only particles inside the kernel support across the interface contribute to the drag.
oil = RectangularShape(particle_spacing, n_particles_per_layer,
                       (0.0, layer_height);
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

struct InterfacialTangentialDrag{T, V}
    coefficient::T
    interface_normal::V
end

function (drag::InterfacialTangentialDrag)(dv, v_system, u_system,
                                           v_neighbor, u_neighbor,
                                           system, neighbor, semi)
    # Preserve the default WCSPH pressure, viscosity, and continuity coupling between the
    # phases, then add the custom drag term below.
    TrixiParticles.interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
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
        velocity_difference = v_b - v_a
        normal_velocity_difference = sum(velocity_difference[i] *
                                         drag.interface_normal[i]
                                         for i in 1:ndims(system))

        rho_a = TrixiParticles.current_density(v_system, system, particle)
        rho_b = TrixiParticles.current_density(v_neighbor, neighbor, neighbor_particle)
        m_b = TrixiParticles.hydrodynamic_mass(neighbor, neighbor_particle)
        kernel = TrixiParticles.smoothing_kernel(system, distance, particle)

        # Symmetric SPH interfacial drag. Only the tangential slip is damped, avoiding an
        # artificial attractive or repulsive normal force between the layers.
        acceleration_factor = drag.coefficient * m_b / (rho_a * rho_b) * kernel
        for i in 1:ndims(system)
            tangential_slip = velocity_difference[i] -
                              normal_velocity_difference * drag.interface_normal[i]
            dv[i, particle] += acceleration_factor * tangential_slip
        end
    end

    return dv
end

drag = InterfacialTangentialDrag(1.0e3, (0.0, 1.0))

# `true` keeps the default interaction, while a callable handles the interaction for that
# ordered pair. This callable calls the default interaction before adding drag. Since the
# matrix is ordered, reciprocal coupling needs both entries.
interaction_matrix = Matrix{Any}(trues(2, 2))
interaction_matrix[1, 2] = drag
interaction_matrix[2, 1] = drag

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(water_system, oil_system;
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          interaction_matrix=interaction_matrix)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35();
            abstol=1e-6,
            reltol=1e-4,
            dtmax=1e-3,
            save_everystep=false,
            callback=callbacks)
