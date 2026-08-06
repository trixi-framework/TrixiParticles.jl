# ==========================================================================================
# 3D Akinci Rolling-Droplet and Two-Way-Adhesion Experiment
#
# Reproduces the mechanisms of Figure 10 in Akinci, Akinci, and Teschner (2013): a strongly
# cohesive drop rolls down a viscous adhesive incline while two rigid figures interact with
# it. One figure has fluid adhesion and the other has none. The articulated ragdolls from the
# paper are represented by rigid, figure-shaped particle bodies supported by this solver.
# https://doi.org/10.1145/2508363.2508395
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

particle_spacing = 0.01
fluid_density = 1000.0
rigid_density = 2000.0
sound_speed = 40.0
gravity = 9.81
tspan = (0.0, 0.4)
solution_saveat = ()
include_rigid_figures = true

incline_angle = deg2rad(20.0)
plane_length = 0.6
plane_width = 0.3
boundary_layers = 3
drop_radius = 0.07

tangent = SVector(cos(incline_angle), 0.0, sin(incline_angle))
cross_slope = SVector(0.0, 1.0, 0.0)
plane_normal = SVector(-sin(incline_angle), 0.0, cos(incline_angle))
plane_center = SVector(0.0, 0.0, 0.0)

plane_corner = plane_center - plane_length / 2 * tangent -
               plane_width / 2 * cross_slope -
               boundary_layers * particle_spacing *
               plane_normal
plane = extrude_geometry((collect(plane_corner),
                          collect(plane_corner + plane_length * tangent),
                          collect(plane_corner + plane_width * cross_slope));
                         particle_spacing, direction=plane_normal,
                         n_extrude=boundary_layers, density=fluid_density)

drop_center = plane_center + 0.15 * tangent +
              (drop_radius + particle_spacing) * plane_normal
drop = SphereShape(particle_spacing, drop_radius, drop_center, fluid_density;
                   sphere_type=VoxelSphere())

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)
smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
viscosity = ArtificialViscosityMonaghan(alpha=0.01, beta=0.0)
surface_tension_coefficient = 3.0
surface_tension = SurfaceTensionAkinci(; surface_tension_coefficient)

fluid_system = WeaklyCompressibleSPHSystem(drop; smoothing_kernel, smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           state_equation, viscosity, surface_tension,
                                           correction=AkinciFreeSurfaceCorrection(fluid_density),
                                           reference_particle_spacing=particle_spacing,
                                           acceleration=(0.0, 0.0, -gravity))

plane_boundary_model = BoundaryModelDummyParticles(plane; fluid_system,
                                                   boundary_density_calculator=AdamiPressureExtrapolation(),
                                                   viscosity=ViscosityAdami(nu=1.0),
                                                   clip_negative_pressure=true)
plane_boundary_system = WallBoundarySystem(plane, plane_boundary_model;
                                           adhesion_coefficient=1.2)

function rigid_figure(center, density, particle_spacing)
    velocity = (0.0, 0.0, 0.0)
    function block(n_particles, block_center)
        min_corner = block_center .- 0.5 * particle_spacing .* n_particles
        return RectangularShape(particle_spacing, n_particles, min_corner;
                                density, velocity)
    end

    torso = block((3, 2, 5), center)
    arms = block((7, 2, 2), center + SVector(0.0, 0.0, 1.5 * particle_spacing))
    left_leg = block((2, 2, 4),
                     center + SVector(-particle_spacing, 0.0, -4 * particle_spacing))
    right_leg = block((2, 2, 4),
                      center + SVector(particle_spacing, 0.0, -4 * particle_spacing))
    head = SphereShape(particle_spacing, 1.5 * particle_spacing,
                       center + SVector(0.0, 0.0, 4 * particle_spacing), density;
                       sphere_type=VoxelSphere(), velocity)

    return union(torso, arms, left_leg, right_leg, head)
end

if include_rigid_figures
    figure_height = 9 * particle_spacing
    figure_center_height = drop_radius + figure_height / 2 + 2 * particle_spacing
    figure_1_center = drop_center + figure_center_height * plane_normal -
                      0.055 * cross_slope
    figure_2_center = drop_center + figure_center_height * plane_normal +
                      0.055 * cross_slope
    adhesive_figure = rigid_figure(figure_1_center, rigid_density, particle_spacing)
    nonadhesive_figure = rigid_figure(figure_2_center, rigid_density, particle_spacing)

    function rigid_boundary_model(shape)
        hydrodynamic_density = fill(fluid_density, nparticles(shape))
        hydrodynamic_mass = fill(fluid_density * particle_spacing^3, nparticles(shape))
        return BoundaryModelDummyParticles(shape; fluid_system,
                                           initial_density=hydrodynamic_density,
                                           hydrodynamic_mass,
                                           boundary_density_calculator=AdamiPressureExtrapolation(),
                                           viscosity=ViscosityAdami(nu=0.01),
                                           clip_negative_pressure=true)
    end

    contact_model = RigidContactModel(; normal_stiffness=2.0e5,
                                      normal_damping=200.0,
                                      contact_distance=2 * particle_spacing)
    adhesive_figure_system = RigidBodySystem(adhesive_figure;
                                             boundary_model=rigid_boundary_model(adhesive_figure),
                                             contact_model,
                                             acceleration=(0.0, 0.0, -gravity),
                                             particle_spacing,
                                             adhesion_coefficient=1.0,
                                             color_value=2)
    nonadhesive_figure_system = RigidBodySystem(nonadhesive_figure;
                                                boundary_model=rigid_boundary_model(nonadhesive_figure),
                                                contact_model,
                                                acceleration=(0.0, 0.0, -gravity),
                                                particle_spacing,
                                                adhesion_coefficient=0.0,
                                                color_value=3)
else
    adhesive_figure_system = nothing
    nonadhesive_figure_system = nothing
end

semi = Semidiscretization(fluid_system, plane_boundary_system,
                          adhesive_figure_system, nonadhesive_figure_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01)
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL49(); abstol=1e-7, reltol=1e-4, dtmax=5e-4,
            save_everystep=false, saveat=solution_saveat, callback=callbacks)
