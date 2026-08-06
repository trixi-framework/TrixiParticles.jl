# ==========================================================================================
# 3D Akinci Stream Flowing over a Sphere
#
# Reproduces Figure 7 of Akinci, Akinci, and Teschner (2013): a cylindrical inflow adheres
# to and flows around a solid sphere. An open boundary continuously feeds the stream, so this
# setup demonstrates fluid-solid adhesion rather than a finite falling water column.
# https://doi.org/10.1145/2508363.2508395
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

particle_spacing = 0.0075
fluid_density = 1000.0
sound_speed = 30.0
gravity = 9.81
tspan = (0.0, 0.5)
solution_saveat = ()

stream_radius = 0.03
stream_speed = 0.5
initial_stream_length = 0.08
sphere_radius = 0.05
sphere_center = (0.0, 0.0, -0.14)
boundary_layers = 3
open_boundary_layers = 6

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

cross_section = SphereShape(particle_spacing, stream_radius, (0.0, 0.0), fluid_density;
                            sphere_type=VoxelSphere())
initial_stream_layers = round(Int, initial_stream_length / particle_spacing)
initial_stream = extrude_geometry(cross_section; direction=SVector(0.0, 0.0, -1.0),
                                  n_extrude=initial_stream_layers,
                                  velocity=(0.0, 0.0, -stream_speed))
initial_stream.coordinates[3, :] .-= particle_spacing

solid_sphere = SphereShape(particle_spacing, sphere_radius, sphere_center, fluid_density;
                           sphere_type=RoundSphere(), n_layers=boundary_layers)

face = ([-stream_radius, -stream_radius, 0.0],
        [stream_radius, -stream_radius, 0.0],
        [-stream_radius, stream_radius, 0.0])
inflow = BoundaryZone(; boundary_face=face, face_normal=[0.0, 0.0, -1.0],
                      density=fluid_density, particle_spacing, open_boundary_layers,
                      boundary_type=InFlow(), reference_density=fluid_density,
                      reference_velocity=[0.0, 0.0, -stream_speed],
                      extrude_geometry=cross_section)
buffer_size = 8 * nparticles(inflow.initial_condition)

smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
viscosity = ArtificialViscosityMonaghan(alpha=0.01, beta=0.0)
surface_tension_coefficient = 1.0
surface_tension = SurfaceTensionAkinci(; surface_tension_coefficient)

fluid_system = WeaklyCompressibleSPHSystem(initial_stream; smoothing_kernel,
                                           smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           state_equation, viscosity, surface_tension,
                                           correction=AkinciFreeSurfaceCorrection(fluid_density),
                                           reference_particle_spacing=particle_spacing,
                                           acceleration=(0.0, 0.0, -gravity), buffer_size)

open_boundary = OpenBoundarySystem(inflow; fluid_system, buffer_size,
                                   boundary_model=BoundaryModelMirroringTafuni(;
                                                                               mirror_method=ZerothOrderMirroring()))

sphere_boundary_model = BoundaryModelDummyParticles(solid_sphere; fluid_system,
                                                    boundary_density_calculator=AdamiPressureExtrapolation(),
                                                    viscosity=ViscosityAdami(nu=0.01),
                                                    clip_negative_pressure=true)
sphere_boundary_system = WallBoundarySystem(solid_sphere, sphere_boundary_model;
                                            adhesion_coefficient=1.0)

min_corner = [-0.2, -0.2, -1.6]
max_corner = [0.2, 0.2, open_boundary_layers * particle_spacing]
neighborhood_search = GridNeighborhoodSearch{3}(;
                                                cell_list=FullGridCellList(; min_corner,
                                                                           max_corner),
                                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, sphere_boundary_system;
                          neighborhood_search)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01)
callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

sol = solve(ode, RDPK3SpFSAL35(); abstol=1e-7, reltol=1e-4, dtmax=1e-3,
            save_everystep=false, saveat=solution_saveat, callback=callbacks)
