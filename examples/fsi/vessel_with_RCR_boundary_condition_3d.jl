using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
particle_spacing = 2e-3

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 5

function pulsatile_flow_rate(t)
    frequency = 1.0
    flow_rate = 1.25e-4 # [m^3/s]
    amplitude = flow_rate / (pi * 0.02^2) # v = Q/A

    t_periodic = mod(t, 1 / frequency)
    return amplitude * sin(pi * frequency * t_periodic)^2
end

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 10.0)

flow_length = 0.3
vessel_diameter = 0.04
vessel_radius = vessel_diameter / 2
vessel_thickness = 0.028

wave_speed = 2 * pulsatile_flow_rate(0.5)
sound_speed_factor = 10
sound_speed = sound_speed_factor * wave_speed

vessel_density = 1000.0 # corresponds to 1 g/cm^3
youngs_modulus = 4e5 # corresponds to 4*10^6 dyn/cm^2
vessel_nu = 0.49 # Poisson's ratio

fluid_density = 1.06 * vessel_density

dynamic_viscosity = 0.004 # corresponds to 0.04 poise

# RCR paramters
R1 = 121.0
R2 = 1212.0
C = 1.5e-5

flow_direction = (1.0, 0.0, 0.0)

n_particles_wall = ceil(Int, vessel_thickness / particle_spacing)
n_particles_length = ceil(Int, flow_length / particle_spacing)
offset_outlet = [n_particles_length * particle_spacing, 0, 0]

wall_2d = SphereShape(particle_spacing, vessel_radius, (0.0, 0.0),
                      vessel_density, n_layers=n_particles_wall, layer_outwards=true,
                      sphere_type=RoundSphere())
# Extend 2d coordinates to 3d by adding x-coordinates
wall_2d_coordinates = hcat(particle_spacing / 2 * ones(nparticles(wall_2d)),
                           wall_2d.coordinates')'

vessel_wall = extrude_geometry(wall_2d_coordinates; particle_spacing,
                               direction=collect(flow_direction), density=vessel_density,
                               n_extrude=n_particles_length)
vessel_clamped_in = extrude_geometry(wall_2d_coordinates; particle_spacing,
                                     direction=-collect(flow_direction),
                                     density=vessel_density, n_extrude=open_boundary_layers)
vessel_clamped_out = extrude_geometry(wall_2d_coordinates .+ [flow_length, 0, 0];
                                      particle_spacing, direction=collect(flow_direction),
                                      n_extrude=open_boundary_layers,
                                      density=vessel_density)

vessel_flexible = setdiff(vessel_wall, vessel_clamped_in, vessel_clamped_out)
vessel = union(vessel_flexible, vessel_clamped_in, vessel_clamped_out)

circle = SphereShape(particle_spacing, vessel_radius, (0.0, 0.0), fluid_density,
                     sphere_type=RoundSphere())
# Extend 2d coordinates to 3d by adding x-coordinates
circle_coordinates = hcat(particle_spacing / 2 * ones(nparticles(circle)),
                          circle.coordinates')'
fluid = extrude_geometry(circle_coordinates; particle_spacing,
                         direction=collect(flow_direction), density=fluid_density,
                         n_extrude=n_particles_length)

inlet = extrude_geometry(circle_coordinates; particle_spacing,
                         direction=-collect(flow_direction), density=fluid_density,
                         n_extrude=open_boundary_layers + 1)
n_buffer_particles = 20 * nparticles(circle)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{3}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = dynamic_viscosity / fluid_density

viscosity = ViscosityAdami(nu=kinematic_viscosity)

background_pressure = 7 * sound_speed_factor / 10 * fluid_density * wave_speed^2
shifting_technique = TransportVelocityAdami(; background_pressure)

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)
fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           buffer_size=n_buffer_particles,
                                           shifting_technique=shifting_technique,
                                           density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                           smoothing_length, viscosity=viscosity)

# ==========================================================================================
# ==== Open Boundary
open_boundary_model = BoundaryModelDynamicalPressureZhang()
# open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())
pressure_model = RCRWindkesselModel(; peripheral_resistance=R2, compliance=C,
                                    characteristic_resistance=R1)

face_in = ([0.0, -vessel_radius, -vessel_radius],
           [0.0, -vessel_radius, vessel_radius],
           [0.0, vessel_radius, vessel_radius])
inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density, particle_spacing,
                      reference_velocity=(pos, t) -> SVector(pulsatile_flow_rate(t), 0.0,
                                                             0.0), initial_condition=inlet)

face_out = ([offset_outlet[1], -vessel_radius, -vessel_radius],
            [offset_outlet[1], -vessel_radius, vessel_radius],
            [offset_outlet[1], vessel_radius, vessel_radius])
outflow = BoundaryZone(; boundary_face=face_out, face_normal=(.-(flow_direction)),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       pressure_model=pressure_model,
                       extrude_geometry=circle_coordinates .+ offset_outlet)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Solid
# For the FSI we need the hydrodynamic masses and densities in the structure boundary model
hydrodynamic_densites = fluid_density * ones(size(vessel.density))
hydrodynamic_masses = hydrodynamic_densites * particle_spacing^ndims(vessel)

boundary_model = BoundaryModelDummyParticles(hydrodynamic_densites, hydrodynamic_masses,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=viscosity,
                                             smoothing_kernel, smoothing_length)

penalty_force = PenaltyForceGanzenmueller(alpha=0.01)
vessel_system = TotalLagrangianSPHSystem(vessel, smoothing_kernel, smoothing_length,
                                         youngs_modulus, vessel_nu,
                                         boundary_model=boundary_model,
                                         penalty_force=penalty_force,
                                         viscosity=ArtificialViscosityMonaghan(alpha=0.01),
                                         n_clamped_particles=nparticles(vessel_clamped_in) +
                                                             nparticles(vessel_clamped_out))

# ==========================================================================================
# ==== Simulation
# Note that we have a flexible pipe
min_corner = minimum(vessel.coordinates .- vessel_radius / 2, dims=2)
max_corner = maximum(vessel.coordinates .+ vessel_radius / 2, dims=2)

nhs = GridNeighborhoodSearch{3}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, vessel_system,
                          neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)

pressure_out(system, data, t) = nothing
function pressure_out(system::OpenBoundarySystem, data, t)
    return system.pressure_model_values[2].pressure[]
end

flow_rate_out(system, data, t) = nothing
function flow_rate_out(system::OpenBoundarySystem, data, t)
    return system.pressure_model_values[2].flow_rate[]
end

output_directory = "out"
postprocess_callback = PostprocessCallback(dt=0.01, output_directory=output_directory,
                                           flow_rate_out=flow_rate_out,
                                           pressure_out=pressure_out,
                                           write_file_interval=5)

callbacks = CallbackSet(info_callback, UpdateCallback(), postprocess_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
