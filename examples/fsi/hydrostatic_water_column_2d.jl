using TrixiParticles
using OrdinaryDiffEq

ddt = false
edac = true

# ==========================================================================================
# ==== Resolution
n_particles_plate_y = 5

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (1.0, 2.0)
plate_size = (1.0, 0.05)

fluid_density = 1000.0
solid_density = 2700.0

# Young's modulus and Poisson ratio
E = 67.5e9
nu = 0.34

sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = edac ? nothing :
                 StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

solid_particle_spacing = plate_size[2] / (n_particles_plate_y - 1)
fluid_particle_spacing = solid_particle_spacing
boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, initial_fluid_size,
                       min_coordinates=(0.0, fluid_particle_spacing / 2),
                       fluid_density, n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, false, false), acceleration=(0.0, -gravity),
                       state_equation=state_equation)

# Beam and clamped particles
n_particles_plate_x = round(Int, plate_size[1] / solid_particle_spacing + 1)
n_particles_per_dimension = (n_particles_plate_x, n_particles_plate_y)

plate = RectangularShape(solid_particle_spacing, n_particles_per_dimension,
                         (0.0, -plate_size[2]), density=solid_density, tlsph=true)

fixed_particles_1 = RectangularShape(solid_particle_spacing, (3, n_particles_plate_y),
                                     (-3solid_particle_spacing, -plate_size[2]),
                                     density=solid_density, tlsph=true)
fixed_particles_2 = RectangularShape(solid_particle_spacing, (3, n_particles_plate_y),
                                     (plate_size[1] + solid_particle_spacing,
                                      -plate_size[2]),
                                     density=solid_density, tlsph=true)
fixed_particles = union(fixed_particles_1, fixed_particles_2)

solid = union(plate, fixed_particles)

# ==========================================================================================
# ==== Solid
smoothing_kernel = WendlandC2Kernel{2}()
smoothing_length_solid = 2 * sqrt(2) * solid_particle_spacing

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites = fluid_density * ones(size(solid.density))
hydrodynamic_masses = hydrodynamic_densites * solid_particle_spacing^ndims(solid)

boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                   hydrodynamic_masses,
                                                   state_equation=state_equation,
                                                   boundary_density_calculator,
                                                   smoothing_kernel, smoothing_length_solid)

solid_system = TotalLagrangianSPHSystem(solid, smoothing_kernel, smoothing_length_solid,
                                        E, nu, boundary_model=boundary_model_solid,
                                        n_fixed_particles=nparticles(fixed_particles),
                                        acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Fluid
smoothing_length_fluid = 2 * sqrt(2) * fluid_particle_spacing

if edac
    fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel,
                                               smoothing_length_fluid,
                                               sound_speed,
                                               acceleration=(0.0, -gravity))
else
    fluid_density_calculator = ContinuityDensity()

    density_diffusion = ddt ? DensityDiffusionMolteniColagrossi(delta=0.1) : nothing
    fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length_fluid,
                                               density_diffusion=density_diffusion,
                                               acceleration=(0.0, -gravity))
end

# ==========================================================================================
# ==== Boundary

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length_fluid)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(solid_system, fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

# Track the position of the particle in the center of the of the beam.
particle_id = Int(n_particles_per_dimension[1] * (n_particles_plate_y + 1) / 2 -
                  (n_particles_per_dimension[1] + 1) / 2 + 1)

function y_deflection(v, u, t, system::TotalLagrangianSPHSystem)
    return TrixiParticles.current_coords(u, system, particle_id)[2] + plate_size[2] / 2
end
y_deflection(v, u, t, system) = nothing

# The pseudostatic midpoint deflection of a 2-D plate
function analytical_sol(v, u, t, system::TotalLagrangianSPHSystem)
    # Flexural rigidity of the plate
    D = E * plate_size[2]^3 / (12 * (1 - nu^2))

    return -0.0026 * gravity *
           (fluid_density * initial_fluid_size[2] + solid_density * plate_size[2]) / D
end
analytical_sol(v, u, t, system) = nothing

saving_callback = SolutionSavingCallback(dt=0.005, prefix="")

pp = PostprocessCallback(; interval=100, filename="hydrostatic_water_column_2d",
                         y_deflection, analytical_sol, kinetic_energy, backup_period=10)

callbacks = CallbackSet(info_callback, saving_callback, pp)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);
