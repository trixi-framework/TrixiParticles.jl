# In this example two circles of water drop to the floor demonstrating the difference
# between the behavior with and without surface tension modelling.
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.0001

boundary_layers = 5
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 5.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.0, 0.0)
tank_size = (0.05, 0.01)

fluid_density = 1000.0
sound_speed = 50
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

sphere_radius = 0.0015

sphere1_center = (0.025, sphere_radius + 0.5 * fluid_particle_spacing)
sphere2_center = (1.5, sphere_radius - 0.5 * fluid_particle_spacing)
sphere1 = SphereShape(fluid_particle_spacing, sphere_radius, sphere1_center,
                      fluid_density, sphere_type=VoxelSphere())
sphere2 = SphereShape(fluid_particle_spacing, sphere_radius, sphere2_center,
                      fluid_density, sphere_type=VoxelSphere())

# ==========================================================================================
# ==== Fluid
# fluid_smoothing_length = 1.2 * fluid_particle_spacing
# fluid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_smoothing_length = 2.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_smoothing_length_2 = 2.5 * fluid_particle_spacing
fluid_smoothing_kernel_2 = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

# water at 20C
#nu=0.00089
nu = 0.00089
# too much 0.00089 -> 0.00045 -> 0.0002 -> 0.0001
# no impact 0.00005

# the following term only holds for 2d sims
# alpha = 8 * nu / (fluid_smoothing_length * sound_speed)
viscosity = ViscosityAdami(nu=nu)
# density_diffusion = DensityDiffusionAntuono(sphere2, delta=0.1)

# with increased smoothing length surface_tension is too small
sphere_surface_tension = WeaklyCompressibleSPHSystem(sphere1, fluid_density_calculator,
                                                     state_equation, fluid_smoothing_kernel,
                                                     fluid_smoothing_length,
                                                     viscosity=viscosity,
                                                     acceleration=(0.0, -gravity),
                                                     surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.011),
                                                     correction=AkinciFreeSurfaceCorrection(fluid_density))

# 0.001
# 0.2 > 90
# 0.15 > 90
# 0.125 > 90
# 0.11 60-90
# 0.1 ~ 45

# 0.0008
# 0.11 ~45-50
# 0.1125 ~90
# 0.115 ~90
# 0.12 > 90


# 0.0005
#0.11 ~100


# 0.00025
# 0.115 ~60 grad aber zu tief in der mitte 0.006
# 0.115 and increase nu to 0.0002

# 0.0001 deg 90 (x-axis: 2mm, y-axis: 1.8mm)
# start x-axis ~1.7mm y-axis: 2.2mm ~110deg distance to boundary to large at adhesion coefficient 1.0
# increase adhesion coefficient to 1.1 x-axis ~1.8mm y-axis: 2mm ~100deg distance still to high
# increase adhesion coefficient to 1.2 x-axis ~2.8 y-axis: 1.7 distance still to high
# decrease adhesion coefficient to 1.15 x-axis 2.4mm y-axis: 1.9 distance high
# decrease adhesion coefficient to 1.125 x-axis 2-2.2mm y-axis: 1.9 distance high ~90+
# decrease surface tension coefficient from 0.115 to 0.11 x-axis: 2.6mm y-axis: 1.7 -> invalid too much adhesion
# increase surface tension coefficient from 0.11 to 0.1125 x-axis: 2-2.2mm y-axis: 1.9
# increase surface tension coefficient from 0.1125 to 0.12 x-axis: 1.9-2mm y-axis: 2.0
# increase viscosity from 0.0001 to 0.0002 x-axis: 1.8mm y-axis:2.2
# increase adhesion coefficient from 1.125 to 1.15 x-axis: 1.8mm y-axis:2.1
# increase adhesion coefficient from 1.15 to 1.2 x-axis: 1.8mm y-axis:2.1
# increase adhesion coefficient from 1.2 to 1.25 x-axis: 2.6mm y-axis:1.7
# increase viscosity from 0.0002 to 0.0003 x-axis: 2.1mm y-axis:1.9 <=== (adh=1.25, surft=0.12, nu=0.0003)
# increase adhesion coefficient from 1.25 to 1.275 x-axis: 2.6mm y-axis:1.7
# decrease adhesion coefficient from 1.275 to 1.26 x-axis: 2.2mm y-axis:1.8
# decrease adhesion coefficient from 1.26 to 1.255 x-axis: 1.8-2.6mm y-axis:1.8
# increase viscosity from 0.0003 to 0.00031 x-axis: 2.2-2.8mm y-axis:1.7
# increase viscosity from 0.00031 to 0.00032 x-axis: 2.2-2.8mm y-axis:1.7

# 0.0001 deg 75 (x-axis: 2.2mm, y-axis: 1.6mm)
#(adh=1.25, surft=0.12, nu=0.0003)
# decrease surft=0.11 x-axis: 2.4-2.6mm y-axis:1.6
# decrease adh=1.2 x-axis: 2.2-2.4mm y-axis:1.8
# increase viscosity = 0.00035 x-axis: 2.4-3.2mm y-axis:1.6
# increase viscosity = 0.0004 x-axis: 2.6-2.8mm y-axis:1.4
# increase viscosity = 0.00045 x-axis: 2.4-3.2mm y-axis:1.4
# decrease adh=1.15 x-axis: 1.8mm y-axis:2.1
# increase adh=1.175 x-axis: 1.8mm y-axis:2.2
# increase adh=1.19 x-axis: 1.8mm y-axis:2.1
#(adh=1.25, surft=0.12, nu=0.0003)
# increase adh=1.3 and  x-axis: 2.4mm y-axis:1.8
# decrease adh=1.275 and  x-axis: 2.4mm y-axis:1.8
# decrease adh=1.26 and  x-axis: 2.4mm y-axis:1.8
# decrease adh=1.255 and  x-axis: 2.4mm y-axis:1.8
# decrease smoothing length from 4 -> 3.75 x-axis: 2mm y-axis:1.9





# 60deg x=2.4, y=1.3
# adh = 1.15, surft=0.115, nu=0.0003, h=3.75, x=1.7-1.8, y=2.2
# adh = 1.2, surft=0.115, nu=0.0003, h=3.75, x=2-2.2, y=1.8
# adh = 1.25, surft=0.115, nu=0.0003, h=3.75, x=2.2, y=1.8
# adh = 1.3, surft=0.115, nu=0.0003, h=3.75, x=2.6, y=1.6
# adh = 1.3, surft=0.12, nu=0.0003, h=3.75, x=2.2-2.6, y=1.6
# adh = 1.3, surft=0.125, nu=0.0003, h=3.75, x=2, y=1.8
# adh = 1.35, surft=0.125, nu=0.0003, h=3.75, x=2.5, y=1.6
# adh = 1.35, surft=0.13, nu=0.0003, h=3.75, x=2-2.2, y=1.8
# adh = 1.4, surft=0.13, nu=0.0003, h=3.75, x=2-2.2, y=1.8
# adh = 1.45, surft=0.13, nu=0.0003, h=3.75, x=2.6, y=1.6
# not possible with smoothing_length -> lower to 3 and switch from C4 to C2
# adh = 1.45, surft=0.13, nu=0.0003, h=3.0, x=1.4, y=2.6
# adh = 1.45, surft=0.12, nu=0.0003, h=3.0, x=1.4, y=2.4
# adh = 1.5, surft=0.12, nu=0.0003, h=3.0, x=1.6, y=2.6
# adh = 1.5, surft=0.11, nu=0.0003, h=3.0, x=1.4, y=2.4
# adh = 1.5, surft=0.10, nu=0.0003, h=3.0, x=1.4, y=2.4
# adh = 1.5, surft=0.05, nu=0.0003, h=3.0, x=3.5, y=2.0
# adh = 1.4, surft=0.075, nu=0.0003, h=3.0, x=3, y=2.0
# adh = 1.4, surft=0.075, nu=0.0003, h=2.0, x=3, y=2.0 crap
# adh = 1.4, surft=0.05, nu=0.0003, h=2.0, x=3, y=2.0 crap
# adh = 1.4, surft=0.01, nu=0.0003, h=2.0, x=3, y=2.0 crap
# adh = 1.4, surft=0.01, nu=0.0001, h=2.0, x=3, y=2.0 crap
# adh = 1.4, surft=0.001, nu=0.00089, h=2.0, x=3, y=2.0 crap
# adh = 1.2, surft=0.002, nu=0.00089, h=2.0, x=3, y=2.0 crap
# adh = 1.2, surft=0.002, nu=0.00089, h=2.5, x=2.5, y=1.4 close but too much adh
# adh = 0.6, surft=0.002, nu=0.00089, h=2.5, x=2.7, y=1.3 close but too much adh
# adh = 0.3, surft=0.002, nu=0.00089, h=2.5, x=2.8, y=1.2
# adh = 0.15, surft=0.002, nu=0.00089, h=2.5, x=2.8, y=1.2
# adh = 0.075, surft=0.002, nu=0.00089, h=2.5, x=3.4, y=1.2
# adh = 0.075, surft=0.004, nu=0.00089, h=2.5, x=2.8, y=1.4
# adh = 0.075, surft=0.01, nu=0.00089, h=2.5, x=2.2, y=1.7
# adh = 0.075, surft=0.007, nu=0.00089, h=2.5, x=2.6, y=1.4 <- okish but form needs more adh
# adh = 0.075, surft=0.008, nu=0.00089, h=2.5, x=2.6, y=1.4
# adh = 0.1, surft=0.01, nu=0.00089, h=2.5, x=2.4, y=1.6 <- okish but form needs more adh
# adh = 0.15, surft=0.01, nu=0.00089, wall_nu=0.00089, h=2.5, x=2.4, y=1.8
# adh = 0.15, surft=0.01, nu=0.00089, h=2.5, wall_nu=0.5*0.00089, x=2.2, y=1.7 <-lower wall viscosity
# adh = 0.15, surft=0.01, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.6, y=1.6 <-lower wall viscosity
# adh = 0.15, surft=0.015, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.2, y=1.8
# adh = 0.2, surft=0.015, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.2, y=1.8
# adh = 0.2, surft=0.0125, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.6, y=1.7
# adh = 0.25, surft=0.014, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.4, y=1.6
# adh = 0.25, surft=0.014, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.8, y=1.5 diverged for up to 3s
# adh = 0.25, surft=0.015, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.5, y=1.6 diverged at 2.722 <----
# adh = 0.25, surft=0.016, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.5, y=1.6
# adh = 0.1, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=20*0.00089, x=2.2, y=1.7 <- 75d
# adh = 0.15, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=20*0.00089, x=2.2, y=1.7
# adh = 0.15, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.3, y=1.7
# adh = 0.2, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.6, y=1.6
# adh = 0.15, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=0.00089, x=2.6, y=1.5
# adh = 0.15, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=5 * 0.00089, x=2.4, y=1.6
# adh = 0.13, surft=0.011,  nu=0.00089, h=2.5, wall_nu=5*0.00089, x=2.6, y=1.5
# adh = 0.11, surft=0.011,  nu=0.00089, h=2.5, wall_nu=5*0.00089, x=2.4, y=1.6


# 75deg (x-axis: 2.2mm, y-axis: 1.6mm)
# adh = 0.20, surft=0.015, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.5, y=1.6
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=20*0.00089, x=2, y=1.7 <- 90d
# adh = 0.1, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=20*0.00089, x=2.2, y=1.7
# adh = 0.1, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=7*0.00089, x=2.2, y=1.6
# adh = 0.08, surft=0.011,  nu=0.00089, h=2.5, wall_nu=7*0.00089, x=2.3, y=1.6
# adh = 0.07, surft=0.011,  nu=0.00089, h=2.5, wall_nu=7*0.00089, x=2.2, y=1.6
# adh = 0.07, surft=0.011,  nu=0.00089, h=2.5, wall_nu=5*0.00089, x=2.3, y=1.6


# 90deg (x-axis: 2mm, y-axis: 1.8mm)
# adh = 0.15, surft=0.015, nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.5, y=1.6
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=0.25*0.00089, x=2.2, y=1.7 after capallariy test
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=0.4*0.00089, x=2.2, y=1.6
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=0.00089, x=2.2, y=1.6
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=2*0.00089, x=2.2, y=1.7
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=4*0.00089, x=2.2, y=1.7
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.2, y=1.7
# adh = 0.05, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=20*0.00089, x=2, y=1.7 <- does not work in capallariy test
# adh = 0.04, surft=0.0125,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.1, y=1.7 <- further testing showed surface tension is about 10-20% too high
# adh = 0.04, surft=0.011,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.2, y=1.7
# adh = 0.03, surft=0.011,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.2, y=1.7 ****
# adh = 0.02, surft=0.011,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.2, y=1.8
# adh = 0.02, surft=0.011,  nu=0.00089, h=2.5, wall_nu=15*0.00089, x=2.2, y=1.7
# adh = 0.01, surft=0.011,  nu=0.00089, h=2.5, wall_nu=10*0.00089, x=2.2, y=1.7

# sphere = WeaklyCompressibleSPHSystem(sphere2, fluid_density_calculator,
#                                      state_equation, fluid_smoothing_kernel_2,
#                                      fluid_smoothing_length_2, viscosity=viscosity,
#                                      density_diffusion=density_diffusion,
#                                      acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel_2,
                                             fluid_smoothing_length_2,
                                             viscosity=ViscosityAdami(nu=5 * nu))

# adhesion_coefficient = 1.0 and surface_tension_coefficient=0.01 for perfect wetting
# adhesion_coefficient = 0.001 and surface_tension_coefficient=2.0 for no wetting
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model, adhesion_coefficient=0.07)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(boundary_system, sphere_surface_tension)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out", prefix="",
                                         write_meta_data=true)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6
            reltol=1e-4, # Default reltol is 1e-3
            dt=1e-4, maxiters=1E12,
            save_everystep=false, callback=callbacks);
