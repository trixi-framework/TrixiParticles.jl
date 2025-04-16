using TrixiParticles
using OrdinaryDiffEq
using Pkg
using Plots

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.02

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81 # gravity verändern
tspan = (0.0, 0.4)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (1.0, 1.0)
tank_size = (2.0, 2.0)

fluid_density = 1000.0

# TODO: Was machen mit dem sound speed?
#sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
sound_speed = 1000.0 

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio)

# Move water column
for i in axes(tank.fluid.coordinates, 2)
    tank.fluid.coordinates[:, i] .+= [0.5 * tank_size[1] - 0.5 * initial_fluid_size[1], 0.1]
end

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = ImplicitIncompressibleSPHSystem(tank.fluid, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity, 
                                           acceleration=(0.0, -gravity))


# ==========================================================================================
# ==== Boundary
boundary_density_calculator = PressureMirroring() #TODO PressureMirroring richtig oder Pressure zeoring oder was ganz anderes???

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=nothing,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, SymplecticEuler(),
            dt = 0.001,
            save_everystep=false, callback=callbacks);
plot(sol)

