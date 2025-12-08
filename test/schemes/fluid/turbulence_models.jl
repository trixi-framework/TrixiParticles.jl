include("../../test_util.jl")

sps = SPSTurbulenceModelDalrymple(; smallest_length_scale=1.0)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid",
                       "hydrostatic_water_column_2d.jl"),
              tspan=(0.0, 0.02))

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity_fluid,
                                           acceleration=system_acceleration,
                                           turbulence_model=sps)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid",
                       "hydrostatic_water_column_2d.jl"),
              fluid_system=fluid_system)

v_ode, u_ode = sol[end].x

v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)

TrixiParticles.calculate_fluid_stress_tensor!(fluid_system,
                                              TrixiParticles.turbulence_model(fluid_system),
                                              v, u, v_ode, u_ode, semi)
