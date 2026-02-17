###############################################################################
# bubble_rise_carreau_yasuda.jl
#
# Demonstration / validation-style example:
#   Run the same bubble-rise setup twice:
#     (A) Newtonian viscosity (constant ν)
#     (B) Carreau–Yasuda viscosity (ν(γ̇), shear-thinning for n < 1)
#
# The goal is to show a clear qualitative and quantitative difference vs.
# a Newtonian fluid, consistent with trends reported in the non-Newtonian
# bubble-rise literature (e.g., Vahabi & Sadeghy, 2014).
#
###############################################################################

using TrixiParticles
using OrdinaryDiffEq
using StaticArrays

# ----------------------------
# User-tunable physical params
# ----------------------------
const NDIMS = 2

# Domain (tank) 
Lx = 1.0
Ly = 1.2

# Particle resolution 
dx = 0.02                      # particle spacing 
h  = 1.3dx                     # smoothing length
kernel = WendlandC2Kernel{NDIMS}()

# Gravity (bubble rises because gas is lighter)
g = SVector(0.0, -9.81)

# Liquid properties (reference)
ρ0_liq = 1000.0
c_liq  = 70.0
γ_eos  = 7.0

# Gas/bubble properties 
ρ0_gas = 50.0
c_gas  = 30.0
γ_gas  = 1.4

# Viscosity: Newtonian baseline (kinematic)
ν_liq_newton = 0.01             # m^2/s 

# Gas viscosity: keep simple & constant (avoid unphysical CY on gas)
ν_gas = 1.0e-5                 # small kinematic viscosity

# Carreau–Yasuda parameters for the *liquid* (kinematic viscosities!)
# ν∞ << ν0 and n << 1 -> EXTREME shear-thinning for visible difference
ν0     = 0.05                  # Zero-shear viscosity (MUCH higher than Newtonian!)
ν∞     = 0.001*ν0             # Infinite-shear viscosity (tiny, 1000:1 ratio!)
λ      = 0.05                 # Time constant (very low -> triggers thinning early)
a_cy   = 2.0
n_cy   = 0.2                  # Very low n -> extreme shear-thinning
ϵ_cy   = 0.1*dx

# Surface tension coefficient 
σ = 0.01

# Simulation time
tspan = (0.0, 2.5)

# Output prefixes 
out_prefix_newton = "out_bubble_newton"
out_prefix_cy     = "out_bubble_carreau_yasuda"

# Bubble placement - centered in the middle
bubble_center = (0.5Lx, 0.5Ly)
bubble_radius = 0.10Lx

# ----------------------------------------
# Geometry + initial condition constructors
# ----------------------------------------
"""
Create particles for a filled rectangular tank of liquid and a circular bubble (gas).
Returns (ic_liq, ic_gas, ic_wall).
"""
function make_tank_with_bubble(; Lx, Ly, dx, bubble_center, bubble_radius, ρ0_liq, ρ0_gas)
    boundary_layers = 2
    tank = RectangularTank(dx, (Lx, Ly), (Lx, Ly), ρ0_liq;
                           n_layers=boundary_layers,
                           faces=(true, true, true, true))

    ic_liq  = tank.fluid
    ic_wall = tank.boundary

    # Bubble (2D "sphere" shape)
    ic_gas = SphereShape(dx, bubble_radius, bubble_center, ρ0_gas)

    # Remove overlap (keep phases disjoint)
    ic_liq = setdiff(ic_liq, ic_gas)

    return ic_liq, ic_gas, ic_wall
end

# ----------------------------------------
# Build one simulation (Newtonian or CY)
# ----------------------------------------
function build_simulation(; viscosity_liquid, viscosity_gas, output_prefix::String)
    # Initial conditions
    ic_liq, ic_gas, ic_wall = make_tank_with_bubble(;
        Lx=Lx, Ly=Ly, dx=dx,
        bubble_center=bubble_center,
        bubble_radius=bubble_radius,
        ρ0_liq=ρ0_liq,
        ρ0_gas=ρ0_gas
    )

    # Density calculators
    dens_liq = ContinuityDensity()
    dens_gas = ContinuityDensity()

    # Equations of state
    eos_liq = StateEquationCole(sound_speed=c_liq,
                               reference_density=ρ0_liq,
                               exponent=γ_eos,
                               background_pressure=0.0,
                               clip_negative_pressure=false)

    eos_gas = StateEquationIdealGas(sound_speed=c_gas,
                                   reference_density=ρ0_gas,
                                   gamma=γ_gas,
                                   background_pressure=0.0,
                                   clip_negative_pressure=false)

    # Density diffusion for stability (increased for better stabilization)
    dd = DensityDiffusionMolteniColagrossi(delta=0.1)

    # Pressure smoothing correction for stability (Akinci free surface correction)
    correction_liq = AkinciFreeSurfaceCorrection(ρ0_liq)
    correction_gas = AkinciFreeSurfaceCorrection(ρ0_gas)

    # Surface tension + normals
    surface_normal  = ColorfieldSurfaceNormal()
    surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=σ)

    # Fluid systems with full stabilization
    fluid_liq = WeaklyCompressibleSPHSystem(
        ic_liq, dens_liq, eos_liq, kernel, h;
        acceleration=g,
        viscosity=viscosity_liquid,
        density_diffusion=dd,
        correction=correction_liq,
        surface_tension=surface_tension,
        surface_normal_method=surface_normal,
        reference_particle_spacing=dx,
    )

    fluid_gas = WeaklyCompressibleSPHSystem(
        ic_gas, dens_gas, eos_gas, kernel, h;
        acceleration=g,
        viscosity=viscosity_gas,
        density_diffusion=dd,
        correction=correction_gas,
        surface_tension=surface_tension,
        surface_normal_method=surface_normal,
        reference_particle_spacing=dx,
    )

    # Boundary system 
    boundary_density_calculator = AdamiPressureExtrapolation()
    viscosity_wall = nothing

    boundary_model = BoundaryModelDummyParticles(ic_wall.density,
                                                 ic_wall.mass,
                                                 boundary_density_calculator,
                                                 kernel, h;
                                                 state_equation=eos_liq,
                                                 correction=nothing,
                                                 reference_particle_spacing=dx,
                                                 viscosity=viscosity_wall)

    wall = WallBoundarySystem(ic_wall, boundary_model)

    # Couple systems
    semi = Semidiscretization(fluid_liq, fluid_gas, wall)

    # Time integration
    ode = semidiscretize(semi, tspan)

    # Callbacks (reduced CFL for better stability)
    info_cb     = InfoCallback(interval=100)
    stepsize_cb = StepsizeCallback(cfl=0.15)
    save_cb     = SolutionSavingCallback(dt=0.02, prefix=output_prefix)

    cb = CallbackSet(info_cb, stepsize_cb, save_cb)

    return ode, cb
end

# -------------------------
# Case A: Newtonian baseline
# -------------------------
visc_liq_newton = ViscosityAdami(nu=ν_liq_newton)
visc_gas_const  = ViscosityAdami(nu=ν_gas)

odeN, cbN = build_simulation(viscosity_liquid=visc_liq_newton,
                             viscosity_gas=visc_gas_const,
                             output_prefix=out_prefix_newton)

solN = solve(odeN, RDPK3SpFSAL49(); callback=cbN, adaptive=true)

# --------------------------------
# Case B: Carreau–Yasuda (liquid)
# --------------------------------
visc_liq_cy = ViscosityCarreauYasuda(nu0=ν0, nu_inf=ν∞, lambda=λ, a=a_cy, n=n_cy, epsilon=ϵ_cy)

odeCY, cbCY = build_simulation(viscosity_liquid=visc_liq_cy,
                               viscosity_gas=visc_gas_const,
                               output_prefix=out_prefix_cy)

solCY = solve(odeCY, RDPK3SpFSAL49(); callback=cbCY, adaptive=true)

println("\nDone.")