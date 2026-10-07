module FSIPairFixtures
using TrixiParticles
using LinearAlgebra: I
using Test: @test_logs

export structure_fluid_pair_state

# Prescribed pair states isolate interaction operators from density/pressure updates.
# Defaults put one fluid particle at the origin and a boundary particle at r/h = 1.5,
# inside the cubic-spline support. Unequal masses, densities, pressures, and velocities
# make swapped arguments or accidental use of material quantities observable.
function structure_fluid_pair_state(; fluid_scheme=:wcsph, structure_kind=:tlsph,
                                    fluid_options=(;),
                                    boundary_density=AdamiPressureExtrapolation(),
                                    distance=1.5, dimensions=2,
                                    boundary_correction=nothing,
                                    monaghan_kajtar=false,
                                    structure_smoothing_length=1.0,
                                    structure_smoothing_kernel=SchoenbergCubicSplineKernel{dimensions}(),
                                    coordinates=reshape([distance; zeros(dimensions - 1)],
                                                        dimensions, 1),
                                    parallelization_backend=SerialBackend())
    smoothing_kernel = SchoenbergCubicSplineKernel{dimensions}()
    smoothing_length = particle_spacing = 1.0
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1.0, clip_negative_pressure=false)
    velocity = reshape([1.0, 0.5, 0.7][1:dimensions], dimensions, 1)
    fluid_ic = InitialCondition(; coordinates=zeros(dimensions, 1), velocity,
                                mass=[1100.0], density=[1005.0], pressure=500.0,
                                particle_spacing)
    options = (; density_calculator=ContinuityDensity(), reference_particle_spacing=1.0,
               fluid_options...)
    fluid = if fluid_scheme == :wcsph
        WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                    state_equation, options...)
    elseif fluid_scheme == :edac
        # Keep absolute pressure by default; individual reduction tests opt in explicitly.
        options = (; average_pressure_reduction=false, options...)
        EntropicallyDampedSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                    sound_speed=10.0, options...)
    else
        # Prescribe pressure for the IISPH pair RHS instead of invoking its PPE solve.
        ImplicitIncompressibleSPHSystem(fluid_ic; smoothing_kernel, smoothing_length,
                                        reference_density=1005.0, time_step=0.001,
                                        viscosity=get(fluid_options, :viscosity, nothing))
    end
    n = size(coordinates, 2)
    structure_velocity = repeat(reshape([0.25, -0.4, 0.2][1:dimensions], dimensions, 1), 1,
                                n)
    # Material masses differ from the boundary model's hydrodynamic masses below.
    # Varying them with particle index also exercises nonuniform force/torque weights.
    structure_ic = InitialCondition(; coordinates, velocity=structure_velocity,
                                    mass=2100.0 .+ 300.0 .* (0:(n - 1)),
                                    density=fill(2000.0, n),
                                    particle_spacing)
    viscosity = get(fluid_options, :viscosity, nothing)
    hydrodynamic_mass = 700.0 .+ 50.0 .* (0:(n - 1))
    # Repulsive particles derive density from mass/spacing^dimensions; at spacing=1,
    # the single MK boundary particle has rho_s=m_s=700 instead of dummy density 950.
    boundary_model = monaghan_kajtar ?
                     BoundaryModelMonaghanKajtar(10.0, 1.0, particle_spacing,
                                                 hydrodynamic_mass) :
                     BoundaryModelDummyParticles(fill(950.0, n), hydrodynamic_mass,
                                                 boundary_density, smoothing_kernel,
                                                 smoothing_length;
                                                 state_equation, viscosity,
                                                 correction=boundary_correction,
                                                 reference_particle_spacing=1.0)
    structure = if structure_kind == :rigid
        RigidBodySystem(structure_ic; boundary_model, adhesion_coefficient=0.25)
    elseif structure_kind == :wall
        WallBoundarySystem(structure_ic, boundary_model)
    else
        TotalLagrangianSPHSystem(structure_ic; smoothing_kernel=structure_smoothing_kernel,
                                 smoothing_length=structure_smoothing_length,
                                 young_modulus=1.0e5, poisson_ratio=0.3, boundary_model)
    end
    # Assert the expected TLS-copy notice and reject any other setup logs or warnings.
    semi = if structure isa TotalLagrangianSPHSystem
        @test_logs (:info,
                    r"^To create the self-interaction neighborhood search of a `TotalLagrangianSPHSystem`") begin
            Semidiscretization(fluid, structure; parallelization_backend)
        end
    else
        @test_logs Semidiscretization(fluid, structure; parallelization_backend)
    end
    ode = semidiscretize(semi, (0.0, 0.01); reset_threads=false)
    # Modify the semidiscretization-owned TLS copy used by the actual pair operators.
    fluid, structure = semi.systems
    v_ode, u_ode = ode.u0.x
    v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
    u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
    v_structure = TrixiParticles.wrap_v(v_ode, structure, semi)
    u_structure = TrixiParticles.wrap_u(u_ode, structure, semi)
    # These states are imposed directly so pressure extrapolation or a density update
    # cannot make the fluid and boundary values equal and hide an argument-order bug.
    TrixiParticles.current_density(v_fluid, fluid) .= 1005.0
    TrixiParticles.current_pressure(v_fluid, fluid) .= 500.0
    if !monaghan_kajtar
        structure.boundary_model.pressure .= 230.0
        # A single fluid neighbor gives the no-slip ghost velocity 2v_s - v_f.
        !isnothing(viscosity) &&
            (structure.boundary_model.cache.wall_velocity .= 2 .* structure_velocity .-
                                                             velocity)
    end
    # Prescribed nonzero shifts and non-unit correction data give independent pair
    # expectations; cache-construction tests must populate their own caches via updates.
    for (field, value) in ((:delta_v, [0.4, -0.3, 0.2][1:dimensions]),
         (:dw_gamma, [0.1, -0.2, 0.15][1:dimensions]),
         (:kernel_correction_coefficient, 1.3), (:pressure_average, 120.0))
        haskey(fluid.cache, field) && (getproperty(fluid.cache, field) .= value)
    end
    if haskey(fluid.cache, :correction_matrix)
        # The off-diagonal entry exposes component mixing for non-radial gradients.
        fluid.cache.correction_matrix[:, :, 1] .= Matrix{Float64}(I, dimensions, dimensions)
        fluid.cache.correction_matrix[1, 2, 1] = 0.3
    end
    return (; fluid, structure, semi, ode, v_fluid, u_fluid, v_structure, u_structure)
end
end
