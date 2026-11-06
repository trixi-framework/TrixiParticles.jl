using OrdinaryDiffEqLowStorageRK

struct PostBoundaryRefreshCounter{S}
    systems::S
    calls::Base.RefValue{Int}
end

function TrixiParticles.update_systems_and_nhs(v, u, semi::PostBoundaryRefreshCounter, t)
    semi.calls[] += 1
    return nothing
end

@testset "Open-boundary caches before native shifting" begin
    function inlet_setup()
        spacing = 0.1
        density = 1000.0
        velocity = SVector(0.1, 0.0)
        fluid_ic = RectangularShape(spacing, (8, 6), (0.0, 0.0); density, velocity)
        inlet_ic = RectangularShape(spacing, (8, 6), (-0.8, 0.0); density, velocity)
        outlet_ic = RectangularShape(spacing, (8, 6), (0.8, 0.0); density, velocity)
        shifting = ParticleShiftingTechniqueSun2017(; sound_speed_factor=1e-4,
                                                    v_max_factor=0)
        equation = StateEquationCole(; sound_speed=10.0, reference_density=density,
                                     exponent=1)
        fluid = WeaklyCompressibleSPHSystem(fluid_ic;
                                            smoothing_kernel=WendlandC6Kernel{2}(),
                                            smoothing_length=2spacing,
                                            density_calculator=SummationDensity(),
                                            state_equation=equation,
                                            correction=MixedKernelGradientCorrection(),
                                            shifting_technique=shifting, buffer_size=1)
        inflow = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 0.6]),
                              face_normal=(1.0, 0.0), particle_spacing=spacing,
                              density, open_boundary_layers=8,
                              reference_velocity=velocity, reference_density=density,
                              initial_condition=inlet_ic)
        outflow = BoundaryZone(; boundary_face=([0.8, 0.0], [0.8, 0.6]),
                               face_normal=(-1.0, 0.0), particle_spacing=spacing,
                               density, open_boundary_layers=8, reference_density=density,
                               initial_condition=outlet_ic)
        boundary_model = BoundaryModelMirroringTafuni(;
                                                      mirror_method=ZerothOrderMirroring())
        open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system=fluid,
                                           boundary_model, buffer_size=2)
        cells = FullGridCellList(; min_corner=(-0.9, -0.1), max_corner=(1.7, 0.7))
        search = GridNeighborhoodSearch{2}(; cell_list=cells,
                                           update_strategy=ParallelUpdate())
        semi = Semidiscretization(fluid, open_boundary; neighborhood_search=search,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1); reset_threads=false)
        return init(ode, RDPK3SpFSAL35(); dt=1e-3, adaptive=false,
                    callback=UpdateCallback(), save_everystep=false)
    end

    function state(integrator)
        semi = integrator.p.semi
        v, u = integrator.u.x
        fluid, boundary = semi.systems
        return (; semi, v, u, fluid, boundary,
                u_fluid=TrixiParticles.wrap_u(u, fluid, semi),
                u_boundary=TrixiParticles.wrap_u(u, boundary, semi))
    end

    function nearest(system, u, position)
        active = collect(TrixiParticles.each_active_particle(system))
        return active[argmin([norm(TrixiParticles.current_coords(u, system, i) - position)
                              for i in active])]
    end

    actual, expected = inlet_setup(), inlet_setup()
    initial = state(actual)
    active_count = initial.fluid.buffer.active_particle_count[]
    first_new = findfirst(!, initial.fluid.buffer.active_particle)
    first_out = nearest(initial.fluid, initial.u_fluid, SVector(0.75, 0.25))
    callback = UpdateCallback().affect!
    for phase in (:activate, :density_change, :reuse)
        for integrator in (actual, expected)
            s = state(integrator)
            y = phase == :activate ? 0.25 : 0.35
            inlet = nearest(s.boundary, s.u_boundary, SVector(-0.05, y))
            if phase == :density_change
                # Prescribed boundary interpolation resets this without any transfer.
                s.boundary.cache.density[inlet] = 2000.0
            else
                outlet = phase == :activate ? first_out : first_new
                s.u_boundary[:, inlet] .= (0.025, y)
                s.u_fluid[:, outlet] .= (0.825, y)
            end
            integrator.tprev = integrator.t
            integrator.t += integrator.dt
        end

        # Fresh-cache reference for the same accepted endpoint and native shifting.
        s = state(expected)
        TrixiParticles.update_systems_and_nhs(s.v, s.u, s.semi, expected.t)
        TrixiParticles.update_open_boundary_eachstep!(s.boundary, s.v, s.u, s.semi,
                                                      expected.t, expected)
        TrixiParticles.update_systems_and_nhs(s.v, s.u, s.semi, expected.t)
        TrixiParticles.particle_shifting_from_callback!(s.u,
                                                        TrixiParticles.shifting_technique(s.fluid),
                                                        s.fluid, s.v, s.semi, expected)
        callback(actual)
        s = state(actual)
        @test s.fluid.buffer.active_particle_count[] == active_count
        @test s.fluid.buffer.active_particle[phase == :reuse ? first_out : first_new]
        @test all(isfinite, actual.u)
        @test actual.u.x[1]≈expected.u.x[1] rtol=1e-12 atol=1e-12
        @test actual.u.x[2]≈expected.u.x[2] rtol=1e-12 atol=1e-12
    end

    # Count only the dispatched post-boundary preparation: skip absent boundaries
    # and absent consumers; multiple consumers still share one global pass.
    s = state(actual)
    for (systems, expected_calls) in (((s.fluid,), 0), ((s.boundary,), 0),
        ((s.boundary, s.fluid, s.fluid), 1))
        probe = PostBoundaryRefreshCounter(systems, Ref(0))
        TrixiParticles.update_systems_and_nhs(s.v, s.u, probe, actual.t, callback)
        @test probe.calls[] == expected_calls
    end
end
