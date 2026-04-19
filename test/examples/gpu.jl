const TRIXIPARTICLES_TEST_ = lowercase(get(ENV, "TRIXIPARTICLES_TEST", "all"))

if TRIXIPARTICLES_TEST_ == "cuda"
    using CUDA
    CUDA.versioninfo()
    parallelization_backend = CUDABackend()
    supports_double_precision = true
    fp64_fastdiv = true
elseif TRIXIPARTICLES_TEST_ == "amdgpu"
    using AMDGPU
    AMDGPU.versioninfo()
    parallelization_backend = ROCBackend()
    supports_double_precision = true
    fp64_fastdiv = false
elseif TRIXIPARTICLES_TEST_ == "metal"
    using Metal
    Metal.versioninfo()
    parallelization_backend = MetalBackend()
    supports_double_precision = false
elseif TRIXIPARTICLES_TEST_ == "oneapi"
    using oneAPI
    oneAPI.versioninfo()
    parallelization_backend = oneAPIBackend()
    # The runners are using an iGPU, which does not support double precision
    supports_double_precision = false
else
    error("Unknown GPU backend: $TRIXIPARTICLES_TEST_")
end

@testset verbose=true "div_fast $TRIXIPARTICLES_TEST_" begin
    @testset verbose=true "CPU Float64" begin
        x = Float64(pi)
        y = rand(Float64, 1024) .+ 1

        # We expect exact equality for `Float64` on the CPU
        @test TrixiParticles.div_fast.(x, y) == x ./ y
    end

    @testset verbose=true "CPU Float32" begin
        x = Float32(pi)
        y = rand(Float32, 1024) .+ 1

        # We don't test `max_error > 0`, since this might be exact on some CPUs
        # (we observed this on ARM CPUs).
        max_error = maximum(abs.(TrixiParticles.div_fast.(x, y) - x ./ y))
        @test max_error < 1.0f-6
    end

    @testset verbose=true "GPU Float32" begin
        x = Float32(pi)
        y = Adapt.adapt(parallelization_backend, rand(Float32, 1024) .+ 1)

        max_error = maximum(abs.(TrixiParticles.div_fast.(x, y) - x ./ y))
        @test max_error < 1.0f-6

        # Make sure that this is actually using a fast division
        @test max_error > 0
    end

    if supports_double_precision
        @testset verbose=true "GPU Float64" begin
            x = Float64(pi)
            y = Adapt.adapt(parallelization_backend, rand(Float64, 1024) .+ 1)

            max_error = maximum(abs.(TrixiParticles.div_fast.(x, y) - x ./ y))

            if fp64_fastdiv
                @test max_error < 1e-15

                # Make sure that this is actually using a fast division
                @test max_error > 0
            else
                # If fast division for Float64 is not supported, we expect exact equality
                @test max_error == 0
            end
        end
    end
end

@testset verbose=true "velocity_and_density $TRIXIPARTICLES_TEST_" begin
    if supports_double_precision
        types = [Float64, Float32]
    else
        types = [Float32]
    end

    @testset verbose=true "use_simd_load $T" for T in types
        # Aligned array
        x = zeros(T, 16)

        # Test different "systems" and density calculators
        @test !TrixiParticles.use_simd_load(x, nothing)
        @test !TrixiParticles.use_simd_load(x, (; density_calculator=ContinuityDensity()))
        @test !TrixiParticles.use_simd_load(x, nothing, SummationDensity())

        # No SIMD load on the CPU
        @test !TrixiParticles.use_simd_load(x, nothing, ContinuityDensity())

        y = Adapt.adapt(parallelization_backend, x)

        # Test different "systems" and density calculators
        @test !TrixiParticles.use_simd_load(y, nothing)
        @test !TrixiParticles.use_simd_load(y, (; density_calculator=ContinuityDensity()))
        @test !TrixiParticles.use_simd_load(y, nothing, SummationDensity())

        # Use SIMD load on the GPU with 4-aligned arrays
        @test TrixiParticles.use_simd_load(y, nothing, ContinuityDensity())
        @test TrixiParticles.use_simd_load(view(y, 5:16), nothing, ContinuityDensity())

        # Unaligned array on the GPU should throw an error
        @test_throws "on GPUs in 3D" TrixiParticles.use_simd_load(view(y, 2:16), nothing,
                                                                  ContinuityDensity())
        @test_throws "on GPUs in 3D" TrixiParticles.use_simd_load(view(y, 3:16), nothing,
                                                                  ContinuityDensity())
        @test_throws "on GPUs in 3D" TrixiParticles.use_simd_load(view(y, 4:16), nothing,
                                                                  ContinuityDensity())
    end

    @testset "velocity_and_density $T" for T in types
        # Aligned array
        x = rand(T, 4, 4)
        y = Adapt.adapt(parallelization_backend, x)

        # Dummy system that will only be used for `ndims`.
        struct MockSystem end
        Base.ndims(::MockSystem) = 3
        system = MockSystem()
        @inline TrixiParticles.current_density(v, ::MockSystem, i) = v[4, i]

        # Test that the SIMD version is consistent with the non-SIMD version.
        # We have 4 particles (with 4 values per particles).
        # In order to test this on the GPU, we need to use a kernel with `@threaded`.
        result = Adapt.adapt(parallelization_backend, zeros(Bool, 4))
        TrixiParticles.@threaded parallelization_backend for i in 1:4
            result[i] = TrixiParticles.velocity_and_density(y, system, Val(false), i) ==
                        TrixiParticles.velocity_and_density(y, system, Val(true), i)
        end
        @test all(result)
    end
end

@testset verbose=true "extract_smatrix_aligned $TRIXIPARTICLES_TEST_" begin
    if supports_double_precision
        types = [Float64, Float32]
    else
        types = [Float32]
    end

    @testset verbose=true "$T" for T in types
        @testset verbose=true "$(N)D" for N in 2:3
            @testset "CPU" begin
                A = rand(T, N, N, 4)

                # Test that the SIMD version is consistent with the non-SIMD version.
                for i in 1:4
                    @test TrixiParticles.extract_smatrix_aligned(A, Val(N), i) ==
                          TrixiParticles.extract_smatrix(A, Val(N), i)
                end
            end

            @testset "GPU" begin
                A = Adapt.adapt(parallelization_backend, rand(T, N, N, 4))
                val = Val(N)

                # Test that the SIMD version is consistent with the non-SIMD version.
                # In order to test this on the GPU, we need to use a kernel with `@threaded`.
                result = Adapt.adapt(parallelization_backend, zeros(Bool, 4))
                TrixiParticles.@threaded parallelization_backend for i in 1:4
                    result[i] = TrixiParticles.extract_smatrix_aligned(A, val, i) ==
                                TrixiParticles.extract_smatrix(A, val, i)
                end
                @test all(result)
            end
        end
    end
end

@trixi_testset "Semidiscretization vector alignment" begin
    # Mock systems
    struct System1 <: TrixiParticles.AbstractSystem{3} end
    struct System2 <: TrixiParticles.AbstractSystem{3} end

    system1 = System1()
    system2 = System2()

    Base.eltype(::System1) = Float64
    Base.eltype(::System2) = Float64
    TrixiParticles.u_nvariables(::System1) = 3
    TrixiParticles.u_nvariables(::System2) = 4
    TrixiParticles.v_nvariables(::System1) = 3
    TrixiParticles.v_nvariables(::System2) = 2
    TrixiParticles.nparticles(::System1) = 2
    TrixiParticles.nparticles(::System2) = 3

    TrixiParticles.compact_support(::System1, neighbor) = 0.2
    TrixiParticles.compact_support(::System2, neighbor) = 0.2

    @testset verbose=true "Constructor" begin
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing,
                                  parallelization_backend=Main.parallelization_backend)

        # These are the ranges that we would expect on the CPU:
        # semi.ranges_u == (1:6, 7:18)
        # semi.ranges_v == (1:6, 7:12)
        # Due to alignment to 64 bytes, the ranges are adjusted to be:
        @test semi.ranges_u == (1:6, 9:20)
        @test semi.ranges_v == (1:6, 9:14)
    end
end

@testset verbose=true "Examples $TRIXIPARTICLES_TEST_" begin
    @testset verbose=true "Fluid" begin
        @trixi_testset "fluid/dam_break_2d_gpu.jl Float64" begin
            if Main.supports_double_precision
                @trixi_test_nowarn trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "dam_break_2d_gpu.jl"),
                                                 tspan=(0.0, 0.1),
                                                 parallelization_backend=Main.parallelization_backend) [
                    r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n",
                    r"┌ Info: The desired tank length in y-direction.*\n",
                    r"└ New tank length in y-direction.*\n"
                ]
                # Since Julia 1.12 has issues with world age, we need to use `@invokelatest`
                # everywhere here. For some reason, this is only necessary in this and the
                # next test, but not in the other tests in this file.
                # Perhaps because this is inside an `if` block?
                @test (@invokelatest (@__MODULE__).semi).neighborhood_searches[1, 1].cell_list isa
                      FullGridCellList
                @test (@invokelatest (@__MODULE__).sol).retcode ==
                      (@invokelatest (@__MODULE__).ReturnCode).Success
                v_ode, u_ode = (@invokelatest (@__MODULE__).sol).u[end].x
                backend = TrixiParticles.KernelAbstractions.get_backend(v_ode)
                @test backend == Main.parallelization_backend
                @test eltype(v_ode) == Float64
                @test eltype(u_ode) == Float64
            else
                error = "Metal does not support Float64 values, try using Float32 instead"
                @test_throws error trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "dam_break_2d_gpu.jl"),
                                                 tspan=(0.0, 0.1),
                                                 parallelization_backend=Main.parallelization_backend)
            end
        end

        @trixi_testset "fluid/dam_break_2d_gpu.jl Float32 + Float64 coordinates" begin
            if Main.supports_double_precision
                @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                                 joinpath(examples_dir(),
                                                                          "fluid",
                                                                          "dam_break_2d_gpu.jl"),
                                                                 tspan=(0.0f0, 0.1f0),
                                                                 parallelization_backend=Main.parallelization_backend) [
                    r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n",
                    r"┌ Info: The desired tank length in y-direction .*\n",
                    r"└ New tank length in y-direction.*\n"
                ]
                # See the comment in the previous test about `@invokelatest`
                @test (@invokelatest (@__MODULE__).semi).neighborhood_searches[1, 1].cell_list isa
                      FullGridCellList
                @test (@invokelatest (@__MODULE__).sol).retcode ==
                      (@invokelatest (@__MODULE__).ReturnCode).Success
                v_ode, u_ode = (@invokelatest (@__MODULE__).sol).u[end].x
                backend = TrixiParticles.KernelAbstractions.get_backend(v_ode)
                @test backend == Main.parallelization_backend
                @test eltype(v_ode) == Float32
                @test eltype(u_ode) == Float64
            else
                error = "Metal does not support Float64 values, try using Float32 instead"
                @test_throws error trixi_include_changeprecision(Float32, @__MODULE__,
                                                                 joinpath(examples_dir(),
                                                                          "fluid",
                                                                          "dam_break_2d_gpu.jl"),
                                                                 tspan=(0.0f0, 0.1f0),
                                                                 parallelization_backend=Main.parallelization_backend)
            end
        end

        @trixi_testset "fluid/dam_break_2d_gpu.jl Float32" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(),
                                                   "fluid", "dam_break_2d.jl");
                                          coordinates_eltype=Float32,
                                          sol=nothing, ode=nothing)

            dam_break_tests = Dict(
                "default" => (),
                "DensityDiffusionMolteniColagrossi" => (density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1f0),),
                "DensityDiffusionFerrari" => (density_diffusion=DensityDiffusionFerrari(),)
            )

            for (test_description, kwargs) in dam_break_tests
                @testset "$test_description" begin
                    println("═"^100)
                    println("$test_description")

                    @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                                     joinpath(examples_dir(),
                                                                              "fluid",
                                                                              "dam_break_2d_gpu.jl");
                                                                     tspan=(0.0f0, 0.1f0),
                                                                     coordinates_eltype=Float32,
                                                                     parallelization_backend=Main.parallelization_backend,
                                                                     kwargs...) [
                        r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n",
                        r"┌ Info: The desired tank length in y-direction.*\n",
                        r"└ New tank length in y-direction.*\n"
                    ]
                    @test semi.neighborhood_searches[1, 1].cell_list isa FullGridCellList
                    @test sol.retcode == ReturnCode.Success
                    v_ode, u_ode = sol.u[end].x
                    backend = TrixiParticles.KernelAbstractions.get_backend(v_ode)
                    @test backend == Main.parallelization_backend
                    @test eltype(v_ode) == Float32
                    @test eltype(u_ode) == Float32
                end
            end
        end

        @trixi_testset "fluid/dam_break_2d_gpu.jl Float32 with ContinuityDensity boundary density" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "dam_break_2d_gpu.jl");
                                                             tspan=(0.0f0, 0.1f0),
                                                             coordinates_eltype=Float32,
                                                             parallelization_backend=Main.parallelization_backend,
                                                             boundary_density_calculator=ContinuityDensity()) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n",
                r"┌ Info: The desired tank length in y-direction.*\n",
                r"└ New tank length in y-direction.*\n"
            ]
            @test semi.neighborhood_searches[1, 1].cell_list isa FullGridCellList
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/dam_break_2d_gpu.jl Float32 BoundaryModelMonaghanKajtar" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(),
                                                   "fluid", "dam_break_2d.jl");
                                          boundary_layers=1, spacing_ratio=3,
                                          coordinates_eltype=Float32,
                                          sol=nothing, ode=nothing)

            boundary_model = BoundaryModelMonaghanKajtar(0.5f0,
                                                         spacing_ratio,
                                                         boundary_particle_spacing,
                                                         tank.boundary.mass)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "dam_break_2d_gpu.jl");
                                                             tspan=(0.0f0, 0.1f0),
                                                             coordinates_eltype=Float32,
                                                             boundary_layers=1,
                                                             spacing_ratio=3,
                                                             boundary_model=boundary_model,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n",
                r"┌ Info: The desired tank length in y-direction.*\n",
                r"└ New tank length in y-direction.*\n"
            ]
            @test semi.neighborhood_searches[1, 1].cell_list isa FullGridCellList
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/dam_break_3d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "dam_break_3d.jl"),
                                          fluid_particle_spacing=0.1,
                                          coordinates_eltype=Float32,
                                          sol=nothing, ode=nothing)

            # Neighborhood search with `FullGridCellList` for GPU compatibility
            min_corner = minimum(tank.boundary.coordinates, dims=2)
            max_corner = maximum(tank.boundary.coordinates, dims=2)
            cell_list = FullGridCellList(; min_corner, max_corner)
            semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                               neighborhood_search=GridNeighborhoodSearch{3}(;
                                                                                             cell_list),
                                               parallelization_backend=Main.parallelization_backend)

            # Note that this simulation only takes 42 time steps on the CPU.
            # TODO This takes 44 time steps on Metal.
            # Maybe related to https://github.com/JuliaGPU/Metal.jl/issues/549
            ismetal = nameof(typeof(Main.parallelization_backend)) == :MetalBackend
            maxiters = ismetal ? 44 : 42
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "dam_break_3d.jl"),
                                                             tspan=(0.0f0, 0.1f0),
                                                             coordinates_eltype=Float32,
                                                             fluid_particle_spacing=0.1,
                                                             semi=semi_fullgrid,
                                                             maxiters=maxiters) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            v_ode, u_ode = sol.u[end].x
            backend = TrixiParticles.KernelAbstractions.get_backend(v_ode)
            @test backend == Main.parallelization_backend
            @test eltype(v_ode) == Float32
            @test eltype(u_ode) == Float32

            @testset "`SymplecticPositionVerlet`" begin
                stepsize_callback = StepsizeCallback(cfl=0.65)
                callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

                @trixi_test_nowarn sol = solve(ode, SymplecticPositionVerlet(),
                                               dt=1, # This is overwritten by the stepsize callback
                                               save_everystep=false, callback=callbacks)
                @test sol.retcode == ReturnCode.Success
                @test maximum(maximum.(abs, sol.u[end].x)) < 2^15
                v_ode, u_ode = sol.u[end].x
                backend = TrixiParticles.KernelAbstractions.get_backend(v_ode)
                @test backend == Main.parallelization_backend
                @test eltype(v_ode) == Float32
                @test eltype(u_ode) == Float32
            end
        end

        # Short tests to make sure that different models and kernels work on GPUs
        @trixi_testset "fluid/hydrostatic_water_column_2d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "hydrostatic_water_column_2d.jl"),
                                          sol=nothing, ode=nothing)

            # Create tank with Float32 coordinates
            tank = RectangularTank(fluid_particle_spacing, initial_fluid_size,
                                   tank_size, fluid_density, n_layers=boundary_layers,
                                   acceleration=(0.0f0, -gravity),
                                   state_equation=state_equation,
                                   coordinates_eltype=Float32)

            hydrostatic_water_column_tests = Dict(
                "WCSPH default" => (),
                "WCSPH with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1.0f-4),),
                "WCSPH with SummationDensity" => (fluid_density_calculator=SummationDensity(),
                                                  clip_negative_pressure=true),
                "WCSPH with ViscosityAdami" => (
                                                # from 0.02*10.0*1.2*0.05/8
                                                viscosity_fluid=ViscosityAdami(nu=0.0015f0),),
                "WCSPH with ViscosityMorris" => (
                                                 # from 0.02*10.0*1.2*0.05/8
                                                 viscosity_fluid=ViscosityMorris(nu=0.0015f0),),
                "WCSPH with ViscosityAdami and SummationDensity" => (
                                                                     # from 0.02*10.0*1.2*0.05/8
                                                                     viscosity_fluid=ViscosityAdami(nu=0.0015f0),
                                                                     fluid_density_calculator=SummationDensity(),
                                                                     maxiters=38, # 38 time steps on CPU
                                                                     clip_negative_pressure=true),
                "WCSPH with SchoenbergQuarticSplineKernel" => (smoothing_length=1.1,
                                                               smoothing_kernel=SchoenbergQuarticSplineKernel{2}()),
                "WCSPH with SchoenbergQuinticSplineKernel" => (smoothing_length=1.1,
                                                               smoothing_kernel=SchoenbergQuinticSplineKernel{2}()),
                "WCSPH with WendlandC2Kernel" => (smoothing_length=1.5,
                                                  smoothing_kernel=WendlandC2Kernel{2}()),
                "WCSPH with WendlandC4Kernel" => (smoothing_length=1.75,
                                                  smoothing_kernel=WendlandC4Kernel{2}()),
                "WCSPH with WendlandC6Kernel" => (smoothing_length=2.0,
                                                  smoothing_kernel=WendlandC6Kernel{2}()),
                "EDAC with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1.0f-4),
                                                    fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                             smoothing_kernel,
                                                                                             smoothing_length,
                                                                                             sound_speed,
                                                                                             viscosity=viscosity_fluid,
                                                                                             density_calculator=ContinuityDensity(),
                                                                                             acceleration=(0.0,
                                                                                                           -gravity))),
                "EDAC with SummationDensity" => (fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                          smoothing_kernel,
                                                                                          smoothing_length,
                                                                                          sound_speed,
                                                                                          viscosity=viscosity_fluid,
                                                                                          density_calculator=SummationDensity(),
                                                                                          acceleration=(0.0,
                                                                                                        -gravity)),)
            )

            for (test_description, kwargs) in hydrostatic_water_column_tests
                @testset "$test_description" begin
                    println("═"^100)
                    println("$test_description")

                    # Create systems with the given keyword arguments
                    trixi_include_changeprecision(Float32, @__MODULE__,
                                                  joinpath(examples_dir(), "fluid",
                                                           "hydrostatic_water_column_2d.jl");
                                                  sol=nothing, ode=nothing, tank=tank,
                                                  kwargs...)

                    # Neighborhood search with `FullGridCellList` for GPU compatibility
                    min_corner = minimum(tank.boundary.coordinates, dims=2)
                    max_corner = maximum(tank.boundary.coordinates, dims=2)
                    cell_list = FullGridCellList(; min_corner, max_corner,
                                                 max_points_per_cell=500)
                    semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                                       neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                                     cell_list),
                                                       parallelization_backend=Main.parallelization_backend)

                    # Run the simulation
                    @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                                     joinpath(examples_dir(),
                                                                              "fluid",
                                                                              "hydrostatic_water_column_2d.jl");
                                                                     semi=semi_fullgrid,
                                                                     tank=tank,
                                                                     tspan=(0.0f0, 0.1f0),
                                                                     kwargs...) [
                        r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n",
                    ]

                    @test sol.retcode == ReturnCode.Success
                    v_ode, u_ode = sol.u[end].x
                    backend = TrixiParticles.KernelAbstractions.get_backend(v_ode)
                    @test backend == Main.parallelization_backend
                    @test eltype(v_ode) == Float32
                    @test eltype(u_ode) == Float32
                end
            end
        end

        # Test periodic neighborhood search
        @trixi_testset "fluid/periodic_channel_2d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "periodic_channel_2d.jl"),
                                          coordinates_eltype=Float32,
                                          sol=nothing, ode=nothing)

            # Neighborhood search with `FullGridCellList` for GPU compatibility
            search_radius = TrixiParticles.compact_support(smoothing_kernel,
                                                           smoothing_length)
            min_corner = minimum(tank.boundary.coordinates, dims=2)
            max_corner = maximum(tank.boundary.coordinates, dims=2) .+ 2 * search_radius
            cell_list = FullGridCellList(; min_corner, max_corner)
            semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                               neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                             cell_list),
                                               parallelization_backend=Main.parallelization_backend)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "periodic_channel_2d.jl"),
                                                             tspan=(0.0f0, 0.1f0),
                                                             coordinates_eltype=Float32,
                                                             semi=semi_fullgrid) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/poiseuille_flow_2d.jl - BoundaryModelDynamicalPressureZhang (WCSPH)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.04f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "poiseuille_flow_2d.jl"),
                                                             use_wcsph=true,
                                                             coordinates_eltype=Float32,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/poiseuille_flow_2d.jl - BoundaryModelDynamicalPressureZhang (EDAC)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.04f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "poiseuille_flow_2d.jl"),
                                                             use_wcsph=false,
                                                             coordinates_eltype=Float32,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        # Test open boundaries and steady-state callback
        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelCharacteristicsLastiwka (WCSPH)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             wcsph=true,
                                                             coordinates_eltype=Float32,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelCharacteristicsLastiwka (EDAC)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             coordinates_eltype=Float32,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelMirroringTafuni (EDAC)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             coordinates_eltype=Float32,
                                                             open_boundary_model=BoundaryModelMirroringTafuni(),
                                                             boundary_type_in=BidirectionalFlow(),
                                                             boundary_type_out=BidirectionalFlow(),
                                                             reference_density_in=nothing,
                                                             reference_pressure_in=nothing,
                                                             reference_density_out=nothing,
                                                             reference_velocity_out=nothing,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelMirroringTafuni (WCSPH)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             wcsph=true, sound_speed=20.0f0,
                                                             coordinates_eltype=Float32,
                                                             open_boundary_model=BoundaryModelMirroringTafuni(;
                                                                                                              mirror_method=ZerothOrderMirroring()),
                                                             boundary_type_in=BidirectionalFlow(),
                                                             boundary_type_out=BidirectionalFlow(),
                                                             reference_density_in=nothing,
                                                             reference_pressure_in=nothing,
                                                             reference_density_out=nothing,
                                                             reference_pressure_out=nothing,
                                                             reference_velocity_out=nothing,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/pipe_flow_2d.jl - steady state reached (`dt`)" begin
            steady_state_reached = SteadyStateReachedCallback(; dt=0.002f0, interval_size=5,
                                                              reltol=1.0f-3)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             coordinates_eltype=Float32,
                                                             open_boundary_model=BoundaryModelCharacteristicsLastiwka(),
                                                             extra_callback=steady_state_reached,
                                                             tspan=(0.0f0, 1.5f0),
                                                             parallelization_backend=Main.parallelization_backend,
                                                             viscosity_boundary=nothing) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]

            # Make sure that the simulation is terminated after a reasonable amount of time
            @test 0.1 < sol.t[end] < 1.0
            @test sol.retcode == ReturnCode.Terminated
        end
    end

    @testset verbose=true "Structure" begin
        @trixi_testset "structure/oscillating_beam_2d.jl" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "structure",
                                                                      "oscillating_beam_2d.jl"),
                                                             coordinates_eltype=Float32,
                                                             tspan=(0.0f0, 0.1f0),
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n",
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "structure/oscillating_beam_2d.jl with PostprocessCallback" begin
            pp = PostprocessCallback(; interval=5, total_mass,
                                     write_file_interval=0)
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "structure",
                                                                      "oscillating_beam_2d.jl"),
                                                             coordinates_eltype=Float32,
                                                             tspan=(0.0f0, 0.1f0),
                                                             saving_callback=pp,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n",
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
            # Check that the postprocess callback was called and computed values
            @test !isempty(pp.affect!.func)
            @test !isempty(pp.affect!.times)
        end
    end

    @testset verbose=true "FSI" begin
        @trixi_testset "fsi/dam_break_gate_2d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fsi",
                                                   "dam_break_gate_2d.jl"),
                                          coordinates_eltype=Float32,
                                          sol=nothing, ode=nothing)

            # Neighborhood search with `FullGridCellList` for GPU compatibility
            min_corner = minimum(tank.boundary.coordinates, dims=2)
            max_corner = maximum(tank.boundary.coordinates, dims=2)
            max_corner[2] = gate_height + movement_function([0, 0], 0.1f0)[2]
            # We need a very high `max_points_per_cell` because the plate resolution
            # is much finer than the fluid resolution.
            cell_list = FullGridCellList(; min_corner, max_corner)
            semi_fullgrid = Semidiscretization(fluid_system, boundary_system_tank,
                                               boundary_system_gate, structure_system,
                                               neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                             cell_list),
                                               parallelization_backend=Main.parallelization_backend)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(), "fsi",
                                                                      "dam_break_gate_2d.jl"),
                                                             coordinates_eltype=Float32,
                                                             tspan=(0.0f0, 0.4f0),
                                                             semi=semi_fullgrid,
                                                             # Needs <1500 steps on the CPU
                                                             maxiters=1500) [
                r"\[ Info: To create the self-interaction neighborhood search.*\n",
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end
    end

    @testset verbose=true "DEM" begin
        @trixi_testset "dem/rectangular_tank_2d.jl" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(), "dem",
                                                                      "rectangular_tank_2d.jl"),
                                                             coordinates_eltype=Float32,
                                                             ode=nothing, sol=nothing)
            # Neighborhood search with `FullGridCellList` for GPU compatibility
            min_corner = minimum(tank.boundary.coordinates, dims=2)
            max_corner = maximum(tank.boundary.coordinates, dims=2)
            cell_list = FullGridCellList(; min_corner, max_corner)
            neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                            update_strategy=ParallelUpdate())

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(), "dem",
                                                                      "rectangular_tank_2d.jl"),
                                                             tspan=(0.0f0, 0.05f0),
                                                             coordinates_eltype=Float32,
                                                             neighborhood_search=neighborhood_search,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"\[ Info: To move data to the GPU, `semidiscretize` creates a deep copy.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end
    end

    @testset verbose=true "Postprocessing $TRIXIPARTICLES_TEST_" begin
        @testset verbose=true "Interpolation" begin
            # Run the dam break example to get a solution
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "dam_break_2d_gpu.jl");
                                          fluid_particle_spacing=0.05f0,
                                          coordinates_eltype=Float32,
                                          tspan=(0.0f0, 0.01f0),
                                          parallelization_backend=Main.parallelization_backend)

            semi_new = sol.prob.p

            @testset verbose=true "Line" begin
                # Interpolation parameters
                n_interpolation_points = 10
                start_point = Float32[0.5, 0.0]
                end_point = Float32[0.5, 0.5]

                result = interpolate_line(start_point, end_point, n_interpolation_points,
                                          semi_new, semi_new.systems[1], sol;
                                          cut_off_bnd=false)

                @test isapprox(Array(result.computed_density),
                               Float32[500.96387, 859.06744, 989.0479, 1001.3735,
                                       1001.30927, 1001.0831, 1000.7325, 1000.3575,
                                       999.8011, 975.98553])

                @test isapprox(Array(result.density),
                               Float32[1002.1891, 1002.0748, 1001.8913, 1001.67126,
                                       1001.4529, 1001.2382, 1001.0298, 1000.81964,
                                       1000.594, 1000.3878])

                @test isapprox(Array(result.pressure),
                               Float32[5154.1177, 4885.1, 4452.6533, 3934.9075, 3420.5737,
                                       2915.0933, 2424.0908, 1929.7888, 1398.8309,
                                       913.2089])
            end

            @testset verbose=true "Plane" begin
                interpolation_start = Float32[0.0, 0.0]
                interpolation_end = Float32[1.0, 1.0]
                resolution = 0.4f0

                result = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi_new, semi_new.systems[1],
                                              sol; cut_off_bnd=false)

                @test isapprox(Array(result.computed_density),
                               Float32[250.47523, 500.98248, 500.49924, 253.77109,
                                       500.0491, 1000.0818, 999.6527, 503.08704])

                @test isapprox(Array(result.density),
                               Float32[1002.2373, 1002.22516, 1001.4339, 999.3567,
                                       1000.8318, 1000.80237, 1000.3408, 999.92017])

                @test isapprox(Array(result.pressure),
                               Float32[5267.867, 5239.2466, 3376.045, -1514.4237,
                                       1958.2637, 1888.739, 802.3146, -187.81871])
            end
        end
    end
end
