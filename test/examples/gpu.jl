const TRIXIPARTICLES_TEST_ = lowercase(get(ENV, "TRIXIPARTICLES_TEST", "all"))

if TRIXIPARTICLES_TEST_ == "cuda"
    using CUDA
    CUDA.versioninfo()
    data_type = CuArray
    supports_double_precision = true
elseif TRIXIPARTICLES_TEST_ == "amdgpu"
    using AMDGPU
    AMDGPU.versioninfo()
    data_type = ROCArray
    supports_double_precision = true
elseif TRIXIPARTICLES_TEST_ == "metal"
    using Metal
    Metal.versioninfo()
    data_type = MtlArray
    supports_double_precision = false
elseif TRIXIPARTICLES_TEST_ == "oneapi"
    using oneAPI
    oneAPI.versioninfo()
    data_type = oneArray
    supports_double_precision = true
end

@testset verbose=true "Examples $TRIXIPARTICLES_TEST_" begin
    @testset verbose=true "Fluid" begin
        @trixi_testset "fluid/dam_break_2d_gpu.jl Float64" begin
            if Main.supports_double_precision
                @test_nowarn_mod trixi_include(@__MODULE__,
                                               joinpath(examples_dir(), "fluid",
                                                        "dam_break_2d_gpu.jl"),
                                               tspan=(0.0, 0.1),
                                               data_type=Main.data_type) [
                    r"┌ Info: The desired tank length in y-direction .*\n",
                    r"└ New tank length in y-direction.*\n"
                ]
                @test semi.neighborhood_searches[1][1].cell_list isa FullGridCellList
                @test sol.retcode == ReturnCode.Success
                @test sol.u[end].x[1] isa Main.data_type
            else
                error = "Metal does not support Float64 values, try using Float32 instead"
                @test_throws error trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "dam_break_2d_gpu.jl"),
                                                 tspan=(0.0, 0.1),
                                                 data_type=Main.data_type)
            end
        end

        @trixi_testset "fluid/dam_break_2d_gpu.jl Float32" begin
            @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
                                                           joinpath(examples_dir(), "fluid",
                                                                    "dam_break_2d_gpu.jl"),
                                                           tspan=(0.0f0, 0.1f0),
                                                           data_type=Main.data_type) [
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n"
            ]
            @test semi.neighborhood_searches[1][1].cell_list isa FullGridCellList
            @test sol.retcode == ReturnCode.Success
            @test sol.u[end].x[1] isa Main.data_type
        end
    end
end
