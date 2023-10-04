@testset verbose=true "InfoCallback" begin
    @testset verbose=true "show" begin
        callback = InfoCallback(interval=10)

        show_compact = "InfoCallback(interval=10)"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ InfoCallback                                                                                     │
        │ ════════════                                                                                     │
        │ interval: ……………………………………………………… 10                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box
    end

    @testset verbose=true "initialize" begin
        callback = InfoCallback()

        # Build a mock `integrator`, which is a `NamedTuple` holding the fields that are
        # accessed in `initialize_info_callback`.
        continuous_callbacks = (:cb1, :cb2)
        discrete_callbacks = (callback, (; (affect!)=:cb3))

        semi = (; systems=(:system1, :system2))

        integrator = (; p=semi,
                      opts=(;
                            callback=(; continuous_callbacks, discrete_callbacks),
                            adaptive=true, abstol=1e-2, reltol=1e-1,
                            controller=:controller),
                      alg=Val(:alg),
                      sol=(; prob=(; tspan=(0.1, 0.5))))

        expected = """

        ████████╗██████╗ ██╗██╗  ██╗██╗██████╗  █████╗ ██████╗ ████████╗██╗ ██████╗██╗     ███████╗███████╗
        ╚══██╔══╝██╔══██╗██║╚██╗██╔╝██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝██║     ██╔════╝██╔════╝
           ██║   ██████╔╝██║ ╚███╔╝ ██║██████╔╝███████║██████╔╝   ██║   ██║██║     ██║     █████╗  ███████╗
           ██║   ██╔══██╗██║ ██╔██╗ ██║██╔═══╝ ██╔══██║██╔══██╗   ██║   ██║██║     ██║     ██╔══╝  ╚════██║
           ██║   ██║  ██║██║██╔╝ ██╗██║██║     ██║  ██║██║  ██║   ██║   ██║╚██████╗███████╗███████╗███████║
           ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝╚══════╝╚══════╝


        (systems = (:system1, :system2),)

        :system1

        :system2

        :cb1

        :cb2

        (affect! = :cb3,)

        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Time integration                                                                                 │
        │ ════════════════                                                                                 │
        │ Start time: ………………………………………………… 0.1                                                              │
        │ Final time: ………………………………………………… 0.5                                                              │
        │ time integrator: …………………………………… Val                                                              │
        │ adaptive: ……………………………………………………… true                                                             │
        │ abstol: …………………………………………………………… 0.01                                                             │
        │ reltol: …………………………………………………………… 0.1                                                              │
        │ controller: ………………………………………………… controller                                                       │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Environment information                                                                          │
        │ ═══════════════════════                                                                          │
        │ #threads: ……………………………………………………… $(@sprintf("%-40d", Threads.nthreads()))                         │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘

        """

        # Redirect `stdout` to a string
        pipe = Pipe()
        redirect_stdout(pipe) do
            TrixiParticles.initialize_info_callback(callback, nothing, nothing, integrator)
        end
        close(pipe.in)
        output = String(read(pipe))

        @test output == expected
    end

    @testset verbose=true "affect! not finished" begin
        callback = InfoCallback()

        # Set `start_time` to -1e109 to make the output independent of the current time
        callback.affect!.start_time = -1e109

        # Build a mock `integrator`, which is a `NamedTuple` holding the fields that are
        # accessed in `initialize_info_callback`.
        integrator = (; t=23.0,
                      stats=(; naccept=453),
                      iter=472,
                      dt=1e-3)

        TrixiParticles.isfinished(::NamedTuple) = false
        TrixiParticles.u_modified!(::NamedTuple, _) = nothing

        expected = "#timesteps:    453 │ Δt: 1.0000e-03 │ sim. time: 2.3000e+01 │ run time: 1.0000e+100 s\n"

        # Redirect `stdout` to a string
        pipe = Pipe()
        redirect_stdout(pipe) do
            callback.affect!(integrator)
        end
        close(pipe.in)
        output = String(read(pipe))

        @test output == expected
    end

    @testset verbose=true "affect! finished" begin
        callback = InfoCallback()

        # Set `start_time` to -1e109 to make the output independent of the current time
        callback.affect!.start_time = -1e109

        # Build a mock `integrator`, which is a `NamedTuple` holding the fields that are
        # accessed in `initialize_info_callback`.
        integrator = (; t=23.0,
                      stats=(; naccept=453),
                      iter=472,
                      dt=1e-3)

        TrixiParticles.isfinished(::NamedTuple) = true
        TrixiParticles.u_modified!(::NamedTuple, _) = nothing

        expected = """
        ────────────────────────────────────────────────────────────────────────────────────────────────────
        Trixi simulation finished.  Final time: 23.0  Time steps: 453 (accepted), 472 (total)
        ────────────────────────────────────────────────────────────────────────────────────────────────────

         ────────────────────────────────────────────────────────────────────
         TrixiParticles.jl          Time                    Allocations"""

        # Redirect `stdout` to a string
        pipe = Pipe()
        redirect_stdout(pipe) do
            callback.affect!(integrator)
        end
        close(pipe.in)
        output = String(read(pipe))

        @test startswith(output, expected)
    end
end
