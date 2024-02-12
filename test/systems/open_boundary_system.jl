@testset verbose=true "OpenBoundarySPHSystem" begin
    @testset verbose=true "Boundary Zone 2D" begin
        boundary_zones = [InFlow(), OutFlow()]
        particle_spacing = 0.2
        open_boundary_layers = 4

        point1s = [[0.0, 0.0], [0.5, -0.5], [1.0, 0.5]]
        point2s = [[0.0, 1.0], [0.2, 2.0], [2.3, 0.5]]

        @testset "$boundary_zone" for boundary_zone in boundary_zones
            @testset "Points $(i)" for i in eachindex(point1s)
                plane_size = point2s[i] - point1s[i]
                flow_directions = [
                    normalize([-plane_size[2], plane_size[1]]),
                    -normalize([-plane_size[2], plane_size[1]]),
                ]

                plane_points = [point1s[i], point2s[i]]

                @testset "Flow Direction $(j)" for j in eachindex(flow_directions)
                    system = OpenBoundarySPHSystem(plane_points, boundary_zone, 1.0;
                                                   flow_direction=flow_directions[j],
                                                   particle_spacing,
                                                   open_boundary_layers,
                                                   density=1.0)
                    zone_width = open_boundary_layers *
                                 system.initial_condition.particle_spacing
                    sign_ = (boundary_zone isa InFlow) ? -1 : 1

                    @test point1s[i] == system.zone_origin
                    @test point2s[i] - system.zone_origin == system.spanning_set[2]
                    @test sign_ * flow_directions[j] ≈ normalize(system.spanning_set[1])
                    @test zone_width ≈ norm(system.spanning_set[1])
                end
            end
        end
    end

    @testset verbose=true "Boundary Zone 3D" begin
        boundary_zones = [InFlow(), OutFlow()]
        particle_spacing = 0.05
        open_boundary_layers = 4

        point1s = [
            [0.0, 0.0, 0.0],
            [0.3113730847835541, 0.19079485535621643, -0.440864622592926],
        ]
        point2s = [
            [1.0, 0.0, 0.0],
            [-0.10468611121177673, 0.252103328704834, -0.44965094327926636],
        ]
        point3s = [
            [0.0, 1.0, 0.0],
            [0.3113730847835541, 0.25057315826416016, -0.02374829351902008],
        ]

        @testset "$boundary_zone" for boundary_zone in boundary_zones
            @testset "Points $(i)" for i in eachindex(point1s)
                edge1 = point2s[i] - point1s[i]
                edge2 = point3s[i] - point1s[i]

                flow_directions = [
                    normalize(cross(edge1, edge2)),
                    -normalize(cross(edge1, edge2)),
                ]

                plane_points = [point1s[i], point2s[i], point3s[i]]

                @testset "Flow Direction $(j)" for j in eachindex(flow_directions)
                    system = OpenBoundarySPHSystem(plane_points, boundary_zone, 1.0;
                                                   flow_direction=flow_directions[j],
                                                   particle_spacing,
                                                   open_boundary_layers,
                                                   density=1.0)
                    zone_width = open_boundary_layers *
                                 system.initial_condition.particle_spacing
                    sign_ = (boundary_zone isa InFlow) ? -1 : 1

                    @test point1s[i] == system.zone_origin
                    @test point2s[i] - system.zone_origin == system.spanning_set[2]
                    @test point3s[i] - system.zone_origin == system.spanning_set[3]
                    @test sign_ * flow_directions[j] ≈ normalize(system.spanning_set[1])
                    @test zone_width ≈ norm(system.spanning_set[1])
                end
            end
        end
    end

    @testset verbose=true "Particle In Boundary Zone 2D" begin
        plane_points = [[-0.2, -0.5], [0.3, 0.5]]
        plane_size = plane_points[2] - plane_points[1]

        flow_direction = normalize([-plane_size[2], plane_size[1]])
        system = OpenBoundarySPHSystem(plane_points, InFlow(), 1.0; flow_direction,
                                       density=1.0,
                                       particle_spacing=0.1, open_boundary_layers=4)

        query_points = Dict(
            "Behind" => ([-1.0, -1.0], false),
            "Before" => ([2.0, 2.0], false),
            "On Point 1" => (plane_points[1], true),
            "On Point 2" => (plane_points[2], true),
            "On Point 3" => (system.spanning_set[1] + system.zone_origin, true),
            "Closely On Point 1" => (plane_points[1] .+ eps() * flow_direction, false))

        @testset "$key" for key in keys(query_points)
            (particle_position, evaluation) = query_points[key]

            @test evaluation ==
                  TrixiParticles.within_boundary_zone(particle_position, system)
        end
    end

    @testset verbose=true "Particle In Boundary Zone 3D" begin
        point1 = [-0.2, -0.5, 0.0]
        point2 = [0.3, 0.5, 0.0]
        point3 = [0.13111173850909402, -0.665555869254547, 0.0]

        flow_direction = normalize(cross(point2 - point1, point3 - point1))
        system = OpenBoundarySPHSystem([point1, point2, point3], InFlow(), 1.0;
                                       flow_direction,
                                       density=1.0, particle_spacing=0.1,
                                       open_boundary_layers=4)

        query_points = Dict(
            "Behind" => ([-1.0, -1.0, 1.2], false),
            "Before" => ([2.0, 2.0, -1.2], false),
            "On Point 1" => (point1, true),
            "On Point 2" => (point2, true),
            "On Point 3" => (point3, true),
            "On Point 4" => (system.spanning_set[1] + system.zone_origin, true),
            "Closely On Point 1" => (point1 .+ eps() * flow_direction, false))

        @testset "$key" for key in keys(query_points)
            (particle_position, evaluation) = query_points[key]

            @test evaluation ==
                  TrixiParticles.within_boundary_zone(particle_position, system)
        end
    end

    @testset verbose=true "Illegal Inputs" begin
        no_rectangular_plane = [[0.2, 0.3, -0.5], [-1.0, 1.5, 0.2], [-0.2, 2.0, -0.5]]
        flow_direction = [0.0, 0.0, 1.0]
        error_str = "the provided points do not span a rectangular plane"
        @test_throws ArgumentError(error_str) system=OpenBoundarySPHSystem(no_rectangular_plane,
                                                                           InFlow(),
                                                                           1.0;
                                                                           flow_direction,
                                                                           particle_spacing=0.1,
                                                                           open_boundary_layers=2,
                                                                           density=1.0)

        rectangular_plane = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        flow_direction = [0.0, 1.0, 0.0]
        error_str = "flow direction and normal vector of " *
                    "InFlow-plane do not correspond"
        @test_throws ArgumentError(error_str) system=OpenBoundarySPHSystem(rectangular_plane,
                                                                           InFlow(),
                                                                           1.0;
                                                                           flow_direction,
                                                                           particle_spacing=0.1,
                                                                           open_boundary_layers=2,
                                                                           density=1.0)
    end
    @testset "show" begin
        system = OpenBoundarySPHSystem(([0.0, 0.0], [0.0, 1.0]), InFlow(), 1.0;
                                       flow_direction=(1.0, 0.0), density=1.0,
                                       particle_spacing=0.05, open_boundary_layers=4)

        show_compact = "OpenBoundarySPHSystem{2}(InFlow()) with 80 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ OpenBoundarySPHSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ #particles: ………………………………………………… 80                                                               │
        │ boundary: ……………………………………………………… InFlow()                                                         │
        │ flow direction: ……………………………………… [1.0, 0.0]                                                       │
        │ width: ……………………………………………………………… 0.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end
