# Overview
The actual API reference is not listed on a single page, like in most Julia packages,
but instead is split into multiple sections that follow a similar structure
as the code files themselves.
In these sections, API docs are combined with explanations of the theoretical background
of these methods.

The following page gives a rough overview of important parts of the code.

## Program flow

To initiate a simulation, the goal is to solve an ordinary differential equation, for example,
by employing the time integration schemes provided by OrdinaryDiffEq.jl. These schemes are then
utilized to integrate ``\mathrm{d}u/\mathrm{d}t`` and ``\mathrm{d}v/\mathrm{d}t``, where ``u``
represents the particles' positions and ``v`` their properties such as velocity and density.
During a single time step or an intermediate step of the time integration scheme, the functions
`drift!` and `kick!` are invoked, followed by the functions depicted in this diagram
(with key parts highlighted in orange/yellow).

```mermaid
%% Make arrows bend at right angles
%%{ init : { "flowchart" : { "curve" : "stepAfter" }}}%%

%% TD means vertical layout
flowchart TD
    %% --- Define color palette and styles ---
    classDef start_node fill:#d9ead3,stroke:#333,color:#333
    classDef time_integration fill:#d9d2e9,stroke:#333,color:#333
    classDef primary_stage fill:#cfe2f3,stroke:#333,color:#333
    classDef update fill:#eeeeee,stroke:#333,color:#333
    classDef updates fill:#fff2cc,stroke:#333,color:#333
    classDef physics fill:#fce5cd,stroke:#333,color:#333

    A(simulation) --> B[time integration];

    %% Add hidden dummy node to branch the arrow nicely
    B --- dummy[ ];
    style dummy width:0;
    dummy --> C["drift!<br/>(update du/dt)"];

    subgraph kick["<div style='padding: 10px; font-weight: bold;'>kick! (update dv/dt)</div>"]
        %% Horizontal layout within this subgraph
        direction LR;

        subgraph updates["<div style='padding: 10px; font-weight: bold;'>update_systems_and_nhs</div>"]
            %% Vertical layout within this subgraph
            direction TB;

            H["update_positions!<br/>(moving boundaries and structures)"];
            I["update_nhs!<br/>(update neighborhood search)"];
            J["update_quantities!<br/>(recalculate density etc.)"];
            K["update_pressure!<br/>(recalculate pressure etc.)"];
            L["update_boundary_interpolation!<br/>(interpolate boundary pressure)"];
            M["update_final!<br/>(update shifting)"];

            H --> I --> J --> K --> L --> M;
        end

        F["system_interaction!<br/>(e.g. momentum/continuity equation)"];
        G["add_source_terms!<br/>(gravity and source terms)"];

        updates --> F --> G;
    end

    dummy --> kick;

    %% Color the sub-tasks by their function
    class A start_node;
    class B time_integration;
    class C primary_stage;
    class kick primary_stage;
    class updates update;
    class H,I,J,K,L,M updates;
    class F,G physics;

    %% Style the arrows
    linkStyle default stroke-width:2px,stroke:#555
```

## Structure
What we refer to as schemes are various models such as Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)
or Total Lagrangian Smoothed Particle Hydrodynamics (TLSPH). These schemes are categorized based on the applicable
physical regimes, namely fluid, solid, gas, and others. Each scheme comprises at least two files: a `system.jl` file
and an `rhs.jl` file. The `system.jl` file provides the data structure holding the particles of this scheme and some
routines, particularly those for allocation and the main update routines, excluding system interactions.
The interactions between particles of this scheme (and with particles of other schemes) are handled in the `rhs.jl` file.
