# [Smoothing Kernels](@id smoothing_kernel)
The following smoothing kernels are currently available:

| Smoothing Kernel                          | Compact Support   | Typ. Smoothing Length | Recommended Application | Stability |
| :---------------------------------------- | :---------------- | :-------------------- | :---------------------- | :-------- |
| [`SchoenbergCubicSplineKernel`](@ref)     | $[0, 2h]$         | $1.1$ to $1.3$        | General + sharp waves   | ++        |
| [`SchoenbergQuarticSplineKernel`](@ref)   | $[0, 2.5h]$       | $1.1$ to $1.5$        | General                 | +++       |
| [`SchoenbergQuinticSplineKernel`](@ref)   | $[0, 3h]$         | $1.1$ to $1.5$        | General                 | +++       |
| [`GaussianKernel`](@ref)                  | $[0, 3h]$         | $1.0$ to $1.5$        | Academic                | +++       |
| [`WendlandC2Kernel`](@ref)                | $[0, 2h]$         | $1.2$ to $2.0$        | General (recommended)   | ++++      |
| [`WendlandC4Kernel`](@ref)                | $[0, 2h]$         | $1.5$ to $2.3$        | General                 | +++++     |
| [`WendlandC6Kernel`](@ref)                | $[0, 2h]$         | $1.7$ to $2.5$        | General                 | +++++     |
| [`Poly6Kernel`](@ref)                     | $[0, 1h]$         | $1.5$ to $2.5$        | Academic                | +         |
| [`SpikyKernel`](@ref)                     | $[0, 1h]$         | $1.5$ to $3.0$        | Academic                | +         |
| [`LaguerreGaussKernel`](@ref)             | $[0, 2h]$         | $1.3$ to $1.5$        | General                 | ++++      |

Any Kernel with a stability rating of more than '+++' doesn't suffer from pairing-instability.

We recommend to use the [`WendlandC2Kernel`](@ref) for most applications.
If less smoothing is needed, try [`SchoenbergCubicSplineKernel`](@ref), for more smoothing try [`WendlandC6Kernel`](@ref).

```@eval

using TrixiParticles                     
using CairoMakie                         

# --- Group the kernels for combined plotting ---  
wendland_kernels = [                     
    ("Wendland C2", WendlandC2Kernel{2}()),  
    ("Wendland C4", WendlandC4Kernel{2}()),  
    ("Wendland C6", WendlandC6Kernel{2}()),  
]                                        

schoenberg_kernels = [                   
    ("Cubic Spline",   SchoenbergCubicSplineKernel{2}()),   
    ("Quartic Spline", SchoenbergQuarticSplineKernel{2}()), 
    ("Quintic Spline", SchoenbergQuinticSplineKernel{2}()), 
]                                        

other_kernels = [                        
    ("Gaussian",       GaussianKernel{2}()),       
    ("Poly6",          Poly6Kernel{2}()),          
    ("Laguerre-Gauss", LaguerreGaussKernel{2}()), 
]                                        

spiky_kernel_group = [                   
    ("Spiky Kernel", SpikyKernel{2}()),  
]                                        

# A list of all kernel groups to be plotted                  
# A boolean flag controls whether to apply the consistent y-range  
kernel_groups = [                                            
    (title="Wendland Kernels",          kernels=wendland_kernels,   use_consistent_range=true),  
    (title="Schoenberg Spline Kernels", kernels=schoenberg_kernels, use_consistent_range=true),  
    (title="Other Kernels",             kernels=other_kernels,      use_consistent_range=true),  
    (title="Spiky Kernel",              kernels=spiky_kernel_group, use_consistent_range=false), 
]                                                            

# --- Pre-calculate global y-ranges for consistency ---      
kernels_for_range_calc = vcat(wendland_kernels, schoenberg_kernels, other_kernels)  

q_range = range(0, 3, length=300)     
h = 1.0                               
min_val, max_val = Inf, -Inf          
min_deriv, max_deriv = Inf, -Inf      

for (_, kernel_obj) in kernels_for_range_calc                             
    kernel_values = [TrixiParticles.kernel(kernel_obj, q, h) for q in q_range]      
    kernel_derivs = [TrixiParticles.kernel_deriv(kernel_obj, q, h) for q in q_range]

    global min_val = min(min_val, minimum(kernel_values))     
    global max_val = max(max_val, maximum(kernel_values))     
    global min_deriv = min(min_deriv, minimum(kernel_derivs)) 
    global max_deriv = max(max_deriv, maximum(kernel_derivs)) 
end                                                                      

# Add 10% padding to the y-limits for better visuals          
y_range_val = (min_val - 0.1 * (max_val - min_val),           
               max_val + 0.1 * (max_val - min_val))           
y_range_deriv = (min_deriv - 0.1 * (max_deriv - min_deriv),   
                 max_deriv + 0.1 * (max_deriv - min_deriv))   

fig = Figure(size = (1000, 1200), fontsize=16)                

for (i, group) in enumerate(kernel_groups)                    
    ax_val = Axis(fig[i, 1],                                  
                  xlabel = "q = r/h", ylabel = "w(q)",        
                  title = group.title)                        

    ax_deriv = Axis(fig[i, 2],                                
                    xlabel = "q = r/h", ylabel = "w'(q)",     
                    title = group.title)                      

    if group.use_consistent_range                             
        ylims!(ax_val, y_range_val)                           
        ylims!(ax_deriv, y_range_deriv)                       
    end                                                       

    hlines!(ax_val, [0.0], linestyle = :dash)                 
    hlines!(ax_deriv, [0.0], linestyle = :dash)               

    for (name, kernel_obj) in group.kernels                   
        kernel_values = [TrixiParticles.kernel(kernel_obj, q, h) for q in q_range]    
        kernel_derivs = [TrixiParticles.kernel_deriv(kernel_obj, q, h) for q in q_range]  

        lines!(ax_val, q_range, kernel_values, label=name, linewidth=2.5)   
        lines!(ax_deriv, q_range, kernel_derivs, label=name, linewidth=2.5) 
    end                                                       

    axislegend(ax_val, position = :rt)                        
    axislegend(ax_deriv, position = :rt)                      
end                                                           

# Add row gaps between the 4 rows (3 gaps total)              
for i in 1:(length(kernel_groups) - 1)                        
    rowgap!(fig.layout, i, 25)                                
end                                                           

CairoMakie.save("smoothing_kernels.png", fig)
                                                           
```

![Radial profiles and derivatives of the available smoothing kernels](smoothing_kernels.png)


!!! note "Usage"
    The kernel can be called as
    ```
    TrixiParticles.kernel(smoothing_kernel, r, h)
    ```
    The length of the compact support can be obtained as
    ```
    TrixiParticles.compact_support(smoothing_kernel, h)
    ```

    Note that ``r`` has to be a scalar, so in the context of SPH, the kernel
    should be used as
    ```math
    W(\Vert r_a - r_b \Vert, h).
    ```

    The gradient required in SPH,
    ```math
        \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)
    ```
    can be called as
    ```
    TrixiParticles.kernel_grad(smoothing_kernel, pos_diff, distance, h)
    ```
    where `pos_diff` is $r_a - r_b$ and `distance` is $\Vert r_a - r_b \Vert$.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "smoothing_kernels.jl")]
```
