using Pixie
using OrdinaryDiffEq

function f(du,u,p,t)
    du[1] = -u[1]
end
u0 = [10.0]
const V = 1
alive_callback = AliveCallback(alive_interval=1)
dt_callback    = StepSizeCallback(callback_interval=1)
prob = ODEProblem(f,u0,(0.0,10.0))
sol = solve(prob,Tsit5(), callback=alive_callback);


