```julia
using Revise
using QuantumOptimalControl
using QuantumOptics
using Flux, DiffEqFlux
using PlotlyJS
using DifferentialEquations: DP5, Tsit5, Vern7, Vern9, BS3
using Random
using ProgressMeter
ProgressMeter.ijulia_behavior(:clear)
```


<div style="padding: 1em; background-color: #f8d6da; border: 1px solid #f5c6cb; font-weight: bold;">
<p>The WebIO Jupyter extension was not detected. See the
<a href="https://juliagizmos.github.io/WebIO.jl/latest/providers/ijulia/" target="_blank">
    WebIO Jupyter integration documentation
</a>
for more information.
</div>






    false




```julia
bs = SpinBasis(1//2)
n_neurons = 8
Random.seed!(1)
ann = FastChain(FastDense(1, n_neurons, tanh), 
                FastDense(n_neurons, n_neurons, tanh),
                FastDense(n_neurons, 1))
θ = Vector{Float64}(initial_params(ann));
n_params = length(θ)
println(n_params)
(t0, t1)=(0.0, 5.0)
```

    97





    (0.0, 5.0)




```julia
bcs(params) = (ann([t0], params)[1])^2 + (ann([t1], params)[1])^2
cost = CostFunction((x,y)-> 1.0-real(x'*y), bcs)
```




    CostFunction(var"#1#2"(), bcs)




```julia
trans = StateTransform(spindown(bs)=>spinup(bs));
```


```julia
coeffs(params, t) = ann([t], params)[1]
tspan = (t0, t1)
H = Hamiltonian(-0.5*sigmaz(bs), [sigmax(bs)], coeffs)
prob = QOCProblem(H, trans, tspan, cost);
```


```julia
@time sol = solve(prob, θ, ADAM(0.08); maxiter=150, abstol=1e-6, reltol=1e-6)
```

    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:01:14[39m
    [34m  distance:    4.953560384102218e-7[39m
    [34m  contraints:  7.116818599940487e-7[39m


    104.283913 seconds (177.93 M allocations: 11.730 GiB, 45.32% gc time, 54.27% compilation time)





    Solution{Float64}([-0.284962152604464, -1.5331911991894618, -1.010023405933112, -0.7365123309741989, -2.5746801648492563, 0.718711589207783, -0.6645587230077273, 0.29561952779882594, 0.614716851881567, 0.18764787090703344  …  -0.11306379618389058, 0.2024893664028441, -0.32691917929989966, 0.45269739829537853, -0.04671211174804547, 0.8082743482458395, -0.5341851468230246, 0.4528547961728511, -0.11834131830299666, 0.0924095845130223], [1.1352716384730854, 1.0255668890862075, 1.0084751278774744, 1.335280825213213, 0.9710824920312312, 0.9004424192532431, 0.7594278767057295, 0.5489853196004976, 0.49894000503020064, 0.5671001448371097  …  4.257042166333491e-6, 3.71450552760777e-6, 3.931315272098779e-6, 3.4457824356071143e-6, 2.3531649397945387e-6, 1.8198556790416376e-6, 1.5427803536915974e-6, 9.538285731247598e-7, 1.35814505686227e-6, 4.953560384102218e-7])




```julia
ts = t0:t1/100:t1
Ω(t) = coeffs(sol.params, t)[1]
f = plot([scatter(x=ts, y=Ω.(ts), name="Ω")],
     Layout(
        xaxis_title_text="Time (a.u)",
        yaxis_title_text="Ω (a.u)",
        legend=attr(x=0, y=1,),
        font=attr(
            size=16
        )
    )
)
savefig(f, "twolevelsys_wfs.eps")
```




    "twolevelsys_wfs.eps"




```julia
tout, psit = schroedinger_dynamic(ts, spindown(bs), H, sol.params);
```


```julia
f = plot(
    [
        scatter(x=tout, y=real(expect(dm(spindown(bs)), psit)), name="|↓⟩")
        scatter(x=tout, y=real(expect(dm(spinup(bs)), psit)), name="|↑⟩")
    ],
    Layout(
        xaxis_title_text="Time (a.u)",
        yaxis_title_text="Population (⟨i|ψ⟩²)",
        legend=attr(x=0.75, y=0.5),
        font=attr(
            size=16
        )
     )
)
savefig(f, "twolevelsys_overlap.eps")
```




    "twolevelsys_overlap.eps"




```julia

```
