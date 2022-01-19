# ``R_x(\pi/2)`` gate


```julia
using Revise
using QuantumOptimalControl
using QuantumOptics
using Flux, DiffEqFlux
using Plots
using Random
using ProgressMeter

ProgressMeter.ijulia_behavior(:clear)
```



```julia
ω₀ = 2π*5.0
η = -2π*300*1e-3
ωlo = ω₀
```





    31.41592653589793




```julia
n_levels = 12
bs = FockBasis(n_levels-1)
a = destroy(bs)
ad = create(bs)
id = identityoperator(bs, bs)
```



```julia
H0 = ω₀*(ad*a + 0.5*id) + (η/12.0)*(a + ad)^4 - η^2 * (a + ad)^6/ω₀/90.0
H1 = 1.0im*(a - ad)
```



```julia
n_neurons = 8
Random.seed!(1)
ann = FastChain(FastDense(1, n_neurons, tanh), 
                FastDense(n_neurons, n_neurons, tanh),
                FastDense(n_neurons, 2))
t0, t1 = 0.0, 4.0

I_guess(t) = @. -2π*exp(-(t-0.5*t1)^2/(0.2*t1)^2)*2.0*(t-0.5*t1) /(0.2*t1)^2
Q_guess(t) = @. -2π*exp(-(t-0.5*t1)^2/(0.2*t1)^2)

tsf32 = Float32(t0):0.001f0:Float32(t1)
Is = Vector{Float32}(I_guess(tsf32))
Qs = Vector{Float32}(Q_guess(tsf32))

ts = Vector{Float64}(tsf32)
function loss(p)
    c = 0.0f0
    for (i,t) in enumerate(tsf32)
        x = ann([t], p)
        c += (x[1] - Is[i])^2
        c += (x[2] - Qs[i])^2
    end
    c
end
res = DiffEqFlux.sciml_train(loss, initial_params(ann), ADAM(0.1f0), maxiters = 500)
θ = Vector{Float64}(res.u);
#θ = Vector{Float64}(initial_params(ann))
```



```julia
coeffs(params, t) = let b = ann([t], params)
                        [b[1]*cos(ωlo*t) + b[2]*sin(ωlo*t)]
                    end
bcs(params) = 0.01*sum(ann([t0], params).^2 + ann([t1], params).^2)
                 
cost = CostFunction((x,y)-> 1.0-real(x'*y), bcs)
```





    CostFunction(var"#1#2"(), bcs)




```julia
trans = UnitaryTransform([fockstate(bs, 0), fockstate(bs, 1)], 
                         [[1.0 -1.0im];[-1.0im 1.0]]/√2)
```



```julia
tspan = (t0, t1)
H = Hamiltonian(H0, [H1], coeffs)
prob = QOCProblem(H, trans, tspan, cost)
```



```julia
@time sol = solve(prob, θ, ADAM(0.02); maxiter=200)
```


    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:17:50[39m
    [34m  distance:     0.004004688599166306[39m
    [34m  constraints:  2.2706680276447027e-5[39m


    1125.732417 seconds (1.21 G allocations: 116.836 GiB, 20.12% gc time, 9.77% compilation time)





    Solution{Float64}([-1.1301956130648005, -0.8415386379718703, -0.62416806565457, -0.05558272975090696, -1.2767618984395774, 0.7777968650411404, -0.79181780541562, 0.4821587381161102, 0.9167613402532108, 1.4252101509955573  …  4.420350868594427, 1.0898734100036007, -1.1077557762412569, 0.1255984053155659, 3.1804582778956734, 0.5059299202762897, -1.260625152717333, 1.7327845621113096, -0.019738132516426414, -0.14246364928080688], [1.174223952158369, 0.9283422899342767, 0.5701455006616576, 0.2822686063136131, 0.13979922387509308, 0.09110572511269543, 0.07913269188721217, 0.08607026770409643, 0.10807190410041634, 0.13809842167715414  …  0.004041516808882928, 0.004037207993353886, 0.004032949443823042, 0.004028745970567349, 0.004024603043437325, 0.004020522923158076, 0.004016501321782884, 0.004012528477546129, 0.00400859313659685, 0.004004688599166306], [0.0002050758343644699, 0.00038902886411267733, 0.0007288376296764478, 0.0013953028946812162, 0.002288254011712841, 0.0032109472633268465, 0.0042329005808591285, 0.005474101930394586, 0.0068460365496430834, 0.008025003132358818  …  2.321000676479122e-5, 2.3119855929475862e-5, 2.3044662113294762e-5, 2.297780872184754e-5, 2.291291217720479e-5, 2.2847376410725797e-5, 2.2784889480739305e-5, 2.2734581962830515e-5, 2.2706633412771438e-5, 2.2706680276447027e-5], Vector{Float64}[])




```julia
plot(sol.trace)
```



    type Solution has no field trace

    

    Stacktrace:

     [1] getproperty(x::Solution{Float64}, f::Symbol)

       @ Base ./Base.jl:42

     [2] top-level scope

       @ In[10]:1

     [3] eval

       @ ./boot.jl:373 [inlined]

     [4] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

       @ Base ./loading.jl:1196



```julia
I_wf(t) = ann([t], sol.params)[1]/2π
Q_wf(t) = ann([t], sol.params)[2]/2π
```





    Q_wf (generic function with 1 method)




```julia
f= plot([
    scatter(x=ts, y=I_wf.(ts), name="I")
    scatter(x=ts, y=Q_wf.(ts), name="Q")
    ],
    Layout(
        xaxis_title_text="Time (ns)",
        yaxis_title_text="Frequency (GHz)",
        legend=attr(x=0, y=1,),
        font=attr(
            size=16,
        )
    )
)

savefig(f, "rxpi2_wfs.eps")
```





    "rxpi2_wfs.eps"




```julia
tout, psit = schroedinger_dynamic(ts, fockstate(bs, 0), H, sol.params)
```



```julia
f = plot([
        scatter(x=ts, y=real(expect(dm(fockstate(bs, i-1)), psit)), name=string(i-1)) for i in 1:12 
    ],
    Layout(
        xaxis_title_text="Time (ns)",
        yaxis_title_text="Population (⟨i|ψ⟩²)",
        legend=attr(x=0, y=0.5,),
        font=attr(
            size=16,
        )
    )
)
savefig(f,"rxpi2_probs.eps")
```





    "rxpi2_probs.eps"




