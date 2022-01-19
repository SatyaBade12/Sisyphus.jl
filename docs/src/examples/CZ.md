```julia
using Revise
using QuantumOptimalControl
using QuantumOptics
using LinearAlgebra
using Flux, DiffEqFlux
using Optim
using NLopt
using PlotlyJS
using DifferentialEquations: Tsit5, DP5
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



$$H = \frac{\Omega(t)}{2}\left[e^{i\phi(t)} (\sigma_+^1 + \sigma_+^2) +  e^{-i\phi(t)} (\sigma_-^1+\sigma_-^2)\right] - \Delta(t) (n_1 + n_2) + V n_1n_2$$


```julia
V = 2π*10.0 # MHz
```




    62.83185307179586




```julia
bs = NLevelBasis(3)
bsys = bs⊗bs

id = identityoperator(bs)

sp1 = transition(bs, 3, 2)⊗id
sm1 = transition(bs, 2, 3)⊗id

sp2 = id⊗transition(bs, 3, 2)
sm2 = id⊗transition(bs, 2, 3)

n1 = transition(bs, 3, 3)⊗id
n2 = id⊗transition(bs, 3, 3);
```


```julia
H0 = V*(n1*n2)
H1 = (n1 + n2)
H2 = (sp1 + sp2 + sm1 + sm2)
H3 = 1.0im*(sp1 + sp2 - sm1 - sm2);
```


```julia
states= [nlevelstate(bs, 1)⊗nlevelstate(bs, 1),
         nlevelstate(bs, 1)⊗nlevelstate(bs, 2),
         nlevelstate(bs, 2)⊗nlevelstate(bs, 1),
         nlevelstate(bs, 2)⊗nlevelstate(bs, 2)]

trans = UnitaryTransform(states, [[1.0 0 0 0 ];[0 1.0 0 0 ];[0 0 1.0 0 ]; [0 0 0 -1.0]]);
```


```julia
n_neurons = 4
ann = FastChain(FastDense(1, n_neurons, tanh), 
                FastDense(n_neurons, n_neurons, tanh), 
                FastDense(n_neurons, n_neurons, tanh), 
                FastDense(n_neurons, 3))
θ = initial_params(ann)     
n_params = length(θ);
Random.seed!(3)
θ = 2.0*rand(n_params).-1.0;
```


```julia
t0, t1 = 0.0, 1.0
sigmoid(x)= @. 2π*7 / (1 + exp(-x))
coeffs(p, t) = let vals = ann([t], p)
                [-vals[1], sigmoid(vals[2])*cos(vals[3]), sigmoid(vals[2])*sin(vals[3])]
               end    
cost = CostFunction((x, y) -> 1.0-real(x'*y),
                     p->2e-2*(sigmoid(ann([t0], p)[2])^2 + sigmoid(ann([t1], p)[2])^2))

H = Hamiltonian(H0, [H1, H2, H3], coeffs)

prob = QOCProblem(H, trans, (t0, t1), cost);
```


```julia
sol = solve(prob, θ, ADAM(0.008); maxiter=2000, abstol=1e-6, reltol=1e-6)
```

    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:14:51[39m
    [34m  distance:    0.0018230279054594067[39m
    [34m  contraints:  0.015043878769374174[39m





    Solution{Float64}([0.8400765737057596, 2.326560388167186, -0.5066669608627995, 1.3279794718994404, 0.586286381587044, -0.00947443251475124, -0.05941918330824658, -0.8357759347813548, -0.205519888534343, -0.7866092471664837  …  0.5845044566908776, -2.874874308004302, -0.6554271832666689, 2.0128445657243086, -1.365934692229344, -1.411928359072274, -0.03514932366826655, -0.5211438023008195, -0.6130441755731763, -0.8939622747325265], [1.1873768263948175, 0.9397324133434627, 0.5490865070101558, 0.17497533265962684, 0.18161259254497908, 0.6207725886058282, 1.066207191786937, 1.1775586608067303, 0.9737677788494261, 0.706244229420213  …  0.001905009136639657, 0.0018938744162585175, 0.0018865705852788572, 0.0018757540507559067, 0.0018682773425667265, 0.0018579324354898452, 0.0018501944930313374, 0.0018403659966037544, 0.0018323651983414013, 0.0018230279054594067])




```julia
cost.constraints = p->(sigmoid(ann([t0], p)[2])^2 + sigmoid(ann([t1], p)[2])^2)
sol1 = solve(prob, sol.params, ADAM(0.008); maxiter=800, abstol=1e-6, reltol=1e-6)
```

    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:05:49[39m
    [34m  distance:    0.003318184688024556[39m
    [34m  contraints:  0.006467681179084656[39m





    Solution{Float64}([0.814908444477573, 2.5266031035921626, -0.5134671050045037, 1.352496438884219, 0.5909015335809052, -0.057306754844247174, -0.0722513911953597, -0.8686619250243353, -0.22060515056775512, -0.7932756712870759  …  0.46930002932817416, -3.3197080347614194, -1.1590946672420352, 2.5434072698861128, -0.8388992487806604, -2.247414612331685, -0.540257418457239, -0.9352701525100026, -0.8597463031671774, -0.8939622734439081], [0.001814810576577286, 0.9302098350526161, 0.2505585189633075, 0.2333318154483398, 0.2507915330313764, 0.026092187117016413, 0.17802370865406444, 0.3128915950395806, 0.22111635618474423, 0.0700818542425225  …  0.003362470530935885, 0.0033575169953351525, 0.0033526064001638822, 0.003347657025287498, 0.003342765110144419, 0.00333781615153661, 0.003332947343398257, 0.00332799282052057, 0.003323154551085644, 0.003318184688024556])




```julia
Ω(t) = sigmoid(ann([t], sol1.params)[2])/2π
Δ(t) = -ann([t], sol1.params)[1]/2π
ϕ(t) = ann([t], sol1.params)[3]
```




    ϕ (generic function with 1 method)




```julia
ts = collect(t0:t1/100:t1)
f = plot(
    [
        scatter(x=ts, y=Ω.(ts), name="Ω/2π"),
        scatter(x=ts, y=Δ.(ts), name="Δ/2π"),
        scatter(x=ts, y=ϕ.(ts), name="ϕ (rad)", yaxis="y2")
    ],
    Layout(
        xaxis_title_text="Time (µs)",
        yaxis_title_text="Frequency (MHz)",
        yaxis2=attr(
            title="Radians",
            overlaying="y",
            side="right"
        ),
        legend=attr(x=0, y=1,),
        font=attr(
            size=16,
        )
    )
)
savefig(f, "cz_wfs.eps")
```




    "cz_wfs.eps"




```julia
tout, psit22 = schroedinger_dynamic(ts, nlevelstate(bs, 2)⊗nlevelstate(bs, 2),
                                    H, sol1.params)
tout, psit21 = schroedinger_dynamic(ts, nlevelstate(bs, 2)⊗nlevelstate(bs, 1),
                                    H, sol1.params)
tout, psit12 = schroedinger_dynamic(ts, nlevelstate(bs, 1)⊗nlevelstate(bs, 2),
                                    H, sol1.params)
tout, psit11 = schroedinger_dynamic(ts, nlevelstate(bs, 1)⊗nlevelstate(bs, 1),
                                    H, sol1.params);
```


```julia
f = plot([
     scatter(x=tout, y=[real((nlevelstate(bs, 2)⊗nlevelstate(bs, 2))'*elm) for elm in psit22], name="|22⟩"),
     scatter(x=tout, y=[real((nlevelstate(bs, 2)⊗nlevelstate(bs, 1))'*elm) for elm in psit21], name="|21⟩"),
     scatter(x=tout, y=[real((nlevelstate(bs, 1)⊗nlevelstate(bs, 2))'*elm) for elm in psit12], name="|12⟩"),
     scatter(x=tout, y=[real((nlevelstate(bs, 1)⊗nlevelstate(bs, 1))'*elm) for elm in psit11], name="|11⟩")        
    ],
    Layout(
        xaxis_title_text="Time (µs)",
        yaxis_title_text="Overlap (⟨ij|ψ⟩)",
        legend=attr(x=0.75, y=0.5,),
        font=attr(
            size=16
        )
    )
)
savefig(f, "cz_overlap.eps")
```




    "cz_overlap.eps"




```julia

```
