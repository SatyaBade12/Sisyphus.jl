# CZ gate in Rydberg atoms

```math
H = \frac{\Omega(t)}{2}\left[e^{i\phi(t)} (\sigma_+^1 + \sigma_+^2) +  e^{-i\phi(t)} (\sigma_-^1+\sigma_-^2)\right] - \Delta(t) (n_1 + n_2) + V n_1n_2
```


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
n2 = id⊗transition(bs, 3, 3)
```



```julia
H0 = V*(n1*n2)
H1 = (n1 + n2)
H2 = (sp1 + sp2 + sm1 + sm2)
H3 = 1.0im*(sp1 + sp2 - sm1 - sm2)
```



```julia
states= [nlevelstate(bs, 1)⊗nlevelstate(bs, 1),
         nlevelstate(bs, 1)⊗nlevelstate(bs, 2),
         nlevelstate(bs, 2)⊗nlevelstate(bs, 1),
         nlevelstate(bs, 2)⊗nlevelstate(bs, 2)]

trans = UnitaryTransform(states, [[1.0 0 0 0 ];[0 1.0 0 0 ];[0 0 1.0 0 ]; [0 0 0 -1.0]])
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
θ = 2.0*rand(n_params).-1.0
```



```julia
t0, t1 = 0.0, 1.0
sigmoid(x)= @. 2π*7 / (1 + exp(-x))
coeffs(p, t) = let vals = ann([t], p)
                [-vals[1], sigmoid(vals[2])*cos(vals[3]), sigmoid(vals[2])*sin(vals[3])]
               end    
cost = CostFunction((x, y) -> 1.0-abs2(x'*y),
                     p->2e-2*(sigmoid(ann([t0], p)[2])^2 + sigmoid(ann([t1], p)[2])^2))

H = Hamiltonian(H0, [H1, H2, H3], coeffs)

prob = QOCProblem(H, trans, (t0, t1), cost)
```



```julia
sol = solve(prob, θ, ADAM(0.008); maxiter=300, abstol=1e-6, reltol=1e-6)
```


    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:04:48[39m
    [34m  distance:     0.007840998244539127[39m
    [34m  constraints:  4.51625262386786[39m





    Solution{Float64}([0.533944835939007, 0.5411706792864116, -0.39370950659263765, 1.1929489876730237, 0.3279335944500537, -0.09391470525329942, -0.00912834687351155, -0.49001287458304843, -0.051594605020617856, -0.6963408978926312  …  1.0250207113666499, 0.41531162673820005, -0.47515090335705434, 1.17813203902096, 0.4099320431840333, -1.0112835820959676, -0.6481359540952559, 1.9574367990159915, -0.49641587491443545, -0.893962279730682], [0.26843114751889097, 0.49414834803686414, 0.5177275011608675, 0.23214670807974766, 0.2673179854334751, 0.6068319927033263, 0.4451347291200315, 0.05643794881100184, 0.40417221837143175, 0.7223618592292169  …  0.008997434128371617, 0.008859576035184336, 0.008724157260282472, 0.008591133923057687, 0.00846046359845029, 0.008332105095932962, 0.008206018260838166, 0.008082163799339626, 0.007960503127558666, 0.007840998244539127], [1532.663776850894, 1449.9969449204282, 1369.1856626391261, 1290.6835019832688, 1214.9441038117075, 1142.3428142220687, 1073.1052796905883, 1007.3289271326232, 945.0895359987063, 886.3894205760935  …  4.743213478891374, 4.717080798116378, 4.691183586715802, 4.665518874204947, 4.640083737660769, 4.614875300945895, 4.589890733934995, 4.565127251743332, 4.5405821139574, 4.51625262386786], Vector{Float64}[])




```julia
cost.constraints = p->(sigmoid(ann([t0], p)[2])^2 + sigmoid(ann([t1], p)[2])^2)
sol1 = solve(prob, sol.params, ADAM(0.008); maxiter=800, abstol=1e-6, reltol=1e-6)
```


    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:16:18[39m
    [34m  distance:     1.2775702440404046e-9[39m
    [34m  constraints:  0.007536927070112456[39m





    Solution{Float64}([0.820542049211993, 0.746318514947916, -0.5985610572275467, 1.020568975409464, 0.5253852049216425, 0.35923985215869386, -0.25455368865782796, -0.37765411716478325, -0.27051985941948453, -0.4068590910858812  …  1.2065389569392322, 0.6336320784910652, -1.0885383304133631, 1.3349253376618715, 0.6280623827150965, -1.6218340523657984, -0.8477871377653223, 2.1753809583303747, -1.101625062702057, -0.8939622795967975], [0.007723611628364602, 0.006180860180573566, 0.005221428847282827, 0.004608665145991453, 0.004180498340466604, 0.0038356018539024705, 0.0035173007609049234, 0.0031991701459576916, 0.002873584718346811, 0.0025433399241110433  …  1.3086068617607083e-9, 1.3050966973793265e-9, 1.3016022426537432e-9, 1.2981233588060803e-9, 1.2946599348140353e-9, 1.291211471077247e-9, 1.2877788280185598e-9, 1.2843608954149488e-9, 1.2809582006223508e-9, 1.2775702440404046e-9], [4.492136127705154, 4.069479412319481, 3.693297623871543, 3.358422251377977, 3.060119804585499, 2.7941551206588855, 2.5567815435367622, 2.3446896288022403, 2.1549558037188774, 1.9849989854233332  …  0.007673146680696182, 0.007657823235919344, 0.007642547244755294, 0.00762731850390435, 0.007612136811181586, 0.007597001965509269, 0.007581913766909584, 0.0075668720164972295, 0.007551876516472185, 0.007536927070112456], Vector{Float64}[])




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
                                    H, sol1.params)
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




