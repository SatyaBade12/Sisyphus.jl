```julia
using QuantumOptimalControl
using QuantumOptics
using LinearAlgebra
using Flux, DiffEqFlux
using PlotlyJS
using ProgressMeter
using Random
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
sx = sigmax(bs)
ni = 0.5*(identityoperator(bs) + sigmaz(bs));
```


```julia
V = 2π*24.0
δe = -2π*4.5
```




    -28.274333882308138




```julia
n_atoms = 12
bsys = tensor([bs for i in 1:n_atoms]...)

H0 = V*sum([embed(bsys, [i, j], [ni, ni])/abs(i-j)^6  for i in 1:n_atoms for j in i+1:n_atoms])
H0 -= δe*sum([embed(bsys, [i], [ni]) for i in [1, n_atoms]])
if n_atoms>8
    H0 -= -2π*1.5*sum([embed(bsys, [i], [ni]) for i in [1, n_atoms]])
    H0 -= -2π*1.5*sum([embed(bsys, [i], [ni]) for i in [4, n_atoms-3]])
end;

H1 = 0.5*sum([embed(bsys, [i], [sx]) for i in 1:n_atoms])
H2 = -sum([embed(bsys, [i], [ni]) for i in 1:n_atoms]);
```


```julia
function GHZ_state(n_atoms)
    state = tensor([spindown(bs)⊗spinup(bs) for i in 1:Int(n_atoms/2)]...) +
            tensor([spinup(bs)⊗spindown(bs) for i in 1:Int(n_atoms/2)]...)
    state/sqrt(2.0)
end 

ground_state(n_atoms) = tensor([spindown(bs) for i in 1:n_atoms]...)
trans = StateTransform(ground_state(n_atoms)=>GHZ_state(n_atoms));
```


```julia
n_neurons = 8
sigmoid(x)= @. 2π*7 / (1 + exp(-x))
Random.seed!(10)
ann = FastChain(FastDense(1, n_neurons, tanh), 
                FastDense(n_neurons, n_neurons, tanh), 
                FastDense(n_neurons, 2))
θ = initial_params(ann)  
n_params = length(θ)
```




    106




```julia
t0, t1 = 0.0, 0.5

tsf32 = Float32(t0):Float32(t1/49):Float32(t1)
Ωs = Vector{Float32}(2π*vcat(0:0.5:4, 5*ones(32), 4:-0.5:0))
Δs = Vector{Float32}(2π*(-5:10/49:5))
ts = Vector{Float64}(tsf32)

function loss(p)
    c = 0.0f0
    for (i,t) in enumerate(tsf32)
        x = ann([t], p)
        c += (abs(x[1]) - Ωs[i])^2
        c += (x[2] - Δs[i])^2
    end
    #println(c)
    c
end

res = DiffEqFlux.sciml_train(loss, initial_params(ann), ADAM(0.1f0), maxiters = 5000)
θ = Vector{Float64}(res.u);
```


```julia
coeffs(params, t) = let vals = ann([t], params)
                        [abs(vals[1]), vals[2]]
                    end    

cost = CostFunction((x, y) -> 1.0 - abs(sum(conj(x).*y)),
                     p->2e-3*(abs(ann([t0], p)[1])+ 5.0*abs(ann([t1], p)[1])))
```




    CostFunction(var"#23#25"(), var"#24#26"())




```julia
H = Hamiltonian(H0, [H1, H2], coeffs);
```


```julia
prob = cu(convert(Float32, QOCProblem(H, trans, (t0, t1), cost)));
```


```julia
@time sol = solve(prob, res.u, ADAM(0.1f0); maxiter=200, abstol=1e-5, reltol=1e-5)
```

    
    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:23:56[39m
    [34m  distance:     0.01712620258331299[39m
    [34m  constraints:  0.008312689546361399[39m


    1474.028825 seconds (5.51 G allocations: 137.420 GiB, 9.97% gc time, 5.27% compilation time)





    Solution{Float32}(Float32[-4.891644, -3.2486708, 3.6208906, -4.285252, -9.749616, -8.146689, 8.857458, -3.5788686, 0.93261456, 1.4900919  …  3.5448282, -6.424723, 7.1616683, -7.219798, 1.7060657, 17.867157, 19.374046, 3.488416, 1.9582546, 6.225233], Float32[0.89193004, 0.8164891, 0.8592043, 0.7698536, 0.79915655, 0.8161896, 0.7245906, 0.6944014, 0.69365954, 0.6437732  …  0.0162189, 0.017019272, 0.020733714, 0.019386351, 0.017857492, 0.01583141, 0.017026603, 0.02026099, 0.018009365, 0.017126203], Float32[0.0110306395, 0.048962075, 0.050294697, 0.03963808, 0.018402087, 0.025846347, 0.017188322, 0.032640316, 0.036211997, 0.02908829  …  0.010170492, 0.005518189, 0.003905754, 0.0028085758, 0.006085555, 0.008987504, 0.005064504, 0.0024155828, 0.0011272868, 0.008312689], Vector{Float32}[])




```julia
plot(sol.distance_trace)
```




<div
    class="webio-mountpoint"
    data-webio-mountpoint="18187880574591109147"
>
    <script>
    (function(){
    // Some integrations (namely, IJulia/Jupyter) use an alternate render pathway than
    // just putting the html on the page. If WebIO isn't defined, then it's pretty likely
    // that we're in one of those situations and the integration just isn't installed
    // correctly.
    if (typeof window.WebIO === "undefined") {
        document
            .querySelector('[data-webio-mountpoint="18187880574591109147"]')
            .innerHTML = (
                '<div style="padding: 1em; background-color: #f8d6da; border: 1px solid #f5c6cb; font-weight: bold;">' +
                '<p><strong>WebIO not detected.</strong></p>' +
                '<p>Please read ' +
                '<a href="https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/" target="_blank">the troubleshooting guide</a> ' +
                'for more information on how to resolve this issue.</p>' +
                '<p><a href="https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/" target="_blank">https://juliagizmos.github.io/WebIO.jl/latest/troubleshooting/not-detected/</a></p>' +
                '</div>'
            );
        return;
    }
    WebIO.mount(
        document.querySelector('[data-webio-mountpoint="18187880574591109147"]'),
        {"props":{},"nodeType":"Scope","type":"node","instanceArgs":{"imports":{"data":[{"name":"Plotly","type":"js","url":"\/assetserver\/8a8e17519ba4665e2917ec2c4ca77fa596f4fb37-plotly.min.js"},{"name":null,"type":"js","url":"\/assetserver\/2130d832dc0717216b9445fc5813a8166285295c-plotly_webio.bundle.js"}],"type":"async_block"},"id":"11502678437862383708","handlers":{"_toImage":["(function (options){return this.Plotly.toImage(this.plotElem,options).then((function (data){return WebIO.setval({\"name\":\"image\",\"scope\":\"11502678437862383708\",\"id\":\"9087831313414027364\",\"type\":\"observable\"},data)}))})"],"__get_gd_contents":["(function (prop){prop==\"data\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"11502678437862383708\",\"id\":\"17826383125228424428\",\"type\":\"observable\"},this.plotElem.data)) : undefined; return prop==\"layout\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"11502678437862383708\",\"id\":\"17826383125228424428\",\"type\":\"observable\"},this.plotElem.layout)) : undefined})"],"_downloadImage":["(function (options){return this.Plotly.downloadImage(this.plotElem,options)})"],"_commands":["(function (args){var fn=args.shift(); var elem=this.plotElem; var Plotly=this.Plotly; args.unshift(elem); return Plotly[fn].apply(this,args)})"]},"systemjs_options":null,"mount_callbacks":["function () {\n    var handler = ((function (Plotly,PlotlyWebIO){PlotlyWebIO.init(WebIO); var gd=this.dom.querySelector(\"#plot-2ef1147f-7120-441f-9efa-4f2c93c328a0\"); this.plotElem=gd; this.Plotly=Plotly; (window.Blink!==undefined) ? (gd.style.width=\"100%\", gd.style.height=\"100vh\", gd.style.marginLeft=\"0%\", gd.style.marginTop=\"0vh\") : undefined; window.onresize=(function (){return Plotly.Plots.resize(gd)}); Plotly.newPlot(gd,[{\"y\":[0.89193004,0.8164891,0.8592043,0.7698536,0.79915655,0.8161896,0.7245906,0.6944014,0.69365954,0.6437732,0.60842705,0.6036384,0.5529903,0.53586024,0.49857795,0.46844023,0.44525927,0.42472363,0.40874428,0.38428348,0.37638587,0.38156152,0.3903494,0.37619424,0.34832925,0.3234939,0.30202562,0.28670627,0.26986206,0.25232118,0.23952287,0.23146361,0.22504658,0.21208143,0.19261384,0.17372179,0.15951133,0.14994895,0.14329761,0.13832843,0.12064266,0.11327362,0.112858534,0.1282506,0.10913271,0.09822065,0.09620446,0.084722996,0.102710426,0.09767103,0.08037555,0.07953566,0.07255,0.07138038,0.08130133,0.06990409,0.058066607,0.05764985,0.049844384,0.054602325,0.06627071,0.066449106,0.064600706,0.058454514,0.048634768,0.04083544,0.047239065,0.04940343,0.045333207,0.05098462,0.057083428,0.047899604,0.041065395,0.04369861,0.042702615,0.039897025,0.044374883,0.042817175,0.034638405,0.033343792,0.034649074,0.03543508,0.044009686,0.043206036,0.030550003,0.026080012,0.02824986,0.026488304,0.03427303,0.03512293,0.025229871,0.023848414,0.027716696,0.028284669,0.03619075,0.036190927,0.025558889,0.023546219,0.026659727,0.02670151,0.026582181,0.028762817,0.023073614,0.024127543,0.025954485,0.02926147,0.028529525,0.022189558,0.021027267,0.022886038,0.026192427,0.026990294,0.022302866,0.019856572,0.02269733,0.026211381,0.024410307,0.021472812,0.019222498,0.02133447,0.026518703,0.025613964,0.021991849,0.019157112,0.021189094,0.025282264,0.022872746,0.021012127,0.018989503,0.020560026,0.025943577,0.024837673,0.020999014,0.018828452,0.02037257,0.023695529,0.02234006,0.019983292,0.01841849,0.02006799,0.024529934,0.023894668,0.019733846,0.018013239,0.0194844,0.022252977,0.021698713,0.022613525,0.018891394,0.019225001,0.019735873,0.024296165,0.022968233,0.018763244,0.018708467,0.018381238,0.022648036,0.021588922,0.017962933,0.017865837,0.017725825,0.022083938,0.021072745,0.017564595,0.017459273,0.017464459,0.021705866,0.021006525,0.01729393,0.016995966,0.017125607,0.02096498,0.0198434,0.019999564,0.017549455,0.016883135,0.017969668,0.021986961,0.020777464,0.017983854,0.016756833,0.017406762,0.020892918,0.019529998,0.01766473,0.016382456,0.017111182,0.021049082,0.019576013,0.017792463,0.0162189,0.017019272,0.020733714,0.019386351,0.017857492,0.01583141,0.017026603,0.02026099,0.018009365,0.017126203],\"type\":\"scatter\",\"x\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200]}],{\"template\":{\"layout\":{\"coloraxis\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}},\"xaxis\":{\"gridcolor\":\"white\",\"zerolinewidth\":2,\"title\":{\"standoff\":15},\"ticks\":\"\",\"zerolinecolor\":\"white\",\"automargin\":true,\"linecolor\":\"white\"},\"hovermode\":\"closest\",\"paper_bgcolor\":\"white\",\"geo\":{\"showlakes\":true,\"showland\":true,\"landcolor\":\"#E5ECF6\",\"bgcolor\":\"white\",\"subunitcolor\":\"white\",\"lakecolor\":\"white\"},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"yaxis\":{\"gridcolor\":\"white\",\"zerolinewidth\":2,\"title\":{\"standoff\":15},\"ticks\":\"\",\"zerolinecolor\":\"white\",\"automargin\":true,\"linecolor\":\"white\"},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"hoverlabel\":{\"align\":\"left\"},\"mapbox\":{\"style\":\"light\"},\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"}},\"autotypenumbers\":\"strict\",\"font\":{\"color\":\"#2a3f5f\"},\"ternary\":{\"baxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"},\"aaxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"}},\"annotationdefaults\":{\"arrowhead\":0,\"arrowwidth\":1,\"arrowcolor\":\"#2a3f5f\"},\"plot_bgcolor\":\"#E5ECF6\",\"title\":{\"x\":0.05},\"scene\":{\"xaxis\":{\"gridcolor\":\"white\",\"gridwidth\":2,\"backgroundcolor\":\"#E5ECF6\",\"ticks\":\"\",\"showbackground\":true,\"zerolinecolor\":\"white\",\"linecolor\":\"white\"},\"zaxis\":{\"gridcolor\":\"white\",\"gridwidth\":2,\"backgroundcolor\":\"#E5ECF6\",\"ticks\":\"\",\"showbackground\":true,\"zerolinecolor\":\"white\",\"linecolor\":\"white\"},\"yaxis\":{\"gridcolor\":\"white\",\"gridwidth\":2,\"backgroundcolor\":\"#E5ECF6\",\"ticks\":\"\",\"showbackground\":true,\"zerolinecolor\":\"white\",\"linecolor\":\"white\"}},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"]},\"data\":{\"barpolar\":[{\"type\":\"barpolar\",\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5}}}],\"carpet\":[{\"aaxis\":{\"gridcolor\":\"white\",\"endlinecolor\":\"#2a3f5f\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\",\"linecolor\":\"white\"},\"type\":\"carpet\",\"baxis\":{\"gridcolor\":\"white\",\"endlinecolor\":\"#2a3f5f\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\",\"linecolor\":\"white\"}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"parcoords\":[{\"line\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}},\"type\":\"parcoords\"}],\"scatter\":[{\"type\":\"scatter\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"histogram2dcontour\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"histogram2dcontour\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contour\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"contour\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"mesh3d\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"mesh3d\"}],\"surface\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"surface\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"histogram\":[{\"type\":\"histogram\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"pie\":[{\"type\":\"pie\",\"automargin\":true}],\"choropleth\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"choropleth\"}],\"heatmapgl\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"heatmapgl\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"bar\":[{\"type\":\"bar\",\"error_y\":{\"color\":\"#2a3f5f\"},\"error_x\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5}}}],\"heatmap\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"heatmap\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"contourcarpet\"}],\"table\":[{\"type\":\"table\",\"header\":{\"line\":{\"color\":\"white\"},\"fill\":{\"color\":\"#C8D4E3\"}},\"cells\":{\"line\":{\"color\":\"white\"},\"fill\":{\"color\":\"#EBF0F8\"}}}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}},\"type\":\"scatter3d\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"histogram2d\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"histogram2d\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}]}},\"margin\":{\"l\":50,\"b\":50,\"r\":50,\"t\":60}},{\"showLink\":false,\"editable\":false,\"responsive\":true,\"staticPlot\":false,\"scrollZoom\":true}); gd.on(\"plotly_hover\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"hover\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"hover\",\"scope\":\"11502678437862383708\",\"id\":\"486914127403127589\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_unhover\",(function (){return WebIO.setval({\"name\":\"hover\",\"scope\":\"11502678437862383708\",\"id\":\"486914127403127589\",\"type\":\"observable\"},{})})); gd.on(\"plotly_selected\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"selected\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"selected\",\"scope\":\"11502678437862383708\",\"id\":\"8851778489081634462\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_deselect\",(function (){return WebIO.setval({\"name\":\"selected\",\"scope\":\"11502678437862383708\",\"id\":\"8851778489081634462\",\"type\":\"observable\"},{})})); gd.on(\"plotly_relayout\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"relayout\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"relayout\",\"scope\":\"11502678437862383708\",\"id\":\"10135904980969585125\",\"type\":\"observable\"},filtered_data.out)) : undefined})); return gd.on(\"plotly_click\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"click\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"click\",\"scope\":\"11502678437862383708\",\"id\":\"9879593663475921091\",\"type\":\"observable\"},filtered_data.out)) : undefined}))}));\n    (WebIO.importBlock({\"data\":[{\"name\":\"Plotly\",\"type\":\"js\",\"url\":\"\/assetserver\/8a8e17519ba4665e2917ec2c4ca77fa596f4fb37-plotly.min.js\"},{\"name\":null,\"type\":\"js\",\"url\":\"\/assetserver\/2130d832dc0717216b9445fc5813a8166285295c-plotly_webio.bundle.js\"}],\"type\":\"async_block\"})).then((imports) => handler.apply(this, imports));\n}\n"],"observables":{"_toImage":{"sync":false,"id":"2170011096654827826","value":{}},"hover":{"sync":false,"id":"486914127403127589","value":{}},"selected":{"sync":false,"id":"8851778489081634462","value":{}},"__gd_contents":{"sync":false,"id":"17826383125228424428","value":{}},"click":{"sync":false,"id":"9879593663475921091","value":{}},"image":{"sync":true,"id":"9087831313414027364","value":""},"__get_gd_contents":{"sync":false,"id":"17252845797983543354","value":""},"_downloadImage":{"sync":false,"id":"11950323811615354761","value":{}},"relayout":{"sync":false,"id":"10135904980969585125","value":{}},"_commands":{"sync":false,"id":"14740290306610665886","value":[]}}},"children":[{"props":{"id":"plot-2ef1147f-7120-441f-9efa-4f2c93c328a0"},"nodeType":"DOM","type":"node","instanceArgs":{"namespace":"html","tag":"div"},"children":[]}]},
        window,
    );
    })()
    </script>
</div>





```julia
Ω(t) = abs(ann([t], sol.params)[1])/2π
Δ(t) = ann([t], sol.params)[2]/2π
ts = collect(t0:0.001:t1)

f = plot(
    [
        scatter(x=ts, y=Ω.(ts), name="Ω/2π"),
        scatter(x=ts, y=Δ.(ts), name="Δ/2π"),
    ],
    Layout(
        xaxis_title_text="Time (µs)",
        yaxis_title_text="Frequency (MHz)",
        legend=attr(x=0, y=1,),
        font=attr(
            size=16
        )
    )
)
savefig(f, "GHZ_12_atoms_wfs.eps")
```




    "GHZ_12_atoms_wfs.eps"




```julia
tout, psit = schroedinger_dynamic(ts, ground_state(n_atoms), H, Vector{Float64}(sol.params));
```


```julia
f = plot(
    [
        scatter(x=ts, y=real(expect(dm(GHZ_state(n_atoms)), psit))),
    ],
    Layout(
        xaxis_title_text="Time (µs)",
        yaxis_title_text="Overlap (|⟨ψ|GHZ⟩|²)",
        font=attr(
            size=16
        )
    )
)
savefig(f,"GHZ_12_atoms_overlap.eps")
```




    "GHZ_12_atoms_overlap.eps"


