```julia
using Revise
using QuantumOptimalControl
using QuantumOptics
using LinearAlgebra
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
id = identityoperator(bs, bs);
```


```julia
H0 = ω₀*(ad*a + 0.5*id) + (η/12.0)*(a + ad)^4 - η^2 * (a + ad)^6/ω₀/90.0
H1 = 1.0im*(a - ad);
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
#θ = Vector{Float64}(initial_params(ann));
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
                         [[1.0 -1.0im];[-1.0im 1.0]]/√2);
```


```julia
tspan = (t0, t1)
H = Hamiltonian(H0, [H1], coeffs)
prob = QOCProblem(H, trans, tspan, cost);
```


```julia
@time sol = solve(prob, θ, ADAM(0.02); maxiter=200)
```

    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:04:23[39m
    [34m  distance:    0.004051665209568678[39m
    [34m  contraints:  2.303269822766253e-5[39m


    278.588450 seconds (1.21 G allocations: 116.153 GiB, 26.66% gc time, 14.32% compilation time)





    Solution{Float64}([-1.1306986611306162, -0.8425831245811041, -0.623546391936392, -0.057293001635333116, -1.2765358851803152, 0.777734717648167, -0.792724247101401, 0.4818008475057589, 0.9160872652599155, 1.424089949015126  …  4.422041178918318, 1.0892082336876976, -1.1083449682726871, 0.12624303517738575, 3.1803943253653464, 0.50464902211469, -1.2606743304546766, 1.7335362727380126, -0.01972996936904131, -0.14262863999400807], [1.1748429261214117, 0.9288213641733225, 0.5708557259503002, 0.2827521701427674, 0.13999372294690765, 0.09114892905631444, 0.07911441026714866, 0.08604308513912307, 0.10809634494604681, 0.1381976596236002  …  0.004088433471982733, 0.004084130869983882, 0.004079878840889062, 0.004075682039990147, 0.004071546118208869, 0.004067473137183475, 0.004063458131355002, 0.0040594917963445165, 0.0040555627256546045, 0.004051665209568678])




```julia
plot(sol.trace)
```




<div
    class="webio-mountpoint"
    data-webio-mountpoint="819187735678944457"
>
    <script>
    (function(){
    // Some integrations (namely, IJulia/Jupyter) use an alternate render pathway than
    // just putting the html on the page. If WebIO isn't defined, then it's pretty likely
    // that we're in one of those situations and the integration just isn't installed
    // correctly.
    if (typeof window.WebIO === "undefined") {
        document
            .querySelector('[data-webio-mountpoint="819187735678944457"]')
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
        document.querySelector('[data-webio-mountpoint="819187735678944457"]'),
        {"props":{},"nodeType":"Scope","type":"node","instanceArgs":{"imports":{"data":[{"name":"Plotly","type":"js","url":"\/assetserver\/8a8e17519ba4665e2917ec2c4ca77fa596f4fb37-plotly.min.js"},{"name":null,"type":"js","url":"\/assetserver\/2130d832dc0717216b9445fc5813a8166285295c-plotly_webio.bundle.js"}],"type":"async_block"},"id":"5978662566714087263","handlers":{"_toImage":["(function (options){return this.Plotly.toImage(this.plotElem,options).then((function (data){return WebIO.setval({\"name\":\"image\",\"scope\":\"5978662566714087263\",\"id\":\"7702278987598093564\",\"type\":\"observable\"},data)}))})"],"__get_gd_contents":["(function (prop){prop==\"data\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"5978662566714087263\",\"id\":\"5017482375428881778\",\"type\":\"observable\"},this.plotElem.data)) : undefined; return prop==\"layout\" ? (WebIO.setval({\"name\":\"__gd_contents\",\"scope\":\"5978662566714087263\",\"id\":\"5017482375428881778\",\"type\":\"observable\"},this.plotElem.layout)) : undefined})"],"_downloadImage":["(function (options){return this.Plotly.downloadImage(this.plotElem,options)})"],"_commands":["(function (args){var fn=args.shift(); var elem=this.plotElem; var Plotly=this.Plotly; args.unshift(elem); return Plotly[fn].apply(this,args)})"]},"systemjs_options":null,"mount_callbacks":["function () {\n    var handler = ((function (Plotly,PlotlyWebIO){PlotlyWebIO.init(WebIO); var gd=this.dom.querySelector(\"#plot-3d14340f-62d4-4d1a-a0b8-c409392faff7\"); this.plotElem=gd; this.Plotly=Plotly; (window.Blink!==undefined) ? (gd.style.width=\"100%\", gd.style.height=\"100vh\", gd.style.marginLeft=\"0%\", gd.style.marginTop=\"0vh\") : undefined; window.onresize=(function (){return Plotly.Plots.resize(gd)}); Plotly.newPlot(gd,[{\"y\":[1.1748429261214117,0.9288213641733225,0.5708557259503002,0.2827521701427674,0.13999372294690765,0.09114892905631444,0.07911441026714866,0.08604308513912307,0.10809634494604681,0.1381976596236002,0.16262863257755644,0.1702265818484503,0.15977794860624372,0.13695198920308865,0.10939518691857014,0.0834889418602125,0.06290879302163305,0.048610376536821154,0.03987235645874021,0.03579225420134413,0.03606597663211658,0.0402640117652992,0.04644047927771777,0.05112520437203699,0.05139819529083578,0.04690757895606945,0.03961385500696307,0.0318723324490755,0.0249914387608291,0.01927175134378195,0.014790270564681685,0.011842106289246646,0.010758722633348927,0.011484268543177967,0.013365936197130002,0.015355715335977072,0.01649796187506053,0.016343912800754035,0.015032902357546285,0.013059281928850586,0.0109542762718714,0.009097967651606531,0.007717375944139737,0.006967762452028015,0.006946511365106878,0.007589247264432997,0.008577226838550889,0.009436059229193217,0.009806737996069848,0.009633932260870504,0.009090336700842583,0.008376777243458533,0.007643754944008263,0.007044142793927788,0.0067353361213184915,0.006767285732158179,0.007005155024394971,0.0072217532429434406,0.007276188389060578,0.0071810259759050865,0.007009358962190737,0.006790741692200064,0.006525114830962353,0.006259045327171697,0.0060864517703759224,0.006057689951090184,0.006118990608999009,0.006165352927047174,0.006138386094834747,0.006052642786149032,0.005942922616363255,0.005821582673669723,0.0056935444701882165,0.005582953633874066,0.005519799775573542,0.005506014300053541,0.00551233203557161,0.005508848730495897,0.005486314219762201,0.00544815367131779,0.0053974983224538975,0.005340177761902265,0.005289739570044527,0.005258009559229504,0.0052424657304239175,0.005230350926743066,0.0052130578360901625,0.005189884969498293,0.0051605616449440506,0.005123627325543945,0.005082972772296301,0.005047790875085223,0.005022562773815631,0.005002009093596049,0.004978978570777504,0.00495301214071614,0.004928002224993788,0.004905481468285899,0.004884620012996832,0.004866279581464572,0.0048524089809891535,0.0048419572326190985,0.004830907274514629,0.0048164368017484804,0.0047988204863191886,0.004779574473733894,0.004760091614575301,0.00474193068728429,0.004726286063272023,0.004712805678648513,0.0047001373420992865,0.00468752141639861,0.004674927099877768,0.0046621729081817165,0.004649161597454654,0.004636579991253387,0.004625171498821579,0.004614539302478904,0.004603593271887418,0.004591913755537469,0.004579886383015652,0.004567838728779583,0.004555877017678345,0.004544301672951667,0.004533490539420648,0.004523452035783082,0.004513929828403984,0.004504791118902951,0.004496021194872024,0.004487529248377309,0.004479197594221329,0.004470944996905479,0.004462609854819499,0.004453960868618323,0.0044449020858692845,0.004435513495743337,0.004425908439008142,0.004416214760980974,0.004406647282549803,0.004397400048091915,0.004388516792243391,0.004379966581186168,0.00437176409256107,0.004363917628101954,0.004356349037883567,0.0043489570729501326,0.004341684830285075,0.00433448477653231,0.004327300135361811,0.004320110061590388,0.004312939543500982,0.004305822751851485,0.004298793596013262,0.004291894442706223,0.0042851551216784856,0.0042785737923515255,0.004272124254068976,0.004265762475505364,0.004259429470725107,0.004253081091761035,0.004246710601512049,0.004240338506339092,0.004233988908791231,0.004227693431477386,0.004221482750162653,0.004215371120460554,0.00420936357706686,0.004203478269702798,0.004197738370001702,0.0041921529874885954,0.004186708253551941,0.004181368504100613,0.0041760843835129124,0.004170810253704582,0.004165522211323325,0.004160222246420153,0.004154930865530537,0.004149677605778002,0.0041444894699064205,0.004139383832737986,0.004134367775436687,0.004129445799943454,0.004124618410301761,0.004119879219590894,0.004115219509236712,0.004110629397250487,0.004106097014632715,0.004101614000624032,0.004097176299107552,0.004092782665620798,0.004088433471982733,0.004084130869983882,0.004079878840889062,0.004075682039990147,0.004071546118208869,0.004067473137183475,0.004063458131355002,0.0040594917963445165,0.0040555627256546045,0.004051665209568678],\"type\":\"scatter\",\"x\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200]}],{\"template\":{\"layout\":{\"coloraxis\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}},\"xaxis\":{\"gridcolor\":\"white\",\"zerolinewidth\":2,\"title\":{\"standoff\":15},\"ticks\":\"\",\"zerolinecolor\":\"white\",\"automargin\":true,\"linecolor\":\"white\"},\"hovermode\":\"closest\",\"paper_bgcolor\":\"white\",\"geo\":{\"showlakes\":true,\"showland\":true,\"landcolor\":\"#E5ECF6\",\"bgcolor\":\"white\",\"subunitcolor\":\"white\",\"lakecolor\":\"white\"},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"yaxis\":{\"gridcolor\":\"white\",\"zerolinewidth\":2,\"title\":{\"standoff\":15},\"ticks\":\"\",\"zerolinecolor\":\"white\",\"automargin\":true,\"linecolor\":\"white\"},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"hoverlabel\":{\"align\":\"left\"},\"mapbox\":{\"style\":\"light\"},\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"}},\"autotypenumbers\":\"strict\",\"font\":{\"color\":\"#2a3f5f\"},\"ternary\":{\"baxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"},\"aaxis\":{\"gridcolor\":\"white\",\"ticks\":\"\",\"linecolor\":\"white\"}},\"annotationdefaults\":{\"arrowhead\":0,\"arrowwidth\":1,\"arrowcolor\":\"#2a3f5f\"},\"plot_bgcolor\":\"#E5ECF6\",\"title\":{\"x\":0.05},\"scene\":{\"xaxis\":{\"gridcolor\":\"white\",\"gridwidth\":2,\"backgroundcolor\":\"#E5ECF6\",\"ticks\":\"\",\"showbackground\":true,\"zerolinecolor\":\"white\",\"linecolor\":\"white\"},\"zaxis\":{\"gridcolor\":\"white\",\"gridwidth\":2,\"backgroundcolor\":\"#E5ECF6\",\"ticks\":\"\",\"showbackground\":true,\"zerolinecolor\":\"white\",\"linecolor\":\"white\"},\"yaxis\":{\"gridcolor\":\"white\",\"gridwidth\":2,\"backgroundcolor\":\"#E5ECF6\",\"ticks\":\"\",\"showbackground\":true,\"zerolinecolor\":\"white\",\"linecolor\":\"white\"}},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"]},\"data\":{\"barpolar\":[{\"type\":\"barpolar\",\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5}}}],\"carpet\":[{\"aaxis\":{\"gridcolor\":\"white\",\"endlinecolor\":\"#2a3f5f\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\",\"linecolor\":\"white\"},\"type\":\"carpet\",\"baxis\":{\"gridcolor\":\"white\",\"endlinecolor\":\"#2a3f5f\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\",\"linecolor\":\"white\"}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"parcoords\":[{\"line\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}},\"type\":\"parcoords\"}],\"scatter\":[{\"type\":\"scatter\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"histogram2dcontour\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"histogram2dcontour\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contour\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"contour\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"mesh3d\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"mesh3d\"}],\"surface\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"surface\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"histogram\":[{\"type\":\"histogram\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"pie\":[{\"type\":\"pie\",\"automargin\":true}],\"choropleth\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"choropleth\"}],\"heatmapgl\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"heatmapgl\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"bar\":[{\"type\":\"bar\",\"error_y\":{\"color\":\"#2a3f5f\"},\"error_x\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5}}}],\"heatmap\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"heatmap\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"contourcarpet\"}],\"table\":[{\"type\":\"table\",\"header\":{\"line\":{\"color\":\"white\"},\"fill\":{\"color\":\"#C8D4E3\"}},\"cells\":{\"line\":{\"color\":\"white\"},\"fill\":{\"color\":\"#EBF0F8\"}}}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}},\"type\":\"scatter3d\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"histogram2d\":[{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0},\"type\":\"histogram2d\",\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"ticks\":\"\",\"outlinewidth\":0}}}]}},\"margin\":{\"l\":50,\"b\":50,\"r\":50,\"t\":60}},{\"showLink\":false,\"editable\":false,\"responsive\":true,\"staticPlot\":false,\"scrollZoom\":true}); gd.on(\"plotly_hover\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"hover\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"hover\",\"scope\":\"5978662566714087263\",\"id\":\"5039108064054116108\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_unhover\",(function (){return WebIO.setval({\"name\":\"hover\",\"scope\":\"5978662566714087263\",\"id\":\"5039108064054116108\",\"type\":\"observable\"},{})})); gd.on(\"plotly_selected\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"selected\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"selected\",\"scope\":\"5978662566714087263\",\"id\":\"10804610336687578113\",\"type\":\"observable\"},filtered_data.out)) : undefined})); gd.on(\"plotly_deselect\",(function (){return WebIO.setval({\"name\":\"selected\",\"scope\":\"5978662566714087263\",\"id\":\"10804610336687578113\",\"type\":\"observable\"},{})})); gd.on(\"plotly_relayout\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"relayout\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"relayout\",\"scope\":\"5978662566714087263\",\"id\":\"15393554840428163965\",\"type\":\"observable\"},filtered_data.out)) : undefined})); return gd.on(\"plotly_click\",(function (data){var filtered_data=WebIO.PlotlyCommands.filterEventData(gd,data,\"click\"); return !(filtered_data.isnil) ? (WebIO.setval({\"name\":\"click\",\"scope\":\"5978662566714087263\",\"id\":\"7814938928318511177\",\"type\":\"observable\"},filtered_data.out)) : undefined}))}));\n    (WebIO.importBlock({\"data\":[{\"name\":\"Plotly\",\"type\":\"js\",\"url\":\"\/assetserver\/8a8e17519ba4665e2917ec2c4ca77fa596f4fb37-plotly.min.js\"},{\"name\":null,\"type\":\"js\",\"url\":\"\/assetserver\/2130d832dc0717216b9445fc5813a8166285295c-plotly_webio.bundle.js\"}],\"type\":\"async_block\"})).then((imports) => handler.apply(this, imports));\n}\n"],"observables":{"_toImage":{"sync":false,"id":"10544416278563009473","value":{}},"hover":{"sync":false,"id":"5039108064054116108","value":{}},"selected":{"sync":false,"id":"10804610336687578113","value":{}},"__gd_contents":{"sync":false,"id":"5017482375428881778","value":{}},"click":{"sync":false,"id":"7814938928318511177","value":{}},"image":{"sync":true,"id":"7702278987598093564","value":""},"__get_gd_contents":{"sync":false,"id":"12038161003071836004","value":""},"_downloadImage":{"sync":false,"id":"16653722641075103269","value":{}},"relayout":{"sync":false,"id":"15393554840428163965","value":{}},"_commands":{"sync":false,"id":"16740319360875791141","value":[]}}},"children":[{"props":{"id":"plot-3d14340f-62d4-4d1a-a0b8-c409392faff7"},"nodeType":"DOM","type":"node","instanceArgs":{"namespace":"html","tag":"div"},"children":[]}]},
        window,
    );
    })()
    </script>
</div>





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
tout, psit = schroedinger_dynamic(ts, fockstate(bs, 0), H, sol.params);
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




```julia

```
