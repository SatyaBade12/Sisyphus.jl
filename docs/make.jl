using Documenter, QuantumOptimalControl

sourcedir = "examples/"
markdowndir = "markdown"
targetpath_examples = "docs/src/examples/"

names = filter(name->endswith(name, ".ipynb"), readdir(sourcedir))

function convert2markdown(name)
    sourcepath = joinpath(sourcedir, name)
    println(sourcepath)
    run(`jupyter-nbconvert --to markdown --output-dir=$markdowndir $sourcepath --template=docs/markdown_template.tpl`)
end

for name in names
    convert2markdown(name)
end

cp(markdowndir, targetpath_examples; force=true)
rm(markdowndir; recursive=true)

makedocs(
    sitename="QuantumOptimalControl.jl",
    format = Documenter.HTML(prettyurls = false),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Introduction" => "introduction.md",
        "Gradient-based QOC" => "gbqoc.md",
        "Open quantum systems" => "noisy.md",
        "Tutorial" => "tutorial.md",
        "Examples" => [
            "DRAG" => "examples/DRAG.md",
            "Two-level system" => "examples/TwoLevelSystem.md",
            "Noisy two-level system" => "examples/TwoLevelSystemNoisy.md",
            "GHZ state" => "examples/GHZState.md",
            "GHZ state (CUDA)" => "examples/GHZStateCUDA.md",
            "CZ" => "examples/CZ.md",
            "Rₓ(π/2)" => "examples/RXpi2.md",
            "√iSWAP" => "examples/SQRTiSWAP.md"
        ],
        "API" => "api.md"
    ])

deploydocs(
    repo = "github.com/SatyaBade12/optimal-control.git",
)
