using Documenter, QuantumOptimalControl

makedocs(
    sitename="QuantumOptimalControl.jl",
    format = Documenter.HTML(prettyurls = false),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Examples" => [
            "DRAG" => "examples/DRAG.md",
            "Two level system" => "examples/TwoLevelSystem.md"
        ]
    ])