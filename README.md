# Sisyphus

A high-performance library for gradient based quantum optimal control


# Installation

Sisyphus is not yet available in the official registry of Julia packages. You can install Sisyphus and it's dependencies in the REPL (press `]` to enter the `Pkg` mode) with the github link,

```
pkg> add git@github.com:SatyaBade12/Sisyphus.jl.git
```

# Documentation

To generate the documentation locally,

1) Clone the repository
   
```shell
git clone git@github.com:SatyaBade12/Sisyphus.jl.git
cd Sisyphus   
```

2) Install `nbconvert`

```shell
pip install nbconvert==5.6.1
```

3) Install `Documenter` and run make.jl to generate the html files

```shell
julia -e "import Pkg; Pkg.add(\"Documenter\");"
julia --project=. docs/make.jl
```

4) Install `LiveServer` and serve the generated pages

```shell
julia -e "import Pkg; Pkg.add(\"LiveServer\");"
julia -e 'using LiveServer; serve(dir="docs/build")'
```

5) Documentation is now accessible in the browser via `http://localhost:8000/`
