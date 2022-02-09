# Sisyphus

A high-performance library for gradient based quantum optimal control


# Installation

Sisyphus is not yet available in the official registry of Julia packages. You can install Sisyphus and it's dependencies in the REPL (press `]` to enter the `Pkg` mode) with the github link,

```
pkg> add git@github.com:entropy-lab/Sisyphus.git
```

# Documentation

To generate the documentation locally,

1) Clone the repository
   
```shell
git clone git@github.com:entropy-lab/Sisyphus.git
cd Sisyphus   
```

2) Install `Documenter` and run make.jl to generate the html files

```shell
julia -e "import Pkg; Pkg.add(\"Documenter\");"
julia --project=. docs/make.jl
```

3) Install `LiveServer` and serve the generated pages

```shell
julia -e "import Pkg; Pkg.add(\"LiveServer\");"
julia -e 'using LiveServer; serve(dir="docs/build")'
```

4) Documentation is now accessible in the browser via `http://localhost:8000/`