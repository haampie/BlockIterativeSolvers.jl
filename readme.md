# BlockIterativeMethods.jl

An attempt to bring efficient, memory-friendly, blocked iterative methods to CPU and GPU :rocket:


Right now very much a work in progress. Unrestarted GMRES works, but it should slowly transform into Krylov-Schur restarted GMRES, and the good bits of it are to be reused in eigenvalue solvers.


```julia
julia> using BlockIterativeMethods

julia> using SparseArrays, LinearAlgebra

julia> A = sprand(1000, 1000, 2/1000) + 4I;

julia> B = rand(1000, 3);

julia> X = rand(1000, 3);

julia> BlockIterativeSolvers.gmres!(X, A, B, steps = 30, block_size = 8, tolerance = 1e-6)

julia> norm(A * X - B)
3.1536626111447993e-7
```

```julia
julia> Ac = sprand(ComplexF64, 1000, 1000, 2/1000) + 4I;

julia> Bc = rand(ComplexF64, 1000, 3);

julia> Xc = rand(ComplexF64, 1000, 3);

julia> BlockIterativeSolvers.gmres!(Xc, Ac, Bc, steps = 30, block_size = 8, tolerance = 1e-6)

julia> norm(Ac * Xc - Bc)
4.2455026885746224e-7
```


