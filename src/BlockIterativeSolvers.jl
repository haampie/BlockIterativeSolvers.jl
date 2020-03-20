module BlockIterativeSolvers

using LinearAlgebra, Random

"""
    sparse_reflector!(A, col, rows) → τ

Implicit representation of a Householder reflection H = I - τ [1, v] [1, vᵀ] such that H*[A[col, col]; A[rows, col]] = e₁β, 1 ≤ real(τ) ≤ 2 and abs(τ - 1) ≤ 1. Updates `A` in-place,
in the sense that: `v := A[rows, col]` and A[col, col] = β.
In the edge case iszero(A[rows, col]) and iszero(imag(A[col, col])) then τ = 0.
Based on LAPACK 3.8 clarfg.
Another nice reference is Mezzadri, Francesco. "How to generate
random matrices from the classical compact groups." arXiv preprint 
math-ph/0609050 (2006).
"""
function sparse_reflector!(A::AbstractMatrix{T}, col, rows) where {T}
    @inbounds begin
        # Diagonal
        α = A[col, col]

        # Norm of the rest
        xnrm = zero(real(T))
        @simd for idx in rows
            xnrm += abs2(A[idx, col])
        end

        iszero(xnrm) && iszero(imag(α)) && return zero(T)

        xnrm = √xnrm

        β = -copysign(hypot(α, xnrm), real(α))
        τ = (β - α) / β
        α = inv(α - β)
       
        # Rescale
        @simd for idx = rows
            A[idx, col] *= α
        end

        A[col, col] = β

        return τ'
    end
end

"""
Apply the householder reflection stored in A to (the columns of) B
"""
function apply_sparse_reflector!(B::AbstractVecOrMat{U}, A::AbstractMatrix{T}, A_col, A_rows, τ) where {T,U}

    iszero(τ) && return nothing

    @inbounds for col = axes(B, 2)
        dot = B[A_col, col]
        @simd for row = A_rows
            dot += A[row, A_col]' * B[row, col]
        end
        dot *= τ
        B[A_col, col] -= dot
        @simd for row = A_rows
            B[row, col] -= dot * A[row, A_col]
        end
    end

    B
end

function gmres!(X::AbstractVecOrMat{T}, A, B; steps = 10, block_size = 5, tolerance = sqrt(eps(real(eltype(T))))) where {T}

    unknown_size = size(X, 2)

    m, n = size(A)

    @assert size(B, 2) == unknown_size
    @assert block_size >= unknown_size
    @assert m == n

    # All allocations. TODO: 

    # High-dimensional bits are allocated on the device

    # V is the orthonormal basis
    V = similar(X, m, block_size * (steps + 1))

    # compact_qr_workspace is a kind of staging thing
    # for a block to be orthogonalized internally
    compact_qr_workspace = similar(X, m, block_size)

    # Low-dimensional stuff is allocated on the host
    
    # Block Hessenberg matrix
    H = zeros(T, block_size * (steps + 1), block_size * steps)

    # Null space of H', so H' * null_space = O. 
    # We have a non-orthogonal version of this for quick
    # updates when the Hessenberg matrix grows, and an
    # orthonormal representation null_space_qr that stores
    # Householder reflectors.
    null_space = zeros(T, block_size * (steps + 1), block_size)
    copyto!(null_space, I)
    null_space_qr = copy(null_space)

    # This is the right-hand side in the least-squares problem
    small_rhs = zeros(T, block_size * (steps + 1), unknown_size)

    # The residual matrix of the least-squares problem
    small_residual = zeros(T, block_size * (steps + 1), unknown_size)

    # This will hold the second projection when classical Gram-Schmidt is repeated
    correction = similar(H, block_size * steps, block_size)

    # First copy the rhs to the compact_qr_workspace block vec
    copyto!(view(compact_qr_workspace, :, 1:unknown_size), B)
    rand!(view(compact_qr_workspace, :, unknown_size+1:block_size))

    # QR the rhs and move it into the V basis.
    qr_block = qr!(compact_qr_workspace)
    copyto!(view(V, :, 1:block_size), I)
    lmul!(qr_block.Q, view(V, :, 1:block_size))

    # Prepare rhs
    copyto!(view(small_rhs, 1:unknown_size, 1:unknown_size), view(qr_block.R, 1:unknown_size, 1:unknown_size))
    copyto!(view(small_residual, 1:unknown_size, 1:unknown_size), view(small_rhs, 1:unknown_size, 1:unknown_size))

    step = 1

    frobenius_norm_residual = typemax(real(T))

    while step <= steps && frobenius_norm_residual > tolerance
        curr_from, curr_to = (step - 1) * block_size + 1, step * block_size
        next_from, next_to = curr_from + block_size, curr_to + block_size

        # Next block vec
        mul!(compact_qr_workspace, A, view(V, :, curr_from:curr_to))

        # Classical Gram-Schmidt repeated twice
        mul!(view(H, 1:curr_to, curr_from:curr_to), view(V, :, 1:curr_to)', compact_qr_workspace)
        mul!(compact_qr_workspace, view(V, :, 1:curr_to), view(H, 1:curr_to, curr_from:curr_to), one(T), -one(T))
        mul!(view(correction, 1:curr_to, :), view(V, :, 1:curr_to)', compact_qr_workspace)
        mul!(compact_qr_workspace, view(V, :, 1:curr_to), view(correction, 1:curr_to, :), one(T), -one(T))
        view(H, 1:curr_to, curr_from:curr_to) .+= view(correction, 1:curr_to, :)

        # Internal orthogonalization
        qr_block = qr!(compact_qr_workspace)
        copyto!(view(V, :, next_from:next_to), I)
        lmul!(qr_block.Q, view(V, :, next_from:next_to))
        copyto!(view(H, next_from:next_to, curr_from:curr_to), qr_block.R)

        # Null space matrix update
        mul!(view(null_space, next_from:next_to, 1:block_size), view(H, 1:curr_to, curr_from:curr_to)', view(null_space, 1:curr_to, 1:block_size), -1, 0)
        ldiv!(UpperTriangular(view(H, next_from:next_to, curr_from:curr_to))', view(null_space, next_from:next_to, 1:block_size))

        # Update the orthonormal basis of the null space matrix
        # and apply immediately to the residual matrix
        copyto!(view(null_space_qr, next_from:next_to, 1:block_size), view(null_space, next_from:next_to, 1:block_size))

        for col in 1:block_size
            τ = sparse_reflector!(null_space_qr, col, next_from:next_to)
            apply_sparse_reflector!(view(null_space_qr, :, col+1:block_size), null_space_qr, col, next_from:next_to, τ)
            apply_sparse_reflector!(small_residual, null_space_qr, col, next_from:next_to, τ)
        end

        # Compute the residual norm
        frobenius_norm_residual = norm(view(small_residual, 1:unknown_size, 1:unknown_size))

        step += 1
    end

    # Solve the least-squares problem

    # Create and apply reflectors to H and the right-hand side.
    last_col = (step - 1) * block_size
    last_row = step * block_size

    for col = 1 : last_col
        reflector_range = col + 1 : min(col + block_size, last_row)
        τ = sparse_reflector!(H, col, reflector_range)
        apply_sparse_reflector!(view(H, :, col+1:last_col), H, col, reflector_range, τ)
        apply_sparse_reflector!(small_rhs, H, col, reflector_range, τ)
    end

    # Do backward substitution
    ldiv!(UpperTriangular(view(H, 1:last_col, 1:last_col)), view(small_rhs, 1:last_col, :))

    # Todo --- move `small_least_squares_solution` to the GPU.

    # And bring back to the high-dimensional world
    mul!(X, view(V, :, 1 : last_col), view(small_rhs, 1:last_col, :))

    return X
end

end # module
