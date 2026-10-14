# Enzyme implementation of example 01: written as you would write it without AD.
# Included by enzyme.qmd; kept in a file so it can also be tested from the REPL.

using Enzyme

# ---------------------------------------------------------------------------- CPU
# The whole pipeline is differentiated: cell ids, counting sort, in-place gather into
# sorted order, and the neighbour loops whose ranges depend on the data.

function sph_loss(x::AbstractMatrix{T}, p::Params{T}) where {T}
    N  = size(x, 2)
    nc = p.nc
    s  = p.L / nc
    rc = 2 * p.h

    # 1. cell id of every particle (integer, zero derivative)
    cell  = Vector{Int}(undef, N)
    start = zeros(Int, nc * nc + 1)
    @inbounds for i in 1:N
        cx = mod(floor(Int, x[1, i] / s), nc)
        cy = mod(floor(Int, x[2, i] / s), nc)
        c = 1 + cx + nc * cy
        cell[i] = c
        start[c + 1] += 1
    end

    # 2. counting sort by cell: start[c] = first sorted index of cell c
    start[1] = 1
    for c in 1:nc * nc
        start[c + 1] += start[c]
    end
    cursor = start[1:end-1]
    perm = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        c = cell[i]
        perm[cursor[c]] = i
        cursor[c] += 1
    end

    # 3. gather positions into sorted order, so each cell is contiguous in memory
    xs = similar(x)
    @inbounds for k in 1:N
        xs[1, k] = x[1, perm[k]]
        xs[2, k] = x[2, perm[k]]
    end

    # 4. density and loss: loop over the 3×3 neighbouring cells; each cell is a
    #    contiguous range start[c]:start[c+1]-1 whose length depends on the data
    loss = zero(T)
    @inbounds for k in 1:N
        xk, yk = xs[1, k], xs[2, k]
        c  = cell[perm[k]] - 1
        cx = c % nc
        cy = c ÷ nc
        ρ = zero(T)
        for dy in -1:1, dx in -1:1
            cn = 1 + mod(cx + dx, nc) + nc * mod(cy + dy, nc)
            for l in start[cn]:start[cn + 1] - 1
                rx = minimg(xk - xs[1, l], p.L)
                ry = minimg(yk - xs[2, l], p.L)
                r2 = rx * rx + ry * ry
                if r2 < rc * rc
                    ρ += p.m * W(sqrt(r2), p.h)
                end
            end
        end
        loss += (ρ - p.ρ0)^2
    end
    return loss
end

function sph_gradient(x::Matrix{T}, p::Params{T}) where {T}
    dx = zero(x)
    Enzyme.autodiff(Reverse, sph_loss, Active, Duplicated(x, dx), Const(p))
    return dx
end

# ---------------------------------------------------------------------------- GPU
# Same algorithm with KernelAbstractions. The cell list (cell ids, sort permutation,
# cell start offsets) is integer-valued, so it is built outside the differentiated
# function; the gather into sorted order and the neighbour loops are differentiated.

using KernelAbstractions, CUDA

# Box size, smoothing length etc. are not differentiated. Saying so lets Params be
# passed to GPU kernels, which do not accept active (differentiable) scalar arguments.
Enzyme.EnzymeRules.inactive_type(::Type{<:Params}) = true

@kernel function cell_ids!(cell, counts, @Const(x), s, nc)
    i = @index(Global)
    @inbounds begin
        cx = mod(unsafe_trunc(Int32, floor(x[1, i] / s)), Int32(nc))
        cy = mod(unsafe_trunc(Int32, floor(x[2, i] / s)), Int32(nc))
        c = Int32(1) + cx + Int32(nc) * cy
        cell[i] = c
        KernelAbstractions.@atomic counts[c + 1] += Int32(1)
    end
end

"Cell list on the GPU: sort permutation, sorted cell ids and cell start offsets."
function build_cells(x::CuMatrix{T}, p::Params{T}) where {T}
    N, nc = size(x, 2), p.nc
    backend = get_backend(x)
    cell = CuVector{Int32}(undef, N)
    counts = CUDA.zeros(Int32, nc * nc + 1)
    cell_ids!(backend)(cell, counts, x, p.L / nc, nc; ndrange=N)
    counts[1:1] .= 1
    start = cumsum(counts)
    perm = sortperm(cell)
    return (; perm=Int32.(perm), cell=cell[perm], start)
end

@kernel function gather!(xs, @Const(x), @Const(perm))
    k = @index(Global)
    @inbounds begin
        j = perm[k]
        xs[1, k] = x[1, j]
        xs[2, k] = x[2, j]
    end
end

@kernel function density_loss!(e, @Const(xs), @Const(cellk), @Const(start), p)
    k = @index(Global)
    T = eltype(xs)
    nc = p.nc
    rc2 = 4 * p.h * p.h
    @inbounds begin
        xk, yk = xs[1, k], xs[2, k]
        c  = cellk[k] - Int32(1)
        cx = c % Int32(nc)
        cy = c ÷ Int32(nc)
        ρ = zero(T)
        for dy in Int32(-1):Int32(1), dx in Int32(-1):Int32(1)
            cn = Int32(1) + mod(cx + dx, Int32(nc)) + Int32(nc) * mod(cy + dy, Int32(nc))
            for l in start[cn]:start[cn + 1] - Int32(1)
                rx = minimg(xk - xs[1, l], p.L)
                ry = minimg(yk - xs[2, l], p.L)
                r2 = rx * rx + ry * ry
                if r2 < rc2
                    ρ += p.m * W(sqrt(r2), p.h)
                end
            end
        end
        e[k] = (ρ - p.ρ0)^2
    end
end

"Per-particle loss terms e (sorted order); the loss is sum(e)."
function sph_terms!(e, xs, x, cells, p)
    backend = get_backend(x)
    N = size(x, 2)
    gather!(backend)(xs, x, cells.perm; ndrange=N)
    density_loss!(backend)(e, xs, cells.cell, cells.start, p; ndrange=N)
    return nothing
end

function sph_loss(x::CuMatrix{T}, p::Params{T}) where {T}
    cells = build_cells(x, p)
    e = similar(x, size(x, 2)); xs = similar(x)
    sph_terms!(e, xs, x, cells, p)
    return sum(e)
end

"Gradient of sum(e): reverse mode with the seed de = 1."
function sph_gradient(x::CuMatrix{T}, p::Params{T}) where {T}
    cells = build_cells(x, p)
    N = size(x, 2)
    e, de = similar(x, N), CUDA.ones(T, N)
    xs, dxs = similar(x), CUDA.zeros(T, size(x))
    dx = CUDA.zeros(T, size(x))
    Enzyme.autodiff(Reverse, sph_terms!, Const,
                    Duplicated(e, de), Duplicated(xs, dxs), Duplicated(x, dx),
                    Const(cells), Const(p))
    return dx
end
