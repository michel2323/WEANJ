# Problem definition shared by the Enzyme and Reactant notebooks of example 01.
#
# N particles in a periodic square box [0, L)², SPH density with a cubic-spline kernel
# of support radius rc = 2h, and the loss Σᵢ (ρᵢ − ρ₀)². The box grows with N so the
# number density (and hence ~31 neighbours per particle) stays fixed.

using WEANJ, Random

const EXAMPLE = "01-particles"

struct Params{T}
    L::T      # box side
    h::T      # smoothing length; kernel support rc = 2h
    m::T      # particle mass
    ρ0::T     # target density
    nc::Int   # cells per side; cell side L/nc ≥ rc
end

const NUMBER_DENSITY = 10.0   # particles per unit area → π·rc²·10 ≈ 31 neighbours

function Params{T}(N::Integer) where {T}
    h = 0.5
    L = sqrt(N / NUMBER_DENSITY)
    nc = floor(Int, L / (2h))
    nc >= 3 || throw(ArgumentError("N=$N too small: need at least 3 cells per side"))
    return Params{T}(T(L), T(h), T(1), T(NUMBER_DENSITY), nc)
end
Params(N::Integer) = Params{Float64}(N)

Base.convert(::Type{Params{T}}, p::Params) where {T} =
    Params{T}(T(p.L), T(p.h), T(p.m), T(p.ρ0), p.nc)

"2D cubic-spline kernel (Monaghan 1992), C², support q = r/h ∈ [0, 2]."
@inline function W(r, h::T) where {T}
    q = r / h
    σ = T(10) / (T(7) * T(π) * h * h)
    if q < 1
        return σ * (1 - T(1.5) * q * q + T(0.75) * q * q * q)
    elseif q < 2
        t = 2 - q
        return σ * T(0.25) * t * t * t
    else
        return zero(T)
    end
end

"Minimum-image displacement in a periodic box of side L."
@inline minimg(d, L) = d - L * round(d / L)

"Uniform random positions (2×N) for timing runs."
positions(::Type{T}, N; seed=1) where {T} = T.(rand(Xoshiro(seed), 2, N) .* Params(N).L)

"The shared correctness-check input, written once and read by all three notebooks."
const N_CHECK = 1000
function ensure_inputs()
    isfile(data_path(EXAMPLE)) && return load_inputs(EXAMPLE)
    x = positions(Float64, N_CHECK; seed=42)
    save_inputs(EXAMPLE, Dict("x" => x))
    return load_inputs(EXAMPLE)
end
