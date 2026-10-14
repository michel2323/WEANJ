# Problem definition shared by the Enzyme and Reactant notebooks of example 03.
#
# N particles move with constant velocities on the periodic interval [0, 1). At each of
# T steps, the particles inside a moving window are selected and deposited with linear
# (cloud-in-cell) weights into a histogram of `nb` bins, tapered by a window weight that
# vanishes at the window edges. Loss: ‖H − target‖², gradient w.r.t. initial positions.

using WEANJ, Random

const EXAMPLE = "03-compaction"

struct Setup{T}
    T::Int        # time steps
    dt::T
    halfwidth::T  # window half-width; the window centre moves from 0 to 1
    nb::Int       # histogram bins
end
Setup{T}() where {T} = Setup{T}(50, T(0.01), T(0.005), 1024)
Setup() = Setup{Float64}()

@inline wrap(x) = x - floor(x)                 # periodic position in [0, 1)
@inline minimg(d) = d - round(d)               # periodic displacement in [-1/2, 1/2)
"Window weight: (1 − |d|/w)² inside the window, 0 outside; C¹ at the edges."
@inline taper(d, w) = (a = 1 - abs(d) / w; a > 0 ? a * a : zero(a))

const N_CHECK = 2000
function ensure_inputs()
    isfile(data_path(EXAMPLE)) && return load_inputs(EXAMPLE)
    rng = Xoshiro(42)
    s = Setup()
    save_inputs(EXAMPLE, Dict("x0" => rand(rng, N_CHECK), "v" => 0.5 .* randn(rng, N_CHECK),
                              "target" => 0.01 .* rand(rng, s.nb)))
    return load_inputs(EXAMPLE)
end

"Random inputs with N particles for timing runs."
function inputs(N; seed=1)
    rng = Xoshiro(seed)
    return rand(rng, N), 0.5 .* randn(rng, N), 0.01 .* rand(rng, Setup().nb)
end
