# Problem definition shared by the Enzyme and Reactant notebooks of example 02.
#
# −Δu = θ on the unit square, n×n interior points, u = 0 on the boundary. u is stored
# with its boundary as an (n+2)×(n+2) array. K Gauss–Seidel sweeps from u = 0, and the
# loss ‖u_K − u_obs‖². We differentiate with respect to the source θ.

using WEANJ, Random

const EXAMPLE = "02-gauss-seidel"
const N_CHECK = 32      # grid side for the shared correctness check
const K_CHECK = 20      # sweeps for the correctness check

"Shared correctness-check input: source θ and observation u_obs (both n×n)."
function ensure_inputs()
    isfile(data_path(EXAMPLE)) && return load_inputs(EXAMPLE)
    rng = Xoshiro(42)
    save_inputs(EXAMPLE, Dict("theta" => randn(rng, N_CHECK, N_CHECK),
                              "uobs" => 0.01 .* randn(rng, N_CHECK, N_CHECK)))
    return load_inputs(EXAMPLE)
end

"Random inputs of grid side n for timing runs."
function inputs(::Type{T}, n; seed=1) where {T}
    rng = Xoshiro(seed)
    return T.(randn(rng, n, n)), T.(0.01 .* randn(rng, n, n))
end
