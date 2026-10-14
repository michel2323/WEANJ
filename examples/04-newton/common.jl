# Problem definition shared by the Enzyme and Reactant notebooks of example 04.
#
# Nonlinear 1D problem F(u, θ) = A u + u³ − θ = 0 with A the Dirichlet Laplacian
# (2u_i − u_{i−1} − u_{i+1})/h². Newton's method until the step is small,
# ‖δ‖∞ ≤ TOL·‖u‖∞; every Newton step solves a tridiagonal system (Thomas algorithm).
# (An absolute residual test cannot work here: rounding in F grows like (n+1)²·eps.) Loss ‖u* − u_obs‖², gradient w.r.t. θ.

using WEANJ, Random

const EXAMPLE = "04-newton"
const N_CHECK = 200
const TOL = 1e-10
const MAXITER = 50

function ensure_inputs()
    isfile(data_path(EXAMPLE)) && return load_inputs(EXAMPLE)
    rng = Xoshiro(42)
    save_inputs(EXAMPLE, Dict("theta" => 50 .* randn(rng, N_CHECK), "uobs" => 0.1 .* randn(rng, N_CHECK)))
    return load_inputs(EXAMPLE)
end

function inputs(n; seed=1)
    rng = Xoshiro(seed)
    return 50 .* randn(rng, n), 0.1 .* randn(rng, n)
end
