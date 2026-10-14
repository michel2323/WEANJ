"""
Shared helpers for the WEANJ example notebooks (Enzyme.jl and Reactant.jl columns).

The Python package `weanj` (python/src/weanj) mirrors this API for the JAX column, so
all three implementations read the same inputs and write results in the same format.
"""
module WEANJ

using JSON3, NPZ, Statistics, LinearAlgebra, Random

export repo_root, data_path, load_inputs, save_inputs,
       time_median, write_results, save_gradient, fd_check, gpu_busy_mib

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"Absolute path of the repository root."
repo_root() = ROOT

"Path of the shared input file for `example` (e.g. `\"01-particles\"`)."
data_path(example::AbstractString) = joinpath(ROOT, "data", "$example.npz")

"Write the deterministic inputs for `example`; all three implementations load this file."
function save_inputs(example::AbstractString, arrays::AbstractDict{<:AbstractString})
    mkpath(dirname(data_path(example)))
    npzwrite(data_path(example), Dict(String(k) => v for (k, v) in arrays))
    return data_path(example)
end

"Load the shared inputs for `example` as a `Dict{String,Any}` of host arrays."
load_inputs(example::AbstractString) = npzread(data_path(example))

"""
    time_median(f; sync=() -> nothing, warmup=2, samples=10)

Median wall time of `f()` in seconds after `warmup` untimed calls. `sync` is called
after every call inside the timed region (pass `CUDA.synchronize` for GPU work).
"""
function time_median(f; sync=() -> nothing, warmup::Int=2, samples::Int=10)
    for _ in 1:warmup
        f(); sync()
    end
    ts = Vector{Float64}(undef, samples)
    for i in 1:samples
        t0 = time_ns()
        f(); sync()
        ts[i] = (time_ns() - t0) / 1e9
    end
    return median(ts)
end

"""
    write_results(example, impl, device, records)

Write `benchmarks/results/<example>_<impl>_<device>.json`. Each record is a NamedTuple
or Dict with keys `dtype, N, forward_s, grad_s, compile_s, peak_mem_bytes`
(use `nothing` for unavailable values).
"""
function write_results(example, impl, device, records)
    dir = joinpath(ROOT, "benchmarks", "results")
    mkpath(dir)
    path = joinpath(dir, "$(example)_$(impl)_$(device).json")
    open(path, "w") do io
        JSON3.pretty(io, Dict("example" => example, "impl" => impl,
                              "device" => device, "records" => collect(records)))
    end
    return path
end

"""
    save_gradient(example, impl, g)

Store the Float64 correctness-check gradient as `benchmarks/grads/<example>_<impl>.npz`,
which the host page compares across implementations.
"""
function save_gradient(example, impl, g::AbstractArray)
    dir = joinpath(ROOT, "benchmarks", "grads")
    mkpath(dir)
    path = joinpath(dir, "$(example)_$(impl).npz")
    npzwrite(path, Dict("grad" => Array{Float64}(Array(g))))
    return path
end

"""
    fd_check(f, x, g; n=5, h=1e-6, rtol=1e-5, rng=Xoshiro(0))

Compare `n` random components of the gradient `g` of scalar `f` at `x` against central
finite differences. Returns the maximum relative error; throws if it exceeds `rtol`.
"""
function fd_check(f, x::AbstractArray{Float64}, g::AbstractArray; n::Int=5, h=1e-6,
                  rtol=1e-5, rng=Xoshiro(0))
    xh = copy(x)
    gh = Array(g)
    worst = 0.0
    for i in rand(rng, eachindex(xh), n)
        xi = xh[i]
        xh[i] = xi + h; fp = f(xh)
        xh[i] = xi - h; fm = f(xh)
        xh[i] = xi
        fd = (fp - fm) / (2h)
        worst = max(worst, abs(fd - gh[i]) / max(abs(fd), abs(gh[i]), 1e-12))
    end
    worst <= rtol || error("finite-difference check failed: max rel. error $worst > $rtol")
    return worst
end

"GPU memory currently in use by all processes, in MiB (via `nvidia-smi`)."
function gpu_busy_mib()
    out = read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String)
    return parse(Int, strip(first(split(out, '\n'))))
end

end # module
