using CUDA, KernelAbstractions, Enzyme
CUDA.versioninfo()
@kernel function square!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i]^2
end
function loss(x)
    y = similar(x)
    square!(get_backend(x))(y, x; ndrange=length(x))
    KernelAbstractions.synchronize(get_backend(x))
    return sum(y)
end
# Loss through a KA kernel, reduced via sum: reverse-mode through the kernel launch.
function f!(y, x)
    square!(get_backend(x))(y, x; ndrange=length(x))
    return nothing
end
x  = CuArray(Float32[1, 2, 3, 4]); dx = CUDA.zeros(Float32, 4)
y  = CUDA.zeros(Float32, 4);       dy = CUDA.ones(Float32, 4)
Enzyme.autodiff(Reverse, f!, Const, Duplicated(y, dy), Duplicated(x, dx))
KernelAbstractions.synchronize(CUDABackend())
println("Enzyme+KA CUDA dx = ", Array(dx), "  expected ", 2 .* Float32[1,2,3,4])
@assert Array(dx) ≈ 2 .* Float32[1,2,3,4]
