using Reactant, Enzyme
Reactant.set_default_backend("gpu")
println(Reactant.devices())
loss(x) = sum(sort(x) .^ 2)
g(x) = Enzyme.gradient(Reverse, loss, x)[1]
x = Reactant.to_rarray(Float32[3, 1, 4, 2])
gc = @compile g(x)
println("Reactant GPU grad = ", Array(gc(x)), "  expected ", 2 .* Float32[3,1,4,2])
@assert Array(gc(x)) ≈ 2 .* Float32[3,1,4,2]
