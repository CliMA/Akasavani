using Test
using Akasavani
using Akasavani.Basis

FT = Float64
poly_order = 5

ξ, ω = lglpoints(FT, poly_order)

D = Basis.spectralderivative(ξ)

@test isapprox(
    D * ones(poly_order + 1, 1),
    zeros(FT, poly_order + 1, 1),
    atol = eps(FT) * 100,
)
