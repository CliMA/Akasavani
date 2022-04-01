module Basis
using DocStringExtensions
using LinearAlgebra
using GaussQuadrature

export Basis1D, lglpoints

"""
    Basis1D{FT,FTA1D,FTA2D}

1-dimensional quadrature mesh, weights, derivative matrices and basis functions.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Basis1D{I,FT,FTA1D,FTA2D}
    "number of quadrature points for mesh 1"
    qg1::I
    "quadrature points for mesh 1"
    qg1_ξ::FTA1D
    "quadrature weights for mesh 1"
    qg1_ω::FTA1D
    "1D basis functions for mesh 1"
    qg1_B::FTA2D
    "1D derivative matrix for mesh 1"
    qg1_D::FTA2D
    "transpose of 1D derivative matrix for mesh 1"
    qg1_Dᵀ::FTA2D
    "number of quadrature points for mesh 2"
    qg2::I
    "quadrature points for mesh 2"
    qg2_ξ::FTA1D
    "quadrature weights for mesh 2"
    qg2_ω::FTA1D
    "1D basis functions on mesh 2"
    qg2_B::FTA2D
    "transpose of 1D basis functions on mesh 2"
    qg2_Bᵀ::FTA2D
    "1D derivative matrix on mesh 2"
    qg2_D::FTA2D
    "transpose of 1D derivative matrix for mesh 2"
    qg2_Dᵀ::FTA2D
end

function Basis1D(poly_order, ::Type{FT}) where {FT}
    qg1_ξ, qg1_ω = lglpoints(FT, poly_order)
    qg1_B = Matrix{FT}(I, poly_order + 1, poly_order + 1)
    qg1_D = spectralderivative(qg1_ξ)
    qg1_Dᵀ = Array(transpose(qg1_D))
    qg1 = length(qg1_ξ)

    qg2_poly_order = cld(3 * poly_order, 2)

    qg2_ξ, qg2_ω = lglpoints(FT, qg2_poly_order)
    imat = interpolationmatrix(qg1_ξ, qg2_ω, qg1_ω)
    qg2_B = imat * qg1_B
    qg2_Bᵀ = Array(transpose(qg2_B))
    qg2_D = imat * qg1_D
    qg2_Dᵀ = Array(transpose(qg2_D))
    qg2 = length(qg2_ξ)

    return Basis1D{typeof(qg1),FT,typeof(qg1_ξ),typeof(qg1_D)}(
        qg1,
        qg1_ξ,
        qg1_ω,
        qg1_B,
        qg1_D,
        qg1_Dᵀ,
        qg2,
        qg2_ξ,
        qg2_ω,
        qg2_B,
        qg2_Bᵀ,
        qg2_D,
        qg2_Dᵀ,
    )
end

"""
    lglpoints(::Type{FT}, N::Integer) where FT <: AbstractFloat
returns the points `r` and weights `w` associated with the `N+1`-point
Gauss-Legendre-Lobatto quadrature rule of type `FT`

Reference:
https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Elements.jl
"""
function lglpoints(::Type{FT}, poly_order::Integer) where {FT<:AbstractFloat}
    @assert poly_order ≥ 1
    GaussQuadrature.legendre(FT, poly_order + 1, GaussQuadrature.both)
end

"""
    glpoints(::Type{FT}, N::Integer) where FT <: AbstractFloat
returns the points `r` and weights `w` associated with the `N+1`-point
Gauss-Legendre quadrature rule of type `FT`

Reference:
https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Elements.jl
"""
function glpoints(::Type{FT}, N::Integer) where {FT<:AbstractFloat}
    GaussQuadrature.legendre(FT, N + 1, GaussQuadrature.neither)
end

"""
    baryweights(r)
returns the barycentric weights associated with the array of points `r`
Reference:
  [Berrut2004]
"""
function baryweights(r::AbstractVector{FT}) where {FT}
    Np = length(r)
    wb = ones(FT, Np)

    for j = 1:Np
        for i = 1:Np
            if i != j
                wb[j] = wb[j] * (r[j] - r[i])
            end
        end
        wb[j] = FT(1) / wb[j]
    end
    return wb
end

"""
    spectralderivative(r::AbstractVector{FT},
                       wb=baryweights(r)::AbstractVector{FT}) where FT
returns the spectral differentiation matrix for a polynomial defined on the
points `r` with associated barycentric weights `wb`

Reference:
 - [Berrut2004]

https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Elements.jl
"""
function spectralderivative(
    r::AbstractVector{FT},
    wb = baryweights(r)::AbstractVector{FT},
) where {FT}
    Np = length(r)
    @assert Np == length(wb)
    D = zeros(FT, Np, Np)

    for k = 1:Np
        for j = 1:Np
            if k == j
                for l = 1:Np
                    if l != k
                        D[j, k] = D[j, k] + FT(1) / (r[k] - r[l])
                    end
                end
            else
                D[j, k] = (wb[k] / wb[j]) / (r[j] - r[k])
            end
        end
    end
    return D
end

"""
    interpolationmatrix(rsrc::AbstractVector{FT}, rdst::AbstractVector{FT},
                        wbsrc=baryweights(rsrc)::AbstractVector{FT}) where FT
returns the polynomial interpolation matrix for interpolating between the points
`rsrc` (with associated barycentric weights `wbsrc`) and `rdst`
Reference:
 - [Berrut2004]

https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Numerics/Mesh/Elements.jl
"""
function interpolationmatrix(
    rsrc::AbstractVector{FT},
    rdst::AbstractVector{FT},
    wbsrc = baryweights(rsrc)::AbstractVector{FT},
) where {FT}
    Npdst = length(rdst)
    Npsrc = length(rsrc)
    @assert Npsrc == length(wbsrc)
    I = zeros(FT, Npdst, Npsrc)
    for k = 1:Npdst
        for j = 1:Npsrc
            I[k, j] = wbsrc[j] / (rdst[k] - rsrc[j])
            if !isfinite(I[k, j])
                I[k, :] .= FT(0)
                I[k, j] = FT(1)
                break
            end
        end
        d = sum(I[k, :])
        I[k, :] = I[k, :] / d
    end
    return I
end

end
