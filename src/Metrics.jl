module Metrics

export AbstractIJac, IJac1D

abstract type AbstractIJac end

struct IJac1D{FT,FTA2D} <: AbstractIJac
    ∂ξ₁∂x₁::FTA2D
    jac::FTA2D
end

function IJac1D(qg, nvert, nelems, ::Type{FT}) where {FT}
    ∂ξ₁∂x₁ = Array{FT}(undef, qg, nelems)
    jac = Array{FT}(undef, qg, nelems)
    return IJac1D{FT,typeof(jac)}(∂ξ₁∂x₁, jac)
end

end
