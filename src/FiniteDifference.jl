module FiniteDifference

export interpolate,
    fd_d1_coeff!,
    fd_d1_coeff,
    fd_d1,
    fd_d2_coeff!,
    fd_d2_coeff,
    fd_d2,
    AbstractFD,
    CentralFD,
    ForwardFD,
    BackwardFD,
    FDStencil,
    stencil_max,
    get_stencil,
    AbstractIF2C,
    IF2CCentered,
    IF2CUpwind,
    AbstractIC2F,
    IC2FCentered,
    IC2FUpwind

include("stencils.jl")


"""
    interpolate(xpts, floc, x)

Compute the first derivative of discrete function `floc` at point `at`.
Reference:
1). Berrut, J., and Trefethen, L. N., 2004. Barycentric Lagrange Interpolation, SIAM REVIEW
Vol. 46, No. 3, pp. 501–517.
DOI. 10.1137/S0036144502417715
"""
Base.@pure function interpolate(
    xpts::AbstractArray{FT,1},
    floc::AbstractArray{FT,1},
    x::FT,
) where {FT<:AbstractFloat}
    l, tot, mi, tol = FT(1), -FT(0), 0, (10 * eps(FT))
    for i in eachindex(xpts)
        dx = x - xpts[i]
        if abs(dx) < tol
            mi = i
            break
        else
            l *= dx
        end
    end
    if mi == 0
        for i in eachindex(xpts)
            xi = xpts[i]
            wi = FT(1)
            for j in eachindex(xpts)
                if j ≠ i
                    wi *= (xi - xpts[j])
                end
            end
            tot += floc[i] * l / ((x - xi) * wi)
        end
    else
        tot = floc[mi]
    end
    return tot
end

# Computation of finite-difference derivatives on a given stencil
# Reference:
#1). WELFERT, B. D. 1997. Generation of pseudospectral differentiation matrices I. SIAM J.
#   Numer. Anal. 34, 4, 1640–1657
#2). A MATLAB Differentiation Matrix Suite
#    J. A. C. WEIDEMAN & S. C. REDDY
#ACM Transactions on Mathematical Software, Vol. 26, No. 4, December 2000, Pages 465–51

#*********************************
#*********************************
#   First derivative
#*********************************
"""
    fd_d1_coeff!(coeffd1, xpts, at)

Compute the coefficients for the first derivative on FD stencil `xpts`, at the point `at`.
The derivative of discrete function `floc` can be computed as `sum(floc .* coeffd1)`.
Here, `xpts` is the view to local finite difference stencil, and `floc` is the view
of discrete function `f` on the stencil `xpts`.
"""
function fd_d1_coeff!(
    coeffd1::AbstractArray{FT,1},
    xpts::AbstractArray{FT,1},
    at::Int,
) where {FT<:AbstractFloat}
    c_at = compute_cvec(xpts, at)
    @inbounds coeffd1[at] = 0
    for i in eachindex(xpts)
        if i ≠ at
            c_i = compute_cvec(xpts, i)
            @inbounds coeffd1[i] = c_at / (c_i * (xpts[at] - xpts[i]))
            @inbounds coeffd1[at] -= coeffd1[i]
        end
    end
    return nothing
end

function fd_d1_coeff(xpts::AbstractArray{FT,1}, at::Int) where {FT<:AbstractFloat}
    coeffd1 = similar(xpts)
    c_at = compute_cvec(xpts, at)
    @inbounds coeffd1[at] = 0
    for i in eachindex(xpts)
        if i ≠ at
            c_i = compute_cvec(xpts, i)
            @inbounds coeffd1[i] = c_at / (c_i * (xpts[at] - xpts[i]))
            @inbounds coeffd1[at] -= coeffd1[i]
        end
    end
    return coeffd1
end

"""
    fd_d1_coeff_at(xpts, at, c_at)

Compute `at`th (diagonal) coefficient for the first derivative on grid `x`, at the point `at`.
"""
function fd_d1_coeff_at(
    xpts::AbstractArray{FT,1},
    at::Int,
    c_at::FT,
) where {FT<:AbstractFloat}
    coeffd1_at = -FT(0)
    for i in eachindex(xpts)
        if i ≠ at
            coeffd1_at -= c_at / (compute_cvec(xpts, i) * (xpts[at] - xpts[i]))
        end
    end
    return coeffd1_at
end

"""
    fd_d1_coeff(xpts, at, i)

Compute `i`th coefficient for the first derivative on grid `x`, at the point `at`.
"""
function fd_d1_coeff(
    xpts::AbstractArray{FT,1},
    at::Int,
    i::Int,
    c_at::FT,
    c_i::FT,
) where {FT<:AbstractFloat}
    return @inbounds (i ≠ at) ? FT(c_at / (c_i * (xpts[at] - xpts[i]))) :
                     fd_d1_coeff_at(xpts, at, c_at)
end
"""
    fd_d1(xpts, floc, at)

Compute the first derivative of discrete function `floc` at point `at`.
"""
Base.@pure function fd_d1(
    xpts::AbstractArray{FT,1},
    floc::AbstractArray{FT,1},
    at::Int,
) where {FT<:AbstractFloat}
    c_at = compute_cvec(xpts, at)
    tot = -FT(0)
    for i in eachindex(xpts)
        if i ≠ at
            c_i = compute_cvec(xpts, i)
            @inbounds tot += (floc[i] - floc[at]) * c_at / (c_i * (xpts[at] - xpts[i]))
        end
    end
    return tot
end
#*********************************
#   Second derivative
#*********************************
"""
    fd_d2_coeff!(coeffd2, xpts, at)

Compute the coefficients for the second derivative on grid `x`, at the specified by
`at`.
"""
function fd_d2_coeff!(
    coeffd2::AbstractArray{FT,1},
    xpts::AbstractArray{FT,1},
    at::Int,
) where {FT<:AbstractFloat}
    npts = length(xpts)
    coeffd1 = zeros(FT, npts)
    fd_d1_coeff!(coeffd1, xpts, at)
    fd_d2_coeff!(coeffd2, coeffd1, xpts, at)
    return nothing
end

function fd_d2_coeff(xpts::AbstractArray{FT,1}, at::Int) where {FT<:AbstractFloat}
    coeffd2 = similar(xpts)
    fd_d2_coeff!(coeffd2, xpts, at)
    return coeffd2
end

function fd_d2_coeff!(
    coeffd2::AbstractArray{FT,1},
    coeffd1::AbstractArray{FT,1},
    xpts::AbstractArray{FT,1},
    at::Int,
) where {FT<:AbstractFloat}
    c_at = compute_cvec(xpts, at)
    @inbounds coeffd2[at] = 0
    for i in eachindex(xpts)
        if i ≠ at
            c_i = compute_cvec(xpts, i)
            @inbounds coeffd2[i] =
                (FT(2) / (xpts[at] - xpts[i])) * ((c_at / c_i) * coeffd1[at] - coeffd1[i])
            @inbounds coeffd2[at] -= coeffd2[i]
        end
    end
    return nothing
end

function fd_d2(
    coeffd2::AbstractArray{FT,1},
    coeffd1::AbstractArray{FT,1},
    xpts::AbstractArray{FT,1},
    floc::AbstractArray{FT,1},
    at::Int,
) where {FT<:AbstractFloat}
    fd_d2_coeff!(coeffd2, coeffd1, xpts, at)
    tot = -FT(0)
    for i in eachindex(floc)
        tot += coeffd2[i] * floc[i]
    end
    return tot
end

Base.@pure function fd_d2(
    xpts::AbstractArray{FT,1},
    floc::AbstractArray{FT,1},
    at::Int,
) where {FT<:AbstractFloat}
    c_at = compute_cvec(xpts, at)
    coeffd1_at = fd_d1_coeff_at(xpts, at, c_at)
    tot = -FT(0)
    for i in eachindex(floc)
        if i ≠ at
            c_i = compute_cvec(xpts, i)
            @inbounds tot +=
                (floc[i] - floc[at]) *
                (FT(2) / (xpts[at] - xpts[i])) *
                ((c_at / c_i) * coeffd1_at - fd_d1_coeff(xpts, at, i, c_at, c_i))
        end
    end
    return tot
end

function fd_d2_coeff_at(
    xpts::AbstractArray{FT,1},
    at::Int,
    c_at::FT,
    coeffd1_at::FT,
) where {FT<:AbstractFloat}
    coeffd2_at = -FT(0)
    for i in eachindex(xpts)
        c_i = compute_cvec(xpts, i)
        @inbounds coeffd2_at -=
            (FT(2) / (xpts[at] - xpts[i])) *
            ((c_at / c_i) * coeffd1_at - fd_d1_coeff(xpts, at, i, c_at, c_i))
    end
end

function fd_d2_coeff(
    xpts::AbstractArray{FT,1},
    at::Int,
    i::Int,
    c_at::FT,
    coeffd1_at::FT,
) where {FT<:AbstractFloat}
    c_i = compute_cvec(xpts, i)
    return @inbounds (i ≠ at) ?
                     (FT(2) / (xpts[at] - xpts[i])) *
                     ((c_at / c_i) * coeffd1_at - fd_d1_coeff(xpts, at, i, c_at, c_i)) :
                     fd_d2_coeff_at(xpts, at, c_at, coeffd1_at)
end

@inline function compute_cvec(xpts::AbstractArray{FT,1}, at::Int) where {FT<:AbstractFloat}
    c, xpts_at = FT(1), xpts[at]
    for i in eachindex(xpts)
        if i ≠ at
            @inbounds c *= (xpts_at - xpts[i])
        end
    end
    return c
end

end
