#*******Interpolation Stencils*******************************************************
# Face to center interpolation
abstract type AbstractIF2C end

struct IF2CCentered{I} <: AbstractIF2C
    ord::I
    nfaces::I
    tail::I
    # inner constructor to restrict paramter value range
    function IF2CCentered(ord::I, nfaces::I) where {I<:Int}
        @assert ord > 0
        ord = (ord % 2 == 1) ? ord + 1 : ord # force even order for central difference
        @assert nfaces ≥ ord # assert mesh size is sufficiently large enough for a given N
        return new{I}(ord, nfaces, div(ord, 2))
    end
end

struct IF2CUpwind{I} <: AbstractIF2C
    ord::I
    nfaces::I
    nups::I
    ndns::I
    # inner constructor to restrict paramter value range
    function IF2CUpwind(ord::I, nfaces::I) where {I<:Int}
        @assert ord > 0
        @assert nfaces ≥ ord # assert mesh size is sufficiently large enough for a given N
        if ord == 1
            nups, ndns = 1, 0
        else
            nups, ndns = ord - 1, 1
        end
        return new{I}(ord, nfaces, nups, ndns)
    end
end

function get_stencil(if2c::IF2CCentered, cloc::Int)
    (; ord, nfaces, tail) = if2c
    if cloc ≥ tail && cloc < nfaces - (tail - 1)
        return (cloc - (tail - 1), cloc + tail)
    elseif cloc < tail
        return (1, 2cloc)
    else
        return (cloc - (nfaces - cloc - 1), nfaces)
    end
end

function get_stencil(if2c::IF2CUpwind, dir::Bool, cloc::Int)
    # direction is true if flow is in forward direction
    (; ord, nfaces, nups, ndns) = if2c
    if dir
        return (max(1, cloc - (nups - 1)), min(nfaces, cloc + ndns))
    else
        return (max(1, cloc - (ndns - 1)), min(nfaces, cloc + nups))
    end
end

# Center to face interpolation
abstract type AbstractIC2F end

struct IC2FCentered{I} <: AbstractIC2F
    ord::I
    ncenter::I
    tail::I
    # inner constructor to restrict paramter value range
    function IC2FCentered(ord::I, ncenter::I) where {I<:Int}
        @assert ord > 0
        ord = (ord % 2 == 1) ? ord + 1 : ord # force even order for central difference
        @assert ncenter ≥ ord # assert mesh size is sufficiently large enough for a given N
        return new{I}(ord, ncenter, div(ord, 2))
    end
end

struct IC2FUpwind{I} <: AbstractIC2F
    ord::I
    ncenter::I
    nups::I
    ndns::I
    # inner constructor to restrict paramter value range
    function IC2FUpwind(ord::I, ncenter::I) where {I<:Int}
        @assert ord > 0
        @assert ncenter ≥ ord # assert mesh size is sufficiently large enough for a given N
        if ord == 1
            nups, ndns = 1, 0
        else
            nups, ndns = ord - 1, 1
        end
        return new{I}(ord, ncenter, nups, ndns)
    end
end

function get_stencil(ic2f::IC2FCentered, floc::Int)
    (; ord, ncenter, tail) = ic2f
    if floc > 1 && floc ≤ ncenter # interior face points
        if floc > tail && floc < ncenter - (tail - 1)
            return (floc - tail, floc + tail - 1)
        elseif floc ≤ tail
            return (1, 2 * (floc - 1))
        else
            return (2floc - ncenter - 1, ncenter)
        end
    else # boundary faces needs to be calculated from boundary conditions
        return (0, 0)
    end
end

function get_stencil(ic2f::IC2FUpwind, dir::Bool, floc::Int)
    # direction is true if flow is in forward direction
    (; ord, ncenter, nups, ndns) = ic2f
    if floc > 1 && floc ≤ ncenter # interior face points
        if dir
            return (max(1, floc - nups), min(ncenter, floc + ndns - 1))
        else
            return (max(1, floc - ndns), min(ncenter, floc + nups - 1))
        end
    else
        return (0, 0)
    end
end
#*******Finite Difference Stencils***************************************************
abstract type AbstractFD end

stencil_max(f::AbstractFD) = f.ord + 1

struct CentralFD{I} <: AbstractFD
    ord::I
    npts::I
    tail::I
    # inner constructor to restrict paramter value range
    function CentralFD(ord::I, npts::I) where {I<:Int}
        @assert ord > 0
        ord = (ord % 2 == 1) ? ord + 1 : ord # force even order for central difference
        @assert npts ≥ ord + 1 # assert mesh size is sufficiently large enough for a given N
        return new{I}(ord, npts, div(ord, 2))
    end
end

struct ForwardFD{I} <: AbstractFD
    order::I
    derivative::I
    npts::I
    # inner constructor to restrict paramter value range
    function ForwardFD(order::I, derivative::I, npts::I) where {I<:Int}
        @assert order > 0
        @assert derivative > 0
        @assert npts ≥ order + 1 # assert mesh size is large enough for a given N
        return new{I}(order, derivative, npts)
    end
end

struct BackwardFD{I} <: AbstractFD
    order::I
    derivative::I
    npts::I
    # inner constructor to restrict paramter value range
    function BackwardFD(order::I, derivative::I, npts::I) where {I<:Int}
        @assert order > 0
        @assert derivative > 0
        @assert npts ≥ order + 1 # assert mesh size is large enough for a given N
        return new{I}(order, derivative, npts)
    end
end

# Central finite difference stencils
function get_stencil(cd::CentralFD, loc::Int)
    (; ord, npts, tail) = cd
    if loc > tail && loc ≤ npts - tail
        return (loc - tail, loc + tail, tail + 1)
    elseif loc ≤ tail
        return (1, ord + 1, loc)
    else
        return (npts - ord, npts, loc - (npts - ord - 1))
    end
end

# Forward finite difference stencils
function get_stencil(fd::ForwardFD, loc::Int)
    (; derivative, order, npts) = fd
    swidth =
        (derivative % 2 == 0) ? min(order + 2, npts - loc + 1) :
        min(order + 1, npts - loc + 1)
    return (loc, loc + swidth - 1, 1)
end

# Backward finite difference stencils
function get_stencil(bd::BackwardFD, loc::Int)
    (; derivative, order) = bd
    swidth = (derivative % 2 == 0) ? min(order + 2, loc) : min(order + 1, loc)
    return (loc - swidth + 1, loc, swidth)
end
