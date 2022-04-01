module Interpolation
using ..FiniteDifference
using ..Grid

export interpolate,
    interpolate!,
    interpolate_face2center,
    interpolate_face2center!,
    interpolate_center2face!

function interpolate_face2center!(
    fcentered::AbstractVector{FT},
    xface::AbstractVector{FT},
    fface::AbstractVector{FT},
    xcentered::AbstractVector{FT},
    if2c::IF2CCentered,
) where {FT<:AbstractFloat}
    for i in eachindex(fcentered)
        st, en = get_stencil(if2c, i)
        fcentered[i] = interpolate(view(xface, st:en), view(fface, st:en), xcentered[i])
    end
    return nothing
end

function interpolate_face2center!(
    fcent::AbstractVector{FT},
    xface::AbstractVector{FT},
    fface::AbstractVector{FT},
    xcent::AbstractVector{FT},
    if2c::IF2CUpwind,
) where {FT<:AbstractFloat}
    for i in eachindex(xcent)
        direction = (interpolate(view(xface, i:i+1), view(fface, i:i+1), xcent[i]) > 0)
        st, en = get_stencil(if2c, direction, i)
        fcent[i] = interpolate(view(xface, st:en), view(fface, st:en), xcent[i])
    end
    return nothing
end

function interpolate_center2face!(
    fface::AbstractVector{FT},
    xcentered::AbstractVector{FT},
    fcentered::AbstractVector{FT},
    xface::AbstractVector{FT},
    ic2f::IC2FCentered,
) where {FT<:AbstractFloat}
    ncentered = length(xface)
    for i = 2:ncentered-1
        st, en = get_stencil(ic2f, i)
        fface[i] = interpolate(view(xcentered, st:en), view(fcentered, st:en), xface[i])
    end
    return nothing
end

function interpolate_center2face!(
    fface::AbstractVector{FT},
    xcentered::AbstractVector{FT},
    fcentered::AbstractVector{FT},
    xface::AbstractVector{FT},
    ic2f::IC2FUpwind,
) where {FT<:AbstractFloat}
    ncentered = length(xface)
    for i = 2:ncentered-1
        direction =
            (interpolate(view(xcentered, i-1:i), view(fcentered, i-1:i), xface[i]) > 0)
        st, en = get_stencil(ic2f, direction, i)
        fface[i] = interpolate(view(xcentered, st:en), view(fcentered, st:en), xface[i])
    end
    return nothing
end

function interpolate_face2center(
    fvar::AbstractArray{FT,3},
    grid,
    if2c::AbstractIF2C,
) where {FT}
    nldof, nvert, nelems = size(fvar)
    cvar = Array{FT}(undef, nldof, nvert - 1, nelems)
    interpolate_face2center!(cvar, fvar, grid, if2c)
    return cvar
end

function interpolate_face2center!(
    cvar::AbstractArray{FT,3},
    fvar::AbstractArray{FT,3},
    grid,
    if2c::AbstractIF2C,
) where {FT}
    nldof, nfaces, nelems = size(fvar)
    @assert size(cvar) == (nldof, nfaces - 1, nelems)
    for ldof = 1:nldof, elem = 1:nelems
        fc = view(cvar, ldof, :, elem)
        ff = view(fvar, ldof, :, elem)
        xc = view(grid.vert_center, ldof, :, elem)
        xf = view(grid.vert_face, ldof, :, elem)
        interpolate_face2center!(fc, xf, ff, xc, if2c)
    end
end

function interpolate_center2face!(
    fvar::AbstractArray{FT,3},
    cvar::AbstractArray{FT,3},
    grid,
    ic2f::AbstractIC2F,
) where {FT}
    nldof, ncenter, nelems = size(cvar)
    @assert size(fvar) == (nldof, ncenter + 1, nelems)
    for ldof = 1:nldof, elem = 1:nelems
        fc = view(cvar, ldof, :, elem)
        ff = view(fvar, ldof, :, elem)
        xc = view(grid.vert_center, ldof, :, elem)
        xf = view(grid.vert_face, ldof, :, elem)
        interpolate_center2face!(ff, xc, fc, xf, ic2f)
    end
end

end
