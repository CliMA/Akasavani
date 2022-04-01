module Grid

using DocStringExtensions

export HybridGrid

"""
    HybridGrid{FT,FTA3D}

Grid data for hybrid solver.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct HybridGrid{FT,FTA3D,FTA4D}
    "Horizontal grid `(nlocaldof horz spectral element, hor_dim, nelems)`"
    horz::FTA3D
    "vertical center grid `(nlocaldof horizontal spectral element, ncenter, nelems)`"
    vert_center::FTA3D
    "vertical face grid `(nlocaldof horizontal spectral element, nfaces, nelems)`"
    vert_face::FTA3D
    "bottom and top outward facing normal for center grid `(nlocaldof horz spectral element, ndims, 2, nelems)`"
    fd_normal_center::FTA4D
    "bottom and top outward facing normal for face grid `(nlocaldof horz spectral element, ndims, 2, nelems)`"
    fd_normal_face::FTA4D
end

HybridGrid(horz, vert_center, vert_face, fd_normal_center, fd_normal_face) =
    HybridGrid{eltype(horz),typeof(horz),typeof(fd_normal_center)}(
        horz,
        vert_center,
        vert_face,
        fd_normal_center,
        fd_normal_face,
    )

function HybridGrid(nlocaldof, hor_dim, nfaces, nelems, ::Type{FT}) where {FT}
    ncenter = nfaces - 1
    horz = Array{FT}(undef, nlocaldof, hor_dim, nelems)
    vert_center = Array{FT}(undef, nlocaldof, ncenter, nelems)
    vert_face = Array{FT}(undef, nlocaldof, nfaces, nelems)
    fd_normal_center = Array{FT}(undef, nlocaldof, hor_dim + 1, 2, nelems)
    fd_normal_face = Array{FT}(undef, nlocaldof, hor_dim + 1, 2, nelems)
    return HybridGrid(horz, vert_center, vert_face, fd_normal_center, fd_normal_face)
end

nelems(hgrid::HybridGrid) = size(hgrid.horz, 3)
nfaces(hgrid::HybridGrid) = size(hgrid.vert_face, 2)
ncenter(hgrid::HybridGrid) = size(hgrid.vert_center, 2)
nlocaldof(hgrid::HybridGrid) = size(hgrid.horz, 1)
horzdim(hgrid::HybridGrid) = size(hgrid.horz, 2)

face_field(hgrid::HybridGrid{FT}, ::Type{T} = FT) where {FT,T} = similar(hgrid.vert_face, T)
center_field(hgrid::HybridGrid{FT}, ::Type{T} = FT) where {FT,T} =
    similar(hgrid.vert_center, T)

horz_center_field(hgrid::HybridGrid{FT}, ::Type{T} = FT) where {FT,T} = similar(
    hgrid.vert_center,
    T,
    size(hgrid.vert_center, 1),
    size(hgrid.vert_center, 2),
    size(hgrid.horz, 2),
    size(hgrid.vert_center, 3),
)

#struct LocalSEIndices{I,IA1D,IA2D}
#    vertex::IA1D
#    face::IA2D
#end

#function LocalIndices(grid::HybridGrid)
#    horz = grid.horz
#    hdim = size(horz, 2)
#    nldof = size(horz, 1)
#end

end
