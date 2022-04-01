module Variables

using DocStringExtensions
using ..Basis
using ..Grid

export HybridPrognostic,
    HybridPrognosticBCType, HybridPrognosticBCVal, HybridDiagnostic, Scratch

"""
    HybridPrognostic{FT,FTA3D,FTA4D}

Prognostic variables for hybrid solver.
"""
struct HybridPrognostic{FT,FTA3D,FTA4D}
    "density, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, nelems)` "
    ρ::FTA3D
    "horizontal momentum, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, hdim, nelems)` "
    ρuₕ::FTA4D
    "vertical momentum, defined on staggered grid, 
    `(nlocaldof horz spectral element, nvert, nelems)` "
    ρuᵥ::FTA3D
    "ρ × energy, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, nelems)` "
    ρe::FTA3D
end

HybridPrognostic(ρ, ρuₕ, ρuᵥ, ρe) =
    HybridPrognostic{eltype(ρ),typeof(ρ),typeof(ρuₕ)}(ρ, ρuₕ, ρuᵥ, ρe)

HybridPrognostic(grid::HybridGrid) = HybridPrognostic(
    Grid.center_field(grid), # ρ
    Grid.horz_center_field(grid), #ρuₕ
    Grid.face_field(grid), #ρuᵥ
    Grid.center_field(grid), # ρe
)

struct HybridPrognosticBCType{UI,UIA3D,UIA4D}
    "boundary condition type for horizontal momentum, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, hdim, nelems)` "
    ρuₕ::UIA4D
    "boundary condition type for vertical momentum, defined on staggered grid, 
    `(nlocaldof horz spectral element, nvert, nelems)` "
    ρuᵥ::UIA3D
    "boundary condition type for ρ × energy, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, nelems)` "
    ρe::UIA3D
end

HybridPrognosticBCType(ρuₕ_bctype, ρuᵥ_bctype, ρe_bctype) =
    HybridPrognosticBCType{eltype(ρuₕ_bctype),typeof(ρuᵥ_bctype),typeof(ρuₕ_bctype)}(
        ρuₕ_bctype,
        ρuᵥ_bctype,
        ρe_bctype,
    )

HybridPrognosticBCType(grid::HybridGrid, ::Type{T} = UInt8) where {T} =
    HybridPrognosticBCType(
        Grid.horz_center_field(grid, T) .* zero(T), #ρuₕ
        Grid.face_field(grid, T) .* zero(T), #ρuᵥ
        Grid.center_field(grid, T) .* zero(T), # ρe
    )

struct HybridPrognosticBCVal{FT,FTA3D,FTA4D}
    "boundary condition value for horizontal momentum, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, hdim, nelems)` "
    ρuₕ::FTA4D
    "boundary condition value for vertical momentum, defined on staggered grid, 
    `(nlocaldof horz spectral element, nvert, nelems)` "
    ρuᵥ::FTA3D
    "boundary condition value for ρ × energy, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, nelems)` "
    ρe::FTA3D
end

HybridPrognosticBCVal(ρuₕ_bcval, ρuᵥ_bcval, ρe_bcval) =
    HybridPrognosticBCVal{eltype(ρuᵥ_bcval),typeof(ρuᵥ_bcval),typeof(ρuₕ_bcval)}(
        ρuₕ_bcval,
        ρuᵥ_bcval,
        ρe_bcval,
    )

HybridPrognosticBCVal(grid::HybridGrid{FT}) where {FT} = HybridPrognosticBCVal(
    Grid.horz_center_field(grid, FT) .* zero(FT), #ρuₕ
    Grid.face_field(grid, FT) .* zero(FT), #ρuᵥ
    Grid.center_field(grid, FT) .* zero(FT), # ρe
)

"""
    HybridDiagnostic{FT,FTA3D}

Diagnostic variables for hybrid solver.
"""
struct HybridDiagnostic{FT,FTA3D}
    "pressure, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, nelems)` "
    pres::FTA3D
    "temperature, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, nelems)` "
    temp::FTA3D
    "potential temperature, defined on center grid, 
    `(nlocaldof horz spectral element, nvert-1, nelems)` "
    θ::FTA3D
end

HybridDiagnostic(pres, temp, θ) = HybridDiagnostic{eltype(pres),typeof(pres)}(pres, temp, θ)

function HybridDiagnostic(grid::HybridGrid)
    pres = Grid.center_field(grid)
    temp = Grid.center_field(grid)
    θ = Grid.center_field(grid)
    return HybridDiagnostic(pres, temp, θ)
end

"""
    Scratch{FT,FTA4D}

Scratch arrays for computations

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Scratch{FT<:AbstractFloat,FTA4D<:AbstractArray{FT,4}}
    "Scratch arrays on grid 1, element center"
    scrg1c::FTA4D
    "Scratch arrays on grid 1, element face"
    scrg1f::FTA4D
    "Scratch arrays on grid 2, element center"
    scrg2c::FTA4D
    "Scratch arrays on grid 2, element face"
    scrg2f::FTA4D
end

Scratch(scrg1c, scrg1f, scrg2c, scrg2f) =
    Scratch{eltype(scrg1c),typeof(scrg1c)}(scrg1c, scrg1f, scrg2c, scrg2f)

function Scratch(b1d::Basis1D, grid::HybridGrid{FT}) where {FT}
    nfaces = Grid.nfaces(grid)
    ncenter = Grid.ncenter(grid)
    nelems = Grid.nelems(grid)
    nscr = (4, 4, 4, 4)
    return Scratch(
        Array{FT}(undef, b1d.qg1, ncenter, nelems, nscr[1]), # center
        Array{FT}(undef, b1d.qg1, nfaces, nelems, nscr[2]),  # face
        Array{FT}(undef, b1d.qg2, ncenter, nelems, nscr[3]), # center
        Array{FT}(undef, b1d.qg2, nfaces, nelems, nscr[4]),  # face
    )
end

end
