module Dycore

using DocStringExtensions
using GCMMesh.Mesh

using ..Basis
using ..Grid
using ..Export
using ..Variables
using ..DSS
using ..Metrics
using ..Mappings

export init_2d_hybrid_solver, alloc_var

"""
    HybridSolver{M<:HybridGrid,B<:Basis1D,J<:AbstractIJac}

The central solver data structure.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct HybridSolver{
    G<:HybridGrid,
    HP<:HybridPrognostic,
    BC<:HybridPrognosticBCType,
    BCV<:HybridPrognosticBCVal,
    HD<:HybridDiagnostic,
    B<:Basis1D,
    J<:AbstractIJac,
    D<:AbstractDSS,
    V<:Vtk,
    S<:Scratch,
}
    "Hybrid grid"
    grid::G
    "Prognostic variables"
    state::HP
    "Boundary condition type for Prognostic variables"
    state_bc::BC
    "Boundary condition value for Prognostic variables"
    state_bcval::BCV
    "reference state"
    ref_state::HP
    "perturbation from reference state"
    ptrb_state::HP
    "Diagnostic variables"
    diag::HD
    "1D quadrature points, basis functions and derivative matrices"
    b1d::B
    "metric terms on grid 1"
    ijac_g1::J
    "FD indices"
    "direct stiffness summation configuration"
    dss_config::D
    "grid information for writing VTK files"
    vtk_info::V
    "Scratch space"
    scratch::S
end

HybridSolver(
    grid,
    state,
    state_bc,
    state_bcval,
    ref_state,
    ptrb_state,
    diag,
    b1d,
    ijac_g1,
    dss_config,
    vtk_info,
    scratch,
) = HybridSolver{
    typeof(grid),
    typeof(state),
    typeof(state_bc),
    typeof(state_bcval),
    typeof(diag),
    typeof(b1d),
    typeof(ijac_g1),
    typeof(dss_config),
    typeof(vtk_info),
    typeof(scratch),
}(
    grid,
    state,
    state_bc,
    state_bcval,
    ref_state,
    ptrb_state,
    diag,
    b1d,
    ijac_g1,
    dss_config,
    vtk_info,
    scratch,
)

function init_2d_hybrid_solver(hmesh::AbstractMesh{FT}, vmesh, poly_order) where {FT}
    return HybridSolver(Mappings.map_2d_hybrid(hmesh, vmesh, poly_order)...)
end

function alloc_var(slv::HybridSolver, gtype::Symbol)
    @assert gtype ∈ (:center, :face, :horzvel, :vertvel)

    FT = eltype(slv.state.ρ)
    nldof, nvertc, nelems = size(slv.state.ρ)

    if gtype == :center
        return Array{FT}(undef, nldof, nvertc, nelems)
    elseif gtype == :face
        return Array{FT}(undef, nldof, nvertc + 1, nelems)
    elseif gtype == :horzvel
        hor_dim = size(slv.state.ρuₕ, 3)
        return Array{FT}(undef, nldof, nvertc, hor_dim, nelems)
    elseif gtype == :vertvel
        return Array{FT}(undef, nldof, nvertc + 1, nelems)
    end
end

end
