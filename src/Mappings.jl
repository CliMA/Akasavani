module Mappings
using GCMMesh.Mesh
using UnPack

using ..Basis
using ..Grid
using ..Variables
using ..Export
using ..Metrics
using ..FastTensorProduct
using ..DSS

function map_2d_hybrid(hmesh::AbstractMesh{I,FT}, vmesh, poly_order) where {I,FT}
    nldof1 = poly_order + 1
    nfaces = length(vmesh)
    ncenter = nfaces - 1
    nelems = hmesh.nelems
    coords = hmesh.coordinates
    hor_dim = 1

    b1d = Basis1D(poly_order, FT)
    grid = HybridGrid(nldof1, hor_dim, nfaces, nelems, FT) # allocate grid
    scr = Scratch(b1d, grid) # allocating scratch arrays

    ξ = b1d.qg1_ξ
    ijac_g1 = IJac1D(b1d.qg1, nfaces, nelems, FT)
    setup_2d_hybrid_mesh!(grid, hmesh, vmesh, b1d, ijac_g1, scr) # build grid
    # compute metric terms
    ftpxv1d!(ijac_g1.jac, view(grid.horz, :, 1, :), b1d, :g1, :∂ξ₁, false) # compute ∂x₁∂ξ₁
    ijac_g1.∂ξ₁∂x₁ .= FT(1) ./ ijac_g1.jac # compute ∂x₁∂ξ₁

    vtk_info = init_vtk_2d_hybrid(grid) # initialize vtk info

    state = HybridPrognostic(grid) # allocating state variables
    ref_state = HybridPrognostic(grid) # allocating reference state
    ptrb_state = HybridPrognostic(grid) # allocating perturbation from reference state
    diag = HybridDiagnostic(grid) # allocating diagnostic variables

    state_bc = HybridPrognosticBCType(grid, UInt8) # allocating state boundary condition types
    state_bcval = HybridPrognosticBCVal(grid) # allocating state boundary condition values

    dss_config = setup_dss1d(hmesh, b1d.qg1) # setup DSS1D

    return grid,
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
    scr
end

function setup_dss1d(hmesh::Mesh1D{I}, qg1) where {I<:Int}
    @unpack nverts, vertex_neighbors = hmesh
    uverts = hmesh.unique_verts
    nuverts = length(uverts)
    if nverts > nuverts # periodic boundary
        vertex_data = Array{I}(undef, nuverts * 4)
        vertex_offset = Array{I}(1:4:nuverts*4+1)
        st, en = 1, nverts - 1
    else # no periodic boundary
        vertex_data = Array{I}(undef, (nverts - 2) * 4)
        vertex_offset = Array{I}(1:4:(nverts-2)*4+1)
        st, en = 2, nverts - 1
    end
    cntr = 1
    for vt = st:en
        vertex_data[cntr:cntr+3] .= vertex_neighbors[vt, :]
        cntr += 4
    end
    vertex_ldof = [1, qg1]
    return DSS1D(vertex_data, vertex_offset, vertex_ldof)
end

function setup_2d_hybrid_mesh!(
    grid,
    hmesh::AbstractMesh{I,FT},
    vmesh,
    b1d,
    ijac,
    scr,
) where {I,FT}
    (; horz, vert_center, vert_face, fd_normal_center, fd_normal_face) = grid
    (; nelems, elem_verts, coordinates) = hmesh
    (; ∂ξ₁∂x₁) = ijac

    ξ = b1d.qg1_ξ
    nldof1 = b1d.qg1
    nfaces = length(vmesh)
    ncenter = nfaces - 1

    # set up mesh
    for el = 1:nelems
        # Horizontal grid 
        v1, v2 = elem_verts[el, :]
        horz[:, 1, el] .=
            coordinates[v2, 1] * (1 .+ ξ) / FT(2) .+ coordinates[v1, 1] * (1 .- ξ) / FT(2)
        # Vertical grid
        for i = 1:nldof1
            vert_face[i, :, el] .= vmesh # face mesh
            for j = 1:ncenter
                vert_center[i, j, el] = (vmesh[j+1] + vmesh[j]) / FT(2) # center mesh
            end
        end
    end
    # computing normals for bottom and top boundaries
    c_∂xᵥ∂x₁_bot = view(scr.scrg1c, :, 1, :, 1)
    c_∂xᵥ∂x₁_top = view(scr.scrg1c, :, 2, :, 1)
    f_∂xᵥ∂x₁_bot = view(scr.scrg1f, :, 1, :, 1)
    f_∂xᵥ∂x₁_top = view(scr.scrg1f, :, 2, :, 1)
    # bottom
    ftpxv1d!(f_∂xᵥ∂x₁_bot, view(vert_face, :, 1, :), b1d, :g1, :∂ξ₁, false)
    f_∂xᵥ∂x₁_bot .*= ∂ξ₁∂x₁

    ftpxv1d!(c_∂xᵥ∂x₁_bot, view(vert_center, :, 1, :), b1d, :g1, :∂ξ₁, false)
    c_∂xᵥ∂x₁_bot .*= ∂ξ₁∂x₁
    # top
    ftpxv1d!(f_∂xᵥ∂x₁_top, view(vert_face, :, nfaces, :), b1d, :g1, :∂ξ₁, false)
    f_∂xᵥ∂x₁_top .*= ∂ξ₁∂x₁

    ftpxv1d!(c_∂xᵥ∂x₁_top, view(vert_center, :, ncenter, :), b1d, :g1, :∂ξ₁, false)
    c_∂xᵥ∂x₁_top .*= ∂ξ₁∂x₁

    for elem = 1:nelems
        for ldof = 1:nldof1
            # bottom, cell grid
            scale = FT(1) / sqrt(c_∂xᵥ∂x₁_bot[ldof, elem]^2 + 1)
            fd_normal_center[ldof, 1, 1, elem] = scale * c_∂xᵥ∂x₁_bot[ldof, elem]
            fd_normal_center[ldof, 2, 1, elem] = -scale
            # bottom, face grid
            scale = FT(1) / sqrt(f_∂xᵥ∂x₁_bot[ldof, elem]^2 + 1)
            fd_normal_face[ldof, 1, 1, elem] = scale * f_∂xᵥ∂x₁_bot[ldof, elem]
            fd_normal_face[ldof, 2, 1, elem] = -scale
            # top, cell grid
            scale = FT(1) / sqrt(c_∂xᵥ∂x₁_top[ldof, elem]^2 + 1)
            fd_normal_center[ldof, 1, 2, elem] = -scale * c_∂xᵥ∂x₁_top[ldof, elem]
            fd_normal_center[ldof, 2, 2, elem] = scale
            # top, face grid
            scale = FT(1) / sqrt(f_∂xᵥ∂x₁_top[ldof, elem]^2 + 1)
            fd_normal_face[ldof, 1, 2, elem] = -scale * f_∂xᵥ∂x₁_top[ldof, elem]
            fd_normal_face[ldof, 2, 2, elem] = scale
        end
    end
end

end
