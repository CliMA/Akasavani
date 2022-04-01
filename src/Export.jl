module Export

using DocStringExtensions
using WriteVTK

using ..Grid
using ..Variables
using ..FiniteDifference
using ..Interpolation

export Vtk, init_vtk_2d_hybrid, write_vtk

struct Vtk{I,FT,IA2D,FTA2D}
    cell_type::I
    cells::IA2D
    points::FTA2D
end

Vtk(cell_type, cells, points) =
    Vtk{eltype(cell_type),eltype(points),typeof(cells),typeof(points)}(
        cell_type,
        cells,
        points,
    )

function init_vtk_2d_hybrid(grid::HybridGrid{FT}) where {FT}
    (; horz, vert_center) = grid
    I = Int
    VTK_QUAD = 9
    nelems = size(horz, 3)
    nldof = size(horz, 1)
    nvert_center = size(vert_center, 2)
    npoints = nldof * nvert_center * nelems
    ncells_el = (nldof - 1) * (nvert_center - 1)
    ncells = ncells_el * nelems
    cell_type = VTK_QUAD
    # build points
    points = zeros(FT, 3, npoints)
    pointno = 1
    for el = 1:nelems
        for v = 1:nvert_center
            for h = 1:nldof
                points[1, pointno] = horz[h, 1, el]
                points[2, pointno] = vert_center[h, v, el]
                pointno += 1
            end
        end
    end
    # building cell list
    clist_local = Array{I}(undef, ncells_el, 4)
    lcellno = 1
    for v = 1:nvert_center-1
        for h = 1:nldof-1
            clist_local[lcellno, 1] = h + (v - 1) * nldof
            clist_local[lcellno, 2] = (h + 1) + (v - 1) * nldof
            clist_local[lcellno, 3] = (h + 1) + (v) * nldof
            clist_local[lcellno, 4] = h + (v) * nldof
            lcellno += 1
        end
    end
    cells = Array{I}(undef, ncells, 4)

    for el = 1:nelems
        pt_shift = (el - 1) * nldof * nvert_center
        cl_range = ((el-1)*ncells_el+1):(el*ncells_el)
        cells[cl_range, :] .= clist_local .+ pt_shift
    end
    return Vtk(cell_type, cells, points)
end

function write_vtk(filename, vtk_info::Vtk, grid, vars...)
    (; cell_type, cells, points) = vtk_info
    ncells = size(cells, 1)
    cells_list = [MeshCell(VTKCellTypes.VTK_QUAD, Vector(cells[cl, :])) for cl = 1:ncells]
    v1 = vars[1]
    vtkfile = vtk_grid(filename, points, cells_list)
    vtkfile["Density", VTKPointData()] = v1[:]
    outfile = vtk_save(vtkfile)

    return nothing
end

function write_vtk(filename, vtk_info::Vtk, grid, vars::HybridPrognostic)
    (; cell_type, cells, points) = vtk_info
    ncells = size(cells, 1)
    cells_list = [MeshCell(VTKCellTypes.VTK_QUAD, Vector(cells[cl, :])) for cl = 1:ncells]
    hor_dim = size(vars.ρuₕ, 3)
    nfaces = Grid.nfaces(grid)
    if2c_order = 2 # interpolation order
    if2c = IF2CCentered(if2c_order, nfaces)
    ρuᵥ = interpolate_face2center(vars.ρuᵥ, grid, if2c) # interpolate to elem center

    vtkfile = vtk_grid(filename, points, cells_list)
    vtkfile["ρ", VTKPointData()] = vars.ρ[:]
    for h = 1:hor_dim
        vtkfile["ρu_h"*"$h", VTKPointData()] = vars.ρuₕ[:, :, h, :][:]
    end
    vtkfile["ρuᵥ", VTKPointData()] = ρuᵥ[:]
    vtkfile["ρe", VTKPointData()] = vars.ρe[:]
    outfile = vtk_save(vtkfile)
    return nothing
end

function write_vtk(filename, vtk_info::Vtk, grid, vars::HybridDiagnostic)
    (; cell_type, cells, points) = vtk_info
    ncells = size(cells, 1)
    cells_list = [MeshCell(VTKCellTypes.VTK_QUAD, Vector(cells[cl, :])) for cl = 1:ncells]

    vtkfile = vtk_grid(filename, points, cells_list)
    vtkfile["pres", VTKPointData()] = vars.pres[:]
    vtkfile["temp", VTKPointData()] = vars.temp[:]
    vtkfile["θ", VTKPointData()] = vars.θ[:]
    outfile = vtk_save(vtkfile)
    return nothing
end

function write_vtk(filename, vtk_info::Vtk, grid, varnames, vars...) # write a variable on center grid
    (; cell_type, cells, points) = vtk_info
    ncells = size(cells, 1)
    cells_list = [MeshCell(VTKCellTypes.VTK_QUAD, Vector(cells[cl, :])) for cl = 1:ncells]
    nvars = length(vars)

    vtkfile = vtk_grid(filename, points, cells_list)
    for i = 1:nvars
        vtkfile[varnames[i], VTKPointData()] = vars[i][:]
    end
    outfile = vtk_save(vtkfile)
    return nothing
end

end
