using GCMMesh.Mesh
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using Akasavani
using Akasavani.Dycore
using Akasavani.Export
using Akasavani.DSS
using Akasavani.Thermodynamics
using Akasavani.Grid
using Akasavani.FiniteDifference
using Akasavani.BCs

using BenchmarkTools

FT = Float64
# constants / parameters
const p_0 = CLIMAParameters.Planet.MSLP(param_set) # mean sea level pressure
const grav = CLIMAParameters.Planet.grav(param_set) # gravitational constant
const R_d = CLIMAParameters.Planet.R_d(param_set) #287.058 # R dry (gas constant / mol mass dry air)
const cp_d = CLIMAParameters.Planet.cp_d(param_set) # heat capacity at constant pressure
const cv_d = CLIMAParameters.Planet.cv_d(param_set) # heat capacity at constant volume
const γ = cp_d / cv_d # heat capacity ratio

#function main()
# domain setup
x1min, x1max = FT(0), FT(1000)
x2min, x2max = FT(0), FT(1000)

poly_order = 5 # polynomial order for horizontal spectral elements
n_hor = 10     # number of horizontal spectral elements
n_ver = 50     # number of vertical levels (finite difference)

hmesh = equispaced_line_mesh(x1min, x1max, n_hor, true) # horizontal outer mesh
vmesh = Vector(range(x2min, x2max, length = n_ver + 1)) # vertical mesh

solver = Dycore.init_2d_hybrid_solver(hmesh, vmesh, poly_order)


horz = solver.grid.horz
vert_center = solver.grid.vert_center

nldof = size(solver.grid.horz, 1)
nelems = size(solver.grid.horz, 3)
nvc = size(solver.grid.vert_center, 2)

ρ = solver.state.ρ
ρ .= FT(1)
# setting up reference conditions

# setting up initial conditions
function initialize_state!(solver)
    @unpack ρ, ρuₕ, ρuᵥ, ρe = solver.state
    vert_center = solver.grid.vert_center
    FT = eltype(ρ)
    xc, zc, rc = FT(500), FT(350), FT(250)
    θc = 0.5 # 0.5 K

    θ = solver.diag.θ
    temp = solver.diag.temp
    pres = solver.diag.pres

    π_exner = alloc_var(solver, :center)

    for el = 1:nelems
        for v = 1:nvc
            for i = 1:nldof
                x, z = horz[i, 1, el], vert_center[i, v, el]
                r = sqrt((x - xc)^2 + (z - zc)^2)
                θp = r > rc ? FT(0) : (θc / FT(2)) * (1 + cos(π * r / rc))
                θ[i, v, el] = 300 + θp
                π_exner[i, v, el] = FT(1) - grav * z / (cp_d * θ[i, v, el]) # exner pressure (Hydrostatic + adiabatic)
                temp[i, v, el] = π_exner[i, v, el] * θ[i, v, el] # temperature
                pres[i, v, el] = p_0 * π_exner[i, v, el]^(cp_d / R_d) # pressure
                ρ[i, v, el] = pres[i, v, el] / R_d / temp[i, v, el] # density
            end
        end
    end
    ρuₕ .= FT(0)
    ρuᵥ .= FT(0)
    energy!(ρe, ρ, ρuₕ, ρuᵥ, temp, solver.scratch, solver.grid, param_set)
end

initialize_state!(solver)

# setting up initial conditions
# velocity
solver.state.ρuₕ .= FT(0)
solver.state.ρuᵥ .= FT(0)

# setting up FD (vertical) boundary conditions
bc = solver.state_bc
bcval = solver.state_bcval
ncenter = Grid.ncenter(solver.grid)
nfaces = Grid.nfaces(solver.grid)
# bottom boundary conditions
bc.ρuₕ[:, 1, :, :] .= BCs.bc_type(FreeSlip()) # no flux for velocity
bc.ρuᵥ[:, 1, :, :] .= BCs.bc_type(FreeSlip()) # no flux for velocity
bc.ρe[:, 1, :, :] .= BCs.bc_type(Neumann())  # no flux for energy 
# top boundary conditions
bc.ρuₕ[:, ncenter, :, :] .= BCs.bc_type(FreeSlip()) # no flux for velocity
bc.ρuᵥ[:, nfaces, :, :] .= BCs.bc_type(FreeSlip()) # no flux for velocity
bc.ρe[:, ncenter, :, :] .= BCs.bc_type(Neumann())  # no flux for energy 

BCs.enforce_fd_bc!(solver.state, solver.state_bc, solver.state_bcval, solver.grid)

write_vtk("state", solver.vtk_info, solver.grid, solver.state)
