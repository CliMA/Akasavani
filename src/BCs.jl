module BCs

using ..FiniteDifference
using ..Grid
using ..Variables

export AbstractBC, Interior, Dirichlet, Neumann, NoSlip, FreeSlip, AbstractBCs

abstract type AbstractBC end

struct Interior <: AbstractBC end

struct Dirichlet <: AbstractBC end

struct Neumann <: AbstractBC end

struct NoSlip <: AbstractBC end

struct FreeSlip <: AbstractBC end

abstract type AbstractBCs end

bc_type(::Interior) = UInt8(0)
bc_type(::Dirichlet) = UInt8(1)
bc_type(::Neumann) = UInt8(2)
bc_type(::NoSlip) = UInt8(3)
bc_type(::FreeSlip) = UInt8(4)


function enforce_fd_bc!(
    state::HybridPrognostic,
    state_bc::HybridPrognosticBCType,
    state_bcval::HybridPrognosticBCVal,
    grid::HybridGrid,
)
    nelems = Grid.nelems(grid)
    nfaces = Grid.nfaces(grid)
    ncenter = Grid.ncenter(grid)
    nldof = Grid.nlocaldof(grid)

@show (nelems, nfaces, ncenter)
    
    for el in 1:nelems
        for ldof in 1:nldof

        end
    end
    return nothing
end


end
