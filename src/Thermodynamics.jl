module Thermodynamics
using CLIMAParameters

using ..Variables
using ..FiniteDifference
using ..Grid
using ..Interpolation
using ..Dycore

export energy!
"""
    temp_to_θ!(θ, temp)

computes potential temperature, given the temperature
"""
function temp_to_θ!(θ, temp, pres, param_set)
    FT = eltype(temp)
    p₀ = FT(CLIMAParameters.Planet.MSLP(param_set)) # mean sea level pressure
    R_d = FT(CLIMAParameters.Planet.R_d(param_set)) # 287.058 # R dry (gas constant / mol mass dry air)
    cp_d = FT(CLIMAParameters.Planet.cp_d(param_set)) # heat capacity at constant pressure

    θ .= temp .* (p₀ ./ pres) .^ (R_d / cp_d)
    return nothing
end

function energy!(ρe, ρ, ρuₕ, ρuᵥ, temp, scratch, grid, param_set)
    FT = eltype(ρ)
    z = grid.vert_center
    grav = FT(CLIMAParameters.Planet.grav(param_set)) # gravitational constant
    cp_d = FT(CLIMAParameters.Planet.cp_d(param_set)) # heat capacity at constant pressure
    cv_d = FT(CLIMAParameters.Planet.cv_d(param_set)) # heat capacity at constant volume
    R_d = FT(CLIMAParameters.Planet.R_d(param_set)) # 287.058 # R dry (gas constant / mol mass dry air)
    γ = cp_d / cv_d
    nfaces = Grid.nfaces(grid)
    if2c_order = 2 # interpolation
    if2c = IF2CCentered(if2c_order, nfaces)
    # kinetic energy
    hdims = size(ρuₕ, 3) # horizontal dimension
    # interpolate ρuᵥ to cell center grid
    ρuᵥ_center = view(scratch.scrg1c, :, :, :, 1) # 
    interpolate_face2center!(ρuᵥ_center, ρuᵥ, grid, if2c)
    ρe .= ρuᵥ_center .* ρuᵥ_center
    for dim = 1:hdims
        ρe .+= view(ρuₕ, :, :, dim, :) .* view(ρuₕ, :, :, dim, :)
    end
    ρe .*= FT(0.5) ./ ρ
    # potential energy
    ρe .+= ρ .* z .* grav
    # internal energy
    ρe .+= ρ .* temp .* cv_d
end

end
