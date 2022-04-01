using Test
using BenchmarkTools

using Akasavani
using Akasavani.FiniteDifference
using Akasavani.Interpolation
using Plots

FT = Float64

xmin, xmax = FT(0), FT(2 * π)
nfaces = 11
ncenter = nfaces - 1

fmesh = Vector{FT}(range(xmin, xmax, length = nfaces))
cmesh = (fmesh[1:end-1] .+ fmesh[2:end]) .* FT(0.5)

func_exact(mesh) = sin.(mesh)

func_fmesh = func_exact(fmesh) # function on face mesh
func_cmesh = func_exact(cmesh) # function on center mesh

ic2f_fmesh = Vector{FT}(undef, nfaces)
if2c_cmesh = Vector{FT}(undef, ncenter)

order = 2 # interpolation order
#order = 4 # interpolation order
#if2c = IF2CCentered(order, nfaces)
if2c = IF2CUpwind(order, nfaces)
#ic2f = IC2FCentered(order, ncenter)
ic2f = IC2FUpwind(order, ncenter)

@show if2c
@show ic2f

interpolate_face2center!(if2c_cmesh, fmesh, func_fmesh, cmesh, if2c)
interpolate_center2face!(ic2f_fmesh, cmesh, func_cmesh, fmesh, ic2f)

plot(
    cmesh,
    if2c_cmesh,
    marker = :plus,
    label = "interpolated " * string(typeof(if2c).name.name),
)
plot!(cmesh, func_cmesh, marker = :circle, label = "exact")
savefig("face to center interpolation")

plot(
    fmesh,
    ic2f_fmesh,
    marker = :plus,
    label = "interpolated " * string(typeof(ic2f).name.name),
)
plot!(fmesh, func_fmesh, marker = :circle, label = "exact")
savefig("center to face interpolation")
