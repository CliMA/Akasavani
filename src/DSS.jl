module DSS
using DocStringExtensions

export AbstractDSS, DSS1D, dss2dhybrid!

abstract type AbstractDSS end

struct DSS1D{IA1D} <: AbstractDSS
    vertex_data::IA1D
    vertex_offset::IA1D
    vertex_ldof::IA1D
end

DSS1D(vertex_data, vertex_offset, vertex_ldof) =
    DSS1D{typeof(vertex_data)}(vertex_data, vertex_offset, vertex_ldof)

function dss2dhybrid!(vars::FTA3D, dss_info::DSS1D) where {FT,FTA3D<:AbstractArray{FT,3}}
    data = dss_info.vertex_data
    offset = dss_info.vertex_offset
    vertex_ldof = dss_info.vertex_ldof

    nlevels = size(vars, 2)
    nelem = size(vars, 3)
    nvert = length(offset) - 1

    for lev = 1:nlevels
        for vt = 1:nvert
            st = offset[vt]
            el1, el2 = data[st], data[st+2]
            ldof1, ldof2 = vertex_ldof[data[st+1:2:st+3]]
            tot = vars[ldof1, lev, el1] + vars[ldof2, lev, el2]
            vars[ldof1, lev, el1] = tot
            vars[ldof2, lev, el2] = tot
        end
    end
    return nothing
end

end
