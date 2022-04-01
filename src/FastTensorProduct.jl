module FastTensorProduct

using ..Basis

export ftpxv1d!

function ftpxv1d!(
    vout::AbstractArray{FT,2},
    vin::AbstractArray{FT,2},
    b1d::Basis1D{I,FT},
    mesh::Symbol,
    opts::Symbol,
    tropts::Bool,
) where {I<:Int,FT<:AbstractFloat}

    if mesh == :g1
        if opts == :∂ξ₁
            phir = (tropts ? b1d.qg1_Dᵀ : b1d.qg1_D)
        elseif opts == :B
            phir = nothing
        else
            error("ftpxv1d: unsupported option $opts; only :∂ξ₁, :B supported")
        end
    elseif mesh == :g2
        if opts == :∂ξ₁
            phir = (tropts ? b1d.qg2_Dᵀ : b1d.qg2_D)
        elseif opts == :B
            phir = (tropts ? b1d.qg2_Bᵀ : b1d.qg2_B)
        else
            error("ftpxv1d: unsupported option $opts; only :∂ξ₁, :B supported")
        end
    else
        error("unsupported mesh type $mesh, only :g1 and :g2 supported")
    end

    ftpxv1d!(vout, vin, phir)
    return nothing
end

function ftpxv1d!(
    vout::AbstractArray{FT,2},
    vin::AbstractArray{FT,2},
    phir::Union{AbstractArray{FT,2},Nothing},
) where {FT<:AbstractFloat}
    Nel = size(vin, 2)
    sr, si = size(phir)
    rflg = phir isa AbstractArray

    if rflg
        for el = 1:Nel
            for r = 1:sr
                tot = -FT(0)
                for i = 1:si
                    tot += phir[r, i] * vin[i, el]
                end
                vout[r, el] = tot
            end
        end
    else
        for i = 1:length(vin)
            vout[i] = vin[i]
        end
    end
    return nothing
end

end
