module Operators
using ..FastTensorProduct
using ..FiniteDifference
using ..Interpolation
using ..BCs

export advect_horizontal!, FluxForm, AdvectiveContravariant, AdvectiveCovariant

struct FluxForm end
struct AdvectiveContravariant end
struct AdvectiveCovariant end


function advect2dh!(vout_h, c_h, c_v, u_h, u_v, scratch, ::FluxForm)
    u_v_c = view(scratch.scrg1c, :, :, :, 1) # scratch space
end

end
