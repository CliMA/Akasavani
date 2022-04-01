using Test
using Akasavani
using BenchmarkTools
using Akasavani.FiniteDifference
using Akasavani.Interpolation
using Plots

# Speed tests

function compute_first_second_fd_derivatives!(df1_comp, df2_comp, cfds, x, f)
    for pt in eachindex(x)
        st, en, at = get_stencil(cfds, pt)
        xpts = view(x, st:en)
        floc = view(f, st:en)
        @inbounds df1_comp[pt] = fd_d1(xpts, floc, at) # first derivative
        @inbounds df2_comp[pt] = fd_d2(xpts, floc, at) # second derivative
    end
end

function test_fd()
    I, FT = Int, Float64

    xmin, xmax = FT(0), FT(1)

    n_vert = 100     # number of vertical levels (finite difference)
    x = Vector{FT}(range(xmin, xmax, length = n_vert + 1)) # vertical mesh
    npts = length(x)
    # function and exact derivatices
    f = sin.(π * x)
    df1_exact = π .* cos.(π .* x)
    df2_exact = -π .* π .* sin.(π .* x)

    df1_comp = zeros(FT, npts)
    df2_comp = zeros(FT, npts)

    # compute second order derivative, central FD
    cfds = CentralFD(4, npts)
    @btime compute_first_second_fd_derivatives!($df1_comp, $df2_comp, $cfds, $x, $f)
    #=
        error1 = abs.(df1_exact .- df1_comp)
        @show maximum(error1[2:end-1])
        plot(x, df1_comp, marker = :circle, label = "numerical")
        plot!(x, df1_exact, label = "exact")
        savefig("first_derivative_comparison.png")

        plot(x, error1, marker = :circle, label = "error in 1st derivative")
        savefig("error_first_derivative.png")
        display(error1')
        println("\n----------------------------")


        error2 = abs.(df2_exact .- df2_comp)
        @show maximum(error2[2:end-1])
        plot(x, df2_comp, marker = :+, label = "numerical")
        plot!(x, df2_exact, label = "exact")
        savefig("second_derivative_comparison.png")

        plot(x, error2, marker = :circle, label = "error in 2nd derivative")
        savefig("error_second_derivative.png")
        display(error2')
        println("\n----------------------------")
    =#
end

function test_interp()
    I, FT = Int, Float64
    xmin, xmax = FT(0), FT(1)
    # source
    nv_src = 50     # number of vertical levels (finite difference)
    x_src = Vector{FT}(range(xmin, xmax, length = nv_src + 1)) # vertical mesh
    npts_src = length(x_src)
    f_src = sin.(π * x_src)

    # destination
    nv_dst = 101     # number of vertical levels (finite difference)
    x_dst = Vector{FT}(range(xmin, xmax, length = nv_dst + 1)) # vertical mesh
    npts_dst = length(x_dst)
    f_dst_exact = sin.(π * x_dst)

    f_dst_comp = zeros(FT, npts_dst)

    x_targ = 0.05
    st, en = 2, 5
    f_targ_comp = interpolate(view(x_src, st:en), view(f_src, st:en), x_targ)
    f_targ_exact = sin(π * x_targ)
    @show (f_targ_comp, f_targ_exact, abs(f_targ_comp - f_targ_exact))

    x_targ = 0.06
    st, en = 2, 5
    f_targ_comp = interpolate(view(x_src, st:en), view(f_src, st:en), x_targ)
    f_targ_exact = sin(π * x_targ)
    @show (f_targ_comp, f_targ_exact, abs(f_targ_comp - f_targ_exact))
end

test_fd()
test_interp()
