using Test
using Akasavani
using Akasavani.FiniteDifference

@testset "second order central finite difference" begin
    FT = Float64
    nvert = 10
    x = Vector{FT}(range(0.0, 1.0, length = nvert + 1)) # vertical mesh
    Δx = x[2] - x[1]
    npts = length(x)
    pt = 5 # testing at an interior point
    # stencil for second order central difference
    cfds = CentralFD(2, npts)
    st, en, at = get_stencil(cfds, pt)
    # first derivative 
    @test fd_d1_coeff(view(x, st:en), at) ≈ [-1.0, 0.0, 1.0] ./ (2 * Δx)
    # second derivative 
    @test fd_d2_coeff(view(x, st:en), at) ≈ [1.0, -2.0, 1.0] ./ (Δx^2)
end

@testset "fourth order central finite difference" begin
    FT = Float64
    nvert = 10
    x = Vector{FT}(range(0.0, 1.0, length = nvert + 1)) # vertical mesh
    Δx = x[2] - x[1]
    npts = length(x)
    pt = 5 # testing at an interior point
    # stencil for second order central difference
    cfds = CentralFD(4, npts)
    st, en, at = get_stencil(cfds, pt)
    # first derivative 
    @test fd_d1_coeff(view(x, st:en), at) ≈ [1.0, -8.0, 0.0, 8.0, -1.0] ./ (12 * Δx)
    # second derivative 
    @test fd_d2_coeff(view(x, st:en), at) ≈ [-1.0, 16.0, -30.0, 16.0, -1.0] ./ (12 * Δx^2)
end

@testset "first order backward finite difference" begin
    FT = Float64
    nvert = 10
    x = Vector{FT}(range(0.0, 1.0, length = nvert + 1)) # vertical mesh
    Δx = x[2] - x[1]
    npts = length(x)
    order = 1
    pt = 5 # testing at an interior point
    # stencil for first derivative
    st, en, at = get_stencil(BackwardFD(order, 1, npts), pt)
    @test fd_d1_coeff(view(x, st:en), at) ≈ [-1.0, 1.0] ./ (Δx)
    # stencil for second derivative
    st, en, at = get_stencil(BackwardFD(order, 2, npts), pt)
    @test fd_d2_coeff(view(x, st:en), at) ≈ [1.0, -2.0, 1.0] ./ (Δx^2)
end

@testset "second order backward finite difference" begin
    FT = Float64
    nvert = 10
    x = Vector{FT}(range(0.0, 1.0, length = nvert + 1)) # vertical mesh
    Δx = x[2] - x[1]
    npts = length(x)
    order = 2
    pt = 5 # testing at an interior point
    # stencil for first derivative
    st, en, at = get_stencil(BackwardFD(order, 1, npts), pt)
    @test fd_d1_coeff(view(x, st:en), at) ≈ [1.0, -4.0, 3.0] ./ (2 * Δx)
    # stencil for second derivative
    st, en, at = get_stencil(BackwardFD(order, 2, npts), pt)
    @test fd_d2_coeff(view(x, st:en), at) ≈ [-1.0, 4.0, -5.0, 2.0] ./ (Δx^2)
end

@testset "first order forward finite difference" begin
    FT = Float64
    nvert = 10
    x = Vector{FT}(range(0.0, 1.0, length = nvert + 1)) # vertical mesh
    Δx = x[2] - x[1]
    npts = length(x)
    order = 1
    pt = 5 # testing at an interior point
    # stencil for first derivative
    st, en, at = get_stencil(ForwardFD(order, 1, npts), pt)
    @test fd_d1_coeff(view(x, st:en), at) ≈ [-1.0, 1.0] ./ (Δx)
    # stencil for second derivative
    st, en, at = get_stencil(ForwardFD(order, 2, npts), pt)
    @test fd_d2_coeff(view(x, st:en), at) ≈ [1.0, -2.0, 1.0] ./ (Δx^2)
end

@testset "second order forward finite difference" begin
    FT = Float64
    nvert = 10
    x = Vector{FT}(range(0.0, 1.0, length = nvert + 1)) # vertical mesh
    Δx = x[2] - x[1]
    npts = length(x)
    order = 2
    pt = 5 # testing at an interior point
    # stencil for first derivative
    st, en, at = get_stencil(ForwardFD(order, 1, npts), pt)
    @test fd_d1_coeff(view(x, st:en), at) ≈ [-3.0, 4.0, -1.0] ./ (2 * Δx)
    # stencil for second derivative
    st, en, at = get_stencil(ForwardFD(order, 2, npts), pt)
    @test fd_d2_coeff(view(x, st:en), at) ≈ [2.0, -5.0, 4.0, -1.0] ./ (Δx^2)
end
