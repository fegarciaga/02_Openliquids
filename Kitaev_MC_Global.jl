using QuantumToolbox
using LinearAlgebra
using DelimitedFiles

function Commutator(A, B)
    return A * B - B * A
end

function Single_term(H, J, ind1, ind2, slist)
    H += -J * slist[ind1] * slist[ind2]
    return H
end

function Single_Heisenberg(H, Jxy, Jz, ind1, ind2, sx_list, sy_list, sz_list)
    H += Jz * sz_list[ind1]*sz_list[ind2]
    H += Jxy * sx_list[ind1]*sz_list[ind2]
    H += Jxy * sy_list[ind2]*sy_list[ind2]
    return H
end

function Build_H(N, hz, Jxy, Jz, θ, sx_list, sy_list, sz_list)
    H = 0
    for n in 1:N
        H += -hz * sz_list[n]
        H += -hz * sx_list[n]
        H += -hz * sy_list[n]
    end

    # Kitaev like terms
    Jkxy = Jxy*cos(θ)
    Jkz = Jz*cos(θ)
    H = Single_term(H, Jkxy, 1, 2, sx_list)
    H = Single_term(H, Jkxy, 2, 3, sy_list)
    H = Single_term(H, Jkxy, 3, 4, sx_list)
    H = Single_term(H, Jkxy, 4, 5, sy_list)
    H = Single_term(H, Jkz, 5, 6, sz_list)
    H = Single_term(H, Jkxy, 6, 7, sx_list)
    H = Single_term(H, Jkxy, 7, 8, sy_list)
    H = Single_term(H, Jkxy, 8, 9, sx_list)
    H = Single_term(H, Jkxy, 9, 10, sy_list)
    H = Single_term(H, Jkz, 3, 8, sz_list)
    H = Single_term(H, Jkz, 1, 10, sz_list)

    # Heisenberg like terms
    Jhxy = Jxy*sin(θ)
    Jhz = Jz*sin(θ)
    H = Single_Heisenberg(H, Jhxy, Jhz, 1, 2, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 2, 3, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 3, 4, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 4, 5, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 5, 6, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 6, 7, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 7, 8, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 8, 9, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 9, 10, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 3, 8, sx_list, sy_list, sz_list)
    H = Single_Heisenberg(H, Jhxy, Jhz, 1, 10, sx_list, sy_list, sz_list)

    return H
end

function Build_spinops(S)
    D = Int(2 * S + 1)
    Sz = zeros(Complex{Float64}, D, D)
    Sx = zeros(Complex{Float64}, D, D)
    Sy = zeros(Complex{Float64}, D, D)

    for i in 1:D
        Sz[i, i] = -S + i - 1
    end

    for i in 1:D-1
        m = -S + i - 1
        Sx[i, i+1] = 0.5 * sqrt(S * (S + 1) - m * (m + 1))
        Sx[i+1, i] = 0.5 * sqrt(S * (S + 1) - m * (m + 1))
        Sy[i, i+1] = 0.5im * sqrt(S * (S + 1) - m * (m + 1))
        Sy[i+1, i] = -0.5im * sqrt(S * (S + 1) - m * (m + 1))
    end

    return to_sparse(Qobj(Sx)), to_sparse(Qobj(Sy)), to_sparse(Qobj(Sz))
end

function Build_manybody(N, si, sx, sy, sz)
    sx_list = []
    sy_list = []
    sz_list = []

    for n in 1:N
        op_list = fill(si, N)
        op_list[n] = sx
        push!(sx_list, tensor(op_list...))

        op_list[n] = sy
        push!(sy_list, tensor(op_list...))

        op_list[n] = sz
        push!(sz_list, tensor(op_list...))
    end

    return sx_list, sy_list, sz_list
end

function Construct_collapse(H, S, A)
    C = 0
    Aux1 = S
    C += 0.1 * A[1] * S
    for i in 1:4
        Aux2 = Commutator(H, Aux1)
        C += 0.1 * A[i+1] * Aux2
        Aux1 = Aux2
    end
    return C
end

function Integrate(N, hz, Jxy, Jz, θ, A, tlist, spin, NTRAJ)
    D_spin = Int(2 * spin + 1)
    si = to_sparse(qeye(D_spin) |> x -> Qobj(Matrix{ComplexF64}(x)))
    sx, sy, sz = Build_spinops(spin)

    sx_list, sy_list, sz_list = Build_manybody(N, si, sx, sy, sz)

    I = tensor(fill(si, N)...)
    Wp1 = sy_list[1]*sz_list[2]*sx_list[3]*sy_list[8]*sz_list[9]*sx_list[10]*64
    Wp2 = sy_list[3]*sz_list[4]*sx_list[5]*sy_list[6]*sz_list[7]*sx_list[8]*64

    P_plus1 = (I + Wp1) / 2
    P_plus2 = (I + Wp2) / 2

    H = Build_H(N, hz, Jxy, Jz, θ, sx_list, sy_list, sz_list)
    H_b = Build_H(N, 0, Jxy, Jz, θ, sx_list, sy_list, sz_list)

    c_list = []
    SX = sx_list[1]
    SY = sy_list[1]
    SZ = sz_list[1]

    for n in 2:N
        SX += sx_list[n]
        SY += sy_list[n]
        SZ += sz_list[n]
    end

    push!(c_list, Construct_collapse(H_b, SX, A))
    push!(c_list, Construct_collapse(H_b, SY, A))
    push!(c_list, Construct_collapse(H_b, SZ, A))

    theta = π / 2
    phi = 0

    psi_list = []
    for i in 1:N
        if i % 2 == 0
            push!(psi_list, Build_single(theta, phi, sx, sy, sz))
        else
            push!(psi_list, Build_single(π - theta, phi + π, sx, sy, sz))
        end
    end

    psi_init = tensor(psi_list...)

    # now the low rank implementation
    # number of the low rank states
    M = N+1

    ϕ = Vector{QuantumObject{KetQuantumObject,Dimensions{M - 1,NTuple{M - 1,Space}},Vector{ComplexF64}}}(undef, M)
    evals, ekets = eigenstates(H, sparse = true)

    psi_init = ekets[1]

    ϕ[1] = P_plus1*psi_init
    ϕ[1] /=norm(ϕ[1])
    ϕ[1] = P_plus2*ϕ[1]
    ϕ[1] /= norm(ϕ[1])

    sz_list = vcat(sz_list, sy_list, sx_list)

    # Include the correlation effects
    for i in 1:N
        for j in 1:i
            push!(sz_list, sz_list[i]*sz_list[j])
            push!(sz_list, sx_list[i]*sx_list[j])
            push!(sz_list, sy_list[i]*sy_list[j])
        end
    end
    push!(sz_list, Wp1)
    push!(sz_list, Wp2)
    push!(sz_list, H)

    println(expect(Wp1, ϕ[1]))
    println(expect(Wp2, ϕ[1]))

    println("Objects calculated, Initial wavefunction computed")

    sol_lr = mcsolve(H, ϕ[1], tlist, c_list; e_ops = sz_list, ntraj=NTRAJ);

    return sol_lr.expect, sol_lr.states
end

function Build_single(theta, phi, sx, sy, sz)
    Sop = cos(theta) * sz + sin(theta) * cos(phi) * sx + sin(theta) * sin(phi) * sy
    evals, ekets = eigenstates(Sop)
    return ekets[1]
end

function Run(N, hz, Jxy, Jz, θ, A, spin, NTRAJ)
    Neg = zeros(9,4)
    Simulation_time = [5, 12, 50, 125, 500, 1250, 5000, 12500, 50000]
    for i in 1:9
        tlist = range(0, stop = 0.02*Simulation_time[i], length=Simulation_time[i])
        expval, states = Integrate(N, hz, Jxy, Jz, θ, A, tlist, spin, NTRAJ)   
        C_half1 = Vector{Bool}(vcat(ones(Int, 5), zeros(Int, 5)))
        C_half2 = Vector{Bool}(vcat(ones(Int, 3), zeros(Int, 7)))
        C_half3 = Vector{Bool}(vcat(ones(Int, 2), zeros(Int, 8)))
        C_half4 = Vector{Bool}(vcat(ones(Int, 3), zeros(Int, 4), ones(Int, 3)))
        
        ρ = states[1][1]*states[1][1]'

        for i in 1:(NTRAJ-1)
            ρ += states[i+1][1]*states[i+1][1]'
        end
    
        ρ/=NTRAJ
    
        ρ_pt = partial_transpose(ρ, C_half1)
        Neg[i,1] = log(real(tr(sqrtm(ρ_pt'*ρ_pt))))
        ρ_pt = partial_transpose(ρ, C_half2)
        Neg[i,2] = log(real(tr(sqrtm(ρ_pt'*ρ_pt))))
        ρ_pt = partial_transpose(ρ, C_half3)
        Neg[i,3] = log(real(tr(sqrtm(ρ_pt'*ρ_pt))))
        ρ_pt = partial_transpose(ρ, C_half4)
        Neg[i,4] = log(real(tr(sqrtm(ρ_pt'*ρ_pt))))
    end
    
    return expval, Neg
end

N = parse(Int64, ARGS[1])
hz = parse(Float64, ARGS[2])
Jxy = parse(Float64, ARGS[3])
θ = parse(Float64, ARGS[4])
Jz = 1
Spin = 0.5
NTRAJ = 1000

A = readdlm("Coefs.txt")

for i in 1:15
    SZ, Neg = Run(N, hz, Jxy, Jz, θ, A[i+2, :], Spin, NTRAJ)

    outfile1 = "results/S_Global_KSL_$(N)_N_$(i)_T_$(hz)_h_$(Jxy)_Jxy_$(Jz)_Jz_$(θ)_theta.txt"
    outfile2 = "results/NEG__Global_KSL_$(N)_N_$(i)_T_$(hz)_h_$(Jxy)_Jxy_$(Jz)_Jz_$(θ)_theta.txt"

    writedlm(outfile1, real.(SZ))
    writedlm(outfile2, real.(Neg))
end
