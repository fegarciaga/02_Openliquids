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
    H += Jxy * sx_list[ind1]*sx_list[ind2]
    H += Jxy * sy_list[ind1]*sy_list[ind2]
    return H
end

function Build_H(N, hz, Jxy, Jz, sx_list, sy_list, sz_list)
    H = 0
    for n in 1:N
        H += -hz * sz_list[n]
        H += -hz * sx_list[n]
        H += -hz * sy_list[n]
    end

    # Kitaev like terms
    H = Single_term(H, Jxy, 1, 2, sx_list)
    H = Single_term(H, Jxy, 2, 3, sy_list)
    H = Single_term(H, Jxy, 3, 4, sx_list)
    H = Single_term(H, Jxy, 4, 5, sy_list)
    H = Single_term(H, Jz, 5, 6, sz_list)
    H = Single_term(H, Jxy, 6, 7, sx_list)
    H = Single_term(H, Jxy, 7, 8, sy_list)
    H = Single_term(H, Jxy, 8, 9, sx_list)
    H = Single_term(H, Jxy, 9, 10, sy_list)
    H = Single_term(H, Jz, 1, 10, sz_list)
    H = Single_term(H, Jz, 3, 8, sz_list)
 
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
        C += 0.1 * (-1)^i * A[i+1] * Aux2
        Aux1 = Aux2
    end
    # Elimiminate small elements 
    return C
end

function Integrate(N, hz, Jxy, Jz, A, tlist, spin, NTRAJ)
    SAVE_STRIDE = 20
    saveat = tlist[1:SAVE_STRIDE:end]
    D_spin = Int(2 * spin + 1)
    si = to_sparse(qeye(D_spin) |> x -> Qobj(Matrix{ComplexF64}(x)))
    sx, sy, sz = Build_spinops(spin)

    sx_list, sy_list, sz_list = Build_manybody(N, si, sx, sy, sz)

    I = tensor(fill(si, N)...)
    Wp1 = sy_list[1]*sz_list[2]*sx_list[3]*sy_list[8]*sz_list[9]*sx_list[10]*64
    Wp2 = sy_list[3]*sz_list[4]*sx_list[5]*sy_list[6]*sz_list[7]*sx_list[8]*64

    P_plus1 = (I + Wp1) / 2
    P_plus2 = (I + Wp2) / 2

    H = Build_H(N, hz, Jxy, Jz, sx_list, sy_list, sz_list)

    c_list = []
    for n in 1:N
        push!(c_list, Construct_collapse(H, sx_list[n], A))
    end
    evals, ekets = eigenstates(H, sparse = true)

    psi_init = ekets[1]

    # Include the observables which will be computed a posteriori
    e_ops = []
    push!(e_ops, Wp1)
    push!(e_ops, sz_list[1]*sz_list[10])

    println(expect(Wp2, psi_init))

    println("Objects calculated, Initial wavefunction computed. Starting MC unravelling")

    sol_lr = mcsolve(H, psi_init, tlist, c_list; ntraj=NTRAJ, saveat = saveat);

    return e_ops, sol_lr.states
end

function avg_reduced_dm(states, subsys, NTRAJ)
    ρ_acc = nothing
    for i in 1:NTRAJ
        ρ_red = ptrace(states[i], subsys)

        ρ_mat = Matrix{ComplexF64}(ρ_red.data)

        if ρ_acc === nothing
            ρ_acc = copy(ρ_mat)
        else
            ρ_acc .+= ρ_mat
        end
    end

    ρ_acc ./= NTRAJ
    return ρ_acc
end

function Run(N, hz, Jxy, Jz, A, spin, NTRAJ, filename)
    Simulation_time = 40000
    expval = 0
    tlist = range(0, stop = 0.05*Simulation_time, length=Simulation_time)
    SAVE_STRIDE = 20
    saveat = tlist[1:SAVE_STRIDE:end]
    expops, states= Integrate(N, hz, Jxy, Jz, A, tlist, spin, NTRAJ)
    println(size(states[1][:]))
    # States are being saved every few time steps
    # Observables are computed as postprocessing operation
    Nobs = 2000
    W = zeros(Nobs)
    SzSz = zeros(Nobs)

    for i in 1:Nobs
        for j in 1:NTRAJ
            W[i] += real(expect(expops[1], states[j][i]))
            SzSz[i] += real(expect(expops[2], states[j][i]))
        end
        W[i] /= NTRAJ
        SzSz[i] /= NTRAJ
    end

    println("Done computing observables")

    #save data
    outfilename = "results/W"*filename*".txt"
    writedlm(outfilename, W)
    outfilename = "results/SzSz"*filename*".txt"
    writedlm(outfilename, SzSz)

    # Now, lets compute the reduced density matrices in an optimal fashion
    # Define subsystems
    C_half1 = (1,2,8,9,10)
    C_half2 = (1,2,3,8,9,10)

    # Just compute reduced density matrices for a chosen set of points 
    Rho_times = [1,2,5,10,20,50,100,200,500,1000,2000]
    for (idx, k) in enumerate(Rho_times)
        states_k = [states[j][k] for j in 1:NTRAJ]

        rho1 = avg_reduced_dm(states_k, C_half1, NTRAJ)
        aux_filename = "_"*string(idx)*"_"*filename
        outfileaux = "results/M1"*aux_filename*"_r.txt"
        writedlm(outfileaux, real.(rho1))
        outfileaux = "results/M1"*aux_filename*"_i.txt"
        writedlm(outfileaux, imag.(rho1))

        rho2 = avg_reduced_dm(states_k, C_half2, NTRAJ) 
        aux_filename = "_"*string(idx)*"_"*filename
        outfileaux = "results/M2"*aux_filename*"_r.txt"
        writedlm(outfileaux, real.(rho2))
        outfileaux = "results/M2"*aux_filename*"_i.txt"
        writedlm(outfileaux, imag.(rho2))
    end
    return
end

N = parse(Int64, ARGS[1])
hz = parse(Float64, ARGS[2])
Jxy = parse(Float64, ARGS[3])
Jz = 1
Spin = 0.5
NTRAJ = 1000

A = readdlm("Coefs.txt")

i = 1
filename = "N_$(N)_T_$(i)_h_$(hz)_Jxy_$(Jxy)"
Run(N, hz, Jxy, Jz, A[i+2, :], Spin, NTRAJ, filename)

i = 10 
filename = "N_$(N)_T_$(i)_h_$(hz)_Jxy_$(Jxy)"
Run(N, hz, Jxy, Jz, A[i+2, :], Spin, NTRAJ, filename)
