function LinearAlgebra.mul!(C::LatticeVector{4,T,AT}, A::LatticeVector{4,T,AT}, B::LatticeVector{4,T,AT}) where {T,AT}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dvector_mul!, C.A, A.A, B.A, C.NC, C.nw, C.PN
    )
    set_halo!(C)
end

function get_4Dindex(i, dims)
    #i = (((it-1)*dims[3]+iz-1)*dims[2]+iy-1)*dims[1]+ix
    Nx, Ny, Nz, Nt = dims
    o = i - 1                      # zero-based offset
    ix = (o % Nx) + 1
    o ÷= Nx
    iy = (o % Ny) + 1
    o ÷= Ny
    iz = (o % Nz) + 1
    o ÷= Nz
    it = o + 1
    return ix, iy, iz, it
end

struct Mulkernel{NC1,NC2,NC3}
end

function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, NC1, NC2, NC3, C.nw, C.PN
    )
    set_halo!(C)
end

function kernel_4Dvector_mul!(i, C, A, B, NC, nw, PN)
    indices = get_4Dindex(i, PN)
    @inbounds for ic = 1:NC
        C[ic, indices...] = A[ic, indices...] * B[ic, indices...]
    end
end

function kernel_4Dmatrix_mul!(i, C, A, B, NC1, NC2, NC3, nw, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    for ic = 1:NC1
        for jc = 1:NC2
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

function expt!(C::LatticeMatrix{4,T,AT,NC1,NC2}, A::LatticeMatrix{4,T1,AT1,NC1,NC2}, t::S=one(S)) where {T,AT,NC1,NC2,S<:Number,T1,AT1}
    @assert NC1 == NC2 "Matrix exponentiation requires square matrices, but got $(NC1) x $(NC2)."
    if NC1 == 3
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt_NC3!, C.A, A.A, C.PN, t
        )
    elseif NC1 == 2
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt_NC2!, C.A, A.A, C.PN, t
        )
    else
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt!, C.A, A.A, C.PN, t, Val(NC1)
        )
    end
    set_halo!(C)
end

function kernel_4Dexpt_NC3!(i, C, A, PN, t)
    ix, iy, iz, it = get_4Dindex(i, PN)
    a11 = A[1, 1, ix, iy, iz, it]
    a12 = A[1, 2, ix, iy, iz, it]
    a13 = A[1, 3, ix, iy, iz, it]
    a21 = A[2, 1, ix, iy, iz, it]
    a22 = A[2, 2, ix, iy, iz, it]
    a23 = A[2, 3, ix, iy, iz, it]
    a31 = A[3, 1, ix, iy, iz, it]
    a32 = A[3, 2, ix, iy, iz, it]
    a33 = A[3, 3, ix, iy, iz, it]

    c11, c12, c13, c21, c22, c23, c31, c32, c33 = exp3x3_pade(a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
    C[1, 1, ix, iy, iz, it] = c11
    C[1, 2, ix, iy, iz, it] = c12
    C[1, 3, ix, iy, iz, it] = c13
    C[2, 1, ix, iy, iz, it] = c21
    C[2, 2, ix, iy, iz, it] = c22
    C[2, 3, ix, iy, iz, it] = c23
    C[3, 1, ix, iy, iz, it] = c31
    C[3, 2, ix, iy, iz, it] = c32
    C[3, 3, ix, iy, iz, it] = c33

end

function kernel_4Dexpt_NC2!(i, C, A, PN, t)
    ix, iy, iz, it = get_4Dindex(i, PN)
    a11 = A[1, 1, ix, iy, iz, it]
    a21 = A[2, 1, ix, iy, iz, it]
    a12 = A[1, 2, ix, iy, iz, it]
    a22 = A[2, 2, ix, iy, iz, it]
    c11, c12, c21, c22 = exp2x2_elem(a11, a12, a21, a22, t)

    C[1, 1, ix, iy, iz, it] = c11
    C[1, 2, ix, iy, iz, it] = c12
    C[2, 1, ix, iy, iz, it] = c21
    C[2, 2, ix, iy, iz, it] = c22
end

function kernel_4Dexpt!(i, C, A, PN, t, ::Val{N}) where {N}
    ix, iy, iz, it = get_4Dindex(i, PN)
    expm_pade13_writeback!(C, A, ix, iy, iz, it, t, Val(N))
    #C[:, :, ix, iy, iz, it] = expm_pade13(A[:, :, ix, iy, iz, it], t)
end




function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, NC1, NC2, NC3, C.nw, C.PN
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, NC1, NC2, NC3, nw, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    for ic = 1:NC1
        for jc = 1:NC2
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end


function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, Adjoint_Lattice{B::LatticeMatrix{4,T3,AT3,NC3,NC2}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, NC1, NC2, NC3, C.nw, C.PN
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_ABdag!(i, C, A, B, NC1, NC2, NC3, nw, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    for ic = 1:NC1
        for jc = 1:NC2
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end