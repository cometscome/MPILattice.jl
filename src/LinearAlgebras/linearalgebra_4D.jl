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
            prod(C.PN), kernel_4Dexpt!, C.A, A.A, C.PN, t
        )
    end
    set_halo!(C)
end

function kernel_4Dexpt_NC3!(i, C, A, PN, t)
    ix, iy, iz, it = get_4Dindex(i, PN)
    C[:, :, ix, iy, iz, it] = exp3x3(A[:, :, ix, iy, iz, it], t)
end

function kernel_4Dexpt_NC2!(i, C, A, PN, t)
    ix, iy, iz, it = get_4Dindex(i, PN)
    C[:, :, ix, iy, iz, it] = exp2x2(A[:, :, ix, iy, iz, it], t)
end

function kernel_4Dexpt!(i, C, A, PN, t)
    ix, iy, iz, it = get_4Dindex(i, PN)
    C[:, :, ix, iy, iz, it] = expm_pade13(A[:, :, ix, iy, iz, it], t)
end



