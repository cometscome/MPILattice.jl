function LinearAlgebra.mul!(C::LatticeVector{4,T,AT}, A::LatticeVector{4,T,AT}, B::LatticeVector{4,T,AT}) where {T,AT}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dvector_mul!, C.A, A.A, B.A, C.NC, C.nw, C.dims
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

function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, NC1, NC2, NC3, C.nw, C.dims
    )
    set_halo!(C)
end

function kernel_4Dvector_mul!(i, C, A, B, NC, nw, dims)
    indices = get_4Dindex(i, dims)
    @inbounds for ic = 1:NC
        C[ic, indices...] = A[ic, indices...] * B[ic, indices...]
    end
end

function kernel_4Dmatrix_mul!(i, C, A, B, NC1, NC2, NC3, nw, dims)
    indices = get_4Dindex(i, dims)
    @inbounds for ic = 1:NC1
        for jc = 1:NC2
            C[ic, jc, indices...] = 0
            for kc = 1:NC3
                C[ic, jc, indices...] += A[ic, kc, indices...] * B[kc, jc, indices...]
            end
        end
    end
end