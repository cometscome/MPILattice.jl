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

function kernel_4Dvector_mul!(i, C, A, B, NC, nw, PN)
    indices = get_4Dindex(i, PN)
    @inbounds for ic = 1:NC
        C[ic, indices...] = A[ic, indices...] * B[ic, indices...]
    end
end


#C = A B 
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

#C = A B α + C β
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}, α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, α, β
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, α, β) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
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



#C = A'*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

#C = α*A'*B+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end


#C = A*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_ABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C = α* A*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_ABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C = A'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C =  α* A'*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2}, A::LatticeMatrix{4,T2,AT2,NC1,NC2}) where {T1,T2,AT1,AT2,NC1,NC2}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute!, C.A, A.A, Val(NC1), Val(NC2), C.nw, C.PN
    )
    set_halo!(C)
end

function kernel_4Dsubstitute!(i, C, A, ::Val{NC1}, ::Val{NC2}, nw, PN) where {NC1,NC2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2}, A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2}}) where {T1,T2,AT1,AT2,NC1,NC2}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_dag!, C.A, A.datA.A, Val(NC1), Val(NC2), C.nw, C.PN
    )
    set_halo!(C)
end

function kernel_4Dsubstitute_dag!(i, C, A, ::Val{NC1}, ::Val{NC2}, nw, PN) where {NC1,NC2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[jc, ic, ix+nw, iy+nw, iz+nw, it+nw]'
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2}, A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2},shift}) where {T1,T2,AT1,AT2,NC1,NC2,shift}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_shift!, C.A, A.data.A, Val(NC1), Val(NC2), C.nw, C.PN, shift
    )
    set_halo!(C)
end
export substitute!

function kernel_4Dsubstitute_shift!(i, C, A, ::Val{NC1}, ::Val{NC2}, nw, PN, shift) where {NC1,NC2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    #println("ix, iy, iz, it = ", (ix, iy, iz, it))
    #println("ix, iy, iz, it = ", (ixp, iyp, izp, itp))
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[ic, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2}, A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2},shift}}) where {T1,T2,AT1,AT2,NC1,NC2,shift}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_shiftdag!, C.A, A.data.data.A, Val(NC1), Val(NC2), C.nw, C.PN, shift
    )
    set_halo!(C)
end
export substitute!

function kernel_4Dsubstitute_shiftdag!(i, C, A, ::Val{NC1}, ::Val{NC2}, nw, PN, shift) where {NC1,NC2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[jc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]'
        end
    end
end

#C = shiftedA*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

#C = α shiftedA*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}, B::LatticeMatrix{4,T3,AT3,NC3,NC2},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

#C = A*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end

#C = α A*shiftedB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end





#C = shiftedA'*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

#C = α*shiftedA'*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

#C = shiftedA*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C = α*shiftedA*B'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C = shiftedA'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C = α*shiftedA'*B'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shift}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end


#C = A'*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end

#C = α*A'*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end


#C = A*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[jc, kc, ixp+nw, iyp+nw, izp+nw, itp+nw]'
            end
        end
    end
end

#C = α*A*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[jc, kc, ixp+nw, iyp+nw, izp+nw, itp+nw]'
            end
        end
    end
end

#C = A'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ixp+nw, iyp+nw, izp+nw, itp+nw]'
            end
        end
    end
end

#C = α*A'*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3}}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shift}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shift, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shift, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ixp+nw, iyp+nw, izp+nw, itp+nw]'
            end
        end
    end
end



#C = shiftA*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = α*shiftA*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = shiftA'*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB}) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = α*shiftA'*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = shiftA*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB}}) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

#C = α* shiftA*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

#C = shiftA'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA}},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB}}) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

#C = α*shiftA'*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3},shiftA}},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2},shiftB}},
    α::Number, β::Number) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), C.nw, C.PN, shiftA, shiftB, α::Number, β::Number
    )
    set_halo!(C)
end


function kernel_4Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, nw, PN, shiftA, shiftB, α::Number, β::Number) where {NC1,NC2,NC3}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

function LinearAlgebra.tr(C::LatticeMatrix{4,T1,AT1,NC1,NC2}) where {T1,AT1,NC1,NC2}
    @assert NC1 == NC2 "Trace is only defined for square matrices"
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_tr_4D, C.A, NC1, C.PN; init=zero(eltype(C.A)))
end

function kernel_tr_4D(i, A, NC1, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    s = zero(eltype(A))
    for ic = 1:NC1
        s += A[ic, ic, ix, iy, iz, it]
    end
    return s
end

function LinearAlgebra.tr(C::LatticeMatrix{4,T1,AT1,3,3}) where {T1,AT1}
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_tr_4D_NC3, C.A, C.PN; init=zero(eltype(C.A)))
end

function kernel_tr_4D_NC3(i, A, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    s = zero(eltype(A))
    for ic = 1:3
        s += A[ic, ic, ix, iy, iz, it]
    end
    return s
end

function partial_trace(C::LatticeMatrix{4,T1,AT1,NC1,NC2}, μ::Int, position::Int=1) where {T1,AT1,NC1,NC2}
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_partial_trace_4D, C.A, NC1, C.PN, μ, position; init=zero(eltype(C.A)))
    return s
end
export partial_trace

function kernel_partial_trace_4D(i, A, NC, PN, μ, position)
    NN = get_4Dindex(i, PN)
    s = zero(eltype(A))
    if NN[μ] == position
        for ic = 1:NC
            s += A[ic, ic, NN...]
        end
    end
    return s
end

# ========== host side ==========
function normalize_matrix!(C::LatticeMatrix{4,T,AT,NC,NC}) where {T,AT,NC}
    if NC == 2
        JACC.parallel_for(prod(C.PN), kernel_normalize_NC2!, C.A, C.PN)
    elseif NC == 3
        JACC.parallel_for(prod(C.PN), kernel_normalize_NC3!, C.A, C.PN)
    else
        # Generic: modified Gram–Schmidt per site (unitarize columns)
        JACC.parallel_for(prod(C.PN), kernel_normalize_generic!, C.A, C.PN, NC)
    end
    set_halo!(C)
end
export normalize_matrix!


function kernel_normalize_NC2!(i, u, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    α = u[1, 1, ix, iy, iz, it]
    β = u[2, 1, ix, iy, iz, it]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, ix, iy, iz, it] = α / detU
    u[2, 1, ix, iy, iz, it] = β / detU
    u[1, 2, ix, iy, iz, it] = -conj(β) / detU
    u[2, 2, ix, iy, iz, it] = conj(α) / detU
end

function kernel_normalize_NC3!(i, u, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    w1 = 0
    w2 = 0
    @inbounds for ic = 1:3
        w1 += u[2, ic, ix, iy, iz, it] * conj(u[1, ic, ix, iy, iz, it])
        w2 += u[1, ic, ix, iy, iz, it] * conj(u[1, ic, ix, iy, iz, it])
    end
    zerock2 = w2
    w1 = -w1 / w2

    x4 = (u[2, 1, ix, iy, iz, it]) + w1 * u[1, 1, ix, iy, iz, it]
    x5 = (u[2, 2, ix, iy, iz, it]) + w1 * u[1, 2, ix, iy, iz, it]
    x6 = (u[2, 3, ix, iy, iz, it]) + w1 * u[1, 3, ix, iy, iz, it]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3

    u[2, 1, ix, iy, iz, it] = x4
    u[2, 2, ix, iy, iz, it] = x5
    u[2, 3, ix, iy, iz, it] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1, ix, iy, iz, it] = u[1, 1, ix, iy, iz, it] * w2
    u[1, 2, ix, iy, iz, it] = u[1, 2, ix, iy, iz, it] * w2
    u[1, 3, ix, iy, iz, it] = u[1, 3, ix, iy, iz, it] * w2
    u[2, 1, ix, iy, iz, it] = u[2, 1, ix, iy, iz, it] * w3
    u[2, 2, ix, iy, iz, it] = u[2, 2, ix, iy, iz, it] * w3
    u[2, 3, ix, iy, iz, it] = u[2, 3, ix, iy, iz, it] * w3

    aa1 = real(u[1, 1, ix, iy, iz, it])
    aa2 = imag(u[1, 1, ix, iy, iz, it])
    aa3 = real(u[1, 2, ix, iy, iz, it])
    aa4 = imag(u[1, 2, ix, iy, iz, it])
    aa5 = real(u[1, 3, ix, iy, iz, it])
    aa6 = imag(u[1, 3, ix, iy, iz, it])
    aa7 = real(u[2, 1, ix, iy, iz, it])
    aa8 = imag(u[2, 1, ix, iy, iz, it])
    aa9 = real(u[2, 2, ix, iy, iz, it])
    aa10 = imag(u[2, 2, ix, iy, iz, it])
    aa11 = real(u[2, 3, ix, iy, iz, it])
    aa12 = imag(u[2, 3, ix, iy, iz, it])

    aa13 =
        aa3 * aa11 - aa4 * aa12 - aa5 * aa9 + aa6 * aa10
    aa14 =
        aa5 * aa10 + aa6 * aa9 - aa3 * aa12 - aa4 * aa11
    aa15 = aa5 * aa7 - aa6 * aa8 - aa1 * aa11 + aa2 * aa12
    aa16 = aa1 * aa12 + aa2 * aa11 - aa5 * aa8 - aa6 * aa7
    aa17 = aa1 * aa9 - aa2 * aa10 - aa3 * aa7 + aa4 * aa8
    aa18 = aa3 * aa8 + aa4 * aa7 - aa1 * aa10 - aa2 * aa9

    u[3, 1, ix, iy, iz, it] = aa13 + im * aa14
    u[3, 2, ix, iy, iz, it] = aa15 + im * aa16
    u[3, 3, ix, iy, iz, it] = aa17 + im * aa18

end



# ========== device side (generic N) ==========
# Normalize columns in-place to form a unitary (QR with Q-only), per lattice site
function kernel_normalize_generic!(i, u, PN, NC)
    # Index decode
    ix, iy, iz, it = get_4Dindex(i, PN)

    # Type helpers
    T = eltype(u)
    rT = real(one(T))
    epsT = sqrt(eps(rT))  # tolerance for near-zero norms

    # Modified Gram–Schmidt over columns j = 1..NC
    @inbounds for j = 1:NC
        # Orthogonalize column j against columns 1..j-1
        for k = 1:j-1
            # inner = ⟨u[:,k], u[:,j]⟩ = sum(conj(u[k]) * u[j])
            inner = zero(T)
            for r = 1:NC
                inner += conj(u[r, k, ix, iy, iz, it]) * u[r, j, ix, iy, iz, it]
            end
            # u[:,j] -= inner * u[:,k]
            for r = 1:NC
                u[r, j, ix, iy, iz, it] -= inner * u[r, k, ix, iy, iz, it]
            end
        end

        # Compute 2-norm of column j
        nrm2 = zero(rT)
        for r = 1:NC
            nrm2 += abs2(u[r, j, ix, iy, iz, it])
        end
        nrm = sqrt(nrm2)

        # Handle near-zero; fall back to a canonical basis vector
        if nrm < epsT
            # Zero column then set j-th row to 1 (produces consistent unitary completion)
            for r = 1:NC
                u[r, j, ix, iy, iz, it] = zero(T)
            end
            u[j, j, ix, iy, iz, it] = one(T)
        else
            # Normalize column j
            invn = one(rT) / nrm
            invnT = convert(T, invn)  # keep type stability for Complex/Real T
            for r = 1:NC
                u[r, j, ix, iy, iz, it] *= invnT
            end
        end
    end

    # Optional: single re-orthogonalization sweep for improved numerical stability
    # (uncomment if needed)
    # @inbounds for j = 1:NC
    #     for k = 1:j-1
    #         inner = zero(T)
    #         for r = 1:NC
    #             inner += conj(u[r,k,ix,iy,iz,it]) * u[r,j,ix,iy,iz,it]
    #         end
    #         for r = 1:NC
    #             u[r,j,ix,iy,iz,it] -= inner * u[r,k,ix,iy,iz,it]
    #         end
    #     end
    #     nrm2 = zero(rT)
    #     for r = 1:NC
    #         nrm2 += abs2(u[r,j,ix,iy,iz,it])
    #     end
    #     nrm = sqrt(nrm2)
    #     invnT = convert(T, one(rT)/max(nrm, epsT))
    #     for r = 1:NC
    #         u[r,j,ix,iy,iz,it] *= invnT
    #     end
    # end

    return nothing
end

function randomize_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2}) where {T,AT,NC1,NC2}
    JACC.parallel_for(prod(C.PN), kernel_randomize_4D!, C.A, C.PN, NC1, NC2)
    set_halo!(C)
end
export randomize_matrix!

function kernel_randomize_4D!(i, u, PN, NC1, NC2)
    ix, iy, iz, it = get_4Dindex(i, PN)

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] = rand() - 0.5 + im * (rand() - 0.5)
        end
    end

end

function clear_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2}) where {T,AT,NC1,NC2}
    JACC.parallel_for(prod(C.PN), kernel_clear_4D!, C.A, C.PN, NC1, NC2)
    set_halo!(C)
end
export clear_matrix!

function kernel_clear_4D!(i, u, PN, NC1, NC2)
    ix, iy, iz, it = get_4Dindex(i, PN)

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] = zero(eltype(u))
        end
    end

end

function makeidentity_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2}) where {T,AT,NC1,NC2}
    JACC.parallel_for(prod(C.PN), kernel_makeidentity_4D!, C.A, C.PN, NC1, NC2)
    set_halo!(C)
end
export makeidentity_matrix!

function kernel_makeidentity_4D!(i, u, PN, NC1, NC2)
    ix, iy, iz, it = get_4Dindex(i, PN)

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] = ifelse(ic == jc, one(eltype(u)), zero(eltype(u)))
        end
    end

end

#C = C+ α*A
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2}, A::LatticeMatrix{4,T1,AT1,NC1,NC2}, α::Number=1) where {T,T1,AT,AT1,NC1,NC2}
    JACC.parallel_for(prod(C.PN), kernel_add_4D!, C.A, A.A, C.PN, NC1, NC2, α)
    set_halo!(C)
end
export add_matrix!

function kernel_add_4D!(i, u, v, PN, NC1, NC2, α)
    ix, iy, iz, it = get_4Dindex(i, PN)

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] += α * v[ic, jc, ix, iy, iz, it]
        end
    end
end

#C = C+ α*shiftA
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2}, A::Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2},shift}, α::Number=1) where {T,T1,AT,AT1,NC1,NC2,shift}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_shift!, C.A, A.A, C.PN, NC1, NC2, α, shift)
    set_halo!(C)
end


function kernel_add_4D_shift!(i, u, v, PN, NC1, NC2, α, shift)
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] += α * v[ic, jc, ixp, iyp, izp, itp]
        end
    end
end

#C = C+ α*Adag
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2}, A::Adjoint_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2}}, α::Number=1) where {T,T1,AT,AT1,NC1,NC2}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_dag!, C.A, A.A, C.PN, NC1, NC2, α)
    set_halo!(C)
end

function kernel_add_4D_dag!(i, u, v, PN, NC1, NC2, α)
    ix, iy, iz, it = get_4Dindex(i, PN)

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] += α * v[jc, ic, ix, iy, iz, it]'
        end
    end
end

#C = C+ α*shiftAdag
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2}, A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2},shift}}, α::Number=1) where {T,T1,AT,AT1,NC1,NC2,shift}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_shiftdag!, C.A, A.A, C.PN, NC1, NC2, α, shift)
    set_halo!(C)
end


function kernel_add_4D_shiftdag!(i, u, v, PN, NC1, NC2, α, shift)
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] += α * v[jc, ic, ixp, iyp, izp, itp]'
        end
    end
end

function applyfunction!(C::LatticeMatrix{4,T,AT,NC1,NC2}, f::Function, variables...) where {T,AT,NC1,NC2}
    JACC.parallel_for(prod(C.PN), kernel_apply_function_4D!, C.A, C.PN, Val(NC1), Val(NC2), f, variables...)
    set_halo!(C)
end
export applyfunction!

function kernel_apply_function_4D!(i, u, PN, ::Val{N1}, ::Val{N2}, f, variables...) where {N1,N2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    At = MMatrix{N1,N2,T}(undef)

    @inbounds for jc = 1:N2
        for ic = 1:N1
            At[ic, jc] = u[ic, jc, ix, iy, iz, it]
        end
    end
    Aout = f(At, variables...)

    for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] = Aout[ic, jc]
        end
    end
end