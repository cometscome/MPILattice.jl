function jaccMPI_kernel_mul!(i, C::AbstractArray{T,3}, A, B, NC,Nwing) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i+Nwing]*B[kc,jc,i+Nwing]
            end
        end
    end
end

function jaccMPI_kernel_mul_shiftedB!(i, C::AbstractArray{T,3}, A, B, NC,Nwing,shift) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i+Nwing]*B[kc,jc,i+Nwing+shift]
            end
        end
    end
end

function jaccMPI_kernel_mul_shiftedA!(i, C::AbstractArray{T,3}, A, B, NC,Nwing,shift) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i+Nwing+shift]*B[kc,jc,i+Nwing]
            end
        end
    end
end

function jaccMPI_kernel_substitute!(i, C::AbstractArray{T,3}, A,  NC1,NC2,Nwing) where {T}
    for jc=1:NC2
        for ic=1:NC1
            C[ic,jc,i+Nwing] = A[ic,jc,i+Nwing]
        end
    end
end


function jaccMPI_kernel_substitute_shiftedA!(i, C::AbstractArray{T,3}, A,  NC1,NC2,Nwing,shift) where {T}
    for jc=1:NC2
        for ic=1:NC1
            C[ic,jc,i+Nwing] = A[ic,jc,i+Nwing+shift]
        end
    end
end

function jaccMPI_kernel_substitute_shiftedcopiedA!(i, C::AbstractArray{T,3}, A,  NC1,NC2,Nwing,shift) where {T}
    for jc=1:NC2
        for ic=1:NC1
            C[ic,jc,i+Nwing] = A[ic,jc,i]
        end
    end
end

function jaccMPI_kernel_mul_shiftedcopiedB!(i, C::AbstractArray{T,3}, A, B, NC,Nwing,shift) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i+Nwing]*B[kc,jc,i]
            end
        end
    end
end

function jaccMPI_kernel_mul_shiftedcopiedA!(i, C::AbstractArray{T,3}, A, B, NC,Nwing,shift) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,i]*B[kc,jc,i+Nwing]
            end
        end
    end
end


function jaccMPI_kernel_mul_shiftedAshiftedB!(i, C::AbstractArray{T,3}, A, B, NC,Nwing,shift1,shift2) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i+Nwing+shift1]*B[kc,jc,i+Nwing+shift2]
            end
        end
    end
end

function jaccMPI_kernel_mul_shiftedAshiftedcopiedB!(i, C::AbstractArray{T,3}, A, B, NC,Nwing,shift1) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i+Nwing+shift1]*B[kc,jc,i]
            end
        end
    end
end

function jaccMPI_kernel_mul_shiftedcopiedAshiftedB!(i, C::AbstractArray{T,3}, A, B, NC,Nwing,shift2) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i]*B[kc,jc,i+Nwing+shift2]
            end
        end
    end
end

function jaccMPI_kernel_mul_shiftedcopiedAshiftedcopiedB!(i, C::AbstractArray{T,3}, A, B, NC,Nwing) where {T}
    for jc=1:NC
        for ic=1:NC
            C[ic,jc,i+Nwing] = 0
            for kc =1:NC
                C[ic,jc,i+Nwing] = A[ic,kc,i]*B[kc,ic,i]
            end
        end
    end
end