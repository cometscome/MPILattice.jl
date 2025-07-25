function jaccMPI_kernel_mul!(i, C::AbstractMatrix, A, B, NC,Nwing)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing]*B[ic,i+Nwing]
    end
end

function jaccMPI_kernel_mul_shiftedB!(i, C::AbstractMatrix, A, B, NC,Nwing,shift)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing]*B[ic,i+Nwing+shift]
    end
end

function jaccMPI_kernel_mul_shiftedA!(i, C::AbstractMatrix, A, B, NC,Nwing,shift)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing+shift]*B[ic,i+Nwing]
    end
end

function jaccMPI_kernel_substitute!(i, C::AbstractMatrix, A,  NC,Nwing)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing]
    end
end


function jaccMPI_kernel_substitute_shiftedA!(i, C::AbstractMatrix, A, NC,Nwing,shift)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing+shift]
    end
end

function jaccMPI_kernel_substitute_shiftedcopiedA!(i, C::AbstractMatrix, A, NC,Nwing,shift)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i]
    end
end

function jaccMPI_kernel_mul_shiftedcopiedB!(i, C::AbstractMatrix, A, B, NC,Nwing,shift)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing]*B[ic,i]
    end
end

function jaccMPI_kernel_mul_shiftedcopiedA!(i, C::AbstractMatrix, A, B, NC,Nwing,shift)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i]*B[ic,i+Nwing]
    end
end


function jaccMPI_kernel_mul_shiftedAshiftedB!(i, C::AbstractMatrix, A, B, NC,Nwing,shift1,shift2)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing+shift1]*B[ic,i+Nwing+shift2]
    end
end

function jaccMPI_kernel_mul_shiftedAshiftedcopiedB!(i, C::AbstractMatrix, A, B, NC,Nwing,shift1)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i+Nwing+shift1]*B[ic,i]
    end
end

function jaccMPI_kernel_mul_shiftedcopiedAshiftedB!(i, C::AbstractMatrix, A, B, NC,Nwing,shift2)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i]*B[ic,i+Nwing+shift2]
    end
end

function jaccMPI_kernel_mul_shiftedcopiedAshiftedcopiedB!(i, C::AbstractMatrix, A, B, NC,Nwing)
    for ic=1:NC
        C[ic,i+Nwing] = A[ic,i]*B[ic,i]
    end
end