######################## 3x3 exp (Complex-ready) via Pade(13) + scaling & squaring ########################
function expt(A, t::S=one(S)) where {S<:Number}
    N1, N2 = size(A)
    if N1 == 1
        return exp(t * A[1, 1])
    elseif N1 == 2
        return exp2x2(A, t)
    elseif N1 == 3
        return exp3x3(A, t)
    else
        return expm_pade13(A, t)
    end
end
export expt


struct Mat3{T<:Number}
    a11::T
    a12::T
    a13::T
    a21::T
    a22::T
    a23::T
    a31::T
    a32::T
    a33::T
end
export Mat3

function Base.Matrix(A::Mat3{T}) where {T<:Number}
    return [A.a11 A.a12 A.a13; A.a21 A.a22 A.a23; A.a31 A.a32 A.a33]
end

# Identity matrix
@inline Mat3{T}(x::T) where {T<:Number} = Mat3{T}(x, zero(T), zero(T), zero(T), x, zero(T), zero(T), zero(T), x)

function exp3x3(A::Matrix{T}, t::S=one(S)) where {T<:Number,S<:Number}
    @assert size(A, 1) == 3 && size(A, 2) == 3
    A2 = Mat3{T}(A[1, 1], A[1, 2], A[1, 3],
        A[2, 1], A[2, 2], A[2, 3],
        A[3, 1], A[3, 2], A[3, 3])
    E2 = exp3x3_pade(A2, t)
    return T[E2.a11 E2.a12 E2.a13;
        E2.a21 E2.a22 E2.a23;
        E2.a31 E2.a32 E2.a33]
end
export exp3x3

# ---- basic operations ----
@inline add3(A::Mat3{T}, B::Mat3{T}) where {T<:Number} =
    Mat3{T}(A.a11 + B.a11, A.a12 + B.a12, A.a13 + B.a13,
        A.a21 + B.a21, A.a22 + B.a22, A.a23 + B.a23,
        A.a31 + B.a31, A.a32 + B.a32, A.a33 + B.a33)

@inline sub3(A::Mat3{T}, B::Mat3{T}) where {T<:Number} =
    Mat3{T}(A.a11 - B.a11, A.a12 - B.a12, A.a13 - B.a13,
        A.a21 - B.a21, A.a22 - B.a22, A.a23 - B.a23,
        A.a31 - B.a31, A.a32 - B.a32, A.a33 - B.a33)

# Allow scalar of any Number type; rely on promotion (e.g., Number * Complex)
@inline function smul3(α, A::Mat3{T}) where {T<:Number}
    S = promote_type(typeof(α), T)
    Mat3{S}(α * A.a11, α * A.a12, α * A.a13,
        α * A.a21, α * A.a22, α * A.a23,
        α * A.a31, α * A.a32, α * A.a33)
end

@inline function mul3(A::Mat3{T}, B::Mat3{T}) where {T<:Number}
    Mat3{T}(
        A.a11 * B.a11 + A.a12 * B.a21 + A.a13 * B.a31,
        A.a11 * B.a12 + A.a12 * B.a22 + A.a13 * B.a32,
        A.a11 * B.a13 + A.a12 * B.a23 + A.a13 * B.a33,
        A.a21 * B.a11 + A.a22 * B.a21 + A.a23 * B.a31,
        A.a21 * B.a12 + A.a22 * B.a22 + A.a23 * B.a32,
        A.a21 * B.a13 + A.a22 * B.a23 + A.a23 * B.a33,
        A.a31 * B.a11 + A.a32 * B.a21 + A.a33 * B.a31,
        A.a31 * B.a12 + A.a32 * B.a22 + A.a33 * B.a32,
        A.a31 * B.a13 + A.a32 * B.a23 + A.a33 * B.a33
    )
end

@inline conjugate3(A::Mat3{T}) where {T<:Number} =
    Mat3{T}(A.a11', A.a21', A.a31',
        A.a12', A.a22', A.a32',
        A.a13', A.a23', A.a33')


# 1-norm (maximum column sum); return real base type RT
@inline function norm1(A::Mat3{T}) where {T<:Number}
    RT = typeof(real(zero(T)))
    c1 = RT(abs(A.a11)) + RT(abs(A.a21)) + RT(abs(A.a31))
    c2 = RT(abs(A.a12)) + RT(abs(A.a22)) + RT(abs(A.a32))
    c3 = RT(abs(A.a13)) + RT(abs(A.a23)) + RT(abs(A.a33))
    max(c1, max(c2, c3))
end

# 3x3 inverse (adjoint / determinant); works for Complex as well
@inline function inv3(A::Mat3{T}) where {T<:Number}
    c11 = (A.a22 * A.a33 - A.a23 * A.a32)
    c12 = -(A.a21 * A.a33 - A.a23 * A.a31)
    c13 = (A.a21 * A.a32 - A.a22 * A.a31)
    c21 = -(A.a12 * A.a33 - A.a13 * A.a32)
    c22 = (A.a11 * A.a33 - A.a13 * A.a31)
    c23 = -(A.a11 * A.a32 - A.a12 * A.a31)
    c31 = (A.a12 * A.a23 - A.a13 * A.a22)
    c32 = -(A.a11 * A.a23 - A.a13 * A.a21)
    c33 = (A.a11 * A.a22 - A.a12 * A.a21)
    det = A.a11 * c11 + A.a12 * c12 + A.a13 * c13
    invdet = one(T) / det
    Mat3{T}(c11 * invdet, c21 * invdet, c31 * invdet,
        c12 * invdet, c22 * invdet, c32 * invdet,
        c13 * invdet, c23 * invdet, c33 * invdet)
end

#=
# ---- Padé(13) coefficients (real constants lifted to T via promotion) ----
@inline function pade13_coeffs(::Type{T}) where {T<:Number}
    # Build in a real type and rely on promotion when used with Complex T
    RT = typeof(real(zero(T)))
    (RT(64764752532480000), RT(32382376266240000), RT(7771770303897600),
        RT(1187353796428800), RT(129060195264000), RT(10559470521600),
        RT(670442572800), RT(33522128640), RT(1323241920),
        RT(40840800), RT(960960), RT(16380),
        RT(182), RT(1))
end
=#

# ---- exp(t*A) via Pade(13) + scaling & squaring (Complex-ready) ----
@inline function exp3x3_pade(A::Mat3{T}, t::S=one(S)) where {T<:Number,S<:Number}
    # Scale: use real base type for thresholds and ldexp
    RT = typeof(real(zero(T)))

    # At = t*A
    At = smul3(t, A)
    n1 = norm1(At)             # ::RT
    if n1 == RT(0)
        return Mat3{T}(one(T))
    end

    θ13 = RT(5.371920351148152)  # Higham threshold (double)
    s = n1 <= θ13 ? 0 : max(0, ceil(Int, log2(n1 / θ13)))
    α = ldexp(RT(1), -s)         # 2^-s in real base type
    As = smul3(α, At)            # promotes to T (works for Complex)

    # powers
    A2 = mul3(As, As)
    A4 = mul3(A2, A2)
    A6 = mul3(A2, A4)

    # coefficients (real) — promoted on use
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = pade13_coeffs(T)
    I = Mat3{T}(one(T))

    # U = As * (A6*(b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I)
    t1 = add3(smul3(b13, A6), add3(smul3(b11, A4), smul3(b9, A2)))
    t2 = add3(mul3(A6, t1),
        add3(add3(smul3(b7, A6), smul3(b5, A4)),
            add3(smul3(b3, A2), smul3(b1, I))))
    U = mul3(As, t2)

    # V = A6*(b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I
    t3 = add3(smul3(b12, A6), add3(smul3(b10, A4), smul3(b8, A2)))
    V = add3(mul3(A6, t3),
        add3(add3(smul3(b6, A6), smul3(b4, A4)),
            add3(smul3(b2, A2), smul3(b0, I))))

    # R = (V - U)^{-1} * (V + U)
    P = add3(V, U)
    Q = sub3(V, U)
    R = mul3(inv3(Q), P)

    # repeated squaring
    for _ in 1:s
        R = mul3(R, R)
    end
    return R
end
###########################################################################################################
export exp3x3_pade
############################################################################################


######################## NxN exp via Pade(13) + scaling & squaring (Complex-ready) ########################
# Pure Base Julia, supports A::Matrix{<:Number} (Real or Complex).
# Function: expm_pade13(A::Matrix{T}, t::Real = 1) -> Matrix{T}

# ---------- helpers ----------
function eye(T::Type, n::Int)
    I = Array{T}(undef, n, n)
    fill!(I, zero(T))
    @inbounds for i = 1:n
        I[i, i] = one(T)
    end
    return I
end

function copy_mat!(B::AbstractMatrix, A::AbstractMatrix)
    n, m = size(A)
    @inbounds for j = 1:m, i = 1:n
        B[i, j] = A[i, j]
    end
    return B
end

# ---------- norms ----------
function norm1(A::AbstractMatrix{T}) where {T<:Number}
    RT = typeof(real(zero(T)))   # real base type
    n, m = size(A)
    maxsum = zero(RT)
    @inbounds for j = 1:m
        s = zero(RT)
        for i = 1:n
            s += abs(A[i, j])
        end
        if s > maxsum
            maxsum = s
        end
    end
    return maxsum
end

# ---------- basic ops ----------
function scal!(α, A::AbstractMatrix)
    n, m = size(A)
    @inbounds for j = 1:m, i = 1:n
        A[i, j] *= α
    end
    return A
end

function axpy!(α, X::AbstractMatrix, Y::AbstractMatrix)
    n, m = size(X)
    @inbounds for j = 1:m, i = 1:n
        Y[i, j] += α * X[i, j]
    end
    return Y
end

function add!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    n, m = size(A)
    @inbounds for j = 1:m, i = 1:n
        C[i, j] = A[i, j] + B[i, j]
    end
    return C
end

function sub!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    n, m = size(A)
    @inbounds for j = 1:m, i = 1:n
        C[i, j] = A[i, j] - B[i, j]
    end
    return C
end

function gemm!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T};
    β=zero(T)) where {T}
    n, k = size(A)
    _k, m = size(B)
    @assert k == _k
    if β == zero(T)
        fill!(C, zero(T))
    elseif β != one(T)
        @inbounds for j = 1:m, i = 1:n
            C[i, j] *= β
        end
    end
    @inbounds for j = 1:m
        for p = 1:k
            bpj = B[p, j]
            for i = 1:n
                C[i, j] += A[i, p] * bpj
            end
        end
    end
    return C
end

# ---------- LU with partial pivoting ----------
function lu_factor!(A::AbstractMatrix{T}, piv::Vector{Int}) where {T<:Number}
    n, m = size(A)
    @assert n == m
    @inbounds for k = 1:n-1
        pk = k
        maxval = abs(A[k, k])
        for i = k+1:n
            v = abs(A[i, k])
            if v > maxval
                maxval = v
                pk = i
            end
        end
        piv[k] = pk
        if pk != k
            A[k, :], A[pk, :] = A[pk, :], A[k, :]
        end
        akk = A[k, k]
        if akk != 0
            invakk = one(T) / akk
            for i = k+1:n
                A[i, k] *= invakk
                lik = A[i, k]
                @inbounds for j = k+1:n
                    A[i, j] -= lik * A[k, j]
                end
            end
        end
    end
    piv[n] = n
    return A
end

function lu_solve!(LU::AbstractMatrix{T}, piv::Vector{Int}, B::AbstractMatrix{T}) where {T<:Number}
    n, m = size(LU)
    @assert size(B, 1) == n
    @inbounds for k = 1:n-1
        pk = piv[k]
        if pk != k
            B[k, :], B[pk, :] = B[pk, :], B[k, :]
        end
    end
    # Forward solve
    @inbounds for i = 2:n
        for k = 1:i-1
            lik = LU[i, k]
            for j = 1:m
                B[i, j] -= lik * B[k, j]
            end
        end
    end
    # Backward solve
    @inbounds for i = n:-1:1
        uii = LU[i, i]
        invuii = one(T) / uii
        for j = 1:m
            s = B[i, j]
            for k = i+1:n
                s -= LU[i, k] * B[k, j]
            end
            B[i, j] = s * invuii
        end
    end
    return B
end

# ---------- Pade(13) coefficients ----------
pade13_coeffs(::Type{T}) where {T<:Number} = (
    64764752532480000, 32382376266240000, 7771770303897600,
    1187353796428800, 129060195264000, 10559470521600,
    670442572800, 33522128640, 1323241920,
    40840800, 960960, 16380,
    182, 1
)

# ---------- main: expm via Pade(13) + scaling & squaring ----------
function expm_pade13(A::Matrix{T}, t::S=one(S)) where {T<:Number,S<:Number}
    n, m = size(A)
    @assert n == m
    if n == 0
        return Array{T}(undef, 0, 0)
    end
    if n == 1
        return reshape(exp(t * A[1, 1]), 1, 1)
    end

    # Scale
    At = copy(A)
    scal!(t, At)
    n1 = norm1(At)   # real
    if n1 == 0
        return eye(T, n)
    end
    θ13 = 5.371920351148152   # Higham threshold (double)
    s = n1 <= θ13 ? 0 : max(0, ceil(Int, log2(n1 / θ13)))
    α = ldexp(1.0, -s)        # real scaling factor
    scal!(α, At)

    # Powers
    A2 = similar(A)
    A4 = similar(A)
    A6 = similar(A)
    gemm!(A2, At, At)
    gemm!(A4, A2, A2)
    gemm!(A6, A2, A4)

    # Coefficients (integers → promoted automatically)
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13 = pade13_coeffs(T)

    I = eye(T, n)
    U = zeros(T, n, n)
    V = zeros(T, n, n)
    T1 = zeros(T, n, n)
    T2 = zeros(T, n, n)

    # U = At * (A6*(b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I)
    copy_mat!(T1, A6)
    scal!(b13, T1)
    axpy!(b11, A4, T1)
    axpy!(b9, A2, T1)
    gemm!(T2, A6, T1)
    axpy!(b7, A6, T2)
    axpy!(b5, A4, T2)
    axpy!(b3, A2, T2)
    axpy!(b1, I, T2)
    gemm!(U, At, T2)

    # V = A6*(b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I
    copy_mat!(T1, A6)
    scal!(b12, T1)
    axpy!(b10, A4, T1)
    axpy!(b8, A2, T1)
    gemm!(V, A6, T1)
    axpy!(b6, A6, V)
    axpy!(b4, A4, V)
    axpy!(b2, A2, V)
    axpy!(b0, I, V)

    # Solve (V - U)X = (V + U)
    P = zeros(T, n, n)
    Q = zeros(T, n, n)
    add!(P, V, U)
    sub!(Q, V, U)
    LU = copy(Q)
    piv = Vector{Int}(undef, n)
    lu_factor!(LU, piv)
    lu_solve!(LU, piv, P)
    R = P

    # repeated squaring
    for _ = 1:s
        gemm!(T1, R, R)
        copy_mat!(R, T1)
    end
    return R
end

export expm_pade13

# ---------------- Example ----------------
# A = ComplexF64[0 1 0; 0 0 1; -1 0 0]
# E = expm_pade13(A, 1.0)
# @show E
###########################################################################################################

######################## 2x2 matrix exponential (Complex-ready, closed-form) ########################
# Stable sinh(z)/z
@inline function sinhc(z)
    # Use series near zero to avoid loss of significance
    if abs(z) < 1e-6
        z2 = z * z
        return 1 + z2 / 6 + (z2 * z2) / 120 + (z2 * z2 * z2) / 5040
    else
        return sinh(z) / z
    end
end

# exp(t*A) for 2x2 Array{<:Number,2} (out-of-place)
function exp2x2(A::Matrix{T}, t::S=one(S)) where {T<:Number,S<:Number}
    @assert size(A, 1) == 2 && size(A, 2) == 2
    a11, a12 = A[1, 1], A[1, 2]
    a21, a22 = A[2, 1], A[2, 2]

    # Trace and determinant
    trA = a11 + a22
    half = trA / 2
    detA = a11 * a22 - a12 * a21

    # δ^2 = (tr/2)^2 - det(A); δ may be imaginary ⇒ promote via complex()
    δ2 = (half * half) - detA
    δ = sqrt(complex(δ2))

    # θ = t * tr(A)/2, τ = t*δ
    θ = t * half
    τ = t * δ

    # c = cosh(τ), s = (sinh(τ)/τ) * t = t*sinhc(τ)
    c = cosh(τ)
    s = t * sinhc(τ)   # <<< IMPORTANT: multiply by t >>>

    # B = A - (tr(A)/2) I
    b11 = a11 - half
    b12 = a12
    b21 = a21
    b22 = a22 - half

    eθ = exp(θ)
    # E = e^{θ} * ( c*I + s*B )
    return T[
        eθ*(c+s*b11) eθ*(s*b12);
        eθ*(s*b21) eθ*(c+s*b22)
    ]
end

export exp2x2

# ---------------- Example ----------------
# A = ComplexF64[ 1+2im  3-1im;
#                 0.5    -2   ]
# E = exp2x2(A, 1.0)
# @show E
###########################################################################################################