# ---------------- Device-safe tiny PRNG (PCG32, stateless per element) ----------------
# All functions are @inline and pure; no dynamic dispatch, no allocations.

#=
# One PCG32 step: returns a new 64-bit state and a 32-bit output
@inline function pcg32_step(state::UInt64, inc::UInt64)
    oldstate = state
    state = oldstate * 0x5851F42D4C957F2D + (inc | 1)   # 6364136223846793005
    xorshifted = UInt32(((oldstate >> 18) ⊻ oldstate) >> 27)
    rot = oldstate >> 59
    out = (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
    return state, out
end
=#

# --- 1) PCG32: make shift counts Int (avoid weird conversions) ---
@inline function pcg32_step(state::UInt64, inc::UInt64)
    oldstate   = state
    state      = oldstate * 0x5851F42D4C957F2D + (inc | 1)
    xorshifted = UInt32(((oldstate >> 18) ⊻ oldstate) >> 27)
    rot        = Int(oldstate >> 59)                     # was UInt64
    out        = (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))
    return state, out
end

# Simple index hashing to make a per-element seed and stream (inc)
@inline function mix_seed(ix,iy,iz,it,ic,jc,seed0::UInt64)
    s = seed0
    s ⊻= UInt64(ix)*0x9E3779B97F4A7C15
    s ⊻= UInt64(iy)*0xBF58476D1CE4E5B9
    s ⊻= UInt64(iz)*0x94D049BB133111EB
    s ⊻= UInt64(it)*0x2545F4914F6CDD1D
    s ⊻= (UInt64(ic) << 32) ⊻ UInt64(jc)
    return s, s ⊻ 0xDEADBEEFCAFEBABE  # (state, inc)
end

# Convert u32 to Float32/Float64 in [0,1)
@inline u01_f32(x::UInt32) = Float32(x) * (1f0 / 2.0f32^32)
@inline function u01_f64(x1::UInt32, x2::UInt32)
    u = (UInt64(x1) << 32) | UInt64(x2)
    return Float64(u) * (1.0 / 2.0^64)
end

