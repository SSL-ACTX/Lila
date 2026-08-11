# Architecture Proposal: Formally Verified XChaCha20-Poly1305 AEAD Engine (RFC-006)

**Document Status:** Technical RFC / Proposal  
**Target Subsystems:** `lirien/stdlib/crypto`, `lirien-verify` (`QF_BV` SMT Backend), `lirien-backend` (SIMD & CLIF Lowering)  
**Author:** Seuriin (Jameel Tutungan)

---

## 1. Executive Summary & Objective

The objective of this proposal is to specify the architecture for **`lirien.crypto`**, a native, formally verified implementation of the **XChaCha20-Poly1305** Authenticated Encryption with Associated Data (AEAD) construction written directly within Lirien.

Implementing cryptographic primitives in standard interpreted languages like Python is historically avoided due to timing side-channel risks, heap-allocation leaks, and memory out-of-bounds vulnerabilities. By leveraging Lirien's core compiler pipeline—specifically **Z3 Bitvector SMT verification (`QF_BV`)**, **128-bit SIMD vector types (`u32x4`, `u8x16`)**, **zero-copy slicing**, and **inline CLIF assembly**—`lirien.crypto` can provide compile-time verified memory safety, constant-time execution, and raw hardware speed inside Python.

---

## 2. Specifications & Reference Materials

This design references the following canonical papers, IETF specifications, and reference implementations:

### Primary Research Papers & Specifications
*   **ChaCha20 Specification (Daniel J. Bernstein, 2008):**  
    [ChaCha, a variant of Salsa20 (PDF)](https://cr.yp.to/chacha/chacha-20080128.pdf)
*   **Poly1305 Specification (Daniel J. Bernstein, 2005):**  
    [The Poly1305-AES Message-Authentication Code (PDF)](https://cr.yp.to/mac/poly1305-20050115.pdf)
*   **IETF Standard ChaCha20-Poly1305 (RFC 8439):**  
    [RFC 8439: ChaCha20 and Poly1305 for IETF Protocols](https://datatracker.ietf.org/doc/html/rfc8439)
*   **IETF Extended-Nonce XChaCha Draft (`draft-irtf-cfrg-xchacha-03`):**  
    [XChaCha: eXtended-nonce ChaCha and AEAD_XChaCha20_Poly1305](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha-03)

### Reference Implementations
*   **`libsodium` (Canonical C Reference):**  
    [jedisct1/libsodium on GitHub](https://github.com/jedisct1/libsodium) — Header: [`crypto_aead_xchacha20poly1305.h`](https://github.com/jedisct1/libsodium/blob/master/src/libsodium/include/sodium/crypto_aead_xchacha20poly1305.h)
*   **`RustCrypto/AEADs` (Pure Rust Reference):**  
    [RustCrypto `chacha20poly1305` on GitHub](https://github.com/RustCrypto/AEADs/tree/master/chacha20poly1305)
*   **Go Standard `x/crypto` Implementation:**  
    [golang/crypto `chacha20poly1305` on GitHub](https://github.com/golang/crypto/tree/master/chacha20poly1305)

---

## 3. Cryptographic Architecture & Primitive Breakdown

XChaCha20-Poly1305 extends standard ChaCha20-Poly1305 by increasing the nonce size from 96 bits (12 bytes) to **192 bits (24 bytes)**, eliminating the risk of nonce reuse in random-nonce environments. 

The implementation in `lirien.crypto` is structured into three discrete, verifiable layers:

```
[ 256-bit Key (K) ]      [ 192-bit Nonce (N_0 || N_1) ]
         │                            │
         ▼                            ▼
  [ HChaCha20 ] ─────────────> (First 16 bytes: N_0)
         │
         ▼
[ Derived Subkey (K') ]  [ Remaining 8 bytes (N_1) ]  [ 32-bit Counter ]
         │                            │                     │
         └────────────────────────────┴─────────────────────┘
                                      │
                                      ▼
                             [ XChaCha20 Keystream ]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   [ Encrypt Plaintext -> Ciphertext ]    [ Poly1305 One-Time Key Generation ]
                                                           │
                                                           ▼
                                               [ Poly1305 MAC Tag Generation ]
```

### 3.1 Primitive 1: HChaCha20 (Subkey Derivation)
HChaCha20 takes a 256-bit key $K$ and the first 16 bytes of the 24-byte nonce ($N_0$). It initializes a 512-bit ChaCha state matrix, runs 10 double-rounds (20 rounds total) of Quarter-Round permutations, and extracts the first and last rows of the matrix to output a 256-bit subkey $K'$:
$$\text{State Matrix} = \begin{pmatrix} c_0 & c_1 & c_2 & c_3 \\ k_0 & k_1 & k_2 & k_3 \\ k_4 & k_5 & k_6 & k_7 \\ n_0 & n_1 & n_2 & n_3 \end{pmatrix}$$

### 3.2 Primitive 2: XChaCha20 Stream Cipher
Using the derived 256-bit subkey $K'$, the remaining 8 bytes of the nonce ($N_1$), and a 64-bit block counter initialized to `0` (or `1` for payload encryption), standard ChaCha20 generates 64-byte keystream blocks to XOR with the plaintext.

### 3.3 Primitive 3: Poly1305 MAC
Poly1305 generates a 128-bit authentication tag over the Associated Authenticated Data (AAD) and Ciphertext. It evaluates a polynomial modulo $2^{130} - 5$ using a one-time key derived from block `0` of the XChaCha20 keystream.

---

## 4. Compiler & Verification Strategy in Lirien

To maintain Lirien's safety guarantees and performance targets, `lirien.crypto` leverages specific compiler features:

### 4.1 SMT BitVector Proofs (`QF_BV`)
ChaCha20 uses Add-Rotate-XOR (ARX) operations:
$$\text{QuarterRound}(a, b, c, d):$$
$$a = a + b; \quad d = (d \oplus a) \lll 16$$
$$c = c + d; \quad b = (b \oplus c) \lll 12$$
$$a = a + b; \quad d = (d \oplus a) \lll 8$$
$$c = c + d; \quad b = (b \oplus c) \lll 7$$

Because ARX logic contains no dynamic branches, memory lookups, or division operations, `lirien-verify` translates these operations directly into Z3 BitVector theory (`QF_BV`). Z3 proves that:
1. Operations cannot produce undefined behavior or unaligned pointer reads.
2. Memory buffer access indices across $N$-byte message blocks remain strictly within buffer capacity bounds.

### 4.2 SIMD Vectorization (`u32x4` & `u8x16`)
The 4 parallel Quarter-Rounds across columns and diagonals map directly to Lirien's 128-bit vector types (`u32x4`). A single `u32x4` register holds a full row or column of the ChaCha state, allowing vector addition, XOR, and bitwise rotation instructions to execute 4 state words simultaneously in hardware.

### 4.3 Inline CLIF Optimization (`with clif:`)
For bitwise rotations ($\lll$), which can sometimes require extra masking instructions in higher-level SSA representations, the core Quarter-Round kernel can be implemented using inline Cranelift IR (`with clif:`):
```python
with clif(inputs={d: v0, a: v1}, outputs={v2: "out"}):
    v3 = v0 ^ v1  # bxor
    v2 = rotl(v3, 16)  # Direct Cranelift bitwise left rotation instruction
```

### 4.4 Zero-Copy Buffer Slicing
Processing byte streams (splitting keys, nonces, AAD, and blocks) utilizes Lirien's zero-copy slicing (`buf[start:end]`). No heap allocations occur when sub-slicing input message blocks.

---

## 5. Proposed Python DSL Interface (`lirien.stdlib.crypto`)

```python
from lirien import verify, Buffer, u8, Refined, V
from lirien.stdlib.crypto import xchacha20poly1305_encrypt, xchacha20poly1305_decrypt

# Strict Refinement Types for Cryptographic Invariants
Key256 = Refined[Buffer[u8], V.len() == 32]
Nonce192 = Refined[Buffer[u8], V.len() == 24]
Tag128 = Refined[Buffer[u8], V.len() == 16]


@verify
def encrypt_message(
    key: Key256,
    nonce: Nonce192,
    plaintext: Buffer[u8],
    aad: Buffer[u8],
    ciphertext_out: Buffer[u8],
    tag_out: Tag128,
):
    """
    Formally verified XChaCha20-Poly1305 AEAD encryption.
    Verified by Z3 to be in-bounds and memory-safe.
    """
    # Requires ciphertext_out to match plaintext capacity
    assert len(ciphertext_out) == len(plaintext)

    xchacha20poly1305_encrypt(key, nonce, plaintext, aad, ciphertext_out, tag_out)
```

---

## 6. Implementation Milestones

1. **Phase 1: Core ARX & Poly1305 Math Primitives**  
   Implement bitwise rotations, `u32x4` ARX Quarter-Rounds, and modulo $2^{130} - 5$ Poly1305 arithmetic in `python/lirien/stdlib/crypto/primitives.py`.

2. **Phase 2: HChaCha20 Subkey Derivation & XChaCha20 Stream Engine**  
   Build HChaCha20 state extraction and block counter management for 192-bit nonces.

3. **Phase 3: Formal SMT Verification & Test Bench Verification**  
   Run `lirien-verify` SMT bitvector checks across all primitives, verify zero-copy slicing, and validate against official RFC 8439 / IETF draft test vectors in `tests/python/stdlib/test_crypto.py`.
