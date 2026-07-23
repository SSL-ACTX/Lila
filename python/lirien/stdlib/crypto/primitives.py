from lirien import verify, Buffer, u8, u32, u64, i64


@verify
def quarter_round(a: u32, b: u32, c: u32, d: u32) -> tuple[u32, u32, u32, u32]:
    a = a + b
    d = d ^ a
    d = (d << 16) | (d >> 16)

    c = c + d
    b = b ^ c
    b = (b << 12) | (b >> 20)

    a = a + b
    d = d ^ a
    d = (d << 8) | (d >> 24)

    c = c + d
    b = b ^ c
    b = (b << 7) | (b >> 25)
    return a, b, c, d


@verify
def load_u32_le(buf: Buffer[u8], offset: i64) -> u32:
    assert offset >= 0
    assert offset <= len(buf) - 4
    return (
        u32(buf[offset])
        | (u32(buf[offset + 1]) << 8)
        | (u32(buf[offset + 2]) << 16)
        | (u32(buf[offset + 3]) << 24)
    )


@verify
def store_u32_le(buf: Buffer[u8], offset: i64, val: u32) -> None:
    assert offset >= 0
    assert offset <= len(buf) - 4
    buf[offset] = u8(val & 0xFF)
    buf[offset + 1] = u8((val >> 8) & 0xFF)
    buf[offset + 2] = u8((val >> 16) & 0xFF)
    buf[offset + 3] = u8((val >> 24) & 0xFF)


@verify
def hchacha20(key: Buffer[u8], nonce: Buffer[u8], out_key: Buffer[u8]) -> None:
    assert len(key) == 32
    assert len(nonce) == 16
    assert len(out_key) == 32

    # Load initial state
    c0 = u32(0x61707865)
    c1 = u32(0x3320646E)
    c2 = u32(0x79622D32)
    c3 = u32(0x6B206574)

    k0 = load_u32_le(key, 0)
    k1 = load_u32_le(key, 4)
    k2 = load_u32_le(key, 8)
    k3 = load_u32_le(key, 12)
    k4 = load_u32_le(key, 16)
    k5 = load_u32_le(key, 20)
    k6 = load_u32_le(key, 24)
    k7 = load_u32_le(key, 28)

    n0 = load_u32_le(nonce, 0)
    n1 = load_u32_le(nonce, 4)
    n2 = load_u32_le(nonce, 8)
    n3 = load_u32_le(nonce, 12)

    # State variables
    s0 = c0
    s1 = c1
    s2 = c2
    s3 = c3
    s4 = k0
    s5 = k1
    s6 = k2
    s7 = k3
    s8 = k4
    s9 = k5
    s10 = k6
    s11 = k7
    s12 = n0
    s13 = n1
    s14 = n2
    s15 = n3

    # 10 double-rounds
    for i in range(10):
        s0, s4, s8, s12 = quarter_round(s0, s4, s8, s12)
        s1, s5, s9, s13 = quarter_round(s1, s5, s9, s13)
        s2, s6, s10, s14 = quarter_round(s2, s6, s10, s14)
        s3, s7, s11, s15 = quarter_round(s3, s7, s11, s15)

        s0, s5, s10, s15 = quarter_round(s0, s5, s10, s15)
        s1, s6, s11, s12 = quarter_round(s1, s6, s11, s12)
        s2, s7, s8, s13 = quarter_round(s2, s7, s8, s13)
        s3, s4, s9, s14 = quarter_round(s3, s4, s9, s14)

    store_u32_le(out_key, 0, s0)
    store_u32_le(out_key, 4, s1)
    store_u32_le(out_key, 8, s2)
    store_u32_le(out_key, 12, s3)
    store_u32_le(out_key, 16, s12)
    store_u32_le(out_key, 20, s13)
    store_u32_le(out_key, 24, s14)
    store_u32_le(out_key, 28, s15)


@verify
def chacha20_block(
    key: Buffer[u8], counter: u64, nonce: Buffer[u8], block_out: Buffer[u8]
) -> None:
    assert len(key) == 32
    assert len(nonce) == 8
    assert len(block_out) == 64

    c0 = u32(0x61707865)
    c1 = u32(0x3320646E)
    c2 = u32(0x79622D32)
    c3 = u32(0x6B206574)

    k0 = load_u32_le(key, 0)
    k1 = load_u32_le(key, 4)
    k2 = load_u32_le(key, 8)
    k3 = load_u32_le(key, 12)
    k4 = load_u32_le(key, 16)
    k5 = load_u32_le(key, 20)
    k6 = load_u32_le(key, 24)
    k7 = load_u32_le(key, 28)

    n0 = load_u32_le(nonce, 0)
    n1 = load_u32_le(nonce, 4)

    cnt_low = u32(counter & 0xFFFFFFFF)
    cnt_high = u32((counter >> 32) & 0xFFFFFFFF)

    s0 = c0
    s1 = c1
    s2 = c2
    s3 = c3
    s4 = k0
    s5 = k1
    s6 = k2
    s7 = k3
    s8 = k4
    s9 = k5
    s10 = k6
    s11 = k7
    s12 = cnt_low
    s13 = cnt_high
    s14 = n0
    s15 = n1

    x0 = s0
    x1 = s1
    x2 = s2
    x3 = s3
    x4 = s4
    x5 = s5
    x6 = s6
    x7 = s7
    x8 = s8
    x9 = s9
    x10 = s10
    x11 = s11
    x12 = s12
    x13 = s13
    x14 = s14
    x15 = s15

    for i in range(10):
        x0, x4, x8, x12 = quarter_round(x0, x4, x8, x12)
        x1, x5, x9, x13 = quarter_round(x1, x5, x9, x13)
        x2, x6, x10, x14 = quarter_round(x2, x6, x10, x14)
        x3, x7, x11, x15 = quarter_round(x3, x7, x11, x15)

        x0, x5, x10, x15 = quarter_round(x0, x5, x10, x15)
        x1, x6, x11, x12 = quarter_round(x1, x6, x11, x12)
        x2, x7, x8, x13 = quarter_round(x2, x7, x8, x13)
        x3, x4, x9, x14 = quarter_round(x3, x4, x9, x14)

    store_u32_le(block_out, 0, x0 + s0)
    store_u32_le(block_out, 4, x1 + s1)
    store_u32_le(block_out, 8, x2 + s2)
    store_u32_le(block_out, 12, x3 + s3)
    store_u32_le(block_out, 16, x4 + s4)
    store_u32_le(block_out, 20, x5 + s5)
    store_u32_le(block_out, 24, x6 + s6)
    store_u32_le(block_out, 28, x7 + s7)
    store_u32_le(block_out, 32, x8 + s8)
    store_u32_le(block_out, 36, x9 + s9)
    store_u32_le(block_out, 40, x10 + s10)
    store_u32_le(block_out, 44, x11 + s11)
    store_u32_le(block_out, 48, x12 + s12)
    store_u32_le(block_out, 52, x13 + s13)
    store_u32_le(block_out, 56, x14 + s14)
    store_u32_le(block_out, 60, x15 + s15)


@verify
def poly1305_process_block(
    b0: u64,
    b1: u64,
    b2: u64,
    b3: u64,
    b4: u64,
    b5: u64,
    b6: u64,
    b7: u64,
    b8: u64,
    b9: u64,
    b10: u64,
    b11: u64,
    b12: u64,
    b13: u64,
    b14: u64,
    b15: u64,
    h0: u64,
    h1: u64,
    h2: u64,
    h3: u64,
    h4: u64,
    r0: u64,
    r1: u64,
    r2: u64,
    r3: u64,
    r4: u64,
    r1_5: u64,
    r2_5: u64,
    r3_5: u64,
    r4_5: u64,
) -> tuple[u64, u64, u64, u64, u64]:
    m0 = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) & 0x3FFFFFF
    m1 = ((b3 >> 2) | (b4 << 6) | (b5 << 14) | (b6 << 22)) & 0x3FFFFFF
    m2 = ((b6 >> 4) | (b7 << 4) | (b8 << 12) | (b9 << 20)) & 0x3FFFFFF
    m3 = ((b9 >> 6) | (b10 << 2) | (b11 << 10) | (b12 << 18)) & 0x3FFFFFF
    m4 = (
        (b12 >> 8) | (b13 << 0) | (b14 << 8) | (b15 << 16) | (u64(1) << 24)
    ) & 0x3FFFFFF

    h0 = h0 + m0
    h1 = h1 + m1
    h2 = h2 + m2
    h3 = h3 + m3
    h4 = h4 + m4

    d0 = h0 * r0 + h1 * r4_5 + h2 * r3_5 + h3 * r2_5 + h4 * r1_5
    d1 = h0 * r1 + h1 * r0 + h2 * r4_5 + h3 * r3_5 + h4 * r2_5
    d2 = h0 * r2 + h1 * r1 + h2 * r0 + h3 * r4_5 + h4 * r3_5
    d3 = h0 * r3 + h1 * r2 + h2 * r1 + h3 * r0 + h4 * r4_5
    d4 = h0 * r4 + h1 * r3 + h2 * r2 + h3 * r1 + h4 * r0

    h0 = d0 & 0x3FFFFFF
    carry = d0 >> 26

    d1 = d1 + carry
    h1 = d1 & 0x3FFFFFF
    carry = d1 >> 26

    d2 = d2 + carry
    h2 = d2 & 0x3FFFFFF
    carry = d2 >> 26

    d3 = d3 + carry
    h3 = d3 & 0x3FFFFFF
    carry = d3 >> 26

    d4 = d4 + carry
    h4 = d4 & 0x3FFFFFF
    carry = d4 >> 26

    h0 = h0 + carry * 5
    carry = h0 >> 26
    h0 = h0 & 0x3FFFFFF
    h1 = h1 + carry

    return h0, h1, h2, h3, h4


@verify
def poly1305_mac_aead(
    aad: Buffer[u8],
    ciphertext: Buffer[u8],
    otk: Buffer[u8],
    tag: Buffer[u8],
) -> None:
    assert len(otk) == 32
    assert len(tag) == 16

    r0 = (
        u64(otk[0]) | (u64(otk[1]) << 8) | (u64(otk[2]) << 16) | (u64(otk[3]) << 24)
    ) & 0x3FFFFFF
    r1 = (
        (u64(otk[3]) >> 2)
        | (u64(otk[4]) << 6)
        | (u64(otk[5]) << 14)
        | (u64(otk[6]) << 22)
    ) & 0x3FFFF03
    r2 = (
        (u64(otk[6]) >> 4)
        | (u64(otk[7]) << 4)
        | (u64(otk[8]) << 12)
        | (u64(otk[9]) << 20)
    ) & 0x3FFC0FF
    r3 = (
        (u64(otk[9]) >> 6)
        | (u64(otk[10]) << 2)
        | (u64(otk[11]) << 10)
        | (u64(otk[12]) << 18)
    ) & 0x3F03FFF
    r4 = (
        (u64(otk[12]) >> 8)
        | (u64(otk[13]) << 0)
        | (u64(otk[14]) << 8)
        | (u64(otk[15]) << 16)
    ) & 0x00FFFFF

    r1_5 = r1 * 5
    r2_5 = r2 * 5
    r3_5 = r3 * 5
    r4_5 = r4 * 5

    h0 = u64(0)
    h1 = u64(0)
    h2 = u64(0)
    h3 = u64(0)
    h4 = u64(0)

    aad_len = len(aad)
    num_aad_blocks = (aad_len + 15) // 16
    for b in range(num_aad_blocks):
        offset = b * 16
        b0 = u64(aad[offset]) if offset < aad_len else u64(0)
        b1 = u64(aad[offset + 1]) if offset + 1 < aad_len else u64(0)
        b2 = u64(aad[offset + 2]) if offset + 2 < aad_len else u64(0)
        b3 = u64(aad[offset + 3]) if offset + 3 < aad_len else u64(0)
        b4 = u64(aad[offset + 4]) if offset + 4 < aad_len else u64(0)
        b5 = u64(aad[offset + 5]) if offset + 5 < aad_len else u64(0)
        b6 = u64(aad[offset + 6]) if offset + 6 < aad_len else u64(0)
        b7 = u64(aad[offset + 7]) if offset + 7 < aad_len else u64(0)
        b8 = u64(aad[offset + 8]) if offset + 8 < aad_len else u64(0)
        b9 = u64(aad[offset + 9]) if offset + 9 < aad_len else u64(0)
        b10 = u64(aad[offset + 10]) if offset + 10 < aad_len else u64(0)
        b11 = u64(aad[offset + 11]) if offset + 11 < aad_len else u64(0)
        b12 = u64(aad[offset + 12]) if offset + 12 < aad_len else u64(0)
        b13 = u64(aad[offset + 13]) if offset + 13 < aad_len else u64(0)
        b14 = u64(aad[offset + 14]) if offset + 14 < aad_len else u64(0)
        b15 = u64(aad[offset + 15]) if offset + 15 < aad_len else u64(0)

        h0, h1, h2, h3, h4 = poly1305_process_block(
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            h0,
            h1,
            h2,
            h3,
            h4,
            r0,
            r1,
            r2,
            r3,
            r4,
            r1_5,
            r2_5,
            r3_5,
            r4_5,
        )

    ct_len = len(ciphertext)
    num_ct_blocks = (ct_len + 15) // 16
    for b in range(num_ct_blocks):
        offset = b * 16
        b0 = u64(ciphertext[offset]) if offset < ct_len else u64(0)
        b1 = u64(ciphertext[offset + 1]) if offset + 1 < ct_len else u64(0)
        b2 = u64(ciphertext[offset + 2]) if offset + 2 < ct_len else u64(0)
        b3 = u64(ciphertext[offset + 3]) if offset + 3 < ct_len else u64(0)
        b4 = u64(ciphertext[offset + 4]) if offset + 4 < ct_len else u64(0)
        b5 = u64(ciphertext[offset + 5]) if offset + 5 < ct_len else u64(0)
        b6 = u64(ciphertext[offset + 6]) if offset + 6 < ct_len else u64(0)
        b7 = u64(ciphertext[offset + 7]) if offset + 7 < ct_len else u64(0)
        b8 = u64(ciphertext[offset + 8]) if offset + 8 < ct_len else u64(0)
        b9 = u64(ciphertext[offset + 9]) if offset + 9 < ct_len else u64(0)
        b10 = u64(ciphertext[offset + 10]) if offset + 10 < ct_len else u64(0)
        b11 = u64(ciphertext[offset + 11]) if offset + 11 < ct_len else u64(0)
        b12 = u64(ciphertext[offset + 12]) if offset + 12 < ct_len else u64(0)
        b13 = u64(ciphertext[offset + 13]) if offset + 13 < ct_len else u64(0)
        b14 = u64(ciphertext[offset + 14]) if offset + 14 < ct_len else u64(0)
        b15 = u64(ciphertext[offset + 15]) if offset + 15 < ct_len else u64(0)

        h0, h1, h2, h3, h4 = poly1305_process_block(
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            h0,
            h1,
            h2,
            h3,
            h4,
            r0,
            r1,
            r2,
            r3,
            r4,
            r1_5,
            r2_5,
            r3_5,
            r4_5,
        )

    l0 = u64(aad_len & 0xFF)
    l1 = u64((aad_len >> 8) & 0xFF)
    l2 = u64((aad_len >> 16) & 0xFF)
    l3 = u64((aad_len >> 24) & 0xFF)
    l4 = u64((aad_len >> 32) & 0xFF)
    l5 = u64((aad_len >> 40) & 0xFF)
    l6 = u64((aad_len >> 48) & 0xFF)
    l7 = u64((aad_len >> 56) & 0xFF)

    l8 = u64(ct_len & 0xFF)
    l9 = u64((ct_len >> 8) & 0xFF)
    l10 = u64((ct_len >> 16) & 0xFF)
    l11 = u64((ct_len >> 24) & 0xFF)
    l12 = u64((ct_len >> 32) & 0xFF)
    l13 = u64((ct_len >> 40) & 0xFF)
    l14 = u64((ct_len >> 48) & 0xFF)
    l15 = u64((ct_len >> 56) & 0xFF)

    h0, h1, h2, h3, h4 = poly1305_process_block(
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
        l14,
        l15,
        h0,
        h1,
        h2,
        h3,
        h4,
        r0,
        r1,
        r2,
        r3,
        r4,
        r1_5,
        r2_5,
        r3_5,
        r4_5,
    )

    carry = h0 >> 26
    h0 = h0 & 0x3FFFFFF
    h1 = h1 + carry

    carry = h1 >> 26
    h1 = h1 & 0x3FFFFFF
    h2 = h2 + carry

    carry = h2 >> 26
    h2 = h2 & 0x3FFFFFF
    h3 = h3 + carry

    carry = h3 >> 26
    h3 = h3 & 0x3FFFFFF
    h4 = h4 + carry

    carry = h4 >> 26
    h4 = h4 & 0x3FFFFFF
    h0 = h0 + carry * 5

    c = (h0 + 5) >> 26
    c = (h1 + c) >> 26
    c = (h2 + c) >> 26
    c = (h3 + c) >> 26
    c = (h4 + c) >> 26

    h0_sub = h0 + 5
    h1_sub = h1 + (h0_sub >> 26)
    h2_sub = h2 + (h1_sub >> 26)
    h3_sub = h3 + (h2_sub >> 26)
    h4_sub = h4 + (h3_sub >> 26)

    h0 = h0_sub & 0x3FFFFFF if c == 1 else h0
    h1 = h1_sub & 0x3FFFFFF if c == 1 else h1
    h2 = h2_sub & 0x3FFFFFF if c == 1 else h2
    h3 = h3_sub & 0x3FFFFFF if c == 1 else h3
    h4 = h4_sub & 0x3FFFFFF if c == 1 else h4

    w0 = (h0 | (h1 << 26)) & 0xFFFFFFFF
    w1 = ((h1 >> 6) | (h2 << 20)) & 0xFFFFFFFF
    w2 = ((h2 >> 12) | (h3 << 14)) & 0xFFFFFFFF
    w3 = ((h3 >> 18) | (h4 << 8)) & 0xFFFFFFFF

    s0 = (
        u64(otk[16]) | (u64(otk[17]) << 8) | (u64(otk[18]) << 16) | (u64(otk[19]) << 24)
    )
    s1 = (
        u64(otk[20]) | (u64(otk[21]) << 8) | (u64(otk[22]) << 16) | (u64(otk[23]) << 24)
    )
    s2 = (
        u64(otk[24]) | (u64(otk[25]) << 8) | (u64(otk[26]) << 16) | (u64(otk[27]) << 24)
    )
    s3 = (
        u64(otk[28]) | (u64(otk[29]) << 8) | (u64(otk[30]) << 16) | (u64(otk[31]) << 24)
    )

    f0 = w0 + s0
    carry = f0 >> 32
    f0 = f0 & 0xFFFFFFFF

    f1 = w1 + s1 + carry
    carry = f1 >> 32
    f1 = f1 & 0xFFFFFFFF

    f2 = w2 + s2 + carry
    carry = f2 >> 32
    f2 = f2 & 0xFFFFFFFF

    f3 = w3 + s3 + carry
    f3 = f3 & 0xFFFFFFFF

    store_u32_le(tag, 0, u32(f0))
    store_u32_le(tag, 4, u32(f1))
    store_u32_le(tag, 8, u32(f2))
    store_u32_le(tag, 12, u32(f3))
