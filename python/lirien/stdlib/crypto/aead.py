from lirien import verify, Buffer, u8, u64
from .primitives import hchacha20, chacha20_block, poly1305_mac_aead


@verify
def xchacha20poly1305_encrypt(
    key: Buffer[u8],
    nonce: Buffer[u8],
    plaintext: Buffer[u8],
    aad: Buffer[u8],
    ciphertext: Buffer[u8],
    tag: Buffer[u8],
) -> None:
    assert len(key) == 32
    assert len(nonce) == 24
    assert len(tag) == 16
    assert len(ciphertext) == len(plaintext)

    sub256 = Buffer.alloc(32, u8)
    n0 = nonce[0:16]
    hchacha20(key, n0, sub256)

    otk_block = Buffer.alloc(64, u8)
    n1 = nonce[16:24]
    chacha20_block(sub256, u64(0), n1, otk_block)
    otk = otk_block[0:32]

    pt_len = len(plaintext)
    num_blocks = (pt_len + 63) // 64
    block_out = Buffer.alloc(64, u8)

    for b in range(num_blocks):
        chacha20_block(sub256, u64(b + 1), n1, block_out)
        offset = b * 64
        rem = pt_len - offset
        limit = 64
        if rem < 64:
            limit = rem

        for i in range(limit):
            ciphertext[offset + i] = plaintext[offset + i] ^ block_out[i]

    poly1305_mac_aead(aad, ciphertext, otk, tag)


@verify
def xchacha20poly1305_decrypt(
    key: Buffer[u8],
    nonce: Buffer[u8],
    ciphertext: Buffer[u8],
    aad: Buffer[u8],
    tag: Buffer[u8],
    plaintext: Buffer[u8],
) -> None:
    assert len(key) == 32
    assert len(nonce) == 24
    assert len(tag) == 16
    assert len(plaintext) == len(ciphertext)

    sub256 = Buffer.alloc(32, u8)
    n0 = nonce[0:16]
    hchacha20(key, n0, sub256)

    otk_block = Buffer.alloc(64, u8)
    n1 = nonce[16:24]
    chacha20_block(sub256, u64(0), n1, otk_block)
    otk = otk_block[0:32]

    computed_tag = Buffer.alloc(16, u8)
    poly1305_mac_aead(aad, ciphertext, otk, computed_tag)

    diff = u8(0)
    for i in range(16):
        diff = diff | (tag[i] ^ computed_tag[i])
    assert diff == 0

    ct_len = len(ciphertext)
    num_blocks = (ct_len + 63) // 64
    block_out = Buffer.alloc(64, u8)

    for b in range(num_blocks):
        chacha20_block(sub256, u64(b + 1), n1, block_out)
        offset = b * 64
        rem = ct_len - offset
        limit = 64
        if rem < 64:
            limit = rem

        for i in range(limit):
            plaintext[offset + i] = ciphertext[offset + i] ^ block_out[i]
