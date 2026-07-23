import unittest
from lirien import u8, Buffer, crypto


class TestStdlibCrypto(unittest.TestCase):
    def test_xchacha20poly1305_vector(self):
        # Key: 32 bytes (80..9f)
        key_bytes = bytes(range(0x80, 0xA0))
        key = Buffer.alloc(32, u8)
        for i in range(32):
            key[i] = key_bytes[i]

        # Nonce: 24 bytes (40..57)
        nonce_bytes = bytes(range(0x40, 0x58))
        nonce = Buffer.alloc(24, u8)
        for i in range(24):
            nonce[i] = nonce_bytes[i]

        # Plaintext: 114 bytes
        plaintext_str = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it."
        plaintext_bytes = plaintext_str.encode("utf-8")
        plaintext = Buffer.alloc(114, u8)
        for i in range(114):
            plaintext[i] = plaintext_bytes[i]

        # AAD: 12 bytes
        aad_bytes = b"PQRS" + bytes([0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7])
        aad = Buffer.alloc(12, u8)
        for i in range(12):
            aad[i] = aad_bytes[i]

        # Ciphertext output buffer
        ciphertext = Buffer.alloc(114, u8)
        tag = Buffer.alloc(16, u8)

        # Encrypt
        crypto.xchacha20poly1305_encrypt(key, nonce, plaintext, aad, ciphertext, tag)

        # Verify Ciphertext
        expected_ct = bytes(
            [
                0xBD,
                0x6D,
                0x17,
                0x9D,
                0x3E,
                0x83,
                0xD4,
                0x3B,
                0x95,
                0x76,
                0x57,
                0x94,
                0x93,
                0xC0,
                0xE9,
                0x39,
                0x57,
                0x2A,
                0x17,
                0x00,
                0x25,
                0x2B,
                0xFA,
                0xCC,
                0xBE,
                0xD2,
                0x90,
                0x2C,
                0x21,
                0x39,
                0x6C,
                0xBB,
                0x73,
                0x1C,
                0x7F,
                0x1B,
                0x0B,
                0x4A,
                0xA6,
                0x44,
                0x0B,
                0xF3,
                0xA8,
                0x2F,
                0x4E,
                0xDA,
                0x7E,
                0x39,
                0xAE,
                0x64,
                0xC6,
                0x70,
                0x8C,
                0x54,
                0xC2,
                0x16,
                0xCB,
                0x96,
                0xB7,
                0x2E,
                0x12,
                0x13,
                0xB4,
                0x52,
                0x2F,
                0x8C,
                0x9B,
                0xA4,
                0x0D,
                0xB5,
                0xD9,
                0x45,
                0xB1,
                0x1B,
                0x69,
                0xB9,
                0x82,
                0xC1,
                0xBB,
                0x9E,
                0x3F,
                0x3F,
                0xAC,
                0x2B,
                0xC3,
                0x69,
                0x48,
                0x8F,
                0x76,
                0xB2,
                0x38,
                0x35,
                0x65,
                0xD3,
                0xFF,
                0xF9,
                0x21,
                0xF9,
                0x66,
                0x4C,
                0x97,
                0x63,
                0x7D,
                0xA9,
                0x76,
                0x88,
                0x12,
                0xF6,
                0x15,
                0xC6,
                0x8B,
                0x13,
                0xB5,
                0x2E,
            ]
        )

        for i in range(114):
            self.assertEqual(
                ciphertext[i], expected_ct[i], f"Ciphertext mismatch at byte {i}"
            )

        # Verify Tag
        expected_tag = bytes(
            [
                0xC0,
                0x87,
                0x59,
                0x24,
                0xC1,
                0xC7,
                0x98,
                0x79,
                0x47,
                0xDE,
                0xAF,
                0xD8,
                0x78,
                0x0A,
                0xCF,
                0x49,
            ]
        )
        for i in range(16):
            self.assertEqual(tag[i], expected_tag[i], f"Tag mismatch at byte {i}")

        # Decrypt
        decrypted = Buffer.alloc(114, u8)
        crypto.xchacha20poly1305_decrypt(key, nonce, ciphertext, aad, tag, decrypted)

        for i in range(114):
            self.assertEqual(
                decrypted[i], plaintext[i], f"Decryption mismatch at byte {i}"
            )


if __name__ == "__main__":
    unittest.main()
