#!/usr/bin/env python3
"""Generate a tiny but structurally-valid `qwen3moe` GGUF file.

Exercises the Quasar GGUF reader, Qwen3-MoE loader, and the byte-level BPE
tokenizer end to end (scalars, floats, strings, int32 + string arrays, 1D/2D/3D
tensors) without downloading a real model. The vocab is the real GPT-2 byte-level
alphabet plus one merge and three special tokens, so encode/decode round-trips.
Tensor *data* is zero-filled; only layout/metadata matter here.
"""
import struct
import sys

GGUF_MAGIC = 0x46554747
GGML_F32 = 0
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)


def bytes_to_unicode():
    """GPT-2 / Qwen byte-level alphabet: byte -> printable unicode char."""
    bs = list(range(33, 127)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def w_str(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b))
    buf += b


class KV:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.n = 0

    def _k(self, k: str) -> None:
        w_str(self.buf, k)
        self.n += 1

    def u32(self, k, v): self._k(k); self.buf += struct.pack("<I", U32); self.buf += struct.pack("<I", v)
    def i32(self, k, v): self._k(k); self.buf += struct.pack("<I", I32); self.buf += struct.pack("<i", v)
    def f32(self, k, v): self._k(k); self.buf += struct.pack("<I", F32); self.buf += struct.pack("<f", v)
    def s(self, k, v):   self._k(k); self.buf += struct.pack("<I", STRING); w_str(self.buf, v)

    def arr_str(self, k, vals):
        self._k(k)
        self.buf += struct.pack("<I", ARRAY) + struct.pack("<I", STRING) + struct.pack("<Q", len(vals))
        for v in vals:
            w_str(self.buf, v)

    def arr_i32(self, k, vals):
        self._k(k)
        self.buf += struct.pack("<I", ARRAY) + struct.pack("<I", I32) + struct.pack("<Q", len(vals))
        for v in vals:
            self.buf += struct.pack("<i", v)


def align(x: int, a: int = 32) -> int:
    return (x + a - 1) & ~(a - 1)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/quasar_tiny.gguf"

    n_embd, n_head, n_head_kv, head_dim = 8, 2, 1, 4
    n_layer, n_expert, n_expert_used, n_ff = 2, 4, 2, 6
    ctx = 32
    q_out, kv_out = n_head * head_dim, n_head_kv * head_dim

    # Vocab: 256 byte-level base tokens (id == byte) + "he" + 3 specials.
    b2u = bytes_to_unicode()
    vocab = [b2u[b] for b in range(256)]
    types = [1] * 256                      # 1 = NORMAL
    vocab.append("he");            types.append(1)       # id 256
    vocab.append("<|endoftext|>"); types.append(3)       # id 257  (3 = CONTROL)
    vocab.append("<|im_start|>");  types.append(3)       # id 258
    vocab.append("<|im_end|>");    types.append(3)       # id 259
    merges = ["h e"]                                      # so "he" forms
    n_vocab = len(vocab)
    bos_id, eos_id = 257, 259

    kv = KV()
    kv.s("general.architecture", "qwen3moe")
    kv.s("general.name", "tiny-qwen3moe-test")
    kv.u32("qwen3moe.block_count", n_layer)
    kv.u32("qwen3moe.context_length", ctx)
    kv.u32("qwen3moe.embedding_length", n_embd)
    kv.u32("qwen3moe.feed_forward_length", n_ff * 4)
    kv.u32("qwen3moe.expert_feed_forward_length", n_ff)
    kv.u32("qwen3moe.attention.head_count", n_head)
    kv.u32("qwen3moe.attention.head_count_kv", n_head_kv)
    kv.u32("qwen3moe.attention.key_length", head_dim)
    kv.u32("qwen3moe.attention.value_length", head_dim)
    kv.f32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
    kv.u32("qwen3moe.expert_count", n_expert)
    kv.u32("qwen3moe.expert_used_count", n_expert_used)
    kv.f32("qwen3moe.rope.freq_base", 1_000_000.0)
    kv.u32("qwen3moe.vocab_size", n_vocab)
    kv.s("tokenizer.ggml.model", "gpt2")
    kv.arr_str("tokenizer.ggml.tokens", vocab)
    kv.arr_i32("tokenizer.ggml.token_type", types)
    kv.arr_str("tokenizer.ggml.merges", merges)
    kv.i32("tokenizer.ggml.bos_token_id", bos_id)
    kv.i32("tokenizer.ggml.eos_token_id", eos_id)

    tensors = [("token_embd.weight", [n_embd, n_vocab])]
    for il in range(n_layer):
        p = f"blk.{il}."
        tensors += [
            (p + "attn_norm.weight",     [n_embd]),
            (p + "attn_q.weight",        [n_embd, q_out]),
            (p + "attn_k.weight",        [n_embd, kv_out]),
            (p + "attn_v.weight",        [n_embd, kv_out]),
            (p + "attn_output.weight",   [q_out, n_embd]),
            (p + "attn_q_norm.weight",   [head_dim]),
            (p + "attn_k_norm.weight",   [head_dim]),
            (p + "ffn_norm.weight",      [n_embd]),
            (p + "ffn_gate_inp.weight",  [n_embd, n_expert]),
            (p + "ffn_gate_exps.weight", [n_embd, n_ff, n_expert]),
            (p + "ffn_up_exps.weight",   [n_embd, n_ff, n_expert]),
            (p + "ffn_down_exps.weight", [n_ff, n_embd, n_expert]),
        ]
    tensors += [("output_norm.weight", [n_embd]), ("output.weight", [n_embd, n_vocab])]

    offsets, sizes, off = [], [], 0
    for _, dims in tensors:
        nb = 4
        for d in dims:
            nb *= d
        off = align(off)
        offsets.append(off)
        sizes.append(nb)
        off += nb

    ti = bytearray()
    for (name, dims), offv in zip(tensors, offsets):
        w_str(ti, name)
        ti += struct.pack("<I", len(dims))
        for d in dims:
            ti += struct.pack("<Q", d)
        ti += struct.pack("<I", GGML_F32)
        ti += struct.pack("<Q", offv)

    header = bytearray()
    header += struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", kv.n)
    header += kv.buf + ti

    data_off = align(len(header))
    blob = bytearray(header)
    blob += b"\x00" * (data_off - len(header))
    for (name, dims), offv, nb in zip(tensors, offsets, sizes):
        start = data_off + offv
        if len(blob) < start:
            blob += b"\x00" * (start - len(blob))
        blob += b"\x00" * nb

    with open(out, "wb") as f:
        f.write(blob)
    print(f"wrote {out}: {len(tensors)} tensors, {kv.n} kv, {n_vocab} vocab, {len(blob)} bytes")


if __name__ == "__main__":
    main()
