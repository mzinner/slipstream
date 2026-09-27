import os
import struct
import ctypes
from pathlib import Path
import numpy as np

# Load ggml library for fast dequantization
GGML_LIB_PATH = "/Users/nitin/Documents/shared-with-google-drive/model-serving/llama.cpp-prism/build/bin/libggml-base.0.21.0.dylib"
if not os.path.exists(GGML_LIB_PATH):
    raise RuntimeError(f"Cannot find ggml library at {GGML_LIB_PATH}")

lib = ctypes.CDLL(GGML_LIB_PATH)
lib.dequantize_row_q4_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
lib.dequantize_row_q4_1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
lib.dequantize_row_q5_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
lib.dequantize_row_q8_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
lib.dequantize_row_q4_K.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
lib.dequantize_row_q6_K.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

class MultiShardGgufReader:
    def __init__(self, model_dir_or_files, sidecars=None):
        if isinstance(model_dir_or_files, (str, Path)):
            p = Path(model_dir_or_files)
            if p.is_dir():
                shard_paths = sorted(p.glob("*.gguf"))
            else:
                shard_paths = [p]
        else:
            shard_paths = [Path(f) for f in model_dir_or_files]

        if sidecars:
            if isinstance(sidecars, (str, Path)):
                sidecars = [sidecars]
            for s in sidecars:
                sp = Path(s)
                if sp.exists() and sp not in shard_paths:
                    shard_paths.append(sp)

        self.shards = []
        self.tensors = {}
        self.metadata = {}

        for shard_idx, shard_path in enumerate(shard_paths):
            fd = open(shard_path, "rb")
            magic = fd.read(4)
            if magic != b"GGUF":
                fd.close()
                continue
            ver, n_tensors, n_kv = struct.unpack("<IQQ", fd.read(20))
            
            # Read KV metadata (on shard 0 or all shards)
            for _ in range(n_kv):
                klen = struct.unpack("<Q", fd.read(8))[0]
                k = fd.read(klen).decode("latin1", errors="replace")
                vtype = struct.unpack("<I", fd.read(4))[0]
                if vtype in (0, 1, 7):
                    val = fd.read(1)
                elif vtype in (2, 3):
                    val = fd.read(2)
                elif vtype in (4, 5, 6):
                    val = fd.read(4)
                elif vtype in (10, 11, 12):
                    val = fd.read(8)
                elif vtype == 8:
                    slen = struct.unpack("<Q", fd.read(8))[0]
                    val = fd.read(slen).decode("latin1", errors="replace")
                elif vtype == 9:
                    etype, count = struct.unpack("<IQ", fd.read(12))
                    sz = 1 if etype in (0,1,7) else 2 if etype in (2,3) else 4 if etype in (4,5,6) else 8
                    if etype == 8:
                        val = []
                        for _ in range(count):
                            slen = struct.unpack("<Q", fd.read(8))[0]
                            val.append(fd.read(slen).decode("latin1", errors="replace"))
                    else:
                        val = fd.read(count * sz)
                self.metadata[k] = val

            # Read tensor metadata
            for _ in range(n_tensors):
                nlen = struct.unpack("<Q", fd.read(8))[0]
                name = fd.read(nlen).decode("latin1", errors="replace")
                ndims = struct.unpack("<I", fd.read(4))[0]
                dims = struct.unpack(f"<{ndims}Q", fd.read(ndims * 8))
                ttype, toffset = struct.unpack("<IQ", fd.read(12))
                self.tensors[name] = {
                    "shard_idx": shard_idx,
                    "dims": dims,
                    "type": ttype,
                    "offset": toffset
                }

            pos = fd.tell()
            align = 32
            pad = (align - pos % align) % align
            data_offset = pos + pad

            self.shards.append({
                "path": shard_path,
                "fd": fd,
                "data_offset": data_offset
            })

    def has(self, name):
        return name in self.tensors

    def read_tensor(self, name):
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found in GGUF shards")
        meta = self.tensors[name]
        shard = self.shards[meta["shard_idx"]]
        fd = shard["fd"]
        offset = shard["data_offset"] + meta["offset"]
        dims = meta["dims"]
        ttype = meta["type"]
        shape = tuple(reversed(dims))
        count = int(np.prod(dims))

        fd.seek(offset)
        if ttype == 0: # F32
            return np.frombuffer(fd.read(count * 4), dtype="<f4").reshape(shape)
        elif ttype == 1: # F16
            return np.frombuffer(fd.read(count * 2), dtype="<f2").astype(np.float32).reshape(shape)
        elif ttype == 30: # BF16
            u16 = np.frombuffer(fd.read(count * 2), dtype="<u2")
            return ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(shape)

        out = np.empty(shape, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.c_void_p)

        if ttype == 2: # Q4_0
            raw = fd.read(count // 32 * 18)
            lib.dequantize_row_q4_0(raw, ptr, count)
        elif ttype == 3: # Q4_1
            raw = fd.read(count // 32 * 20)
            lib.dequantize_row_q4_1(raw, ptr, count)
        elif ttype == 6: # Q5_0
            raw = fd.read(count // 32 * 22)
            lib.dequantize_row_q5_0(raw, ptr, count)
        elif ttype == 8: # Q8_0
            raw = fd.read(count // 32 * 34)
            lib.dequantize_row_q8_0(raw, ptr, count)
        elif ttype == 12: # Q4_K
            raw = fd.read(count // 256 * 144)
            lib.dequantize_row_q4_K(raw, ptr, count)
        elif ttype == 14: # Q6_K
            raw = fd.read(count // 256 * 210)
            lib.dequantize_row_q6_K(raw, ptr, count)
        else:
            raise ValueError(f"Unsupported ggml type {ttype} for tensor {name}")

        return out

    def close(self):
        for s in self.shards:
            s["fd"].close()
