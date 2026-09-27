#include "model/ShardedGgufFile.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char **argv) {
  try {
    std::string modelPath = (argc > 1) ? argv[1] : "/Users/nitin/models/qwen38-flash-next-v3";
    std::string sidecarPath = (argc > 2) ? argv[2] : "/Users/nitin/models/qwen38-flash-next-mtp/mtp-shared-Q4_K_M.gguf";

    std::cout << "Loading ShardedGgufFile from: " << modelPath << "\n";
    splash::model::ShardedGgufFile gguf(modelPath);

    std::cout << "Loaded " << gguf.shards().size() << " shards successfully!\n";
    for (size_t i = 0; i < gguf.shards().size(); ++i) {
      std::cout << "  Shard " << i << ": " << gguf.shards()[i].path.filename().string()
                << " (" << (gguf.shards()[i].fileSize / (1024 * 1024)) << " MB, dataOffset: "
                << gguf.shards()[i].dataOffset << ")\n";
    }

    std::cout << "Architecture: " << gguf.architecture() << "\n";
    std::cout << "Context length: " << gguf.unsignedValue("qwen4exp.context_length").value_or(0) << "\n";
    std::cout << "Block count: " << gguf.unsignedValue("qwen4exp.block_count").value_or(0) << "\n";
    std::cout << "Total tensors in main model: " << gguf.tensors().size() << "\n";

    if (std::filesystem::exists(sidecarPath)) {
      std::cout << "\nAdding MTP sidecar: " << sidecarPath << "\n";
      gguf.addSidecar(sidecarPath);
      std::cout << "Total tensors after sidecar: " << gguf.tensors().size() << "\n";
    }

    // Inspect critical tensors
    const char *checkTensors[] = {
      "output.weight",
      "token_embd.weight",
      "per_layer_token_embd.weight",
      "blk.0.hc_attn_norm.weight",
      "blk.0.attn_qkv.weight",
      "blk.0.ffn_gate_exps.weight",
      "blk.0.ffn_down_exps.weight",
      "blk.3.attn_output.weight",
      "blk.48.attn_q.weight",
      "blk.48.nextn.fc_embd.weight",
    };

    std::cout << "\nInspecting sample tensors:\n";
    for (const char *name : checkTensors) {
      const auto *t = gguf.find(name);
      if (!t) {
        std::cout << "  MISSING: " << name << "\n";
        continue;
      }
      std::cout << "  Found " << std::left << std::setw(32) << t->name
                << " shape=[";
      for (size_t d = 0; d < t->dims.size(); ++d) {
        if (d > 0) std::cout << ", ";
        std::cout << t->dims[d];
      }
      std::cout << "] type=" << splash::model::ggmlTypeName(t->type)
                << " (" << (t->bytes / 1024) << " KB) in shard " << t->shardIndex << "\n";

      // Test reading first 64 bytes
      std::vector<uint8_t> sample(std::min<size_t>(t->bytes, 64));
      gguf.readTensorRange(*t, 0, sample);
    }

    std::cout << "\nAll checks passed successfully!\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
