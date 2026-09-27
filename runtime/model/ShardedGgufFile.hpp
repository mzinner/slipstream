#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

class GgufError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// ggml type ids as stored in GGUF tensor infos.
namespace ggml {
inline constexpr uint32_t kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ4_1 = 3,
                          kQ8_0 = 8, kQ3_K = 11, kQ4_K = 12,
                          kQ5_K = 13, kQ6_K = 14, kIQ4_NL = 20, kIQ3_S = 21,
                          kIQ4_XS = 23, kBF16 = 30;
}

struct GgmlTypeTraits {
  const char *name;
  uint32_t blockElements;
  uint32_t blockBytes;
};

[[nodiscard]] const GgmlTypeTraits *ggmlTypeTraits(uint32_t type) noexcept;
[[nodiscard]] std::string ggmlTypeName(uint32_t type);

struct ShardedGgufTensor {
  std::string name;
  uint32_t type = 0;
  std::vector<uint64_t> dims; // dims[0] is fastest (columns)
  uint64_t offset = 0;        // byte offset in shard's data section
  uint64_t bytes = 0;         // total data bytes
  size_t shardIndex = 0;      // which shard owns this tensor

  [[nodiscard]] uint64_t columns() const noexcept { return dims.empty() ? 0 : dims[0]; }
  [[nodiscard]] uint64_t rows() const;
  [[nodiscard]] uint64_t elements() const;
};

struct GgufShard {
  std::filesystem::path path;
  int fd = -1;
  uint64_t fileSize = 0;
  uint64_t dataOffset = 0;
  uint32_t splitNo = 0;
};

class ShardedGgufFile final {
public:
  // Opens a GGUF file or directory of shards (e.g. dir/ or *-00001-of-00003.gguf).
  explicit ShardedGgufFile(const std::filesystem::path &path);
  ~ShardedGgufFile();

  ShardedGgufFile(const ShardedGgufFile &) = delete;
  ShardedGgufFile &operator=(const ShardedGgufFile &) = delete;
  ShardedGgufFile(ShardedGgufFile &&) noexcept;
  ShardedGgufFile &operator=(ShardedGgufFile &&) noexcept;

  // Add an auxiliary/sidecar GGUF (e.g. mtp draft sidecar).
  void addSidecar(const std::filesystem::path &sidecarPath);

  [[nodiscard]] const std::string &architecture() const noexcept { return architecture_; }
  [[nodiscard]] const std::vector<GgufShard> &shards() const noexcept { return shards_; }

  [[nodiscard]] std::optional<uint64_t> unsignedValue(std::string_view key) const;
  [[nodiscard]] std::optional<std::string> stringValue(std::string_view key) const;
  [[nodiscard]] std::optional<double> floatValue(std::string_view key) const;
  [[nodiscard]] std::optional<std::span<const double>> numericArray(std::string_view key) const;

  [[nodiscard]] const std::vector<ShardedGgufTensor> &tensors() const noexcept { return tensors_; }
  [[nodiscard]] bool has(std::string_view name) const noexcept;
  [[nodiscard]] const ShardedGgufTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const ShardedGgufTensor &require(std::string_view name) const;

  // Read data of a tensor into destination.
  void readTensor(const ShardedGgufTensor &tensor, std::span<uint8_t> dst) const;
  void readTensorRange(const ShardedGgufTensor &tensor, uint64_t fromByte, std::span<uint8_t> dst) const;

private:
  void parseShard(size_t shardIndex, bool readMetadata);

  std::vector<GgufShard> shards_;
  std::string architecture_;
  std::map<std::string, uint64_t, std::less<>> unsigned_;
  std::map<std::string, std::string, std::less<>> strings_;
  std::map<std::string, double, std::less<>> floats_;
  std::map<std::string, std::vector<double>, std::less<>> arrays_;
  std::vector<ShardedGgufTensor> tensors_;
  std::map<std::string, size_t, std::less<>> index_;
};

} // namespace splash::model
