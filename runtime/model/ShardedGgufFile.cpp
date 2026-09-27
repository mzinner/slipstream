#include "model/ShardedGgufFile.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <regex>

namespace splash::model {
namespace {

uint64_t checkedMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw GgufError("GGUF size overflows uint64");
  return a * b;
}

constexpr std::array<std::pair<uint32_t, GgmlTypeTraits>, 30> kTypes{{
    {0, {"F32", 1, 4}},         {1, {"F16", 1, 2}},         {2, {"Q4_0", 32, 18}},
    {3, {"Q4_1", 32, 20}},      {6, {"Q5_0", 32, 22}},      {7, {"Q5_1", 32, 24}},
    {8, {"Q8_0", 32, 34}},      {9, {"Q8_1", 32, 36}},      {10, {"Q2_K", 256, 84}},
    {11, {"Q3_K", 256, 110}},   {12, {"Q4_K", 256, 144}},   {13, {"Q5_K", 256, 176}},
    {14, {"Q6_K", 256, 210}},   {15, {"Q8_K", 256, 292}},   {16, {"IQ2_XXS", 256, 66}},
    {17, {"IQ2_XS", 256, 74}},  {18, {"IQ3_XXS", 256, 98}}, {19, {"IQ1_S", 256, 50}},
    {20, {"IQ4_NL", 32, 18}},   {21, {"IQ3_S", 256, 110}},  {22, {"IQ2_S", 256, 82}},
    {23, {"IQ4_XS", 256, 136}}, {24, {"I8", 1, 1}},         {25, {"I16", 1, 2}},
    {26, {"I32", 1, 4}},        {27, {"I64", 1, 8}},        {28, {"F64", 1, 8}},
    {29, {"IQ1_M", 256, 56}},   {30, {"BF16", 1, 2}},       {39, {"MXFP4", 32, 17}},
}};

enum ValueType : uint32_t {
  kUint8 = 0, kInt8 = 1, kUint16 = 2, kInt16 = 3, kUint32 = 4, kInt32 = 5,
  kFloat32 = 6, kBool = 7, kString = 8, kArray = 9, kUint64 = 10, kInt64 = 11,
  kFloat64 = 12,
};

void preadBytes(int fd, uint64_t offset, std::span<uint8_t> dst) {
  uint8_t *p = dst.data();
  size_t remaining = dst.size();
  while (remaining > 0) {
    const ssize_t n = pread(fd, p, remaining, static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw GgufError("failed to read from GGUF file descriptor");
    p += n;
    offset += n;
    remaining -= n;
  }
}

class Reader {
public:
  Reader(int fd, uint64_t fileSize) : fd_(fd), fileSize_(fileSize) {}

  void bytes(void *destination, uint64_t count) {
    requireRemaining(count);
    auto *to = static_cast<uint8_t *>(destination);
    while (count) {
      if (position_ < bufferStart_ || position_ >= bufferStart_ + buffered_) fill();
      const uint64_t at = position_ - bufferStart_;
      const uint64_t part = std::min(count, buffered_ - at);
      std::memcpy(to, buffer_.data() + at, part);
      to += part;
      count -= part;
      position_ += part;
    }
  }

  void skip(uint64_t count) {
    requireRemaining(count);
    position_ += count;
  }

  template <class T> T scalar() {
    T value{};
    bytes(&value, sizeof value);
    return value;
  }

  std::string string() {
    const uint64_t length = scalar<uint64_t>();
    if (length > (8u << 20) || retainedStrings_ > (8u << 20) - length)
      throw GgufError("GGUF string metadata exceeds bounds");
    retainedStrings_ += length;
    std::string value(length, '\0');
    if (length) bytes(value.data(), length);
    return value;
  }

  void skipString() { skip(scalar<uint64_t>()); }
  [[nodiscard]] uint64_t position() const noexcept { return position_; }

private:
  void requireRemaining(uint64_t count) const {
    if (count > fileSize_ - position_) throw GgufError("GGUF header is truncated");
  }

  void fill() {
    bufferStart_ = position_;
    buffered_ = std::min<uint64_t>(buffer_.size(), fileSize_ - position_);
    preadBytes(fd_, bufferStart_, std::span(buffer_).first(buffered_));
  }

  int fd_;
  uint64_t fileSize_;
  std::vector<uint8_t> buffer_ = std::vector<uint8_t>(1024 * 1024);
  uint64_t bufferStart_ = 0, buffered_ = 0;
  uint64_t position_ = 0;
  uint64_t retainedStrings_ = 0;
};

uint64_t scalarBytes(uint32_t type) {
  switch (type) {
  case kUint8: case kInt8: case kBool: return 1;
  case kUint16: case kInt16: return 2;
  case kUint32: case kInt32: case kFloat32: return 4;
  case kUint64: case kInt64: case kFloat64: return 8;
  default: throw GgufError("unknown GGUF value type " + std::to_string(type));
  }
}

constexpr uint64_t kMaximumKeptArray = 4096;

double numericValue(Reader &reader, uint32_t type) {
  switch (type) {
  case kUint8: return reader.scalar<uint8_t>();
  case kInt8: return reader.scalar<int8_t>();
  case kBool: return reader.scalar<uint8_t>() != 0;
  case kUint16: return reader.scalar<uint16_t>();
  case kInt16: return reader.scalar<int16_t>();
  case kUint32: return reader.scalar<uint32_t>();
  case kInt32: return reader.scalar<int32_t>();
  case kFloat32: return reader.scalar<float>();
  case kUint64: return double(reader.scalar<uint64_t>());
  case kInt64: return double(reader.scalar<int64_t>());
  case kFloat64: return reader.scalar<double>();
  default: throw GgufError("unknown GGUF value type " + std::to_string(type));
  }
}

void skipValue(Reader &reader, uint32_t type, unsigned depth);

void skipArray(Reader &reader, uint32_t element, uint64_t count, unsigned depth) {
  if (element == kString || element == kArray) {
    for (uint64_t i = 0; i < count; ++i) skipValue(reader, element, depth + 1);
  } else {
    reader.skip(checkedMultiply(count, scalarBytes(element)));
  }
}

void skipValue(Reader &reader, uint32_t type, unsigned depth) {
  if (depth > 16) throw GgufError("GGUF metadata nesting is too deep");
  if (type == kString) {
    reader.skipString();
  } else if (type == kArray) {
    const uint32_t element = reader.scalar<uint32_t>();
    skipArray(reader, element, reader.scalar<uint64_t>(), depth);
  } else {
    reader.skip(scalarBytes(type));
  }
}

} // namespace

const GgmlTypeTraits *ggmlTypeTraits(uint32_t type) noexcept {
  for (const auto &[id, traits] : kTypes)
    if (id == type) return &traits;
  return nullptr;
}

std::string ggmlTypeName(uint32_t type) {
  const GgmlTypeTraits *traits = ggmlTypeTraits(type);
  return traits ? traits->name : "type-" + std::to_string(type);
}

uint64_t ShardedGgufTensor::rows() const {
  uint64_t rows = 1;
  for (size_t i = 1; i < dims.size(); ++i) rows = checkedMultiply(rows, dims[i]);
  return rows;
}

uint64_t ShardedGgufTensor::elements() const {
  uint64_t elements = 1;
  for (uint64_t dim : dims) elements = checkedMultiply(elements, dim);
  return elements;
}

ShardedGgufFile::ShardedGgufFile(const std::filesystem::path &path) {
  std::vector<std::filesystem::path> shardPaths;
  if (std::filesystem::is_directory(path)) {
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
      if (entry.path().extension() == ".gguf") {
        shardPaths.push_back(entry.path());
      }
    }
    std::sort(shardPaths.begin(), shardPaths.end());
    if (shardPaths.empty())
      throw GgufError("directory contains no .gguf files: " + path.string());
  } else if (std::filesystem::is_regular_file(path)) {
    // Check if part of a split pattern like name-00001-of-00003.gguf
    std::string stem = path.stem().string();
    std::regex splitRegex(R"((.*)-([0-9]{5})-of-([0-9]{5})$)");
    std::smatch match;
    if (std::regex_match(stem, match, splitRegex)) {
      std::string base = match[1].str();
      int total = std::stoi(match[3].str());
      const auto parent = path.parent_path();
      for (int i = 1; i <= total; ++i) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%05d-of-%05d.gguf", i, total);
        std::filesystem::path p = parent / (base + "-" + buf);
        if (std::filesystem::exists(p)) shardPaths.push_back(p);
      }
    }
    if (shardPaths.empty()) shardPaths.push_back(path);
  } else {
    throw GgufError("path does not exist: " + path.string());
  }

  for (const auto &p : shardPaths) {
    int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw GgufError("failed to open GGUF shard: " + p.string());
    struct stat st{};
    if (::fstat(fd, &st) < 0) {
      ::close(fd);
      throw GgufError("failed to stat GGUF shard: " + p.string());
    }
    GgufShard shard;
    shard.path = p;
    shard.fd = fd;
    shard.fileSize = static_cast<uint64_t>(st.st_size);
    shards_.push_back(std::move(shard));
  }

  for (size_t i = 0; i < shards_.size(); ++i) {
    parseShard(i, /*readMetadata=*/(i == 0));
  }
}

ShardedGgufFile::~ShardedGgufFile() {
  for (auto &shard : shards_) {
    if (shard.fd >= 0) ::close(shard.fd);
    shard.fd = -1;
  }
}

ShardedGgufFile::ShardedGgufFile(ShardedGgufFile &&other) noexcept
    : shards_(std::move(other.shards_)),
      architecture_(std::move(other.architecture_)),
      unsigned_(std::move(other.unsigned_)),
      strings_(std::move(other.strings_)),
      floats_(std::move(other.floats_)),
      arrays_(std::move(other.arrays_)),
      tensors_(std::move(other.tensors_)),
      index_(std::move(other.index_)) {
  for (auto &s : other.shards_) s.fd = -1;
}

ShardedGgufFile &ShardedGgufFile::operator=(ShardedGgufFile &&other) noexcept {
  if (this != &other) {
    for (auto &shard : shards_) {
      if (shard.fd >= 0) ::close(shard.fd);
    }
    shards_ = std::move(other.shards_);
    architecture_ = std::move(other.architecture_);
    unsigned_ = std::move(other.unsigned_);
    strings_ = std::move(other.strings_);
    floats_ = std::move(other.floats_);
    arrays_ = std::move(other.arrays_);
    tensors_ = std::move(other.tensors_);
    index_ = std::move(other.index_);
    for (auto &s : other.shards_) s.fd = -1;
  }
  return *this;
}

void ShardedGgufFile::addSidecar(const std::filesystem::path &sidecarPath) {
  int fd = ::open(sidecarPath.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw GgufError("failed to open GGUF sidecar: " + sidecarPath.string());
  struct stat st{};
  if (::fstat(fd, &st) < 0) {
    ::close(fd);
    throw GgufError("failed to stat GGUF sidecar: " + sidecarPath.string());
  }
  GgufShard shard;
  shard.path = sidecarPath;
  shard.fd = fd;
  shard.fileSize = static_cast<uint64_t>(st.st_size);
  size_t idx = shards_.size();
  shards_.push_back(std::move(shard));
  parseShard(idx, /*readMetadata=*/false);
}

void ShardedGgufFile::parseShard(size_t shardIndex, bool readMetadata) {
  GgufShard &shard = shards_[shardIndex];
  Reader reader(shard.fd, shard.fileSize);
  char magic[4];
  reader.bytes(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0)
    throw GgufError("not a GGUF file: " + shard.path.string());
  const uint32_t version = reader.scalar<uint32_t>();
  if (version != 2 && version != 3)
    throw GgufError("unsupported GGUF version " + std::to_string(version) + ": " + shard.path.string());

  const uint64_t tensorCount = reader.scalar<uint64_t>();
  const uint64_t keyCount = reader.scalar<uint64_t>();
  if (tensorCount > 32768 || keyCount > 16384)
    throw GgufError("implausible GGUF counts in " + shard.path.string());

  for (uint64_t i = 0; i < keyCount; ++i) {
    const std::string key = reader.string();
    const uint32_t type = reader.scalar<uint32_t>();
    if (readMetadata) {
      switch (type) {
      case kUint8: unsigned_[key] = reader.scalar<uint8_t>(); break;
      case kUint16: unsigned_[key] = reader.scalar<uint16_t>(); break;
      case kUint32: unsigned_[key] = reader.scalar<uint32_t>(); break;
      case kUint64: unsigned_[key] = reader.scalar<uint64_t>(); break;
      case kInt8: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int8_t>()); break;
      case kInt16: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int16_t>()); break;
      case kInt32: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int32_t>()); break;
      case kInt64: unsigned_[key] = static_cast<uint64_t>(reader.scalar<int64_t>()); break;
      case kBool: unsigned_[key] = reader.scalar<uint8_t>() != 0; break;
      case kString: strings_[key] = reader.string(); break;
      case kFloat32: floats_[key] = reader.scalar<float>(); break;
      case kFloat64: floats_[key] = reader.scalar<double>(); break;
      case kArray: {
        const uint32_t element = reader.scalar<uint32_t>();
        const uint64_t count = reader.scalar<uint64_t>();
        if (count > kMaximumKeptArray || element == kString || element == kArray) {
          skipArray(reader, element, count, 0);
          break;
        }
        auto &values = arrays_[key];
        values.reserve(count);
        for (uint64_t j = 0; j < count; ++j) values.push_back(numericValue(reader, element));
        break;
      }
      default: skipValue(reader, type, 0); break;
      }
    } else {
      if (key == "split.no") shard.splitNo = reader.scalar<uint16_t>();
      else skipValue(reader, type, 0);
    }
  }

  const uint64_t alignment = unsignedValue("general.alignment").value_or(32);
  if (alignment == 0 || alignment > 65536 || (alignment & (alignment - 1)))
    throw GgufError("invalid GGUF alignment");

  if (readMetadata) {
    architecture_ = stringValue("general.architecture").value_or("");
  }

  for (uint64_t i = 0; i < tensorCount; ++i) {
    ShardedGgufTensor tensor;
    tensor.name = reader.string();
    const uint32_t dimensions = reader.scalar<uint32_t>();
    if (dimensions == 0 || dimensions > 4)
      throw GgufError("invalid tensor rank for " + tensor.name);
    for (uint32_t d = 0; d < dimensions; ++d) {
      const uint64_t dim = reader.scalar<uint64_t>();
      if (!dim) throw GgufError("zero tensor dimension for " + tensor.name);
      tensor.dims.push_back(dim);
    }
    (void)tensor.elements();
    tensor.type = reader.scalar<uint32_t>();
    tensor.offset = reader.scalar<uint64_t>();
    tensor.shardIndex = shardIndex;

    const GgmlTypeTraits *traits = ggmlTypeTraits(tensor.type);
    if (!traits)
      throw GgufError("unknown ggml type " + std::to_string(tensor.type) + " for " + tensor.name);
    if (tensor.columns() % traits->blockElements)
      throw GgufError("tensor row is not block aligned: " + tensor.name);
    tensor.bytes = checkedMultiply(
        checkedMultiply(tensor.rows(), tensor.columns() / traits->blockElements),
        traits->blockBytes);

    if (!index_.emplace(tensor.name, tensors_.size()).second) {
      // If already present (e.g. from shard 0), overwrite or log
    } else {
      tensors_.push_back(std::move(tensor));
    }
  }

  const uint64_t headerEnd = reader.position();
  const uint64_t padding = (alignment - headerEnd % alignment) % alignment;
  if (headerEnd + padding > shard.fileSize)
    throw GgufError("GGUF data section is truncated in " + shard.path.string());
  shard.dataOffset = headerEnd + padding;
}

std::optional<uint64_t> ShardedGgufFile::unsignedValue(std::string_view key) const {
  auto it = unsigned_.find(key);
  if (it == unsigned_.end()) return std::nullopt;
  return it->second;
}

std::optional<std::string> ShardedGgufFile::stringValue(std::string_view key) const {
  auto it = strings_.find(key);
  if (it == strings_.end()) return std::nullopt;
  return it->second;
}

std::optional<double> ShardedGgufFile::floatValue(std::string_view key) const {
  auto it = floats_.find(key);
  if (it == floats_.end()) return std::nullopt;
  return it->second;
}

std::optional<std::span<const double>> ShardedGgufFile::numericArray(std::string_view key) const {
  auto it = arrays_.find(key);
  if (it == arrays_.end()) return std::nullopt;
  return std::span<const double>(it->second);
}

bool ShardedGgufFile::has(std::string_view name) const noexcept {
  return index_.find(name) != index_.end();
}

const ShardedGgufTensor *ShardedGgufFile::find(std::string_view name) const noexcept {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

const ShardedGgufTensor &ShardedGgufFile::require(std::string_view name) const {
  const ShardedGgufTensor *tensor = find(name);
  if (!tensor) throw GgufError("GGUF is missing tensor: " + std::string(name));
  return *tensor;
}

void ShardedGgufFile::readTensor(const ShardedGgufTensor &tensor, std::span<uint8_t> dst) const {
  readTensorRange(tensor, 0, dst);
}

void ShardedGgufFile::readTensorRange(const ShardedGgufTensor &tensor, uint64_t fromByte,
                                     std::span<uint8_t> dst) const {
  if (fromByte > tensor.bytes || dst.size() > tensor.bytes - fromByte)
    throw GgufError("read out of bounds for tensor: " + tensor.name);
  const GgufShard &shard = shards_[tensor.shardIndex];
  preadBytes(shard.fd, shard.dataOffset + tensor.offset + fromByte, dst);
}

} // namespace splash::model
