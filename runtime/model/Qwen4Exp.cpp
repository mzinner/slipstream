#include "Qwen4Exp.hpp"

#include <string_view>

namespace splash::model {
namespace {

constexpr std::string_view kNextHeadMagic = "MDFN0002";
constexpr std::string_view kNextEmbeddingMagic = "MDFN0003";
constexpr std::string_view kNextNgramMagic = "MDFN0004";

void requireLayout(const Qwen4ExpLayout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.attentionQueryHeads || !layout.attentionKvHeads ||
      !layout.attentionHeadDimension || !layout.rotaryPairs ||
      !(layout.rotaryTheta > 0.0F) || !layout.fullAttentionPeriod ||
      !layout.experts || !layout.expertsPerToken ||
      !layout.expertIntermediateSize || !layout.expertStorageN ||
      !layout.hyperConnectionCount || !layout.hyperConnectionLowRank ||
      !layout.indexerHeads || !layout.indexerKvHeads ||
      !layout.indexerHeadDimension || !layout.ngramVocabularySize ||
      !layout.ngramEmbeddingSize || !layout.ngramHeadsPerOrder ||
      !layout.ngramShards || !layout.pleConvolutionTaps ||
      !layout.ngramVocabularyBase || !layout.ngramHeads() ||
      !layout.ngramHeadDimension()) {
    throw WeightStoreError("qwen4exp layout contains a zero dimension");
  }
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.attentionWidth !=
          layout.gdnValueHeads * layout.gdnHeadDimension ||
      layout.packedFullWidth != layout.actualFullWidth() ||
      layout.packedGdnWidth < layout.actualGdnWidth() ||
      layout.packedGdnWidth % kQ4StorageN ||
      layout.expertsPerToken > layout.experts ||
      layout.ngramLayer >= layout.layers ||
      layout.hiddenCaptureLayers.back() >= layout.layers ||
      !layout.q8Layout().valid() || !layout.gdnStateLayout().valid()) {
    throw WeightStoreError("qwen4exp layout is inconsistent");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
  // Experts and the indexer tile narrower; both are multiples of 128 only.
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.indexerProjectionWidth(), layout.hiddenSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.hyperConnectionWidth(), layout.ngramEmbeddingSize);
  validateQ4Layout(layout.hiddenSize, layout.ngramEmbeddingSize);
  if (layout.ngramEmbeddingSize % layout.ngramHeads() ||
      layout.ngramHeadDimension() % kQ4FineGroupElements ||
      layout.ngramVocabularySize % layout.ngramShards) {
    throw WeightStoreError("qwen4exp n-gram geometry is inconsistent");
  }
}

Qwen4ExpHyperConnection readHyperConnection(WeightFile &file,
                                            const Qwen4ExpLayout &layout,
                                            std::string_view label,
                                            bool withInject) {
  const uint64_t width =
      checkedWeightMultiply(layout.hyperConnectionWidth(), kBFloat16Bytes,
                            "hyper-connection width bytes");
  const uint64_t mix =
      checkedWeightMultiply(layout.hyperConnectionLowRank, width,
                            "hyper-connection mix bytes");
  const std::string prefix(label);
  Qwen4ExpHyperConnection result;
  result.norm = file.section(width, prefix + "-norm");
  result.mixDown = file.section(mix, prefix + "-mix-down");
  result.mixUp = file.section(mix, prefix + "-mix-up");
  if (withInject) {
    result.blockInject =
        file.section(checkedWeightMultiply(layout.hyperConnectionCount, width,
                                           "hyper-connection inject bytes"),
                     prefix + "-inject");
  }
  return result;
}

Qwen4ExpIndexer readIndexer(WeightFile &file, metal::MetalBackend &backend,
                            const Qwen4ExpLayout &layout) {
  const uint64_t normBytes =
      checkedWeightMultiply(layout.indexerHeadDimension, kBFloat16Bytes,
                            "indexer norm bytes");
  ops::Q4Projection projection = readQ4Projection(
      file, backend, layout.indexerProjectionWidth(), layout.hiddenSize,
      "indexer-qk", layout.expertStorageN);
  return {projection, file.section(normBytes, "indexer-query-norm"),
          file.section(normBytes, "indexer-key-norm")};
}

void readExperts(WeightFile &file, metal::MetalBackend &backend,
                 const Qwen4ExpLayout &layout, ops::MoeWeights &ffn) {
  const uint32_t n = layout.expertStorageN;
  ffn.router = readQ8Projection(file, backend, layout.experts,
                                layout.hiddenSize, "router");
  ffn.expertGate = readExpertQ4Projection(
      file, layout.experts, layout.expertIntermediateSize, layout.hiddenSize,
      "experts-gate", n);
  ffn.expertUp = readExpertQ4Projection(
      file, layout.experts, layout.expertIntermediateSize, layout.hiddenSize,
      "experts-up", n);
  ffn.expertDown = readExpertQ4Projection(
      file, layout.experts, layout.hiddenSize, layout.expertIntermediateSize,
      "experts-down", n);
  ffn.sharedGate = readExpertQ4Projection(
      file, 1, layout.expertIntermediateSize, layout.hiddenSize,
      "shared-expert-gate", n);
  ffn.sharedUp = readExpertQ4Projection(
      file, 1, layout.expertIntermediateSize, layout.hiddenSize,
      "shared-expert-up", n);
  ffn.sharedDown = readExpertQ4Projection(
      file, 1, layout.hiddenSize, layout.expertIntermediateSize,
      "shared-expert-down", n);
  ffn.sharedExpertGate = readQ8Projection(
      file, backend, kQ4StorageN, layout.hiddenSize,
      "shared-expert-scalar-gate");
}

} // namespace

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend,
                                    const std::filesystem::path &directory,
                                    Qwen4ExpLayout layout) {
  requireLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Qwen4ExpWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    const std::string filename =
        "layer-" + std::to_string(layerIndex) + ".bin";
    WeightFile file(backend, directory / filename, "target/" + filename,
                    Qwen4ExpLayout::layerMagic, layerIndex,
                    fullAttention ? 1U : 0U);
    auto &layer = result.layers.emplace_back();
    layer.attentionHyperConnection =
        readHyperConnection(file, layout, "attention-hyper", true);
    layer.mixer =
        readQwenMixer(file, backend, layout.mixerGeometry(), fullAttention);
    if (fullAttention) layer.indexer = readIndexer(file, backend, layout);
    layer.mlpHyperConnection = readHyperConnection(file, layout, "mlp-hyper", true);
    readExperts(file, backend, layout, layer.ffn);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file(backend, directory / "head.bin", "target/head.bin",
                    kNextHeadMagic, layout.layers, 2);
    result.hyperConnectionMixer =
        readHyperConnection(file, layout, "hyper-mixer", false);
    result.finalNorm = file.section(
        checkedWeightMultiply(layout.hiddenSize, kBFloat16Bytes,
                              "qwen4exp norm bytes"),
        "final-norm");
    result.logitsProjection = readQ4Projection(
        file, backend, layout.vocabularySize, layout.hiddenSize, "logits");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file(backend, directory / "embedding.bin",
                    "target/embedding.bin", kNextEmbeddingMagic,
                    layout.vocabularySize, layout.hiddenSize);
    result.tokenEmbedding = readQ4ProjectionComponents(
        file, layout.vocabularySize, layout.hiddenSize, "embedding");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    // The per-layer embedding, in its own file: the table is gathered per
    // token like the token embedding, so it is stored the same way, and at
    // 26.8 GiB it does not belong inside the layer it serves.
    WeightFile file(backend, directory / "ngram.bin", "target/ngram.bin",
                    kNextNgramMagic, layout.ngramShards,
                    layout.ngramHeadDimension());
    auto &ple = result.perLayerEmbedding;
    // One row per hashed n-gram per head, a head wide, in finer groups.
    ple.table = readQ4ProjectionComponents(
        file, layout.ngramVocabularySize, layout.ngramHeadDimension(), "ngram",
        kQ4FineGroupElements);
    const uint64_t headBytes = checkedWeightMultiply(
        layout.ngramHeads(), 8, "n-gram head table bytes");
    ple.headOffsets = file.section(headBytes, "ngram-head-offsets");
    ple.headVocabularySizes =
        file.section(headBytes, "ngram-head-vocabulary-sizes");
    ple.layerMultipliers = file.section(
        checkedWeightMultiply(layout.ngramSize, 8,
                              "n-gram layer multiplier bytes"),
        "ngram-layer-multipliers");
    ple.keyProjection = readQ4Projection(
        file, backend, layout.hyperConnectionWidth(),
        layout.ngramEmbeddingSize, "ple-key");
    ple.valueProjection = readQ4Projection(
        file, backend, layout.hiddenSize, layout.ngramEmbeddingSize,
        "ple-value");
    const uint64_t width = checkedWeightMultiply(
        layout.hyperConnectionWidth(), kBFloat16Bytes, "PLE norm bytes");
    ple.keyNorm = file.section(width, "ple-key-norm");
    ple.queryNorm = file.section(width, "ple-query-norm");
    ple.convolutionNorm = file.section(width, "ple-conv-norm");
    // Depthwise, so one row of taps per channel.
    ple.convolutionWeights = file.section(
        checkedWeightMultiply(width, layout.pleConvolutionTaps,
                              "PLE convolution bytes"),
        "ple-convolution");
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
