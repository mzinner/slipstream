// GDN verify decode against a direct CPU reference: the four-tap convolution,
// taped k/v/gates, recurrence and convolution carry as each lane folds its
// retained rows on the following step, across both compiled geometries and
// lane counts. The recurrence is checked from the kernel's taped operands
// after those were checked against their projection reference.
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/GDN.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kMaxLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kLayers = 2;
// The last entry is qwen4exp: the same dimensions as the first, gated with a
// sigmoid rather than silu.
constexpr std::array kShapes{GdnShape{16, 48, 128, 10240, 16640},
                             GdnShape{16, 32, 128, 8192, 12544},
                             GdnShape{16, 48, 128, 10240, 16640, true}};

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid GDN request was accepted");
}

uint16_t toBfloat(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

float fromBfloat(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

double roundBfloat(double value) {
  return fromBfloat(toBfloat(static_cast<float>(value)));
}

// One bf16 unit in the last place at the reference's magnitude.
double bfloatUlp(double reference) {
  int exponent = 0;
  std::frexp(std::fabs(reference), &exponent);
  return std::ldexp(1.0, exponent - 8);
}

bool closeBfloat(uint16_t got, double reference, double ulps, double floor) {
  return std::fabs(double(fromBfloat(got)) - reference) <=
         ulps * bfloatUlp(reference) + floor;
}

bool closeFloat(float got, double reference, double tolerance) {
  return std::fabs(double(got) - reference) <=
         tolerance * (1.0 + std::fabs(reference));
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

uint64_t align16k(uint64_t bytes) { return (bytes + 16383) & ~uint64_t{16383}; }

double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }

// The production state cell layout: every layer's conv rows, then every
// layer's recurrent state, both padded to 16 KiB.
struct Cell final {
  uint64_t convLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convBytes = 0;
  uint64_t bytes = 0;

  explicit Cell(const GdnShape &shape)
      : convLayerBytes(align16k(uint64_t{3} * shape.convolutionDimension * 2)),
        recurrentLayerBytes(align16k(uint64_t{shape.valueHeads} * kHeadDim *
                                     kHeadDim * 4)),
        convBytes(kLayers * convLayerBytes),
        bytes(convBytes + kLayers * recurrentLayerBytes) {}

  GdnStateStrides strides() const {
    return {convLayerBytes, recurrentLayerBytes, convBytes};
  }
  const uint16_t *conv(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const uint16_t *>(cell + layer * convLayerBytes);
  }
  const float *recurrent(const uint8_t *cell, uint32_t layer) const {
    return reinterpret_cast<const float *>(cell + convBytes +
                                           layer * recurrentLayerBytes);
  }
};

struct Fixture final {
  const GdnShape &shape;
  Cell cell;
  uint32_t lanes;
  MetalBuffer packed, convWeights, decayWeights, timeBias, recurrent,
      mixerNorm, hidden, arrived, generation, tape;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::array<GdnTapeLane, kMaxLanes> tapeLanes{};
  std::vector<MetalBuffer> packedLayer;
  uint64_t packedStride, tapeLayerStride, tapeSlotStride;

  Fixture(MetalBackend &backend, const GdnShape &geometry, uint32_t laneCount)
      : shape(geometry), cell(geometry), lanes(laneCount),
        packedStride(uint64_t{kRows} * shape.packedWidth),
        tapeLayerStride(gdnTapeLayerBytes(shape)),
        tapeSlotStride(kLayers * tapeLayerStride) {
    Random random(0x6D4E1234ULL + laneCount);
    auto alloc = [&](uint64_t bytes, const char *label) {
      return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
    };
    auto fill = [&](MetalBuffer &buffer, float scale) {
      auto *values = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
        values[index] = toBfloat(random.unit() * scale);
    };
    packed = alloc(kLayers * kMaxLanes * packedStride * 2, "gdn packed");
    fill(packed, 1.0F);
    convWeights = alloc(uint64_t{shape.convolutionDimension} * 4 * 2,
                        "gdn conv weights");
    fill(convWeights, 0.5F);
    decayWeights = alloc(uint64_t{shape.valueHeads} * 4, "gdn decay weights");
    for (uint32_t head = 0; head < shape.valueHeads; ++head)
      static_cast<float *>(decayWeights.contents())[head] =
          -std::exp(random.unit() * 1.5F);
    timeBias = alloc(uint64_t{shape.valueHeads} * 2, "gdn time bias");
    fill(timeBias, 0.5F);
    tape = alloc(kMaxLanes * 2 * tapeSlotStride, "gdn tape");
    const uint64_t rowBytes =
        uint64_t{kMaxLanes} * kRows * shape.valueHeads * kHeadDim * 2;
    recurrent = alloc(rowBytes, "gdn recurrent rows");
    hidden = alloc(rowBytes, "gdn hidden");
    mixerNorm = alloc(uint64_t{kHeadDim} * 2, "gdn mixer norm");
    fill(mixerNorm, 1.0F);
    arrived = alloc(kMaxLanes * 4, "gdn arrived");
    generation = alloc(kMaxLanes * 4, "gdn generation");
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      tapeLanes[lane] = {uint64_t{lane} * 2 * tapeSlotStride,
                         (uint64_t{lane} * 2 + 1) * tapeSlotStride, 0};
      current[lane] = alloc(cell.bytes, "gdn current");
      next[lane] = alloc(cell.bytes, "gdn next");
      auto *bytes = static_cast<uint8_t *>(current[lane].contents());
      auto *conv = reinterpret_cast<uint16_t *>(bytes);
      for (uint64_t index = 0; index < cell.convBytes / 2; ++index)
        conv[index] = toBfloat(random.unit());
      auto *state = reinterpret_cast<float *>(bytes + cell.convBytes);
      for (uint64_t index = 0; index < (cell.bytes - cell.convBytes) / 4;
           ++index)
        state[index] = random.unit() * 0.5F;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      packedLayer.push_back(backend.view(packed,
                                         layer * kMaxLanes * packedStride * 2,
                                         kMaxLanes * packedStride * 2));
    clear();
  }

  void clear() {
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
      std::memset(next[lane].contents(), 0, cell.bytes);
    for (MetalBuffer *buffer : {&recurrent, &hidden, &arrived, &generation})
      std::memset(buffer->contents(), 0, buffer->sizeBytes());
  }

  GdnDecodeBuffers decodeBuffers(uint32_t layer) const {
    return {packedLayer[layer], convWeights, current, next, tape, tapeLanes,
            decayWeights, timeBias, recurrent, mixerNorm, hidden, arrived,
            generation};
  }

  // Row `token` of lane `lane` in layer `layer` of the packed projection.
  const uint16_t *packedRow(uint32_t layer, uint32_t lane,
                            uint32_t token) const {
    return static_cast<const uint16_t *>(packed.contents()) +
           (uint64_t{layer} * kMaxLanes + lane) * packedStride +
           uint64_t{token} * shape.packedWidth;
  }
  const uint8_t *tapeLayer(uint32_t layer, uint64_t slot) const {
    return static_cast<const uint8_t *>(tape.contents()) + slot +
           uint64_t{layer} * tapeLayerStride;
  }
  const uint16_t *kvRow(uint32_t layer, uint64_t slot,
                        uint32_t token) const {
    const uint32_t kvWidth =
        (shape.keyHeads + shape.valueHeads) * shape.headDimension;
    return reinterpret_cast<const uint16_t *>(
               tapeLayer(layer, slot) +
               gdn_tape_kv_offset(shape.convolutionDimension)) +
           uint64_t{token} * kvWidth;
  }
  const float *decayRow(uint32_t layer, uint64_t slot,
                        uint32_t token) const {
    const uint32_t kvWidth =
        (shape.keyHeads + shape.valueHeads) * shape.headDimension;
    return reinterpret_cast<const float *>(
               tapeLayer(layer, slot) +
               gdn_tape_decay_offset(shape.convolutionDimension, kvWidth)) +
           uint64_t{token} * shape.valueHeads;
  }
  const uint16_t *betaRow(uint32_t layer, uint64_t slot,
                          uint32_t token) const {
    const uint32_t kvWidth =
        (shape.keyHeads + shape.valueHeads) * shape.headDimension;
    return reinterpret_cast<const uint16_t *>(
               tapeLayer(layer, slot) +
               gdn_tape_beta_offset(shape.convolutionDimension, kvWidth,
                                    shape.valueHeads)) +
           uint64_t{token} * shape.valueHeads;
  }
  const uint16_t *inputRow(uint32_t layer, uint64_t slot,
                           uint32_t token) const {
    return reinterpret_cast<const uint16_t *>(tapeLayer(layer, slot)) +
           uint64_t{token} * shape.convolutionDimension;
  }
  const uint16_t *rowsOf(const MetalBuffer &buffer, uint32_t lane,
                         uint32_t token, uint32_t head) const {
    return static_cast<const uint16_t *>(buffer.contents()) +
           ((uint64_t{lane} * kRows + token) * shape.valueHeads + head) *
               kHeadDim;
  }
  const uint8_t *cellBytes(const MetalBuffer &buffer) const {
    return static_cast<const uint8_t *>(buffer.contents());
  }
};

// Four-tap causal convolution of one channel at one token, three carried
// rows then the command's rows, rounded to bf16 and gated by SiLU.
double convolutionSilu(const Fixture &fixture, uint32_t layer, uint32_t lane,
                       uint32_t token, uint32_t channel,
                       uint32_t pendingRows = 0, uint64_t pendingSlot = 0) {
  const uint16_t *weights =
      static_cast<const uint16_t *>(fixture.convWeights.contents()) +
      uint64_t{channel} * 4;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  double value = 0.0;
  for (uint32_t tap = 0; tap < 4; ++tap) {
    const uint32_t position = token + tap;
    uint16_t input;
    if (position < 3) {
      const uint32_t source = pendingRows + position;
      input = source < 3
                  ? carried[source * fixture.shape.convolutionDimension + channel]
                  : fixture.inputRow(layer, pendingSlot, source - 3)[channel];
    } else {
      input = fixture.packedRow(layer, lane, position - 3)[channel];
    }
    value += double(fromBfloat(input)) * fromBfloat(weights[tap]);
  }
  value = roundBfloat(value);
  return roundBfloat(value * sigmoid(value));
}

uint16_t convolutionCarry(const Fixture &fixture, uint32_t layer,
                          uint32_t lane, uint32_t pendingRows,
                          uint64_t pendingSlot, uint32_t row,
                          uint32_t channel) {
  const uint32_t source = pendingRows + row;
  const uint16_t *carried =
      fixture.cell.conv(fixture.cellBytes(fixture.current[lane]), layer);
  return source < 3
             ? carried[source * fixture.shape.convolutionDimension + channel]
             : fixture.inputRow(layer, pendingSlot, source - 3)[channel];
}

void checkConvolution(const Fixture &fixture, uint32_t layer, uint32_t lane,
                      const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  for (uint32_t token = 0; token < kRows; ++token) {
    std::vector<double> conv(shape.convolutionDimension);
    for (uint32_t channel = 0; channel < shape.convolutionDimension; ++channel)
      conv[channel] = convolutionSilu(
          fixture, layer, lane, token, channel,
          fixture.tapeLanes[lane].pendingRows,
          fixture.tapeLanes[lane].pendingSlot);
    const uint16_t *kv =
        fixture.kvRow(layer, fixture.tapeLanes[lane].stepSlot, token);
    for (uint32_t head = 0; head < shape.keyHeads; ++head) {
      const uint32_t convBase = keyWidth + head * kHeadDim;
      double squares = 0.0;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        squares += conv[convBase + dim] * conv[convBase + dim];
      const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double normalized =
            roundBfloat(conv[convBase + dim] * inverse);
        require(closeBfloat(kv[head * kHeadDim + dim],
                            normalized * 0.08838834765, 3.0, 1e-6),
                where + ": tape key row mismatch");
      }
    }
    const uint16_t *inputs =
        fixture.inputRow(layer, fixture.tapeLanes[lane].stepSlot, token);
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    for (uint32_t channel = 0; channel < shape.convolutionDimension;
         ++channel)
      require(inputs[channel] == packed[channel],
              where + ": convolution input tape mismatch");
    for (uint32_t head = 0; head < shape.valueHeads; ++head)
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        require(closeBfloat(kv[keyWidth + head * kHeadDim + dim],
                            conv[2 * keyWidth + head * kHeadDim + dim], 3.0,
                            1e-6),
                where + ": tape value row mismatch");
  }
}

void checkGates(const Fixture &fixture, uint32_t layer, uint32_t lane,
                const std::string &where) {
  const GdnShape &shape = fixture.shape;
  const uint32_t bOffset = shape.convolutionDimension +
                           shape.valueHeads * kHeadDim;
  const uint32_t aOffset = bOffset + shape.valueHeads;
  const auto *decayWeights =
      static_cast<const float *>(fixture.decayWeights.contents());
  const auto *timeBias =
      static_cast<const uint16_t *>(fixture.timeBias.contents());
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    const float *decay =
        fixture.decayRow(layer, fixture.tapeLanes[lane].stepSlot, token);
    const uint16_t *beta =
        fixture.betaRow(layer, fixture.tapeLanes[lane].stepSlot, token);
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const double b = fromBfloat(packed[bOffset + head]);
      require(closeBfloat(beta[head], sigmoid(b), 2.0, 1e-6),
              where + ": beta mismatch");
      const double x = roundBfloat(double(fromBfloat(packed[aOffset + head])) +
                                   fromBfloat(timeBias[head]));
      const double softplus =
          std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
      // The kernel rounds softplus to bf16 with fast transcendentals, so a
      // value near a rounding boundary may land one bf16 step away.
      bool matched = false;
      const uint16_t rounded = toBfloat(static_cast<float>(softplus));
      for (int step = -1; step <= 1 && !matched; ++step) {
        const double candidate =
            fromBfloat(static_cast<uint16_t>(rounded + step));
        matched = closeFloat(decay[head],
                             std::exp(double(decayWeights[head]) * candidate),
                             1e-4);
      }
      require(matched, where + ": decay mismatch");
    }
  }
}

// Advance one tape's rows for a head. The output uses q reconstructed from
// this step's convolution history; state-only pending folds need no q.
void advanceRecurrence(const Fixture &fixture, uint32_t layer, uint32_t lane,
                       uint32_t head, uint64_t slot, uint32_t tokens,
                       uint32_t pendingRows, uint64_t pendingSlot,
                       std::vector<double> &state,
                       std::vector<double> *rows) {
  const GdnShape &shape = fixture.shape;
  const uint32_t keyWidth = shape.keyHeads * kHeadDim;
  const uint32_t keyHead = head / (shape.valueHeads / shape.keyHeads);
  if (rows)
    rows->assign(uint64_t{tokens} * kHeadDim, 0.0);
  for (uint32_t token = 0; token < tokens; ++token) {
    const uint16_t *kv = fixture.kvRow(layer, slot, token);
    const uint16_t *key = kv + keyHead * kHeadDim;
    const uint16_t *value = kv + keyWidth + head * kHeadDim;
    std::vector<double> query(kHeadDim);
    if (rows) {
      const uint32_t qBase = keyHead * kHeadDim;
      std::vector<double> qConv(kHeadDim);
      double qSquares = 0.0;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        qConv[dim] = convolutionSilu(fixture, layer, lane, token,
                                     qBase + dim, pendingRows, pendingSlot);
        qSquares += qConv[dim] * qConv[dim];
      }
      const double qScale = 1.0 / std::sqrt(qSquares / kHeadDim + 1e-6);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        query[dim] = roundBfloat(roundBfloat(qConv[dim] * qScale) *
                                 0.0078125);
    }
    const double decay = fixture.decayRow(layer, slot, token)[head];
    const double beta =
        fromBfloat(fixture.betaRow(layer, slot, token)[head]);
    for (uint32_t valueDim = 0; valueDim < kHeadDim; ++valueDim) {
      double *row = state.data() + uint64_t{valueDim} * kHeadDim;
      double memory = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        row[keyDim] *= decay;
        memory += row[keyDim] * fromBfloat(key[keyDim]);
      }
      const double delta = (fromBfloat(value[valueDim]) - memory) * beta;
      double output = 0.0;
      for (uint32_t keyDim = 0; keyDim < kHeadDim; ++keyDim) {
        row[keyDim] += fromBfloat(key[keyDim]) * delta;
        if (rows)
          output += row[keyDim] * query[keyDim];
      }
      if (rows)
        (*rows)[uint64_t{token} * kHeadDim + valueDim] = output;
    }
  }
}

std::vector<double> recurrence(const Fixture &fixture, const MetalBuffer &source,
                               uint32_t layer, uint32_t lane, uint32_t head,
                               uint64_t slot, uint32_t tokens,
                               std::vector<double> *rows,
                               uint32_t pendingRows = 0,
                               uint64_t pendingSlot = 0) {
  const float *stateIn =
      fixture.cell.recurrent(fixture.cellBytes(source), layer) +
      uint64_t{head} * kHeadDim * kHeadDim;
  std::vector<double> state(stateIn, stateIn + kHeadDim * kHeadDim);
  if (pendingRows)
    advanceRecurrence(fixture, layer, lane, head, pendingSlot, pendingRows, 0, 0,
                      state, nullptr);
  advanceRecurrence(fixture, layer, lane, head, slot, tokens, pendingRows,
                    pendingSlot, state, rows);
  return state;
}

void checkState(const Fixture &fixture, const MetalBuffer &buffer,
                uint32_t layer, uint32_t head,
                const std::vector<double> &expected, const std::string &where) {
  const float *state =
      fixture.cell.recurrent(fixture.cellBytes(buffer), layer) +
      uint64_t{head} * kHeadDim * kHeadDim;
  for (uint32_t index = 0; index < kHeadDim * kHeadDim; ++index)
    require(closeFloat(state[index], expected[index], 1e-4),
            where + ": recurrent state mismatch");
}

void checkCarry(const Fixture &fixture, const MetalBuffer &buffer,
                uint32_t layer, uint32_t lane, const std::string &where) {
  const uint16_t *carry = fixture.cell.conv(fixture.cellBytes(buffer), layer);
  for (uint32_t row = 0; row < 3; ++row)
    for (uint32_t channel = 0; channel < fixture.shape.convolutionDimension;
         ++channel)
      require(carry[row * fixture.shape.convolutionDimension + channel] ==
                  convolutionCarry(fixture, layer, lane,
                                   fixture.tapeLanes[lane].pendingRows,
                                   fixture.tapeLanes[lane].pendingSlot, row,
                                   channel),
              where + ": convolution carry mismatch");
}

void checkDecode(const Fixture &fixture, uint32_t layer, uint32_t lane) {
  const GdnShape &shape = fixture.shape;
  const std::string where = "decode layer " + std::to_string(layer) +
                            " lane " + std::to_string(lane);
  checkConvolution(fixture, layer, lane, where);
  checkGates(fixture, layer, lane, where);
  checkCarry(fixture, fixture.next[lane], layer, lane, where);
  // Every layer writes recurrent rows and hidden rows into shared scratch;
  // these retain the last layer's values after the full graph finishes.
  const bool lastLayer = layer + 1 == kLayers;
  std::vector<double> rows;
  for (uint32_t head = 0; head < shape.valueHeads; ++head) {
    const std::vector<double> state = recurrence(
        fixture, fixture.current[lane], layer, lane, head,
        fixture.tapeLanes[lane].stepSlot, 0, nullptr,
        fixture.tapeLanes[lane].pendingRows,
        fixture.tapeLanes[lane].pendingSlot);
    checkState(fixture, fixture.next[lane], layer, head, state, where);
    if (!lastLayer)
      continue;
    static_cast<void>(recurrence(fixture, fixture.current[lane], layer, lane,
                                 head, fixture.tapeLanes[lane].stepSlot, kRows,
                                 &rows, fixture.tapeLanes[lane].pendingRows,
                                 fixture.tapeLanes[lane].pendingSlot));
    for (uint32_t token = 0; token < kRows; ++token) {
      const uint16_t *recurrent =
          fixture.rowsOf(fixture.recurrent, lane, token, head);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim)
        if (!closeBfloat(recurrent[dim], rows[token * kHeadDim + dim], 16.0,
                         1e-6)) {
          throw std::runtime_error(
              where + ": recurrent row mismatch got=" +
              std::to_string(fromBfloat(recurrent[dim])) + " expected=" +
              std::to_string(rows[token * kHeadDim + dim]));
        }
    }
  }
  if (!lastLayer)
    return;
  // The gated RMSNorm, from the kernel's own recurrent rows.
  const auto *norm =
      static_cast<const uint16_t *>(fixture.mixerNorm.contents());
  const uint32_t zOffset = shape.convolutionDimension;
  for (uint32_t token = 0; token < kRows; ++token) {
    const uint16_t *packed = fixture.packedRow(layer, lane, token);
    for (uint32_t head = 0; head < shape.valueHeads; ++head) {
      const uint16_t *recurrent =
          fixture.rowsOf(fixture.recurrent, lane, token, head);
      const uint16_t *hidden =
          fixture.rowsOf(fixture.hidden, lane, token, head);
      double squares = 0.0;
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double row = fromBfloat(recurrent[dim]);
        squares += row * row;
      }
      const double inverse = 1.0 / std::sqrt(squares / kHeadDim + 1e-6);
      for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const double normalized = roundBfloat(
            fromBfloat(recurrent[dim]) * inverse * fromBfloat(norm[dim]));
        const double gate = fromBfloat(packed[zOffset + head * kHeadDim + dim]);
        require(closeBfloat(hidden[dim],
                            normalized * (fixture.shape.sigmoidGate
                                              ? sigmoid(gate)
                                              : gate * sigmoid(gate)),
                            2.0,
                            1e-6),
                where + ": hidden mismatch");
      }
    }
  }
}

void requireUntouched(const Fixture &fixture, uint32_t lane,
                      const std::string &where) {
  const uint8_t *bytes = fixture.cellBytes(fixture.next[lane]);
  for (uint64_t index = 0; index < fixture.cell.bytes; ++index)
    require(bytes[index] == 0, where + ": idle lane cell was written");
}

void runDecode(MetalBackend &backend, const GdnShape &shape, uint32_t lanes) {
  Fixture fixture(backend, shape, lanes);
  CommandGraph graph;
  for (uint32_t layer = 0; layer < kLayers; ++layer)
    GDN::addDecode(graph, fixture.decodeBuffers(layer), shape, lanes, layer,
                   fixture.cell.strides());
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const std::string where = "lanes " + std::to_string(lanes);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
    const auto *generation =
        static_cast<const uint32_t *>(fixture.generation.contents());
    const auto *arrived =
        static_cast<const uint32_t *>(fixture.arrived.contents());
    require(arrived[lane] == 0, where + ": completion counter not reset");
    if (lane >= lanes) {
      require(generation[lane] == 0, where + ": idle lane completed");
      requireUntouched(fixture, lane, where);
      continue;
    }
    require(generation[lane] == kLayers, where + ": layer completion count");
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      checkDecode(fixture, layer, lane);
  }

  // Rotate state/tape parity and exercise every retained-row count. Each
  // following step folds its lane's retained rows into the incoming cell.
  for (uint32_t retained = 1; retained <= kRows; ++retained) {
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      std::swap(fixture.current[lane], fixture.next[lane]);
      const uint64_t previousStep = fixture.tapeLanes[lane].stepSlot;
      fixture.tapeLanes[lane].pendingSlot = previousStep;
      fixture.tapeLanes[lane].stepSlot =
          uint64_t{lane} * 2 * fixture.tapeSlotStride +
          (previousStep == uint64_t{lane} * 2 * fixture.tapeSlotStride
               ? fixture.tapeSlotStride
               : 0);
      fixture.tapeLanes[lane].pendingRows =
          1 + ((retained + lane - 1) % kRows);
      std::memset(fixture.next[lane].contents(), 0, fixture.cell.bytes);
    }
    CommandGraph nextStep;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      GDN::addDecode(nextStep, fixture.decodeBuffers(layer), shape, lanes,
                     layer, fixture.cell.strides());
    static_cast<void>(backend.submitCommand(nextStep.dispatches()));
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        checkDecode(fixture, layer, lane);
    }
  }
}

void rejectsInvalid(MetalBackend &backend) {
  const GdnShape &shape = kShapes[1];
  Fixture fixture(backend, shape, 1);
  CommandGraph graph;
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, 0, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0), shape, kMaxLanes + 1, 0,
                   fixture.cell.strides());
  });
  rejects([&] {
    GDN::addDecode(graph, fixture.decodeBuffers(0),
                   GdnShape{16, 40, 128, 9216, 14400}, 1, 0,
                   fixture.cell.strides());
  });
  auto invalidLanes = fixture.tapeLanes;
  invalidLanes[0].pendingRows = kRows + 1;
  GdnDecodeBuffers invalidBuffers = fixture.decodeBuffers(0);
  invalidBuffers.tapeLanes = invalidLanes;
  rejects([&] {
    GDN::addDecode(graph, invalidBuffers, shape, 1, 0,
                   fixture.cell.strides());
  });
  require(graph.empty(), "invalid GDN request partially encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: gdn-decode METALLIB");
    MetalBackend backend(argv[1]);
    rejectsInvalid(backend);
    for (const GdnShape &shape : kShapes)
      for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
        runDecode(backend, shape, lanes);
    std::cout << "gdn_decode_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn_decode_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
