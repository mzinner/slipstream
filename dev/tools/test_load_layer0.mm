#include "metal/MetalBackend.hpp"
#include "models/qwen4exp/Qwen4Exp.hpp"
#include <iostream>

int main() {
  try {
    splash::metal::MetalBackend backend;
    splash::model::Qwen4ExpLayout layout;
    std::filesystem::path dir = "/tmp";
    std::cout << "Opening /tmp/test_layer-0.bin with WeightFile...\n";
    splash::model::WeightFile file(backend, dir / "test_layer-0.bin", "target/layer-0.bin",
                                  splash::model::Qwen4ExpLayout::layerMagic, 0, 0);
    splash::model::Qwen4ExpLayerWeights layer;
    layer.attentionHyperConnection =
        readHyperConnection(file, backend, layout, "attention-hyper", true);
    layer.mixer =
        readQwenMixer(file, backend, layout.mixerGeometry(), false, true);
    layer.mlpHyperConnection =
        readHyperConnection(file, backend, layout, "mlp-hyper", true);
    readExperts(file, backend, layout, layer.ffn);
    file.finish();
    std::cout << "SUCCESS! WeightFile verified and finished /tmp/test_layer-0.bin perfectly!\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
}
