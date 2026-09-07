// Run from repo root: ./build/test_visualizer

#include <iostream>
#include <string>
#include <variant>

#include "common/status.hpp"
#include "detector.hpp"
#include "visualization.hpp"

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

namespace {

int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

}  // namespace

int main() {
    VisualizerConfig::class_labels_file_path =
        std::string{PROJECT_ROOT} + "/assets/labels/coco.names";

    // 1. The shipped assets agree with the compiled-in class count.
    try {
        Visualizer visualizer{2, DetectorFixedParams::num_classes};
        std::cout << "labels file matches DetectorFixedParams::num_classes (" << DetectorFixedParams::num_classes << ")\n";
    } catch (const Status::FatalException& e) {
        return fail(std::string{"shipped assets are inconsistent: "} + e.what());
    }

    // 2. A missing labels file is fatal at construction, not at first draw.
    VisualizerConfig::class_labels_file_path = std::string{PROJECT_ROOT} + "/assets/labels/does_not_exist.names";
    try {
        Visualizer visualizer{2, DetectorFixedParams::num_classes};
        return fail("expected a FatalException for a missing labels file");
        
    } catch (const Status::FatalException& e) {
        if (e.error().origin != Status::Stage::Visualization) {
            return fail("missing labels file reported the wrong stage");
        }
        std::cout << "missing labels file -> Fatal[Visualization]\n";
    }

    std::cout << "PASS\n";
    return 0;
}
