#include "PE/units/AddTreeUnit.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <math.h>
#include <sys/types.h>

namespace PE {

size_t AddTreeUnit::compute_tree_height(size_t unit_num) const {
    if (unit_num == 0) return 0;
    return static_cast<size_t>(ceil(log2(unit_num)));
}

AddTreeUnit::AddTreeUnit(PipelineFactory add_pipe_factory, size_t input_num)
    : input_num(input_num),
      tree_height(compute_tree_height(input_num)) {
    // Build the tree structure
    tree.resize(tree_height);
    size_t level_size = input_num / 2;

    for (size_t level = 0; level < tree_height; ++level) {
        tree[level].reserve(level_size);
        for (size_t i = 0; i < level_size; ++i) {
            PipelinePtr<Number> p = add_pipe_factory();
            tree[level].push_back(std::make_unique<AddUnit>(std::move(p)));
        }
        level_size /= 2;
    }
    stage_outputs.assign(tree_height + 1, std::deque<Number>());
    stage_valids.assign(tree_height + 1, false);
}

void AddTreeUnit::reset() {
    for (size_t level = 0; level < tree_height; ++level) {
        for (auto& unit : tree[level]) {
            unit->reset();
        }
    }
    stage_outputs.clear();
    stage_outputs.resize(tree_height + 1, std::deque<Number>());
    stage_valids.clear();
    stage_valids.resize(tree_height + 1, false);
    cycle_count = 0;

    input_valid = false;
}

void AddTreeUnit::load_inputs(const std::vector<Number>& inputs, const bool valid) {
    if (inputs.size() != input_num) {
        throw std::runtime_error("AddTreeUnit: Input size does not match input_num");
    }
    // Load inputs into the first stage
    for (const auto& input : inputs) {
        stage_outputs[0].push_back(input);
    }
    stage_valids[0] = valid;
    if (!valid) {
        stage_outputs[0].clear();
    }
}

bool AddTreeUnit::is_active() const {

    for (const auto& valid : stage_valids) {
        if (valid) return true;
    }

    for (const auto& level : tree) {
        for (const auto& unit : level) {
            if (unit && unit->is_active()) return true;
        }
    }
    return false;
}

bool AddTreeUnit::has_output() const {
    return !outputs.empty();
}

Number AddTreeUnit::get_output() {
    if (!has_output()) {
        throw std::runtime_error("AddTreeUnit::get_output: no output available");
    }
    Number val = outputs.front();
    outputs.pop_front();
    return val;
}


void AddTreeUnit::clock_cycle() {
    cycle_count++;

    for (int i = tree_height - 1; i >= 0; --i) {
        bool add_layer_valid = false;
        stage_outputs[i + 1].clear();
        for (auto &add : tree[i]) {

            Number a = Number(); // Default 0
            Number b = Number(); // Default 0

            if (stage_valids[i] && !stage_outputs[i].empty()) {
                a = stage_outputs[i].front();
                stage_outputs[i].pop_front();
            }

            if (stage_valids[i] && !stage_outputs[i].empty()) {
                b = stage_outputs[i].front();
                stage_outputs[i].pop_front();
            }
            bool valid = stage_valids[i];
            
            add->load_operands(
                a, 
                b, 
                valid
            );

            add->clock_cycle();
            if (add->has_output()) {
                Number result = add->get_result();
                stage_outputs[i + 1].push_back(result);
                add_layer_valid = true;
            }
        }
        if (add_layer_valid) {
            stage_valids[i + 1] = true;
        } else {
            stage_valids[i + 1] = false;
            stage_outputs[i + 1].clear();
        }
    }

    if (stage_valids.back() && !stage_outputs.back().empty()) {
        outputs.push_back(stage_outputs.back().front());
        stage_outputs.back().pop_front();
    }
    
}

} //namespace PE