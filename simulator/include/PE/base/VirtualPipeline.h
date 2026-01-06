#pragma once

#include "PE/base/Pipeline.h"
#include "PE/base/PipelineInput.h"
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <vector>
#include <stdexcept>

namespace PE {


template <typename T>
using ComputeFunc = std::function<std::vector<T>(const PipelineInput&)>;

template <typename T>
class VirtualPipeline : public Pipeline<uint16_t> {
public:
    VirtualPipeline(uint32_t latency, ComputeFunc<T> compute_function)
        : latency(latency), compute_function(compute_function) {
        reset();
    }

    ~VirtualPipeline() override = default;

    void reset() override {
        outputs.clear();
        delay_line.clear();
        cycle_count = 0;
    }

    void clock_cycle(const PipelineInput& input) override {
        cycle_count++;

        if (!delay_line.empty()) {
            InFlightData& task = delay_line.front();

            if (cycle_count >= task.ready_cycle) {
                for (const T& val : task.results) {
                    outputs.push_back(val);
                }
                delay_line.pop_front();
            }
        }

        std::vector<T> results = compute_function(input);

        if (!results.empty()) {
            InFlightData task;
            task.results = std::move(results);
            task.ready_cycle = cycle_count + latency;
            delay_line.push_back(std::move(task));
        }

    }

    const std::deque<T>& get_outputs() const override {
        return outputs;
    }

    T pop_output() override {
        if (outputs.empty()) {
            throw std::runtime_error("No outputs available to pop.");
        }
        T val = outputs.front();
        outputs.pop_front();
        return val;
    }

    bool is_active() const override {
        return !delay_line.empty() || !outputs.empty();
    }

private:
    uint32_t latency;
    ComputeFunc<T> compute_function;

    struct InFlightData {
        std::vector<T> results;
        uint32_t ready_cycle;
    };

    std::deque<InFlightData> delay_line;
};


} // namespace PE