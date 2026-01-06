#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include "PE/base/ComputeComponent.h"
#include "PE/base/Pipeline.h"

namespace PE {

class Backend {
public:
    virtual ComputeComponent* onCreate(const std::string& name, std::vector<PipelinePtr<uint16_t>>&& pipes) const;

public:
    class Creator {
    public:
        virtual ComputeComponent* onCreate(std::vector<PipelinePtr<uint16_t>>&& pipes) const = 0;
    };

    static void addCreator(const std::string& name, Creator* creator);

    static void initCreatorMap();

private:
    static std::map<std::string, Backend::Creator*> * gCreator;

};

#define REGISTER_COMPONENT(name, creatorType) \
    void ___register_##creatorType() { \
        static creatorType _temp;\
        PE::Backend::addCreator(name, &_temp); \
    } \


} // namespace PE