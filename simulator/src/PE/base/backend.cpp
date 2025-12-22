#include "PE/base/backend.h"
#include "PE/base/ComputeComponent.h"
#include <stdexcept>
#include <utility>
#include <mutex>
namespace PE {

std::map<std::string, Backend::Creator*> * Backend::gCreator = nullptr;

void Backend::initCreatorMap() {
    gCreator = new std::map<std::string, Backend::Creator*>();
}

void Backend::addCreator(const std::string& name, Creator *creator) {
    auto map = gCreator;
    if (map->find(name) != map->end()) {
        throw std::invalid_argument("Creator with name " + name + " already exists.");
    }
    map->insert(std::make_pair(name, creator)); 
}

ComputeComponent* Backend::onCreate(const std::string& name, std::vector<PipelinePtr>&& pipes) const {
    if (!gCreator) return nullptr;
    auto it = gCreator->find(name);
    if (it == gCreator->end()) {
        throw std::runtime_error("Component not registered: " + name);
    }
    return it->second->onCreate(std::move(pipes));
}

extern void registerComponents();

static std::once_flag s_flag;
void registerBackend() {
    std::call_once(s_flag, [&]() {
        Backend::initCreatorMap();
        registerComponents();
    });
}

} // namespace PE