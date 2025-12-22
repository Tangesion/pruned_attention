#ifndef PE_H
#define PE_H

#include <cstdint>
#include <vector>

namespace pe {

class PE {
public:
    PE();
    void reset();
    void clock_cycle(uint16_t bf16_a, uint16_t bf16_b, bool valid);
    const std::vector<uint16_t>& get_outputs() const;
    bool is_active() const;
};

} // namespace pe


#endif // PE_H