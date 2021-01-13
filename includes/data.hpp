#pragma once

#include <vector>
#include <cstdio>
#include <cstdint>

class Data
{
public:
    Data();
    ~Data();
    void set_feature_vector(std::vector<uint8_t> );
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enumerated_label(int);

    int get_feature_vector_size() const;
    uint8_t get_label() const;
    uint8_t get_enumerated_label() const;

    std::vector<uint8_t> &get_feature_vector();
    const std::vector<uint8_t> &get_feature_vector() const;

    Data operator/(uint32_t val){
        Data res;

        return res;
    }

private:
    std::vector<uint8_t> feature_vector; // no class at end.
    uint8_t label;
    int enum_label; // A -> 1, B -> 2
};