#pragma once

#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

#include "data.hpp"

using byte = unsigned char;

class DataHandler
{
public:

    DataHandler();
    ~DataHandler();

    void read_feature_vector(const std::string &);
    void read_feature_labels(const std::string &);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(byte*);

    const std::vector<Data> &get_training_data();
    const std::vector<Data> &get_test_data();
    const std::vector<Data> &get_validataion_data();

private:
    std::vector<Data> data_array; //all of the data (pre-split)
    std::vector<Data> training_data;
    std::vector<Data> test_data;
    std::vector<Data> validation_data;

    int num_classes;
    int feature_vector_size;
    std::map<uint8_t, int> class_map;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_PERCENT = 0.05;
};