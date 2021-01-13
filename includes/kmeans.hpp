#pragma once
#include <vector>
#include "data.hpp"
#include <unordered_map>
class kmeans
{
public:
    kmeans();
    ~kmeans();

    void set_training_data(std::vector<Data> vec);
    void set_test_data(std::vector<Data> vec);
    void set_validation_data(std::vector<Data> vec);

    std::vector<Data> get_training_data();
    std::vector<Data> get_test_data();
    std::vector<Data> get_validation_data();

    void train();

    uint8_t predict(const Data &query);

    double test_performance();
    double validation_performance();

private:
    std::vector<Data> training_data;
    std::vector<Data> test_data;
    std::vector<Data> validation_data;
    std::unordered_map<uint8_t, Data> label_to_focus;
};