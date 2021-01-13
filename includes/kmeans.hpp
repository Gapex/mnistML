#pragma once
#include <vector>
#include "data.hpp"
#include <unordered_map>

class cluster
{
public:
    cluster();
    ~cluster();

    void push(const Data &point);
    
    uint8_t get_most_calss();

    double query(const Data &point);

private:
    void set_most(uint8_t label, uint32_t freq);

    std::vector<double> centroid;
    std::vector<Data> points;
    std::unordered_map<uint8_t, uint32_t> class_cnt;

    uint8_t most_class;
    uint32_t most_freq;
};

class kmeans
{
public:
    kmeans();
    ~kmeans();

    void set_training_data(std::vector<Data> vec);
    void set_test_data(std::vector<Data> vec);
    void set_validation_data(std::vector<Data> vec);

    const std::vector<Data> &get_training_data();
    const std::vector<Data> &get_test_data();
    const std::vector<Data> &get_validation_data();

    void train();

    uint8_t predict(const Data &query);

    double test_performance();
    double validation_performance();

private:
    std::vector<Data> training_data;
    std::vector<Data> test_data;
    std::vector<Data> validation_data;

    std::unordered_map<uint8_t, cluster> model;
};