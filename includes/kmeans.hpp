#pragma once
#include <vector>
#include <unordered_set>

#include "cluster.hpp"

class kmeans
{
public:
    kmeans(int k);
    ~kmeans();

    void init_clusters();
    void init_clusters_for_each_class();
    void train();
    double distance(const std::vector<double> &, const Data &);
    double validate();
    double test();

    void set_training_data(std::vector<Data> vec);
    void set_test_data(std::vector<Data> vec);
    void set_validation_data(std::vector<Data> vec);

    const std::vector<Data> &get_training_data();
    const std::vector<Data> &get_test_data();
    const std::vector<Data> &get_validation_data();

private:
    std::vector<Data> training_data;
    std::vector<Data> test_data;
    std::vector<Data> validation_data;

    int num_clusters;
    std::vector<cluster> clusters;
    std::unordered_set<int> used_indexes;
};