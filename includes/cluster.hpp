#pragma once
#include "data.hpp"
#include <unordered_map>

class cluster
{
public:
    cluster(Data initial_point);
    ~cluster();

    void add_to_cluster(const Data &point);

    void set_most_frequent_class();

    uint8_t get_most_calss() const;

    const std::vector<double> &get_centroid() const;

private:
    std::vector<double> centroid;
    std::vector<Data> cluster_points;
    std::unordered_map<uint8_t, uint32_t> class_cnt;

    int most_frequent_class;
};