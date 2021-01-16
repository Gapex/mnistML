#include "cluster.hpp"
#include <cmath>

cluster::cluster(Data inital_point) : centroid(inital_point.get_feature_vector().begin(), inital_point.get_feature_vector().end())
{
    cluster_points.push_back(inital_point);
    class_cnt[inital_point.get_label()] = 1;
    most_frequent_class = inital_point.get_label();
}

cluster::~cluster()
{
}

void cluster::add_to_cluster(const Data &point)
{
    size_t previous_size = cluster_points.size();
    cluster_points.push_back(point);
    size_t new_size = cluster_points.size();
    size_t i = 0;
    for (double &value : centroid)
    {
        double val = value * previous_size;
        val += point.get_feature_vector().at(i++);
        val /= new_size;
        value = val;
    }
    ++class_cnt[point.get_label()];
    set_most_frequent_class();
}

void cluster::set_most_frequent_class()
{
    int best_class;
    int freq = 0;
    for (auto &kv : class_cnt)
    {
        if (kv.second > freq)
        {
            freq = kv.second;
            best_class = kv.first;
        }
    }
    most_frequent_class = best_class;
}

const std::vector<double> &cluster::get_centroid() const
{
    return centroid;
}

uint8_t cluster::get_most_calss() const
{
    return most_frequent_class;
}