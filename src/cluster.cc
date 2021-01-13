#include "kmeans.hpp"
#include <cmath>

cluster::cluster() : most_class(-1), most_freq(0)
{
}

cluster::~cluster()
{
}

void cluster::push(const Data &point)
{
    const auto &feature = point.get_feature_vector();
    size_t old_size = points.size();
    points.push_back(point);
    if (centroid.empty() == true)
    {
        most_class = point.get_label();
        most_freq = 1;
        centroid.reserve(feature.size());
        for (auto val : feature)
        {
            centroid.push_back(val);
        }
        return;
    }
    size_t ndimension = centroid.size();
    if (ndimension != point.get_feature_vector_size())
    {
        printf("vector dimension mismatch: %lu != %lu\n", ndimension, feature.size());
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < ndimension; ++i)
    {
        double value = centroid.at(i);
        value *= old_size;
        value += feature.at(i);
        value /= points.size();
        centroid.at(i) = value;
    }
    uint32_t new_freq = ++class_cnt[point.get_label()];
    set_most(point.get_label(), new_freq);
}

double cluster::query(const Data &point)
{
    const auto &feature = point.get_feature_vector();
    size_t ndimension = centroid.size();
    if (ndimension != feature.size())
    {
        printf("[cluster::query] vector dimension mismatch: %lu != %lu\n", ndimension, feature.size());
        exit(EXIT_FAILURE);
    }
    double res = 0;
    for (size_t i = 0; i < ndimension; ++i)
    {
        res += std::pow(feature[i] - centroid[i], 2);
    }
    res = std::sqrt(res);
    return res;
}

void cluster::set_most(uint8_t label, uint32_t freq)
{
    if (freq > most_freq)
    {
        most_freq = freq;
        most_class = label;
    }
}

uint8_t cluster::get_most_calss()
{
    return most_class;
}