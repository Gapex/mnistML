#include "kmeans.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

kmeans::kmeans(int k) : num_clusters(k)
{
    clusters.reserve(k);
}

kmeans::~kmeans()
{
}

/**
 * 从训练集中随机选取k个点作为初始cluster的重心
 */
void kmeans::init_clusters()
{
    for (int i = 0; i < num_clusters; ++i)
    {
        int index = (std::rand() % training_data.size());
        while (used_indexes.find(index) != used_indexes.end())
        {
            index = (std::rand() % training_data.size());
        }
        used_indexes.insert(index);
        cluster c(training_data.at(index));
        clusters.emplace_back(std::move(c));
    }
}

/**
 * 从训练集自动发现所有类别并创建初始cluster
 */
void kmeans::init_clusters_for_each_class()
{
    std::unordered_set<int> used_class;
    int index = 0;
    for (const Data &data : training_data)
    {
        if (used_class.find(data.get_label()) == used_class.end())
        {
            cluster c(data);
            clusters.emplace_back(c);
            used_class.insert(data.get_label());
            used_indexes.insert(index);
        }
        ++index;
    }
}

void kmeans::train()
{
    std::srand(time(NULL));
    int index = 0;
    for (int index = 0; index < training_data.size(); ++index)
    {
        if (used_indexes.find(index) != used_indexes.end())
        {
            continue;
        }
        printf("\r tarining: %d", index);
        fflush(stdout);
        double min_dist = std::numeric_limits<double>::max();
        cluster *best_cluster = nullptr;
        std::for_each(clusters.begin(), clusters.end(), [&](cluster &c) {
            double dist = distance(c.get_centroid(), training_data.at(index));
            if (dist <= min_dist)
            {
                min_dist = dist;
                best_cluster = &c;
            }
        });
        best_cluster->add_to_cluster(training_data.at(index));
    }
}

double kmeans::distance(const std::vector<double> &centroid, const Data &point)
{
    double res = 0;
    for (int i = 0; i < centroid.size(); ++i)
    {
        res += std::pow(centroid.at(i) - point.get_feature_vector().at(i), 2);
    }
    res = std::sqrt(res);
    return res;
}

double kmeans::validate()
{
    double res = 0;
    int correct_cnt = 0, total_cnt = 0;
    for (const Data &point : validation_data)
    {
        double min_dist = std::numeric_limits<double>::max();
        const cluster *best_cluster = nullptr;
        std::for_each(clusters.cbegin(), clusters.cend(), [&](const cluster &c) {
            double dist_to_cluster = distance(c.get_centroid(), point);
            if (dist_to_cluster <= min_dist)
            {
                min_dist = dist_to_cluster;
                best_cluster = &c;
            }
        });

        if (best_cluster->get_most_calss() == point.get_label())
        {
            ++correct_cnt;
        }
        ++total_cnt;
    }
    res = 100.0 * correct_cnt / total_cnt;
    return res;
}

double kmeans::test()
{
    double res = 0;
    int correct_cnt = 0, total_cnt = 0;
    for (const Data &point : validation_data)
    {
        double min_dist = std::numeric_limits<double>::max();
        const cluster *best_cluster = nullptr;
        std::for_each(clusters.cbegin(), clusters.cend(), [&](const cluster &c) {
            double dist_to_cluster = distance(c.get_centroid(), point);
            if (dist_to_cluster <= min_dist)
            {
                min_dist = dist_to_cluster;
                best_cluster = &c;
            }
        });

        if (best_cluster->get_most_calss() == point.get_label())
        {
            ++correct_cnt;
        }
        ++total_cnt;
        printf("\rvalidating: %u %u %lu %.2lf", correct_cnt, total_cnt, test_data.size(), 100.0 * correct_cnt / total_cnt);
    }
    res = 100.0 * correct_cnt / total_cnt;
    return res;
}

void kmeans::set_training_data(std::vector<Data> vec)
{
    training_data = std::move(vec);
}

void kmeans::set_test_data(std::vector<Data> vec)
{
    test_data = std::move(vec);
}

void kmeans::set_validation_data(std::vector<Data> vec)
{
    validation_data = std::move(vec);
}

const std::vector<Data> &kmeans::get_training_data()
{
    return training_data;
}

const std::vector<Data> &kmeans::get_test_data()
{
    return test_data;
}

const std::vector<Data> &kmeans::get_validation_data()
{
    return validation_data;
}