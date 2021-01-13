#include "kmeans.hpp"

kmeans::kmeans()
{
}

kmeans::~kmeans()
{
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

void kmeans::train()
{
    for (const Data &data : training_data)
    {
        model[data.get_label()].push(data);
    }
    printf("聚类完成\n");
}

uint8_t kmeans::predict(const Data &query)
{
    double dist = std::numeric_limits<double>().max();
    uint8_t cluster_label = -1;
    for (auto &p : model)
    {
        double dist_to_cluster = p.second.query(query);
        if (dist_to_cluster < dist)
        {
            dist = dist_to_cluster;
            cluster_label = p.first;
        }
    }
    return cluster_label;
}

double kmeans::test_performance()
{
    uint32_t cnt = 0, total = 0;
    for (const Data &point : test_data)
    {
        uint8_t actual_label = point.get_label(), predict_label = predict(point);
        if (actual_label == predict_label)
        {
            ++cnt;
        }
        ++total;
        printf("\rkeans current test performance: %u / %u, %.3lf%%", cnt, total, cnt * 100.0 / total);
        fflush(stdout);
    }
    double res = cnt * 100.0 / total;
    printf("\nkemans test_performance: %.3lf%%\n", res);
    return res;
}
double kmeans::validation_performance()
{
    uint32_t cnt = 0, total = 0;
    for (const Data &point : validation_data)
    {
        uint8_t actual_label = point.get_label(), predict_label = predict(point);
        if (actual_label == predict_label)
        {
            ++cnt;
        }
        ++total;
        printf("\rkeans current validation performance: %u / %u, %.3lf%%", cnt, total, cnt * 100.0 / total);
        fflush(stdout);
    }
    double res = cnt * 100.0 / total;
    printf("\nkemans validation_performance: %.3lf%%\n", res);
    return res;
}