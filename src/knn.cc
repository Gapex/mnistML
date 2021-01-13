#include "knn.hpp"

#include <cmath>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <numeric>
#include <mutex>

knn::knn() : k(75)
{
}
knn::knn(int val) : k(val)
{
}
knn::~knn()
{
}

void knn::set_k(int val)
{
    k = val;
}

int knn::get_k()
{
    return k;
}

//在训练集train_data中找到query_point的k近邻点
//使用大根堆实现
struct Wrapper
{
    size_t index_in_train_set;
    double dist_to_query_point;

    Wrapper(size_t index, double dist) : index_in_train_set(index), dist_to_query_point(dist) {}

    bool operator<(const Wrapper &ano) const
    {
        if (dist_to_query_point != ano.dist_to_query_point)
        {
            return dist_to_query_point < ano.dist_to_query_point;
        }
        //如果距离相等，就按照下标递增排序
        return index_in_train_set < ano.index_in_train_set;
    }
};

std::vector<uint32_t> knn::find_knearest(const Data &query_point)
{
    std::priority_queue<Wrapper> q;
    for (size_t index = 0; index < training_data.size(); ++index)
    {
        double dist = calculate_distance(query_point, training_data.at(index));
        Wrapper wrapper(index, dist);
        if (q.size() <= k || dist <= q.top().dist_to_query_point)
        {
            q.emplace(wrapper);
            if (q.size() > k)
                q.pop();
        }
    }
    std::vector<uint32_t> neighbors;
    neighbors.reserve(q.size());
    while (q.empty() == false)
    {
        auto &wrapper = q.top();
        neighbors.push_back(wrapper.index_in_train_set);
        q.pop();
    }
    return std::move(neighbors);
}

void knn::set_training_data(std::vector<Data> data)
{
    training_data = std::move(data);
}

void knn::set_test_data(std::vector<Data> data)
{
    test_data = std::move(data);
}

void knn::set_validation_data(std::vector<Data> data)
{
    validation_data = std::move(data);
}

std::vector<Data> knn::get_training_data()
{
    return training_data;
}

std::vector<Data> knn::get_test_data()
{
    return test_data;
}
std::vector<Data>  knn::get_validation_data()
{
    return validation_data;
}

int knn::predict(const std::vector<uint32_t> neighbors)
{
    std::unordered_map<uint8_t, int> class_freq;
    for (uint32_t index : neighbors)
    {
        Data &neighbor = training_data.at(index);
        ++class_freq[neighbor.get_label()];
    }
    uint8_t res_label = -1;
    int max_freq = 0;
    for (auto &p : class_freq)
    {
        if (p.second >= max_freq)
        {
            max_freq = p.second;
            res_label = p.first;
        }
    }
    return res_label;
}

double knn::calculate_distance(const Data &d1, const Data &d2)
{
    int ndimension = d1.get_feature_vector_size();
    if (ndimension != d2.get_feature_vector_size())
    {
        printf("feature vector size mismatch: %u != %u\n", ndimension, d2.get_feature_vector_size());
        exit(EXIT_FAILURE);
    }
    double res = 0;
    for (int i = 0; i < ndimension; ++i)
    {
        uint8_t val1 = d1.get_feature_vector().at(i);
        uint8_t val2 = d2.get_feature_vector().at(i);
        res += std::pow(val1 - val2, 2);
    }
    res = std::sqrt(res);
    return res;
}

double knn::validate_performance()
{
    double performance = 0;
    uint32_t cnt = 0;
    for (Data &query_point : validation_data)
    {
        std::vector<uint32_t> neighbors = find_knearest(query_point);
        uint8_t prediction = predict(std::move(neighbors));
        if (prediction == query_point.get_label())
        {
            ++cnt;
        }
    }
    performance = cnt * 100.0 / validation_data.size();
    printf("\rvalidation performance: %.3lf%%", performance);
    return performance;
}

double knn::test_performance()
{
    std::mutex stdout_mtx;
    uint32_t cnt = 0, nstep = 0, nitems = test_data.size();
    auto inc_cnt = [&cnt, &stdout_mtx, nitems, &nstep](int inc, int step) {
        std::lock_guard<std::mutex> lk(stdout_mtx);
        cnt += inc;
        nstep += step;
        printf("\r%u/%u = %.3lf%%, total: %u", cnt, nstep, cnt * 100.0 / nstep, nitems);
        fflush(stdout);
    };
    auto predict_task = [this, &stdout_mtx, &inc_cnt](std::vector<Data>::iterator begin,
                                                      std::vector<Data>::iterator end) {
        std::thread::id id = std::this_thread::get_id();
        double performance = 0;
        uint32_t total_size = end - begin;
        for (auto iter = begin; iter != end; ++iter)
        {
            Data &query_point = *iter;
            std::vector<uint32_t> neighbors = find_knearest(query_point);
            uint8_t prediction = predict(std::move(neighbors));
            int inc = prediction == query_point.get_label();
            inc_cnt(inc, 1); //update the outer counter.
        }
        return;
    };
    uint32_t ncpus = std::thread::hardware_concurrency();
    std::vector<double> part_res(ncpus);
    std::vector<std::thread> ts;
    uint32_t chunk_size = test_data.size() / ncpus;

    uint32_t i = 0;
    for (std::vector<Data>::iterator start = test_data.begin(), final_end = test_data.end();
         start != final_end;)
    {
        std::vector<Data>::iterator end = start + chunk_size;
        if (end > final_end)
            end = final_end;
        std::thread t(predict_task, start, end);
        ts.emplace_back(std::move(t));
        start = end;
    }
    for (std::thread &t : ts)
    {
        t.join();
    }
    double res = cnt * 100 / nstep;
    printf("test performance: %.3lf%%\n", res);
    return res;
}
