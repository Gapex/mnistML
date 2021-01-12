#include "knn.hpp"

#include <cmath>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <iostream>

knn::knn() : k(75), neighbors(nullptr),
             training_data(nullptr),
             test_data(nullptr),
             validation_data(nullptr)
{
}
knn::knn(int val) : k(val)
{
}
knn::~knn()
{
    //free allocated STUFF
}

void knn::set_k(int val)
{
    this->k = val;
}

int knn::get_k()
{
    return this->k;
}

void knn::find_knearest(Data *query_point)
{
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

    std::priority_queue<Wrapper> q;
    for (size_t index = 0; index < this->training_data->size(); ++index)
    {
        double dist = this->calculate_distance(query_point, training_data->at(index));
        Wrapper wrapper(index, dist);
        if (q.size() <= k || dist <= q.top().dist_to_query_point)
        {
            q.emplace(wrapper);
            if (q.size() > k)
            {
                q.pop();
            }
        }
    }
    if (!neighbors)
        neighbors = new std::vector<Data *>();
    else
        neighbors->clear();
    neighbors->reserve(q.size());
    while (q.empty() == false)
    {
        auto &wrapper = q.top();
        Data *data = training_data->at(wrapper.index_in_train_set);
        neighbors->push_back(data);
        q.pop();
    }
    std::reverse(neighbors->begin(), neighbors->end());
}

void knn::set_training_data(std::vector<Data *> *data)
{
    this->training_data = data;
}
void knn::set_test_data(std::vector<Data *> *data)
{
    this->test_data = data;
}
void knn::set_validation_data(std::vector<Data *> *data)
{
    this->validation_data = data;
}

std::vector<Data *> *knn::get_training_data()
{
    return training_data;
}
std::vector<Data *> *knn::get_test_data()
{
    return test_data;
}
std::vector<Data *> *knn::get_validation_data()
{
    return validation_data;
}

int knn::predict()
{
    std::unordered_map<uint8_t, int> class_freq;
    for (Data *neighbor : *neighbors)
    {
        if(neighbor)
            ++class_freq[neighbor->get_label()];
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

double knn::calculate_distance(Data *d1, Data *d2)
{
    int ndimension = d1->get_feature_vector_size();
    if (ndimension != d2->get_feature_vector_size())
    {
        printf("feature vector size mismatch: %u != %u\n", ndimension, d2->get_feature_vector_size());
        exit(EXIT_FAILURE);
    }
    double res = 0;
    for (int i = 0; i < ndimension; ++i)
    {
        uint8_t val1 = d1->get_feature_vector()->at(i);
        uint8_t val2 = d2->get_feature_vector()->at(i);
        res += std::pow(val1 - val2, 2);
    }
    res = std::sqrt(res);
    return res;
}

double knn::validate_performance()
{
    double performance = 0;
    uint32_t cnt = 0;
    for (Data *query_point : *validation_data)
    {
        find_knearest(query_point);
        uint8_t prediction = predict();
        if (prediction == query_point->get_label())
        {
            ++cnt;
        }
    }
    performance = cnt * 100.0 / validation_data->size();
    printf("\rvalidation performance: %.3lf%%", performance);
    return performance;
}

double knn::test_performance()
{
    auto predict_task = [this](uint32_t from, uint32_t to, double *res) {
        std::thread::id id = std::this_thread::get_id();
        double performance = 0;
        uint32_t cnt = 0, index = 0;
        for (uint32_t i = from; i < to; ++i)
        {
            Data *query_point = this->get_test_data()->at(i);
            find_knearest(query_point);
            uint8_t prediction = predict();
            if (prediction == query_point->get_label())
            {
                ++cnt;
            }
            ++index;
            {
                
            }
        }
        performance = cnt * 100.0 / test_data->size();
        std::cout << id << ' ';
        printf("test performance: %.3lf%%\n", performance);
        *res = performance;
    };
    uint32_t ncpus = std::thread::hardware_concurrency();
    printf("%u CPU detcted\n", ncpus);
    double *res_ptr = new double[ncpus];
    auto _ = std::make_shared<double*>(res_ptr);
    std::vector<std::thread> ts;
    uint32_t chunk_size = this->test_data->size() / ncpus;
    uint32_t start_index = 0;
    for (uint32_t i = 0; i < ncpus; ++i)
    {
        uint32_t end_index = std::min(static_cast<size_t>(start_index + chunk_size), test_data->size());
        std::thread t(predict_task, start_index, end_index, &res_ptr[i]);
        std::cout << "thread(" << t.get_id() << ")";
        printf(" %u ~ %u\n", start_index, end_index);
        ts.emplace_back(std::move(t));
        start_index = end_index;
    }
    for(std::thread &t : ts){
        t.join();
    }
    double res = 0;
    for(uint32_t i = 0; i<ncpus; ++i){
        res += res_ptr[i];
    }
    res /= ncpus;
    return res;
}
