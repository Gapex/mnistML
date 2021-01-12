#include "data_handler.hpp"

#include <memory>
#include <algorithm>
#include <random>
#include <unordered_map>

DataHandler::DataHandler()
{
    data_array = new std::vector<Data *>();
    training_data = new std::vector<Data *>();
    test_data = new std::vector<Data *>();
    validation_data = new std::vector<Data *>();
}

DataHandler::~DataHandler()
{
    // Free dynamically allocated STUFF.
}

void DataHandler::read_feature_vector(const std::string &path)
{
    uint32_t header[4]; // magic|nimages|nrows|ncols
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
    {
        perror("打开失败");
        exit(EXIT_FAILURE);
    }
    size_t nread;

    nread = fread(header, sizeof(uint32_t), sizeof(header) / sizeof(uint32_t), f);
    if (nread != 4)
    {
        perror("header读取失败");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 4; ++i)
    {
        header[i] = convert_to_little_endian(reinterpret_cast<byte *>(header + i));
    }
    printf("done read header: %u %u %u %u\n", header[0], header[1], header[2], header[3]);
    uint32_t image_size = header[2] * header[3];
    byte image_buf[image_size]; //图片数据的缓冲区
    uint32_t nimages = header[1];
    for (int i = 0; i < nimages; ++i)
    { //header[1]表示文件中图片的个数
        Data *data = new Data();
        nread = fread(image_buf, sizeof(uint8_t), image_size, f);
        if (nread != image_size)
        {
            printf("read incomplete image data: %u\n", i);
            continue;
        }
        std::vector<uint8_t> *vec = new std::vector<uint8_t>(
            image_buf,
            image_buf + nread);
        data->set_feature_vector(vec);
        data_array->push_back(data);
    }
    fclose(f);
    printf("done read feature vectors: %lu\n", data_array->size());
}

void DataHandler::read_feature_labels(const std::string &path)
{
    uint32_t header[2]; // magic | nitem
    FILE *f = fopen(path.c_str(), "rb");
    auto closeFile = [](FILE *f) {
        fclose(f);
    };
    std::unique_ptr<FILE, decltype(closeFile)> fPtr(f, closeFile); //auto close
    if (!f)
    {
        perror("could not open lable file");
        exit(EXIT_FAILURE);
    }
    size_t nread;
    nread = fread(header, sizeof(uint32_t), 2, f);
    if (nread != 2)
    {
        perror("could not read lable file header");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < 2; ++i)
    {
        header[i] = convert_to_little_endian(reinterpret_cast<byte *>(header + i));
    }
    size_t nitems = header[1];
    if (nitems != data_array->size())
    {
        perror("label set size mismatch with feature number");
        exit(EXIT_FAILURE);
    }
    uint8_t *labels = new uint8_t[nitems];
    auto ptr = std::make_shared<uint8_t *>(labels);    // auto free
    nread = fread(labels, sizeof(uint8_t), nitems, f); // 一次性读完
    if (nread != nitems)
    {
        perror("incomplete lable file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nitems; ++i)
    {
        data_array->at(i)->set_label(labels[i]);
    }
    printf("done read label file: %lu\n", nitems);
}

void DataHandler::split_data()
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data_array->begin(), data_array->end(), g);

    uint32_t n_train_data = data_array->size() * TRAIN_SET_PERCENT;
    uint32_t n_test_data = data_array->size() * TEST_SET_PERCENT;

    training_data = new std::vector<Data *>();
    training_data->reserve(n_train_data);
    std::copy(data_array->end() - n_train_data, data_array->end(), std::back_inserter(*training_data)); //最后n_train_data个feature放到train_data里
    data_array->erase(data_array->end() - n_train_data, data_array->end());

    test_data = new std::vector<Data *>();
    test_data->reserve(n_test_data);
    std::copy(data_array->end() - n_test_data, data_array->end(), std::back_inserter(*test_data)); //最后n_test_data个feature放到test_data里
    data_array->erase(data_array->end() - n_test_data, data_array->end());

    validation_data = new std::vector<Data *>();
    validation_data->reserve(data_array->size());
    validation_data->assign(data_array->begin(), data_array->end());
    data_array->clear();

    printf("done split data: train(%lu) test(%lu) validation(%lu)\n",
           training_data->size(),
           test_data->size(),
           validation_data->size());
}

void DataHandler::count_classes()
{
    std::unordered_map<uint8_t, int> class_map;
    uint32_t count = 0;
    for (Data *data : *data_array)
    {
        if (class_map.find(data->get_label()) == class_map.end())
        {
            class_map[data->get_label()] = count;
            data->set_enumerated_label(count);
            ++count;
        }
    }
    num_classes = count;
    printf("%u classess detected\n", num_classes);
}

uint32_t DataHandler::convert_to_little_endian(byte *ptr)
{
    uint32_t val = *(reinterpret_cast<uint32_t *>(ptr));
    val = __bswap_32(val);
    return val;
}

std::vector<Data *> *DataHandler::get_training_data()
{
    return training_data;
}
std::vector<Data *> *DataHandler::get_test_data()
{
    return test_data;
}
std::vector<Data *> *DataHandler::get_validataion_data()
{
    return validation_data;
}