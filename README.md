## 使用C++11实现MNIST上的分类任务

### 下载mnist数据集
```bash
cd minst
sh download-minist.sh
```


### 构建
```bash
cd build
cmake ..
make
```

### 已经实现
1. KNN **准确率95%**
2. Kmeans **准确率81%**