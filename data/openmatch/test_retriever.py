import faiss
import numpy as np

def main():
    d = 64  # 向量的维度
    nb = 20000  # 数据库中的向量数量
    np.random.seed(1234)

    # 生成一些随机数据向量
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    # 创建一个 IndexShards 对象
    index = faiss.IndexShards(d, True, True)

    # 设定使用的 GPU 列表
    ngpus = faiss.get_num_gpus()

    # 为每个 GPU 创建一个索引，并添加到 IndexShards
    for i in range(ngpus):
        sub_index = faiss.IndexFlatL2(d)  # 每个 GPU 上的索引类型
        gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), i, sub_index)
        index.add_shard(gpu_index)

    # 将数据添加到索引
    index.add(xb)

    # 搜索参数
    nq = 5  # 查询向量数量
    k = 4   # 返回的最近邻数量
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # 进行搜索
    D, I = index.search(xq, k)

    print('Distances\n', D)
    print('Indices\n', I)

if __name__ == '__main__':
    main()