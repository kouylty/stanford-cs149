### CS149GPT 笔记

kouylty



#### 【写在前面】

gpt149 是一个基于 transformer 的深度神经网络（deep neural network, DNN），可以生成莎士比亚的文本。在 gpt149 中，transformer 的核心就是注意力机制（attention mechanism），公式为 $O=\mathrm{softmax}(QK^T)V$，其中包含两次矩阵乘法和一次 $\mathrm{softmax}$ 运算。其中 $Q,K,V$ 都是张量（tensor），因为此处我们还使用了多头注意力（multi-head attention, MHA），因此张量的形状都是 `[batch_size, num_heads, seq_len, hidden_dim]`，记为 $[B,H,N,d]$。



#### 【Part 1】

这一部分就是最简单的实现 attention，遍历 $B,H$，做一次矩阵乘法、一次 $\mathrm{softmax}$ 和第二次矩阵乘法。其中张量 `QK_t` 用于保存第一步乘法 $QK^T$ 的中间结果。另外，在 $\mathrm{softmax}$ 阶段要大量计算 $e$ 指数，为了提升精度，我们可以先将 `QK_t` 矩阵的每个元素减去行最大值再计算。



#### 【Part 2】

对于每一个 $b,h$，我们做朴素矩阵乘法时（以 $QK^T$ 为例），我们都是对于目标矩阵的某个位置 $(i,j)$，取出 $Q$ 的第 $i$ 行和 $K$ 的第 $j$ 行进行内积运算。这需要大量读写内存，并且每次读写的位置都不连续，也就是说缓存的命中率很低。为了提升读写效率，我们要努力让每次读写尽量连续。这样，我们想到可以把矩阵分块，对每一块矩阵，它们在内存中都是连续的，读写起来也更快。分块矩阵乘的朴素算法为

```c
for(int ib=0;ib<N;ib+=b)
    for(int jb=0;jb<M;jb+=b)
        for(int kb=0;kb<K;kb+=b)
        {
            for(int i=0;i<b;i++)
                for(int j=0;j<b;j++)
                    for(int k=0;k<b;k++)
                        c[ib+i][jb+j] += a[ib+i][kb+k]*b[kb+k][jb+j];
		}
```

在 attention 的 tensor 计算中，只要把读写换成他提供的接口即可。

测试了 `blocksize` 等于 $8,16,32,64$ 的情况，发现 $16,32$ 最快，大体原因是本机器的 $L1$ 缓存为 $2.0MB$，对应 `blocksize` 正好能完整放进缓存中。

另外，我们还发现对于每一组 $b,h$，内部的 attention 计算都是完全独立的，因此可以为每一组 $b,h$ 创建一个线程来并行计算。这样可以把效率提升大约 $1.5$ 倍。

<img src="https://raw.githubusercontent.com/kouylty/cs149gpt/assets/test2_ref.png" style="zoom:50%;" /> <img src="https://raw.githubusercontent.com/kouylty/cs149gpt/assets/test2_stu.png" style="zoom:50%;" />



#### 【Part 3】

再次观察 attention 的公式，我们发现，因为 $\mathrm{softmax}$ 是行相关的，所以我们只要能算出 $QK^T$ 的一整行，就可以根据公式算出对应 $O$ 矩阵的一整行。因此我们可以按行计算，不需要存储完整的 $QK^T$ 矩阵，只要每次存储 $QK^T$ 的一整行即可。这就是 "fused"，即把多种不同算子融合起来计算。这样做了以后不仅提高内存读写的效率，而且我们惊喜的发现对于 $B,H,N$ 三维都是可并行的，比之前的并行多了一维。

在这里我们使用的是 $\mathrm{OpenMP}$ 并行策略，因为我们的三重循环都是可并行的，我们只需要在循环前加上

```c
#pragma omp parallel for collapse(3)
```

就好。这种并行策略下的效率大约是 Part 2 并行的 $3$ 倍。



#### 【Part 4】

最后就是 Flash Attention。Flash Attention 是在 Part 2 分块矩阵乘法的基础上，努力让整个 Attention 计算变成可分块的（此处的术语是 tiling）。整个流程中 tiling 的瓶颈在于 $\mathrm{softmax}$ 运算，因为这个运算每次都要一整行的数据。根据 Flash Attention 算法，他是把 $\mathrm{softmax}$ 转化成了递推，这样就可以做到完全分块。

测试时我们发现，Part 4 的用时其实是更长的，但是内存使用大大减少，这是因为中间使用的张量大小都是 tile 级别的。

<img src="https://raw.githubusercontent.com/kouylty/cs149gpt/assets/test2_ref.png" style="zoom:50%;" /> 

其实我们的 Part 4 还能进一步优化，比如并行化 $B,H$ 维度，内部做更加细粒的递推处理，减少迭代次数。
