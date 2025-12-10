### Programming Assignment 3 笔记

kouylty



#### 【写在前面】

Assignment 3 主要使用 CUDA 完成一些基础的计算任务，例如 saxpy 和 scan，然后再用 CUDA 实现一个简易的图形渲染器（renderer）。主要对应第七次和第八次 lecture，主要介绍了 CUDA 编程的基础语法和 GPU 的基本架构，以及一些数据并行（data parallel）思想。

GPU 中有大量的核，每一个核都可以执行并行计算任务，尤其是 SIMD 或 SPMD。对于线程以及显存的管理，GPU 也有多层体系结构。GPU 的每个线程都有一份自己独有的内存空间，在此之上 GPU 还会把若干个（一般是 $32$ 个）连续线程统一成一个线程束（warp），这是 GPU 最小的线程调度单元。在此基础上，GPU 会把几个 warp，也就是若干个（通常是 $128$ 或 $256$ 个）线程，打包成一个线程块（block），每一个 block 有自己内部共享的内存空间。最后，GPU 还会把许多线程块打包成一个线程格（grid），每一个 grid 一般都是在平面上按顺序分布的，也有自己的共享内存空间。一般情况下，到 grid 层就已经是我们的原始任务了，也就是包含所有 block，因为在 GPU 上大部分计算的都是图形或矩阵运算，都是二维结构。

CUDA 是一种类 C 的语言，主要用于大规模并行计算领域，统一调度 CPU 和 GPU 进行计算。在 $\tt{Linux}$ 中使用 `sudo apt install nvidia-cuda-toolkit` 安装 CUDA 支持，并使用 `nvcc` 编译。CUDA 一般是在 block 级编程，我们也可以更精细地下沉到 warp 级。首先，CUDA 有一些限定符。`__host__` 表示此函数是在主机（也就是 CPU）上执行（这也是默认情况）。`__global__` 表示此函数在 GPU 的内核里执行，需要通过主机函数调用，语法是 `kernelFunc<<<BlockNum, BlockSize>>>()`。`__device__` 是在 GPU 计算单元上运行的函数，一般是 kernel function 的辅助函数，只能由 kernel 或其他 device 调用。除此之外，还有 `__shared__` 限定符，这是内存限定符，表示变量是 block 的共享变量，存储在 block 的共享内存中，只保留一份。其次，因为 CUDA 涉及到两个设备（CPU 和 GPU），无可避免会涉及到数据的传输与通信，CUDA 提供了独有的内存管理函数，例如 `cudaMalloc`、`cudaMemcpy`、`cudaMemset` 等，但在单一设备上操控就直接用基础的就好。最后就是 CUDA 里的一些线程管理函数，同步操作包括 warp 级的 `__syncwarp()`、block 级的 `__syncthreads()` 以及跨设备的 `cudaDeviceSynchronize()` 等等，除此之外还有原子操作（atomics）和更灵活的 group 级同步（cooperate groups）在此不多赘述。

最后是一些数据并行思想，$\mathrm{NumPy}$ 是一个经典的数据并行库，由 `map`、`fold` 等操作引入，主要讨论了并行扫描（parallel scan），使用了类似于树状数组和倍增的想法，在总操作次数还是 $O(N)$ 级的情况下分出了 $O(\log N)$ 层可并行的操作，本人不明觉厉，实际编程的时候只能根据伪代码实现。



#### 【Part A】

这就是要并行的实现向量的数乘和加法：$z=Ax+y$。因为这每一项都是完全独立的，并行化很容易。我们只需要实现好 CPU 和 GPU 之间的数据传输，在每一个线程中根据 `blockIdx` 和 `threadIdx` 算出对应的 `index` 即可，注意判断边界。



#### 【Part B】

这一部分要先利用 parallel scan 思想来计算数组的前项前缀和（exclusive prefix sum），再寻找一个数组的相邻重复项（find repeats）。

**Step 1**

我们先来做 exclusive scan。这就根据 lecture 讲的伪代码实现并行化就好，分为 up-sweep 和 down-sweep 阶段，再次不明觉厉。

实际上， lecture 应该是还讲了一种方法，是从 warp 级层层递进，先 scan warp，再 scan block，最后把所有 block 整合，本人看的云里雾里，也没有选择这种方式实现。

**Step 2**

接着我们来看 find repeats。普通的线性想法，是建立一个 `output` 数组，然后把 `input` 数组从前往后扫，如果 `input[i]==input[i+1]`，就把 `i` 加入当前 `output` 数组的末尾。如果想要并行化，就要并行处理每一个下标 `i`，然后并行放入 `output` 数组中。我们发现，这里并行化的主要挑战是，确定每一个合法的 `i` 在 `output` 数组里的位置。

在这里我们就可以考虑使用 exclusive scan 来帮助我们。我们可以先维护一个 `mask[]` 数组用于记录每个下标是否合法（判断 `i` 是否合法是独立的，很好并行化）。我们想，假设最后 `output[k]=i`，那 `k` 其实也就是 `mask[:i]` 之间 $1$ 的个数，这其实也是 `mask[]` 数组的前项前缀和。因此我们可以对 `mask[]` 数组做一次 exclusive scan，得到 `scanedMask[]`。这样，我们就有 `output[scanedMask[i]]=i`，当然前提是 `mask[i]=1`。最后在统计总共合法下标个数时有一个小细节，因为我们算的是前项前缀和，所以 `mask[]` 的最后一项其实不在 `scanedMask[]` 里，所以统计总数时要单独考虑它，也就是 `tot=scanedMask[N-1]+mask[N-1]`。



#### 【Part C】

这就到了本次 Assignment 的核心部分，实现一个简易的图形渲染器，这个渲染器的功能是画圆。每个像素点（pixel）都是一个 $1\times1$ 的小方格，判定像素点是否要染色的算法也很简单，染色当且仅当像素中心在圆内，具体地说，我们会先找到每个圆所在方块，然后枚举判断方块内的所有像素。只不过，因为圆是透明的，我们画圆必须严格按照给定的顺序。也就是说，我们不能直接把每个圆并行处理，这是因为不同并行线程的执行顺序是随机的（完全取决于硬件调度），我们无法保证圆是按顺序绘制的。这也就是 "convince yourself that the given implementation is incorrect"。

然后，我们就能很自然的转换思路，既然对圆并行化不行，那就对像素并行化。每个像素的渲染一定是独立的，这种方法显然是可行的。我们给每个像素分配一个线程，对于每个像素，按顺序遍历每个圆，依次判断是否在圆内并染色。测试这个代码的时候，虽然正确，但跑的特别慢，与 reference 有十倍左右的差距。究其原因，是因为圆的个数很多，但是覆盖某一个像素点的圆又比较少（远少于圆的总数），这就会遍历大量不需要的圆，造成大量的浪费。我们想，如果把可能覆盖某个像素点的圆先列举出来，再遍历这个选出来的圆的列表，总工作量就会少很多。

这时，GPU的线程分块（block）就派上用场了。我们可以先把整个图像分成若干个 block，先去找与每一个 block 相交的圆（判定两个矩形是否相交），这些圆才有可能覆盖 block 内的像素点，之后再对 block 内的每个像素点遍历这些选出来的圆并进行染色。实现时我们设置每个block 是 $16\times16$ 的大小，也就是包含 $256$ 个线程，$256$ 个像素。第一层优化就是我们不用对每个像素点都遍历所有的圆，而是对每一个 block 遍历一次所有圆。对一个 $1024\times1024$ 的图像，与每一个 block 相交的圆大约只占总数的 $10\%$，这是第二层优化。

想要找出与一个 block 相交的所有圆，可以先并行的计算每一个圆是否相交，记为 `mask[]`，然后再使用 exclusive scan，得到一个相交圆的下标数组 `circleIndices[]`。但是，圆的总数太多了（最多的一个测试点有 $1\mathrm{M}$ 个圆），如果直接这样做，数组长度要开到 $10^6$，这会 MLE（block 的共享内存空间不够大）。为此，我们可以再次利用一个 block 中的 $256$ 个线程，把这些圆分批次处理，每一批处理 $256$ 个圆（一个线程计算一个圆），这样就可以了，只是要注意 block 内线程的同步，只有当这一批次的圆都处理完，才能开启下一批。

最后测试结果如下。

<img src="https://raw.githubusercontent.com/kouylty/stanford-cs149/main/asst3/render/cudaout_0000.png" style="zoom:33%;" />

<img src="https://raw.githubusercontent.com/kouylty/stanford-cs149/main/asst3/render/score.png" style="zoom:50%;" />
