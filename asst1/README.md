### Programming Assignment 1 笔记

kouylty



#### 【写在前面】

前三次 lecture 初步介绍了并行计算的几种思想：超标量（superscalar）、SIMD（single instruction multiple data）、多线程（multi-thread）。超标量是在硬件层面找到可以并行执行的命令然后同步计算，使用的是同一套运行上下文（execution context）。SIMD 利用了处理器中的向量化（vectorized）寄存器和 ALU，同时对相同指令处理多组数据。多线程是一种调度和同步思想，将可以并行计算的代码分配给不同核、不同 exection context 进行计算，可以掩盖访存延迟等问题。Assignment 1 主要是模拟并测试上述几种思想对程序的加速效果。

本机硬件属性：Intel i7 8th CPU，四核，每个核支持双线程（hyper threads）。



#### 【Prog 1】

此题要求用多线程并行的生成 Mandel Set 图像。如果要分出 $n$ 个线程，那就把要求的复数域从上到下等分成 $n$ 块，分给 $n$ 个线程分别同步计算。

下图是线程数与加速效率之间的关系图。

<img src="raw.githubusercontent.com\kouylty\stanford-cs149\asst1\prog1_mandelbrot_threads\line_graph.png" style="zoom:20%;" />

我们发现，当线程数量为 $3$ 时，加速效率有下降。这是因为复数域中每个点的迭代次数不同， ```thread 1``` 计算的图像中间部分耗时多，出现了短板效应。

为了解决这个问题，可以重新划分图像，一行一行交替进行进程分配。In other words, turn block assignment into interleaved assignment. 此时，八线程的加速效果约为 $7.24\mathrm{x}$。

特别的，我们注意到线程数为 $8$ 时，加速效果出现下降，推测原因是个人的 CPU 线程调度的时间已经大于单个线程的计算时间了。

当线程数提升到 $16$ 时，加速效果没有明显变化，这是因为硬件最多只能同时运行 $8$ 个线程。



#### 【Prog 2】

此题提供了一套 SIMD simulator API，要求用向量操作计算一列数的幂。

对于一个 vector 中的底数和指数，使用 ```vgt_int``` 把指数是否大于零作为 ```maskAct```，使用 ```vmult_float``` 进行顺次乘积操作，以及 ```vsub_int``` 把指数减一。溢出判断时，使用 ```vgt_float``` 生成 ```maskOverflow```，再利用 ```vset_float``` 设置上界。考虑到 $n$ 可能不是 ```VECTOR_WIDTH``` 的整数倍，处理的时候先处理前面的 ```VECTOR_WIDTH``` 的整块，最后余下 ```len``` 个数单独处理（利用 ```init_ones(len)```）。

对于 ```VECTOR_WIDTH``` 为 $2,4,8,16$ 的情况，分别用 ```./myexp -s 10000``` 进行测试，结果如下。

<img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog2_vecintrin\test2.png" style="zoom:50%;" /><img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog2_vecintrin\test4.png" style="zoom:50%;" />

<img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog2_vecintrin\test8.png" style="zoom:50%;" /><img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog2_vecintrin\test16.png" style="zoom:50%;" />

我们发现 vector utilization 是很高的，说明我们很好的利用了 SIMD 的性质。另外，total vector instructions 与 vector width 大致成反比，符合预期。

我们还发现，随着 ```VECTOR_WIDTH``` 增加，vector utilization 在逐渐降低。这是因为，对于每一组向量中每一个工作域（lane）的数，都要操作 $m=\max\{exp[i]\}$ 次，其他数就会产生 $m-exp[i]$ 个空操作，而向量里的数越多，空操作的总数也就越多。

此外，此题还有 extra credit，要求我们计算一列数的和。调用 API 中的 ```hadd``` 和 ```interleave```，可以在对数复杂度下计算一列数的和。只需要交替使用这两个函数，最后的部分和会存在 ```vec[0]``` 中。



#### 【Prog 3】

从此题开始，需要使用 ISPC 语言编写和测试并行程序。ISPC 是 Intel 发行的一种支持自动并行计算的语言。为实现并行计算，ISPC 中会划分出很多程序实例（a gang of program instances），在运行时生成许多任务（tasks）分配给不同的执行上下文（execution context）进行计算。每个 instance 有 ```ProgramIndex``` 和 ```ProgramCount``` 两个属性，支持手动将每次循环划分进不同 instance。同时，ISPC 还支持一种更抽象的写法 ```foreach```，让编译器决定如何划分 tasks。ISPC 中的 task 与 C++ 中的 thread 不同之处在于，启动 $10000$ 个 threads，会实际创建这 $10000$ 个线程运行，这是很慢的，因为大量时间消耗在了线程调度上。而启动 $10000$ 个 tasks，编译器会根据硬件情况等创建合适数量的线程，这 $10000$ 个任务组成任务池（task pool），每个线程从池中取出未做的任务完成。

在搭建环境方面，使用语句 ```sudo apt install ispc``` 来安装 ISPC 编译器。

第一部分，理论上使用 ISPC 计算 mandel set 会获得 $8$ 倍加速，因为 SIMD 向量操作的宽度为 $8$。但实际测试中加速只有约 $5.05\mathrm{x}$，推测原因是向量运算不均衡而导致向量利用率低于 $100\%$。

第二部分，ISPC launch 了若干任务。当有两个任务时，加速约 $9.80\mathrm{x}$。当任务数量为 $8$ 时，加速效果最好，如下图。

<img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog3_mandelbrot_ispc\test2.png" style="zoom:50%;" /><img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog3_mandelbrot_ispc\test8.png" style="zoom:50%;" />



#### 【Prog 4】

本题要求使用牛顿法迭代计算平方根。

一般情况下，加速效率的测试结果如下图。

<img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog4_sqrt\test1.png" style="zoom:50%;" />

可见，SIMD 带来的加速效果约为 $4\mathrm{x}$，multicore/tasks 带来的加速效果也约为 $4\mathrm{x}$。理论上，后者带来的加速效果应该接近 $8\mathrm{x}$，效果减半应该与本人虚拟机的硬件设置有关。

想要获得最大的加速效果，就是要最大化向量利用率（vector utilization）。可以将所有数设置成相同的值（如 ```2.9999f```），这样向量利用率可以接近 $100\%$。相反，想要获得最差的加速效果，就要尽可能降低向量利用率，可以每八个数中有一个是 ```2.9999f```，其余全是 ```1.f```。两种情况的测试效果如下图。

<img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog4_sqrt\test2.png" style="zoom:50%;" /><img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog4_sqrt\test3.png" style="zoom:50%;" />

特别的，我们发现在最差情况下 SIMD 的加速效果小于 $1$，推测是因为在向量利用率降低至 $10\%$ 左右时，SIMD 硬件调度的消耗已经不能忽略。



#### 【Prog 5】

此题用线性变换的背景探讨了影响效率（efficiency）的另一个关键因素：带宽（bandwidth）。

对于线性变换来说，每次访存后进行的计算操作很少（一次乘法、一次加法）。但是两次 ALU 计算后却要进行 $4$ 次 I/O 操作（读 $x,y$，读 $result$，写 $result$）。

因为带宽涉及硬件中的存储器（memory）、总线（bus）以及数据传输等问题，所以无法通过简单的重写代码完成加速，能做的只有增加每次访存后的运算次数，以掩盖访存的延迟（hide the latency）。



#### 【Prog 6】

此题需要我们通过并行化来优化聚类分析算法。

在搭建环境方面，我的虚拟机不允许直接用 ```pip``` 下载 python 库，需要使用 ```sudo``` 提升权限。例如，如果想安装 numpy 库，需要命令 ```sudo apt install python3-numpy```。

首先，根据 ```main.cpp``` 中的注释，在本地生成数据集 ```data.dat```。根据要求，我们只能修改 ```computeAssignments```，```computeCentroids```，```computeCost``` 三个函数中的一个，因此我们要先对三个函数的耗时进行分析，结果如下图。

<img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog6_kmeans\test1.1.png" style="zoom:50%;" /><img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog6_kmeans\test1.2.png" style="zoom:50%;" />

我们发现，函数 ```computeAssignments``` 用时最长，考虑对这个函数进行并行化。这个函数实现的功能是，对每一个点，分配给距离它最近的中心（centroid）。由此可见，每个点是独立的、可并行化的（parallelizeable）。因此，将 $M$ 个点等分成 $8$ 份，分配给 $8$ 个线程进行独立计算。优化后的运行结果如下图。

<img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog6_kmeans\test2.1.png" style="zoom:50%;" /><img src="E:\NJU\并行计算（cmu15-418,cs149）\asst1\prog6_kmeans\test2.2.png" style="zoom:50%;" />

总加速效果约为 $1.82\mathrm{x}$。
