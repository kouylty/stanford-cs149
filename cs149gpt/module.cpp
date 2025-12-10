#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <cmath>
#include <thread>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

inline float oneDimRead(std::vector<float> &tensor, int &x) {
    return tensor[x];
}

inline void oneDimWrite(std::vector<float> &tensor, int &x, float &val) {
    tensor[x] = val;
}

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for(int b=0;b<B;b++)
    for(int h=0;h<H;h++)
    for(int i=0;i<N;i++)
    for(int j=0;j<d;j++)
    {
        float val = 0.0;
        fourDimWrite(O, b,h,i,j, H,N,d, val);
    }
    for(int b=0;b<B;b++)
    {
        for(int h=0;h<H;h++)
        {
            /* Calculate QK^T */
            for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
            {
                float val = 0.0;
                twoDimWrite(QK_t, i,j, N, val);
            }
            for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
            {
                float valqkt = 0.0;
                for(int k=0;k<d;k++)
                {
                    float valq = fourDimRead(Q, b,h,i,k, H,N,d);
                    float valkt = fourDimRead(K, b,h,j,k, H,N,d);
                    valqkt += valq*valkt;
                }
                twoDimWrite(QK_t, i,j, N, valqkt);
            }
            /* Calculate Softmax */
            for(int i=0;i<N;i++)
            {
                float sum = 0.0, mx = 0.0;
                for(int j=0;j<N;j++)
                {
                    float val = twoDimRead(QK_t, i,j, N);
                    mx = std::max(mx, val);
                }
                for(int j=0;j<N;j++)
                {
                    float val = twoDimRead(QK_t, i,j, N);
                    sum += exp(val-mx);
                }
                for(int j=0;j<N;j++)
                {
                    float val = twoDimRead(QK_t, i,j, N);
                    val = exp(val-mx) / sum;
                    twoDimWrite(QK_t, i,j, N, val);
                }
            }
            /* Calculate O = AV */
            for(int i=0;i<N;i++)
            for(int j=0;j<d;j++)
            {
                float valo = 0.0;
                for(int k=0;k<N;k++)
                {
                    float vala = twoDimRead(QK_t, i,k, N);
                    float valv = fourDimRead(V, b,h,k,j, H,N,d);
                    valo += vala * valv;
                }
                fourDimWrite(O, b,h,i,j, H,N,d, valo);
            }
        }
    }
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    for(int b=0;b<B;b++)
    for(int h=0;h<H;h++)
    for(int i=0;i<N;i++)
    for(int j=0;j<d;j++)
    {
        float val = 0.0;
        fourDimWrite(O, b,h,i,j, H,N,d, val);
    }
    
    std::vector<std::thread> threads;
    auto fthread = [&](int b, int h)
    {
        std::vector<float> QK_t = formatTensor(QK_tTensor);
        /* Calculate QK^T, Blocked */
        for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
        {
            float val = 0.0;
            twoDimWrite(QK_t, i,j, N, val);
        }
        int blocksize = 32;
        for(int kb=0;kb<d;kb+=blocksize)
        for(int ib=0;ib<N;ib+=blocksize)
        for(int jb=0;jb<N;jb+=blocksize)
        {
            for(int i=0;i<blocksize;i++)
            for(int j=0;j<blocksize;j++)
            {
                if(ib+i>=N || jb+j>=N) continue;
                int reali = ib+i, realj = jb+j;
                float valqkt = twoDimRead(QK_t, reali,realj, N);
                for(int k=0;k<blocksize;k++)
                    if(kb+k<d)
                    {
                        int realk = kb+k;
                        float valq = fourDimRead(Q, b,h,reali,realk, H,N,d);
                        float valkt = fourDimRead(K, b,h,realj,realk, H,N,d);
                        valqkt += valq*valkt;
                    }
                twoDimWrite(QK_t, reali,realj, N, valqkt);
            }
        }
        /* Calculate Softmax */
        for(int i=0;i<N;i++)
        {
            float sum = 0.0, mx = 0.0;
            for(int j=0;j<N;j++)
            {
                float val = twoDimRead(QK_t, i,j, N);
                mx = std::max(mx, val);
            }
            for(int j=0;j<N;j++)
            {
                float val = twoDimRead(QK_t, i,j, N);
                sum += exp(val-mx);
            }
            for(int j=0;j<N;j++)
            {
                float val = twoDimRead(QK_t, i,j, N);
                val = exp(val-mx) / sum;
                twoDimWrite(QK_t, i,j, N, val);
            }
            }
            /* Calculate O = AV */
            for(int i=0;i<N;i++)
            for(int j=0;j<d;j++)
            {
                float valo = 0.0;
                for(int k=0;k<N;k++)
                {
                    float vala = twoDimRead(QK_t, i,k, N);
                    float valv = fourDimRead(V, b,h,k,j, H,N,d);
                    valo += vala * valv;
                }
                fourDimWrite(O, b,h,i,j, H,N,d, valo);
            }
    };

    for(int b=0;b<B;b++)
    {
        for(int h=0;h<H;h++)
        {
            threads.push_back(std::thread(fthread,b,h));
        /* Calculate QK^T, Blocked */
        // for(int i=0;i<N;i++)
        // for(int j=0;j<N;j++)
        // {
        //     float val = 0.0;
        //     twoDimWrite(QK_t, i,j, N, val);
        // }
        // int blocksize = 32;
        // for(int kb=0;kb<d;kb+=blocksize)
        // for(int ib=0;ib<N;ib+=blocksize)
        // for(int jb=0;jb<N;jb+=blocksize)
        // {
        //     for(int i=0;i<blocksize;i++)
        //     for(int j=0;j<blocksize;j++)
        //     {
        //         if(ib+i>=N || jb+j>=N) continue;
        //         int reali = ib+i, realj = jb+j;
        //         float valqkt = twoDimRead(QK_t, reali,realj, N);
        //         for(int k=0;k<blocksize;k++)
        //             if(kb+k<d)
        //             {
        //                 int realk = kb+k;
        //                 float valq = fourDimRead(Q, b,h,reali,realk, H,N,d);
        //                 float valkt = fourDimRead(K, b,h,realj,realk, H,N,d);
        //                 valqkt += valq*valkt;
        //             }
        //         twoDimWrite(QK_t, reali,realj, N, valqkt);
        //     }
        // }
        // /* Calculate Softmax */
        // for(int i=0;i<N;i++)
        // {
        //     float sum = 0.0, mx = 0.0;
        //     for(int j=0;j<N;j++)
        //     {
        //         float val = twoDimRead(QK_t, i,j, N);
        //         mx = std::max(mx, val);
        //     }
        //     for(int j=0;j<N;j++)
        //     {
        //         float val = twoDimRead(QK_t, i,j, N);
        //         sum += exp(val-mx);
        //     }
        //     for(int j=0;j<N;j++)
        //     {
        //         float val = twoDimRead(QK_t, i,j, N);
        //         val = exp(val-mx) / sum;
        //         twoDimWrite(QK_t, i,j, N, val);
        //     }
        //     }
        //     /* Calculate O = AV */
        //     for(int i=0;i<N;i++)
        //     for(int j=0;j<d;j++)
        //     {
        //         float valo = 0.0;
        //         for(int k=0;k<N;k++)
        //         {
        //             float vala = twoDimRead(QK_t, i,k, N);
        //             float valv = fourDimRead(V, b,h,k,j, H,N,d);
        //             valo += vala * valv;
        //         }
        //         fourDimWrite(O, b,h,i,j, H,N,d, valo);
        //     }
        }
    }
    for(auto &th : threads) th.join();

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    #pragma omp parallel for collapse(3)
    // We give you a template of the first three loops for your convenience
    for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++)
    {
        for (int i = 0; i < N ; i++)
        {
            // YRow is moved inside so each OpenMP thread gets a local copy.
            at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
            std::vector<float> ORow = formatTensor(ORowTensor);
            //YOUR CODE HERE
            int xxx = 0;
            for(int j=0;j<N;j++)
            {
                float val = 0.0;
                twoDimWrite(ORow, xxx,j, N, val);
            }
            float sum = 0.0, mx = 0.0;
            for(int j=0;j<N;j++)
            {
                float valoraw = 0.0;
                for(int k=0;k<d;k++)
                {
                    float valq = fourDimRead(Q, b,h,i,k, H,N,d);
                    float valkt = fourDimRead(K, b,h,j,k, H,N,d);
                    valoraw += valq*valkt;
                }
                twoDimWrite(ORow, xxx,j, N, valoraw);
                mx = std::max(mx, valoraw);
            }
            for(int j=0;j<N;j++)
            {
                float valoraw = twoDimRead(ORow, xxx,j, N);
                sum += exp(valoraw-mx);
            }
            for(int j=0;j<N;j++)
            {
                float valoraw = twoDimRead(ORow, xxx,j, N);
                valoraw = exp(valoraw-mx) / sum;
                twoDimWrite(ORow, xxx,j, N, valoraw);
            }
            for(int j=0;j<d;j++)
            {
                float valo = 0.0;
                for(int k=0;k<N;k++)
                {
                    float vala = twoDimRead(ORow, xxx,k, N);
                    float valv = fourDimRead(V, b,h,k,j, H,N,d);
                    valo += vala * valv;
                }
                fourDimWrite(O, b,h,i,j, H,N,d, valo);
            }
        }
	}
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    int xxx = 0, yyy = 1;
    for(int b=0;b<B;b++)
    for(int h=0;h<H;h++)
    {
        for(int jb=0;jb<N;jb+=Bc)
        {
            /* Load Vj and Kj */
            for(int j=0;j<Br;j++)
            {
                int realj = jb+j;
                float valk = 0.0, valv = 0.0;
                for(int k=0;k<d;k++)
                {
                    if(realj<N)
                    {
                        valk = fourDimRead(K, b,h,realj,k, H,N,d);
                        valv = fourDimRead(V, b,h,realj,k, H,N,d);
                    }
                    twoDimWrite(Kj, j,k, d, valk);
                    twoDimWrite(Vj, j,k, d, valv);
                }
            }
            for(int ib=0;ib<N;ib+=Br)
            {
                /* Initialize */
                for(int i=0;i<Br;i++)
                {
                    float val = 0.0;
                    oneDimWrite(lij, i, val);
                    for(int j=0;j<Bc;j++)
                    {
                        twoDimWrite(lnew, i,j, Bc, val);
                    }
                }
                /* Load Qi, Oi */
                for(int i=0;i<Br;i++)
                {
                    int reali = ib+i;
                    for(int k=0;k<d;k++)
                    {
                        float valq = 0.0, valo = 0.0;
                        if(reali<N)
                        {
                            valq = fourDimRead(Q, b,h,reali,k, H,N,d);
                            valo = fourDimRead(O, b,h,reali,k, H,N,d);
                        }
                        twoDimWrite(Oi, i,k, d, valo);
                        twoDimWrite(Qi, i,k, d, valq);
                    }
                }
                /* Compute Sij = Qi*Kj^T, Pij =  exp(Sij) and Lij = \sum{Pij}*/
                for(int i=0;i<Br;i++)
                {
                    for(int j=0;j<Bc;j++)
                    {
                        int reali = ib+i, realj = jb+j;
                        if(reali>=N || realj>=N) continue;
                        float valsij = 0.0;
                        for(int k=0;k<d;k++)
                        {
                            float valq = twoDimRead(Qi, i,k, d);
                            float valkt = twoDimRead(Kj, j,k, d);
                            valsij += valq*valkt;
                        }
                        float valpij = exp(valsij);
                        twoDimWrite(Pij, i,j, Bc, valpij);
                        float vallij = oneDimRead(lij, i);
                        vallij += valpij;
                        oneDimWrite(lij, i, vallij);
                    }
                }
                /* Update Li and Lnew */
                for(int i=0;i<Br;i++)
                {
                    int reali = ib+i;
                    if(reali>=N) continue;
                    float vallij = oneDimRead(lij, i);
                    float valli = oneDimRead(l, reali);
                    float valnew = valli+vallij;
                    twoDimWrite(lnew, i,xxx, yyy, valnew);
                }
                /* Update Oi */
                for(int i=0;i<Br;i++)
                {
                    int reali = ib+i;
                    if(reali>=N) continue;
                    for(int j=0;j<d;j++)
                    {
                        float valpv = 0.0;
                        float valo = twoDimRead(Oi, i,j, d);
                        float valli = twoDimRead(l, reali,xxx, yyy);
                        valo = valo*valli;
                        for(int k=0;k<Bc;k++)
                        {
                            int realj = jb+k;
                            if(realj>=N) continue;
                            float valpij = twoDimRead(Pij, i,k, Bc);
                            float valvj = twoDimRead(Vj, k,j, d);
                            valpv += valpij*valvj;
                        }
                        valo += valpv;
                        float vallnew = twoDimRead(lnew, i,xxx, yyy);
                        valo = valo / vallnew;
                        twoDimWrite(Oi, i,j, d, valo);
                    }
                }
                /* Write back Oi and Lnew to O and L */
                for(int i=0;i<Br;i++)
                {
                    int reali = ib+i;
                    if(reali>=N) continue;
                    for(int k=0;k<d;k++)
                    {
                        float valo = twoDimRead(Oi, i,k, d);
                        fourDimWrite(O, b,h,reali,k, H,N,d, valo);
                    }
                    float vallnew = oneDimRead(lnew, i);
                    oneDimWrite(l, reali, vallnew);
                }
            }
        }
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
