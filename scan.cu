#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

#define PAGE_SIZE (64*KB) // 定义页面大小为4KB
#define ELEMENTS 4112// 数组的元素数量
#define Round 60000
#define test_ele 16
# define GB (1024LU*1024LU*1024LU)
# define MB (1024LU*1024LU)
# define KB (1024LU)
__device__ __host__
const int TOTAL_LEN = 4112;//总TLB条目数


__device__ __host__
const int THREADS = 32;

__device__ __host__
const int THREADS_per_block = 16;

__device__ __host__
const int ITEMS_PER_THREAD = (4112 + THREADS - 1) /THREADS;

//计算L3 tlb索引
__device__ unsigned long calculate_L3tlb_index(unsigned long *ptr) {
    unsigned long addr = (unsigned long)ptr;
    unsigned long index = 0;

    // 按位异或计算 TLB 索引
    index |= ((addr >> 20) & 1) ^ ((addr >> 28) & 1) ^ ((addr >> 36) & 1) ^ ((addr >> 44) & 1); // 第1位
    index |= (((addr >> 21) & 1) ^ ((addr >> 29) & 1) ^ ((addr >> 37) & 1) ^ ((addr >> 45) & 1)) << 1; // 第2位
    index |= (((addr >> 22) & 1) ^ ((addr >> 30) & 1) ^ ((addr >> 38) & 1) ^ ((addr >> 46) & 1)) << 2; // 第3位
    index |= (((addr >> 23) & 1) ^ ((addr >> 31) & 1) ^ ((addr >> 39) & 1)) << 3; // 第4位
    index |= (((addr >> 24) & 1) ^ ((addr >> 32) & 1) ^ ((addr >> 40) & 1)) << 4; // 第5位
    index |= (((addr >> 25) & 1) ^ ((addr >> 33) & 1) ^ ((addr >> 41) & 1))  << 5; // 第6位
    index |= (((addr >> 26) & 1) ^ ((addr >> 34) & 1) ^ ((addr >> 42) & 1)) << 6; // 第7位
    index |= (((addr >> 27) & 1) ^ ((addr >> 35) & 1) ^ ((addr >> 43) & 1)) << 7; // 第8位

    return index; // 返回计算得到的 TLB 索引值
}

//计算L2 tlb索引
__device__ unsigned long calculate_L2tlb_index(unsigned long *ptr) {
    unsigned long addr = (unsigned long)ptr;
    unsigned long index = 0;

    index |= ((addr >> 20) & 1) ^ ((addr >> 28) & 1) ^ ((addr >> 36) & 1) ^ ((addr >> 44) & 1); // 第1位
    index |= (((addr >> 21) & 1) ^ ((addr >> 29) & 1) ^ ((addr >> 37) & 1) ^ ((addr >> 45) & 1)) << 1; // 第2位
    index |= (((addr >> 22) & 1) ^ ((addr >> 30) & 1) ^ ((addr >> 38) & 1) ^ ((addr >> 46) & 1)) << 2; // 第3位
    index |= (((addr >> 23) & 1) ^ ((addr >> 31) & 1) ^ ((addr >> 39) & 1)) << 3; // 第4位
    index |= (((addr >> 24) & 1) ^ ((addr >> 32) & 1) ^ ((addr >> 40) & 1)) << 4; // 第5位
    index |= (((addr >> 25) & 1) ^ ((addr >> 33) & 1) ^ ((addr >> 41) & 1))  << 5; // 第6位
    index |= (((addr >> 26) & 1) ^ ((addr >> 34) & 1) ^ ((addr >> 42) & 1)) << 6; // 第7位
    index |= (((addr >> 27) & 1) ^ ((addr >> 35) & 1) ^ ((addr >> 43) & 1)) << 7; // 第8位

    return index; // 返回计算得到的 TLB 索引值
}






// 初始化数组的函数
__device__ __managed__ unsigned long d_thread_entry[THREADS];//入口数组
__global__ void initialize_array(unsigned long *d_array,int *count) {
    int countall = 0;
    unsigned long start_L2_index = calculate_L2tlb_index(d_array);
    int l3_index_counts[256] = {0},l2_counts[256] = {0};
    int l2tlb_count = 0,l3tlb_count = 0;
    int L2_temp,L3_temp;
    unsigned long tem_addr = 0;
    long long i = 0;
    unsigned long l3_heads[256] = {0};
    unsigned long l3_tails[256] = {0};  // 每组尾部指针
    //选择256组不同L3索引的元素，每组8个
    while (l3tlb_count < 256 * 8) {
    i += PAGE_SIZE / sizeof(unsigned long);
    int L3_temp = calculate_L3tlb_index(&d_array[i]);
    if (l3_index_counts[L3_temp] < 8) {
        if (l3_index_counts[L3_temp] == 0) {
            l3_heads[L3_temp] = i;
        } else {
            d_array[l3_tails[L3_temp]] = i;
        }
        l3_tails[L3_temp] = i;
        l3_index_counts[L3_temp]++;
        l3tlb_count++;
    }
    }
    //把他们连起来
    unsigned long final_tail = 0;
    bool first = true;
    for (int j = 0; j < 256; j++) {
        if (l3_index_counts[j] > 0) {
            if (!first) {
                d_array[final_tail] = l3_heads[j];
            } else {
                first = false;
            }
            final_tail = l3_tails[j];
        }
    }

    //选择128组不同L2索引的元素，每组8个
    while(l2tlb_count<256*8){
        i+=PAGE_SIZE/sizeof(unsigned long);
        L2_temp = calculate_L2tlb_index(&d_array[i]);
        if (l2_counts[L2_temp]<8)
        {
            d_array[tem_addr] = i;
            tem_addr = i;
            l2_counts[L2_temp]++;
            l2tlb_count++;
            // printf("L2:%d\n",L2_temp);
            // printf("%p",&d_array[i]);
        }
        if (i*sizeof(unsigned long)/PAGE_SIZE>250000)
        {
            break;
        }
        *count = l3tlb_count;
        
        
    }
    //最后选择16个页面填充L1
    for(int u= 0;u<16;u++){
        i+=PAGE_SIZE/sizeof(unsigned long);
        d_array[tem_addr] = i;
        tem_addr = i;
        

    }
    // 记录每个线程的访问起点
    unsigned long walker = 0;
    //int index = 0;
    int thread_id = 0;
    int visited = 0;

    while (visited < 4112 && thread_id < THREADS) {
        if (visited % ITEMS_PER_THREAD == 0) {
            d_thread_entry[thread_id] = walker;
            thread_id++;
        }
        walker = d_array[walker];
        visited++;
        
    }

     
    
        
        
        
        
    }
    
    
        

//填充TLB
// __global__ void prime_array(unsigned long *d_array, unsigned long *d_time,int round) {
//     // 仅使用单个线程
//     volatile unsigned long p = 0;
    
//     unsigned long start_time, end_time;
//     for (int i = 0; i < 3088; i++) {

//             // 访问当前元素
//         start_time = clock64();
//         p = d_array[p];
//         end_time = clock64();
                
    
//         d_time[round * 3088+i] = end_time-start_time;
//         //printf("%ld",calculate_L3tlb_index(&d_array[p])); 
//     }
      
// }


//多线程填充TLB
__global__ void prime_array(unsigned long *d_array, unsigned long *d_time,
                            unsigned long *d_thread_entry, int round) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid%THREADS_per_block!=0)
    {
        return;
    }
    volatile unsigned long p = d_thread_entry[tid/THREADS_per_block]; 
    volatile unsigned long* v_array=d_array; // 每个线程从自己的入口开始
    int offset = (tid/THREADS_per_block) * ITEMS_PER_THREAD; 

    for (int i = 0; i < ITEMS_PER_THREAD && i<4112; i++) {
        unsigned long start = clock64();
        p = v_array[p];
        unsigned long end = clock64();
        d_time[round * 4112  + offset + i] = end - start;
    
    }
    
}

// 获取当前时间（秒 + 微秒），作为 double 类型
double get_time_now() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // 高精度系统时间
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// 写入日志文件（追加模式）
void write_log(const char *tag) {
    FILE *fp = fopen("run.log", "a");
    if (fp == NULL) {
        perror("fopen");
        return;
    }
    double t = get_time_now();
    fprintf(fp, "[%s] %.6f\n", tag, t);
    fclose(fp);
}






int main() {
    unsigned long *d_array, *d_time={0};
    unsigned long *h_time = new unsigned long[(ELEMENTS+2) * PAGE_SIZE];
    int count=0;
    int device_index = 0;

    // 设置当前使用的 GPU
    cudaError_t wrong = cudaSetDevice(device_index);
    if (wrong != cudaSuccess) {
        fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(wrong));
        return 1;
    }

    // 获取设备属性
    cudaDeviceProp prop;
    wrong = cudaGetDeviceProperties(&prop, device_index);
    if (wrong != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(wrong));
        return 1;
    }

    printf("Device index: %d\n", device_index);
    printf("Device name: %s\n", prop.name);
    printf("Total memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // 获取 GPU UUID 使用 nvidia-smi
    char command[128];
    snprintf(command, sizeof(command),
             "nvidia-smi --query-gpu=uuid --format=csv,noheader --id=%d", device_index);

    FILE *fp1 = popen(command, "r");
    if (fp1 == NULL) {
        fprintf(stderr, "Failed to run nvidia-smi to get UUID.\n");
        return 1;
    }

    char uuid[128];
    if (fgets(uuid, sizeof(uuid), fp1) != NULL) {
        // 去除末尾换行符
        uuid[strcspn(uuid, "\n")] = 0;
        printf("GPU UUID: %s\n", uuid);
    } else {
        fprintf(stderr, "Could not read UUID from nvidia-smi output.\n");
    }

    pclose(fp1);
    //////////////////
    //////////////////
    int *count_gpu;
    cudaMalloc(&count_gpu, sizeof(int));
    cudaMemset(count_gpu, 0, sizeof(int));

    size_t array_size = (size_t)(250000) * (size_t)PAGE_SIZE; 

    

    // 在GPU上分配内存
    cudaMalloc(&d_array, array_size);
    cudaMalloc(&d_time, sizeof(unsigned long) * Round*4112);
    
    

    //初始化目標數組
    initialize_array<<<1, 1>>>(d_array,count_gpu);

    cudaDeviceSynchronize ();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    FILE *fp = fopen("result/result.csv","w");
    if (fp == NULL){
        printf("error open");
        return;
    }

    

    // 启动CUDA内核，遍历数组并记录耗时
    printf("scan start!!!\n");
    write_log("SCAN START");
    unsigned long start_time = clock();
    for (int i = 0; i < Round; i++)
    {   
        prime_array<<<32, 16>>>(d_array, d_time,d_thread_entry,i);

    
    
        // 等待CUDA完成计算
        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        
    }
  
    }
    unsigned long end_time = clock();
    printf("scan end!!!\n");
    write_log("SCAN END");
     // 将结果从设备拷贝到主机
        cudaMemcpy(h_time, d_time, 4112*Round*sizeof(unsigned long), cudaMemcpyDeviceToHost);
        for (int i = 0; i < Round; i++){
            //printf("Round%d===============================================================\n",i);
            //fprintf(fp,"Round%d===============================================================\n",i);
            for (int k = 0; k < 2048; k++) {
            
                
                fprintf(fp,"%lu\n",h_time[i*4112+k]);
             
            } 
        }
        
        
        
    
    printf("Total:%.3fms\n",(float)(end_time-start_time)/CLOCKS_PER_SEC*1000);
        
    

    // 释放设备内存
    cudaFree(d_array);
    cudaFree(d_time);
    cudaFree(count_gpu);

    // 释放主机内存
    delete[] h_time;
    fclose(fp);

    return 0;
}

