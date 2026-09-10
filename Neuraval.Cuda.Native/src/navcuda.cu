#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cfloat>
#include <mutex>

#if defined(_WIN32)
#define NCB_API extern "C" __declspec(dllexport)
#else
#define NCB_API extern "C" __attribute__((visibility("default")))
#endif

#define NCB_TILE 16
#define NCB_REDUCE_THREADS 256
#define NCB_ELEMENTWISE_THREADS 256

enum NcbStatus
{
    NCB_OK = 0,
    NCB_ERROR_NO_DEVICE = 1,
    NCB_ERROR_INIT_FAILED = 2,
    NCB_ERROR_ALLOC_FAILED = 3,
    NCB_ERROR_COPY_FAILED = 4,
    NCB_ERROR_LAUNCH_FAILED = 5,
    NCB_ERROR_NOT_INITIALIZED = 6,
    NCB_ERROR_INVALID_ARGUMENT = 7
};

static int g_lastStatus = NCB_OK;
static bool g_initialized = false;
static int g_deviceId = -1;
static std::recursive_mutex g_mutex;

static void ncb_set_status(cudaError_t err, NcbStatus fallback)
{
    g_lastStatus = (err == cudaSuccess) ? NCB_OK : fallback;
}

NCB_API int ncb_cuda_device_count()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
    {
        g_lastStatus = NCB_ERROR_NO_DEVICE;
        return 0;
    }
    g_lastStatus = NCB_OK;
    return count;
}

NCB_API int ncb_cuda_get_device_name(int deviceId, char* buffer, int bufferLen)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, deviceId);
    if (err != cudaSuccess || buffer == nullptr || bufferLen <= 0)
    {
        g_lastStatus = NCB_ERROR_NO_DEVICE;
        return 0;
    }

    std::strncpy(buffer, props.name, bufferLen - 1);
    buffer[bufferLen - 1] = '\0';
    g_lastStatus = NCB_OK;
    return 1;
}

NCB_API int ncb_cuda_init(int deviceId)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess)
    {
        g_lastStatus = NCB_ERROR_INIT_FAILED;
        g_initialized = false;
        return 0;
    }

    g_deviceId = deviceId;
    g_initialized = true;
    g_lastStatus = NCB_OK;
    return 1;
}

NCB_API void* ncb_cuda_alloc(size_t bytes)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    ncb_set_status(err, NCB_ERROR_ALLOC_FAILED);
    return (err == cudaSuccess) ? ptr : nullptr;
}

NCB_API void ncb_cuda_free(void* ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (ptr == nullptr)
    {
        g_lastStatus = NCB_OK;
        return;
    }

    cudaError_t err = cudaFree(ptr);
    ncb_set_status(err, NCB_ERROR_ALLOC_FAILED);
}

NCB_API void ncb_cuda_copy_h2d(void* dst, const float* src, size_t bytes)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    ncb_set_status(err, NCB_ERROR_COPY_FAILED);
}

NCB_API void ncb_cuda_copy_d2h(float* dst, const void* src, size_t bytes)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    ncb_set_status(err, NCB_ERROR_COPY_FAILED);
}

__global__ void ncb_matmul_kernel(const float* a, const float* b, float* c, int m, int k, int n)
{
    __shared__ float tileA[NCB_TILE][NCB_TILE];
    __shared__ float tileB[NCB_TILE][NCB_TILE];

    int row = blockIdx.y * NCB_TILE + threadIdx.y;
    int col = blockIdx.x * NCB_TILE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (k + NCB_TILE - 1) / NCB_TILE;

    for (int t = 0; t < numTiles; t++)
    {
        int aCol = t * NCB_TILE + threadIdx.x;
        int bRow = t * NCB_TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < NCB_TILE; i++)
        {
            acc += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[row * n + col] = acc;
    }
}

NCB_API void ncb_cuda_matmul_device(const void* devA, const void* devB, void* devC, int m, int k, int n)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (m <= 0 || k <= 0 || n <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 block(NCB_TILE, NCB_TILE);
    dim3 grid((n + NCB_TILE - 1) / NCB_TILE, (m + NCB_TILE - 1) / NCB_TILE);

    ncb_matmul_kernel<<<grid, block>>>(
        static_cast<const float*>(devA),
        static_cast<const float*>(devB),
        static_cast<float*>(devC),
        m, k, n);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

NCB_API void ncb_cuda_matmul(const float* hostA, const float* hostB, float* hostC, int m, int k, int n)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    size_t bytesA = static_cast<size_t>(m) * k * sizeof(float);
    size_t bytesB = static_cast<size_t>(k) * n * sizeof(float);
    size_t bytesC = static_cast<size_t>(m) * n * sizeof(float);

    float* devA = static_cast<float*>(ncb_cuda_alloc(bytesA));
    float* devB = static_cast<float*>(ncb_cuda_alloc(bytesB));
    float* devC = static_cast<float*>(ncb_cuda_alloc(bytesC));

    if (devA == nullptr || devB == nullptr || devC == nullptr)
    {
        ncb_cuda_free(devA);
        ncb_cuda_free(devB);
        ncb_cuda_free(devC);
        g_lastStatus = NCB_ERROR_ALLOC_FAILED;
        return;
    }

    ncb_cuda_copy_h2d(devA, hostA, bytesA);
    ncb_cuda_copy_h2d(devB, hostB, bytesB);

    ncb_cuda_matmul_device(devA, devB, devC, m, k, n);
    cudaDeviceSynchronize();

    ncb_cuda_copy_d2h(hostC, devC, bytesC);

    ncb_cuda_free(devA);
    ncb_cuda_free(devB);
    ncb_cuda_free(devC);
}

NCB_API void ncb_cuda_synchronize()
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    cudaError_t err = cudaDeviceSynchronize();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

__global__ void ncb_matmul_transpose_b_kernel(const float* a, const float* b, float* c, int m, int k, int n, float scale)
{
    __shared__ float tileA[NCB_TILE][NCB_TILE];
    __shared__ float tileB[NCB_TILE][NCB_TILE];

    int row = blockIdx.y * NCB_TILE + threadIdx.y;
    int col = blockIdx.x * NCB_TILE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (k + NCB_TILE - 1) / NCB_TILE;

    for (int t = 0; t < numTiles; t++)
    {
        int aCol = t * NCB_TILE + threadIdx.x;
        int bCol = t * NCB_TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        tileB[threadIdx.x][threadIdx.y] = (col < n && bCol < k) ? b[col * k + bCol] : 0.0f;

        __syncthreads();

        for (int i = 0; i < NCB_TILE; i++)
        {
            acc += tileA[threadIdx.y][i] * tileB[threadIdx.x][i];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[row * n + col] = acc * scale;
    }
}

NCB_API void ncb_cuda_matmul_transpose_b_device(const void* devA, const void* devB, void* devC, int m, int k, int n, float scale)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (m <= 0 || k <= 0 || n <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 block(NCB_TILE, NCB_TILE);
    dim3 grid((n + NCB_TILE - 1) / NCB_TILE, (m + NCB_TILE - 1) / NCB_TILE);

    ncb_matmul_transpose_b_kernel<<<grid, block>>>(
        static_cast<const float*>(devA),
        static_cast<const float*>(devB),
        static_cast<float*>(devC),
        m, k, n, scale);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

NCB_API void ncb_cuda_matmul_transpose_b(const float* hostA, const float* hostB, float* hostC, int m, int k, int n, float scale)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    size_t bytesA = static_cast<size_t>(m) * k * sizeof(float);
    size_t bytesB = static_cast<size_t>(n) * k * sizeof(float);
    size_t bytesC = static_cast<size_t>(m) * n * sizeof(float);

    float* devA = static_cast<float*>(ncb_cuda_alloc(bytesA));
    float* devB = static_cast<float*>(ncb_cuda_alloc(bytesB));
    float* devC = static_cast<float*>(ncb_cuda_alloc(bytesC));

    if (devA == nullptr || devB == nullptr || devC == nullptr)
    {
        ncb_cuda_free(devA);
        ncb_cuda_free(devB);
        ncb_cuda_free(devC);
        g_lastStatus = NCB_ERROR_ALLOC_FAILED;
        return;
    }

    ncb_cuda_copy_h2d(devA, hostA, bytesA);
    ncb_cuda_copy_h2d(devB, hostB, bytesB);

    ncb_cuda_matmul_transpose_b_device(devA, devB, devC, m, k, n, scale);
    cudaDeviceSynchronize();

    ncb_cuda_copy_d2h(hostC, devC, bytesC);

    ncb_cuda_free(devA);
    ncb_cuda_free(devB);
    ncb_cuda_free(devC);
}

__global__ void ncb_matmul_transpose_a_kernel(const float* a, const float* b, float* c, int p, int m, int n)
{
    __shared__ float tileA[NCB_TILE][NCB_TILE];
    __shared__ float tileB[NCB_TILE][NCB_TILE];

    int row = blockIdx.y * NCB_TILE + threadIdx.y;
    int col = blockIdx.x * NCB_TILE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (p + NCB_TILE - 1) / NCB_TILE;

    for (int t = 0; t < numTiles; t++)
    {
        int aRow = t * NCB_TILE + threadIdx.x;
        int bRow = t * NCB_TILE + threadIdx.y;

        tileA[threadIdx.x][threadIdx.y] = (aRow < p && row < m) ? a[aRow * m + row] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < p && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < NCB_TILE; i++)
        {
            acc += tileA[i][threadIdx.y] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[row * n + col] = acc;
    }
}

NCB_API void ncb_cuda_matmul_transpose_a_device(const void* devA, const void* devB, void* devC, int p, int m, int n)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (p <= 0 || m <= 0 || n <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 block(NCB_TILE, NCB_TILE);
    dim3 grid((n + NCB_TILE - 1) / NCB_TILE, (m + NCB_TILE - 1) / NCB_TILE);

    ncb_matmul_transpose_a_kernel<<<grid, block>>>(
        static_cast<const float*>(devA),
        static_cast<const float*>(devB),
        static_cast<float*>(devC),
        p, m, n);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

NCB_API void ncb_cuda_matmul_transpose_a(const float* hostA, const float* hostB, float* hostC, int p, int m, int n)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    size_t bytesA = static_cast<size_t>(p) * m * sizeof(float);
    size_t bytesB = static_cast<size_t>(p) * n * sizeof(float);
    size_t bytesC = static_cast<size_t>(m) * n * sizeof(float);

    float* devA = static_cast<float*>(ncb_cuda_alloc(bytesA));
    float* devB = static_cast<float*>(ncb_cuda_alloc(bytesB));
    float* devC = static_cast<float*>(ncb_cuda_alloc(bytesC));

    if (devA == nullptr || devB == nullptr || devC == nullptr)
    {
        ncb_cuda_free(devA);
        ncb_cuda_free(devB);
        ncb_cuda_free(devC);
        g_lastStatus = NCB_ERROR_ALLOC_FAILED;
        return;
    }

    ncb_cuda_copy_h2d(devA, hostA, bytesA);
    ncb_cuda_copy_h2d(devB, hostB, bytesB);

    ncb_cuda_matmul_transpose_a_device(devA, devB, devC, p, m, n);
    cudaDeviceSynchronize();

    ncb_cuda_copy_d2h(hostC, devC, bytesC);

    ncb_cuda_free(devA);
    ncb_cuda_free(devB);
    ncb_cuda_free(devC);
}

__global__ void ncb_cast_float_to_half_kernel(const float* src, half* dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    dst[idx] = __float2half(src[idx]);
}

NCB_API void ncb_cuda_cast_float_to_half_device(const void* devSrcFloat, void* devDstHalf, int n)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (n <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 block(NCB_ELEMENTWISE_THREADS);
    dim3 grid((n + NCB_ELEMENTWISE_THREADS - 1) / NCB_ELEMENTWISE_THREADS);

    ncb_cast_float_to_half_kernel<<<grid, block>>>(
        static_cast<const float*>(devSrcFloat),
        static_cast<half*>(devDstHalf),
        n);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

__global__ void ncb_matmul_half_b_kernel(const float* a, const half* b, float* c, int m, int k, int n)
{
    __shared__ float tileA[NCB_TILE][NCB_TILE];
    __shared__ float tileB[NCB_TILE][NCB_TILE];

    int row = blockIdx.y * NCB_TILE + threadIdx.y;
    int col = blockIdx.x * NCB_TILE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (k + NCB_TILE - 1) / NCB_TILE;

    for (int t = 0; t < numTiles; t++)
    {
        int aCol = t * NCB_TILE + threadIdx.x;
        int bRow = t * NCB_TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? __half2float(b[bRow * n + col]) : 0.0f;

        __syncthreads();

        for (int i = 0; i < NCB_TILE; i++)
        {
            acc += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[row * n + col] = acc;
    }
}

NCB_API void ncb_cuda_matmul_half_b_device(const void* devA, const void* devBHalf, void* devC, int m, int k, int n)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (m <= 0 || k <= 0 || n <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 block(NCB_TILE, NCB_TILE);
    dim3 grid((n + NCB_TILE - 1) / NCB_TILE, (m + NCB_TILE - 1) / NCB_TILE);

    ncb_matmul_half_b_kernel<<<grid, block>>>(
        static_cast<const float*>(devA),
        static_cast<const half*>(devBHalf),
        static_cast<float*>(devC),
        m, k, n);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

__global__ void ncb_matmul_transpose_b_half_kernel(const float* a, const half* b, float* c, int m, int k, int n, float scale)
{
    __shared__ float tileA[NCB_TILE][NCB_TILE];
    __shared__ float tileB[NCB_TILE][NCB_TILE];

    int row = blockIdx.y * NCB_TILE + threadIdx.y;
    int col = blockIdx.x * NCB_TILE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (k + NCB_TILE - 1) / NCB_TILE;

    for (int t = 0; t < numTiles; t++)
    {
        int aCol = t * NCB_TILE + threadIdx.x;
        int bCol = t * NCB_TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        tileB[threadIdx.x][threadIdx.y] = (col < n && bCol < k) ? __half2float(b[col * k + bCol]) : 0.0f;

        __syncthreads();

        for (int i = 0; i < NCB_TILE; i++)
        {
            acc += tileA[threadIdx.y][i] * tileB[threadIdx.x][i];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        c[row * n + col] = acc * scale;
    }
}

NCB_API void ncb_cuda_matmul_transpose_b_half_device(const void* devA, const void* devBHalf, void* devC, int m, int k, int n, float scale)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (m <= 0 || k <= 0 || n <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 block(NCB_TILE, NCB_TILE);
    dim3 grid((n + NCB_TILE - 1) / NCB_TILE, (m + NCB_TILE - 1) / NCB_TILE);

    ncb_matmul_transpose_b_half_kernel<<<grid, block>>>(
        static_cast<const float*>(devA),
        static_cast<const half*>(devBHalf),
        static_cast<float*>(devC),
        m, k, n, scale);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

__global__ void ncb_softmax_rows_kernel(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    const float* rowIn = input + (size_t)row * cols;
    float* rowOut = output + (size_t)row * cols;

    float localMax = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        localMax = fmaxf(localMax, rowIn[j]);
    }
    shared[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float rowMax = shared[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float e = expf(rowIn[j] - rowMax);
        rowOut[j] = e;
        localSum += e;
    }
    shared[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float rowSum = shared[0];
    __syncthreads();

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        rowOut[j] /= rowSum;
    }
}

NCB_API void ncb_cuda_softmax_rows_device(const void* devInput, void* devOutput, int rows, int cols)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (rows <= 0 || cols <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 grid(rows);
    dim3 block(NCB_REDUCE_THREADS);
    size_t sharedBytes = NCB_REDUCE_THREADS * sizeof(float);

    ncb_softmax_rows_kernel<<<grid, block, sharedBytes>>>(
        static_cast<const float*>(devInput),
        static_cast<float*>(devOutput),
        rows, cols);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

NCB_API void ncb_cuda_softmax_rows(const float* hostInput, float* hostOutput, int rows, int cols)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    float* devInput = static_cast<float*>(ncb_cuda_alloc(bytes));
    float* devOutput = static_cast<float*>(ncb_cuda_alloc(bytes));

    if (devInput == nullptr || devOutput == nullptr)
    {
        ncb_cuda_free(devInput);
        ncb_cuda_free(devOutput);
        g_lastStatus = NCB_ERROR_ALLOC_FAILED;
        return;
    }

    ncb_cuda_copy_h2d(devInput, hostInput, bytes);

    ncb_cuda_softmax_rows_device(devInput, devOutput, rows, cols);
    cudaDeviceSynchronize();

    ncb_cuda_copy_d2h(hostOutput, devOutput, bytes);

    ncb_cuda_free(devInput);
    ncb_cuda_free(devOutput);
}

__global__ void ncb_layernorm_rows_kernel(const float* input, const float* gamma, const float* beta, float* output, float* meanOut, float* stdOut, int rows, int cols, float epsilon)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    const float* rowIn = input + (size_t)row * cols;
    float* rowOut = output + (size_t)row * cols;

    float localSum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        localSum += rowIn[j];
    }
    shared[threadIdx.x] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = shared[0] / cols;
    __syncthreads();

    float localVarSum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float diff = rowIn[j] - mean;
        localVarSum += diff * diff;
    }
    shared[threadIdx.x] = localVarSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float variance = shared[0] / cols;
    float std = sqrtf(variance + epsilon);

    if (threadIdx.x == 0)
    {
        meanOut[row] = mean;
        stdOut[row] = std;
    }

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
    {
        float normalized = (rowIn[j] - mean) / std;
        rowOut[j] = gamma[j] * normalized + beta[j];
    }
}

NCB_API void ncb_cuda_layernorm_rows_device(const void* devInput, const void* devGamma, const void* devBeta, void* devOutput, void* devMean, void* devStd, int rows, int cols, float epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (rows <= 0 || cols <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 grid(rows);
    dim3 block(NCB_REDUCE_THREADS);
    size_t sharedBytes = NCB_REDUCE_THREADS * sizeof(float);

    ncb_layernorm_rows_kernel<<<grid, block, sharedBytes>>>(
        static_cast<const float*>(devInput),
        static_cast<const float*>(devGamma),
        static_cast<const float*>(devBeta),
        static_cast<float*>(devOutput),
        static_cast<float*>(devMean),
        static_cast<float*>(devStd),
        rows, cols, epsilon);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

NCB_API void ncb_cuda_layernorm_rows(const float* hostInput, const float* hostGamma, const float* hostBeta, float* hostOutput, float* hostMean, float* hostStd, int rows, int cols, float epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    size_t bytesRow = static_cast<size_t>(rows) * cols * sizeof(float);
    size_t bytesCol = static_cast<size_t>(cols) * sizeof(float);
    size_t bytesRowVec = static_cast<size_t>(rows) * sizeof(float);

    float* devInput = static_cast<float*>(ncb_cuda_alloc(bytesRow));
    float* devGamma = static_cast<float*>(ncb_cuda_alloc(bytesCol));
    float* devBeta = static_cast<float*>(ncb_cuda_alloc(bytesCol));
    float* devOutput = static_cast<float*>(ncb_cuda_alloc(bytesRow));
    float* devMean = static_cast<float*>(ncb_cuda_alloc(bytesRowVec));
    float* devStd = static_cast<float*>(ncb_cuda_alloc(bytesRowVec));

    if (devInput == nullptr || devGamma == nullptr || devBeta == nullptr ||
        devOutput == nullptr || devMean == nullptr || devStd == nullptr)
    {
        ncb_cuda_free(devInput);
        ncb_cuda_free(devGamma);
        ncb_cuda_free(devBeta);
        ncb_cuda_free(devOutput);
        ncb_cuda_free(devMean);
        ncb_cuda_free(devStd);
        g_lastStatus = NCB_ERROR_ALLOC_FAILED;
        return;
    }

    ncb_cuda_copy_h2d(devInput, hostInput, bytesRow);
    ncb_cuda_copy_h2d(devGamma, hostGamma, bytesCol);
    ncb_cuda_copy_h2d(devBeta, hostBeta, bytesCol);

    ncb_cuda_layernorm_rows_device(devInput, devGamma, devBeta, devOutput, devMean, devStd, rows, cols, epsilon);
    cudaDeviceSynchronize();

    ncb_cuda_copy_d2h(hostOutput, devOutput, bytesRow);
    ncb_cuda_copy_d2h(hostMean, devMean, bytesRowVec);
    ncb_cuda_copy_d2h(hostStd, devStd, bytesRowVec);

    ncb_cuda_free(devInput);
    ncb_cuda_free(devGamma);
    ncb_cuda_free(devBeta);
    ncb_cuda_free(devOutput);
    ncb_cuda_free(devMean);
    ncb_cuda_free(devStd);
}

__global__ void ncb_adam_update_kernel(float* params, const float* grads, float* m, float* v, int n,
    float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grads[idx];

    float mNew = beta1 * m[idx] + (1.0f - beta1) * g;
    float vNew = beta2 * v[idx] + (1.0f - beta2) * g * g;

    m[idx] = mNew;
    v[idx] = vNew;

    float mHat = mNew / biasCorrection1;
    float vHat = vNew / biasCorrection2;

    params[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}

NCB_API void ncb_cuda_adam_update_device(void* devParams, const void* devGrads, void* devM, void* devV, int n,
    float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    if (n <= 0)
    {
        g_lastStatus = NCB_ERROR_INVALID_ARGUMENT;
        return;
    }

    dim3 block(NCB_ELEMENTWISE_THREADS);
    dim3 grid((n + NCB_ELEMENTWISE_THREADS - 1) / NCB_ELEMENTWISE_THREADS);

    ncb_adam_update_kernel<<<grid, block>>>(
        static_cast<float*>(devParams),
        static_cast<const float*>(devGrads),
        static_cast<float*>(devM),
        static_cast<float*>(devV),
        n, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);

    cudaError_t err = cudaGetLastError();
    ncb_set_status(err, NCB_ERROR_LAUNCH_FAILED);
}

NCB_API void ncb_cuda_adam_update(float* hostParams, const float* hostGrads, float* hostM, float* hostV, int n,
    float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
{
    std::lock_guard<std::recursive_mutex> lock(g_mutex);

    if (!g_initialized)
    {
        g_lastStatus = NCB_ERROR_NOT_INITIALIZED;
        return;
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* devParams = static_cast<float*>(ncb_cuda_alloc(bytes));
    float* devGrads = static_cast<float*>(ncb_cuda_alloc(bytes));
    float* devM = static_cast<float*>(ncb_cuda_alloc(bytes));
    float* devV = static_cast<float*>(ncb_cuda_alloc(bytes));

    if (devParams == nullptr || devGrads == nullptr || devM == nullptr || devV == nullptr)
    {
        ncb_cuda_free(devParams);
        ncb_cuda_free(devGrads);
        ncb_cuda_free(devM);
        ncb_cuda_free(devV);
        g_lastStatus = NCB_ERROR_ALLOC_FAILED;
        return;
    }

    ncb_cuda_copy_h2d(devParams, hostParams, bytes);
    ncb_cuda_copy_h2d(devGrads, hostGrads, bytes);
    ncb_cuda_copy_h2d(devM, hostM, bytes);
    ncb_cuda_copy_h2d(devV, hostV, bytes);

    ncb_cuda_adam_update_device(devParams, devGrads, devM, devV, n, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
    cudaDeviceSynchronize();

    ncb_cuda_copy_d2h(hostParams, devParams, bytes);
    ncb_cuda_copy_d2h(hostM, devM, bytes);
    ncb_cuda_copy_d2h(hostV, devV, bytes);

    ncb_cuda_free(devParams);
    ncb_cuda_free(devGrads);
    ncb_cuda_free(devM);
    ncb_cuda_free(devV);
}

NCB_API int ncb_cuda_last_status()
{
    return g_lastStatus;
}

NCB_API const char* ncb_cuda_status_message(int status)
{
    switch (status)
    {
        case NCB_OK: return "OK";
        case NCB_ERROR_NO_DEVICE: return "No se detectó ninguna GPU CUDA";
        case NCB_ERROR_INIT_FAILED: return "No se pudo inicializar el dispositivo CUDA";
        case NCB_ERROR_ALLOC_FAILED: return "Fallo al reservar memoria en el dispositivo";
        case NCB_ERROR_COPY_FAILED: return "Fallo al copiar memoria entre host y dispositivo";
        case NCB_ERROR_LAUNCH_FAILED: return "Fallo al lanzar el kernel CUDA";
        case NCB_ERROR_NOT_INITIALIZED: return "El dispositivo CUDA no fue inicializado";
        case NCB_ERROR_INVALID_ARGUMENT: return "Argumento inválido para la operación CUDA";
        default: return "Estado desconocido";
    }
}
