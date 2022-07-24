#include <iostream>
#include <limits>
#include <bit>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/uio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

// Defines:
// HIPFIZZBUZZ_VMSPLICE (0/1) (default: 1)
//     If 1, use vmsplice() instead of write() to print the output. If the output is not consumed
//     at sufficient rate, it might be corrupted. Disable hwen piping the output to slower programs.
// HIPFIZZBUZZ_PRINT_SYNC (0/1) (default: 0)
//     Print a buffer directly after computing it instead of swapping buffers. Useful for debugging.
// HIPFIZZBUZZ_OVERRIDE_PIPESZ (0/1) (default: 1)
//     Override the pipe size. When this is enabled, some programs cannot consume the pipe correctly.
//     Useful to disable for debugging.
// HIPFIZZBUZZ_COPY_MEM (0/1) (default: 0)
//     Manually copy vram to main memory before printing it, instead of directly printing automatically
//     migrating data.
// HIPFIZZBUZZ_DEBUG (0/1) (default: 0)
//     Enable "debug mode": This enables HIPFIZZBUZZ_PRINT_SYNC and HIPFIZZBUZZ_COPY_MEM,
//     disables HIPFIZZBUZZ_VMSPLICE and HIPFIZZBUZZ_OVERRIDE_PIPESZ.

#ifdef HIPFIZZBUZZ_DEBUG
    #undef HIPFIZZBUZZ_VMSPLICE
    #undef HIPFIZZBUZZ_PRINT_SYNC
    #undef HIPFIZZBUZZ_OVERRIDE_PIPESZ
    #undef HIPFIZZBUZZ_COPY_MEM

    #define HIPFIZZBUZZ_VMSPLICE 0
    #define HIPFIZZBUZZ_PRINT_SYNC 1
    #define HIPFIZZBUZZ_OVERRIDE_PIPESZ 0
    #define HIPFIZZBUZZ_COPY_MEM 1
#else
    #ifndef HIPFIZZBUZZ_VMSPLICE
        #define HIPFIZZBUZZ_VMSPLICE 1
    #endif

    #ifndef HIPFIZZBUZZ_PRINT_SYNC
        #define HIPFIZZBUZZ_PRINT_SYNC 0
    #endif

    #ifndef HIPFIZZBUZZ_OVERRIDE_PIPESZ
        #define HIPFIZZBUZZ_OVERRIDE_PIPESZ 1
    #endif

    #ifndef HIPFIZZBUZZ_COPY_MEM
        #define HIPFIZZBUZZ_COPY_MEM 0
    #endif
#endif

#define HIP_CHECK(expr) do { \
    hipError_t status = (expr); \
    if (status != hipSuccess) { \
       std::cerr << "error at line " << __LINE__ << ": " << hipGetErrorName(status) << " (" << status << ")" << std::endl; \
       exit(1); \
    } \
} while (0)

constexpr size_t num_swap_buffers = 3;

// Size to override the stdout pipe with, if enabled
constexpr int pipe_size = 512 * 1024;

// Max digits we need, floor(log10(2^64))
constexpr int max_digits = 19;
constexpr int max_line_size = max_digits + 1;

 // Number of lines in on "fizzbuzz group": the pattern repeats after 15 lines.
constexpr unsigned lines_per_group = 15;
constexpr unsigned block_size = 256;

constexpr unsigned groups_per_thread = 1;
constexpr unsigned groups_per_block = block_size * groups_per_thread;
constexpr unsigned groups_per_batch = groups_per_block * 120;

constexpr size_t lines_per_batch = groups_per_batch * lines_per_group;
// Upper bound for the memory we need: every line needs at most max(8, max_digits) bytes, plus one for the newline.
constexpr size_t bytes_per_batch = lines_per_batch * max_line_size;
constexpr unsigned blocks_per_batch = groups_per_batch / groups_per_block;

constexpr const char* initial_group  = "1\n2\nfizz\n4\nbuzz\nfizz\n7\n8\nfizz\nbuzz\n";
constexpr size_t initial_group_bytes = 35;

constexpr uint64_t digits_to_base_offset[] = {
    0ULL,
    2ULL,
    35ULL,
    413ULL,
    4673ULL,
    52073ULL,
    574073ULL,
    6274073ULL,
    68074073ULL,
    734074073ULL,
    7874074073ULL,
    84074074073ULL,
    894074074073ULL,
    9474074074073ULL,
    100074074074073ULL,
    1054074074074073ULL,
    11074074074074073ULL,
    116074074074074073ULL,
    1214074074074074073ULL,
    12674074074074074073ULL,
};

constexpr uint64_t digits_to_base_power[] = {
    1ULL,
    1ULL,
    10ULL,
    100ULL,
    1000ULL,
    10000ULL,
    100000ULL,
    1000000ULL,
    10000000ULL,
    100000000ULL,
    1000000000ULL,
    10000000000ULL,
    100000000000ULL,
    1000000000000ULL,
    10000000000000ULL,
    100000000000000ULL,
    1000000000000000ULL,
    10000000000000000ULL,
    100000000000000000ULL,
    1000000000000000000ULL,
};

template <uint64_t n>
__device__
void barret_divmod(uint64_t& q, uint64_t& t, uint64_t x) {
    static_assert(std::popcount(n) != 1, "n must not be a power of 2");

    constexpr uint64_t k = 64;
    constexpr uint64_t m = std::numeric_limits<uint64_t>::max() / n;

    const uint64_t q0 = __umul64hi(x, m);
    t = x - q0 * n;
    q = q0;

    if (t == n) {
        ++q;
        t = 0;
    }
}

template <uint64_t n>
__host__
void barret_divmod(uint64_t& q, uint64_t& t, uint64_t x) {
    static_assert(std::popcount(n) != 1, "n must not be a power of 2");

    constexpr uint64_t k = 64;
    constexpr uint64_t m = std::numeric_limits<uint64_t>::max() / n;

    const uint64_t q0 = (uint64_t)((__uint128_t)(x) * (__uint128_t)(m) >> k);
    t = x - q0 * n;
    q = q0;

    if (t == n) {
        ++q;
        t = 0;
    }
}

template <uint64_t n>
__device__
uint64_t barett_div(uint64_t x) {
    uint64_t q, r;
    barret_divmod<n>(q, r, x);
    return q;
}

template <uint64_t n>
__host__
uint64_t barett_div(uint64_t x) {
    uint64_t q, r;
    barret_divmod<n>(q, r, x);
    return q;
}

template <int digits>
__host__ __device__
constexpr uint64_t fizzbuzz_bytes_for_groups(uint64_t groups) {
    const uint64_t bytes_per_group =
        // each group contains...
        4 * 4 // 4 fizzes...
        + 2 * 4 // 2 buzzes...
        + 8 // 1 fizzbuzz...
        + 8 * digits // 8 digits...
        + 15; // and 15 newlines.

    return bytes_per_group * groups;
}

template <int digits>
__host__ __device__
uint64_t fizzbuzz_group_byte_offset(uint64_t group) {
    const uint64_t base = digits_to_base_power[digits];
    return base + initial_group_bytes + fizzbuzz_bytes_for_groups<digits>(group);
}

template <unsigned digits>
__global__
__launch_bounds__(block_size)
void fizzbuzz_kernel(
    const uint64_t batch_start_group,
    const uint64_t batch_end_group,
    const uint64_t batch_start_byte,
    uint8_t* const output
) {
    constexpr unsigned shared_bytes_per_group = fizzbuzz_bytes_for_groups<digits>(1);
    constexpr unsigned shared_bytes_per_block = shared_bytes_per_group * block_size;

    static_assert(shared_bytes_per_block < 65536, "Shared memory exceeds limit of 64KiB");

    const unsigned tid = hipThreadIdx_x;
    const unsigned gid = hipBlockIdx_x * block_size + tid;
    const unsigned bid = hipBlockIdx_x;

    const uint64_t block_start_group = batch_start_group + bid * groups_per_block;

    __shared__ uint8_t shared_output[shared_bytes_per_block];

    #pragma unroll
    for (unsigned i = 0; i < groups_per_thread; ++i) {

        const uint64_t stripe_start_group = block_start_group + i * block_size;
        const uint64_t stripe_end_group   = std::min(stripe_start_group + groups_per_block, batch_end_group);

        const uint64_t stripe_start_byte = fizzbuzz_group_byte_offset<digits>(stripe_start_group) - batch_start_byte;
        const uint64_t stripe_end_byte   = fizzbuzz_group_byte_offset<digits>(stripe_end_group) - batch_start_byte;

        const uint64_t group = stripe_start_group + tid;

        const int stripe_bytes = static_cast<int>(stripe_end_byte - stripe_start_byte);
        int offset             = shared_bytes_per_group * tid;

        const auto print_num = [&](uint64_t num) {
            offset += digits;
            int num_offset = offset - 1;

            #pragma unroll
            for (int j = 0; j < digits; ++j) {
                uint64_t digit;
                barret_divmod<10>(num, digit, num);
                shared_output[num_offset--] = static_cast<uint8_t>(digit) + '0';
            }
        };

        const auto print_fizz = [&] {
            shared_output[offset++] = 'f';
            shared_output[offset++] = 'i';
            shared_output[offset++] = 'z';
            shared_output[offset++] = 'z';
        };

        const auto print_buzz = [&] {
            shared_output[offset++] = 'b';
            shared_output[offset++] = 'u';
            shared_output[offset++] = 'z';
            shared_output[offset++] = 'z';
        };

        const auto print_fizzbuzz = [&] {
            print_fizz();
            print_buzz();
        };

        const auto print_nl = [&] {
            shared_output[offset++] = '\n';
        };

        const uint64_t v = group * lines_per_group;

        // Note: Because the kernels start printing at 10 rather than 0,
        // the group starts at offset 10 in the fizzbuzz pattern and then
        // loops around.
        print_num(v + 11); print_nl();
        print_fizz(); print_nl();
        print_num(v + 13); print_nl();
        print_num(v + 14); print_nl();
        print_fizzbuzz(); print_nl();
        print_num(v + 16); print_nl();
        print_num(v + 17); print_nl();
        print_fizz(); print_nl();
        print_num(v + 19); print_nl();
        print_buzz(); print_nl();
        print_fizz(); print_nl();
        print_num(v + 22); print_nl();
        print_num(v + 23); print_nl();
        print_fizz(); print_nl();
        print_buzz(); print_nl();

        __syncthreads();

        int j = tid;
        for (; j + block_size < stripe_bytes; j += block_size) {
            output[stripe_start_byte + j] = shared_output[j];
        }

        if (j < stripe_bytes) {
            output[stripe_start_byte + j] = shared_output[j];
        }
    }
}

void print_buffer(const uint8_t* buffer, size_t size) {
    #if HIPFIZZBUZZ_VMSPLICE
        iovec iov = {
            const_cast<void*>(static_cast<const void*>(buffer)),
            static_cast<size_t>(size)
        };

        do {
            ssize_t left = vmsplice(1, &iov, 1, 0);
            if (left < 0) {
                std::cerr << "vmsplice() failed: " << strerror(errno) << std::endl;
                exit(1);
            }
            iov.iov_base = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(iov.iov_base) + left);
            iov.iov_len -= left;
        } while (iov.iov_len > 0);
    #else
        ssize_t left = static_cast<ssize_t>(size);
        do {
            ssize_t write_result = write(1, static_cast<const void*>(buffer), left);
            if (write_result < 0) {
                std::cerr << "write() failed: " << strerror(errno) << std::endl;
                exit(1);
            }
            left -= write_result;
        } while (left > 0);
    #endif
}

struct SwapBuffer {
    uint8_t* d_buffer;
    uint8_t* h_buffer;
    hipEvent_t event;
    size_t current_bytes;

    SwapBuffer(uint8_t* d_buffer, uint8_t* h_buffer): d_buffer(d_buffer), h_buffer(h_buffer), current_bytes(0) {
        HIP_CHECK(hipEventCreate(&this->event));
    }

    ~SwapBuffer() {
        HIP_CHECK(hipEventSynchronize(this->event));
        HIP_CHECK(hipEventDestroy(this->event));
    }

    SwapBuffer& operator=(const SwapBuffer&) = delete;
    SwapBuffer&& operator=(SwapBuffer&&) = delete;

    SwapBuffer(const SwapBuffer&) = delete;
    SwapBuffer(SwapBuffer&&) = delete;

    void print() const {
        if (current_bytes > 0) {
            // Wait until the batch is ready
            HIP_CHECK(hipEventSynchronize(this->event));
            #if HIPFIZZBUZZ_COPY_MEM
                print_buffer(this->h_buffer, this->current_bytes);
            #else
                print_buffer(this->d_buffer, this->current_bytes);
            #endif
        }
    }

    template <unsigned digits>
    void launch(hipStream_t stream, uint64_t batch_start_group, uint64_t batch_end_group) {
        #if !HIPFIZZBUZZ_PRINT_SYNC
            this->print();
            print();
        #endif

        const uint64_t batch_groups     = batch_end_group - batch_start_group;
        const uint64_t batch_start_byte = fizzbuzz_group_byte_offset<digits>(batch_start_group);
        const uint64_t batch_end_byte   = fizzbuzz_group_byte_offset<digits>(batch_end_group);
        const uint64_t blocks           = (batch_groups + groups_per_block - 1) / groups_per_block;

        this->current_bytes = batch_end_byte - batch_start_byte;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(fizzbuzz_kernel<digits>),
            dim3(blocks),
            dim3(block_size),
            0,
            stream,
            batch_start_group,
            batch_end_group,
            batch_start_byte,
            this->d_buffer
        );
        HIP_CHECK(hipGetLastError());
        #if HIPFIZZBUZZ_COPY_MEM
            HIP_CHECK(hipMemcpyAsync(this->h_buffer, this->d_buffer, this->current_bytes, hipMemcpyDeviceToHost));
        #endif

        HIP_CHECK(hipEventRecord(this->event, stream));

        #if HIPFIZZBUZZ_PRINT_SYNC
            this->print();
        #endif
    }
};

template <unsigned digits>
void generate(hipStream_t stream, size_t& batch_index, std::array<SwapBuffer, num_swap_buffers>& swap_buffers) {
    const uint64_t digit_start_line = digits_to_base_power[digits];
    const uint64_t digit_end_line = digit_start_line * 10;
    const uint64_t digit_lines = digit_end_line - digit_start_line;

    const uint64_t digit_start_group = digit_start_line / lines_per_group;
    const uint64_t digit_end_group   = digit_end_line / lines_per_group;

    const uint64_t groups = digit_end_group - digit_start_group;
    const uint64_t batches = (groups + groups_per_batch - 1) / groups_per_batch;

    for (size_t batch = 0; batch < batches; ++batch) {
        const uint64_t batch_start_group = digit_start_group + batch * groups_per_batch;
        const uint64_t batch_end_group   = std::min(batch_start_group + groups_per_batch, digit_end_group);

        swap_buffers[batch_index++ % num_swap_buffers].launch<digits>(stream, batch_start_group, batch_end_group);
    }

    if constexpr (digits < max_digits) {
        generate<digits + 1>(stream, batch_index, swap_buffers);
    }
}

template <typename T>
constexpr T align_addr_forward(T addr, size_t align) {
    if (addr % align == 0) {
        return addr;
    }
    return addr + align - addr % align;
}

template <typename T>
T* align_forward(T* ptr, size_t align) {
    return reinterpret_cast<T*>(align_addr_forward(reinterpret_cast<uintptr_t>(ptr), align));
}

int main() {
    std::cerr << "generating " << lines_per_batch << " lines per batch" << std::endl;
    std::cerr << "generating " << bytes_per_batch << " bytes per batch" << std::endl;

    #if HIPFIZZBUZZ_OVERRIDE_PIPESZ
        if (fcntl(1, F_SETPIPE_SZ, pipe_size) < 0) {
            std::cerr << "failed to set pipe size: " << strerror(errno) << std::endl;
            return EXIT_FAILURE;
        }
    #endif

    // Round up allocations to a multiple of the huge page size
    constexpr size_t huge_page_size = 2 * 1024 * 1024;
    constexpr size_t swap_buffer_size = align_addr_forward(bytes_per_batch, huge_page_size);
    const size_t total_buffer_size = swap_buffer_size * num_swap_buffers;
    // Allocate twice the amount so we are guaranteed that we can find
    // a suitably aligned offset in the buffers.
    const size_t alloc_size = 2 * total_buffer_size;

    uint8_t* d_buffer;
    uint8_t* h_buffer;

    #if HIPFIZZBUZZ_COPY_MEM
        HIP_CHECK(hipMalloc(&d_buffer, alloc_size));
    #else
        HIP_CHECK(hipMallocManaged(&d_buffer, alloc_size));
    #endif
    HIP_CHECK(hipHostMalloc(&h_buffer, alloc_size, 0));

    // Align to huge page size...
    d_buffer = align_forward(d_buffer, huge_page_size);
    h_buffer = align_forward(h_buffer, huge_page_size);

    #if !HIPFIZZBUZZ_COPY_MEM
        if (madvise(d_buffer, total_buffer_size, MADV_HUGEPAGE) < 0) {
            std::cerr << "failed to madvise(): " << strerror(errno) << std::endl;
        }
    #endif

    if (madvise(h_buffer, total_buffer_size, MADV_HUGEPAGE) < 0) {
        std::cerr << "failed to madvise(): " << strerror(errno) << std::endl;
    }

    auto swap_buffers = [&]<size_t... i>(std::index_sequence<i...>) {
        return std::array{SwapBuffer(d_buffer + i * swap_buffer_size, h_buffer + i * swap_buffer_size)...};
    }(std::make_index_sequence<num_swap_buffers>());

    hipStream_t stream = 0; // Default stream

    // Print the first 10 lines manually. This allows us to process the next lines in groups of 15
    // which all have the same digits.
    print_buffer(reinterpret_cast<const uint8_t*>(initial_group), initial_group_bytes);

    size_t batch_index = 0;
    generate<2>(stream, batch_index, swap_buffers);

    HIP_CHECK(hipDeviceSynchronize());

    return EXIT_SUCCESS;
}
