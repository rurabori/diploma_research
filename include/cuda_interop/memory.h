#pragma once
#include <memory>

#include <cuda.h>
#include <cuda_interop/error.h>
#include <cuda_runtime_api.h>

namespace cui {

struct deleter
{
    template<typename Ty>
    void operator()(Ty* ptr) {
        cuda_try ::cudaFree(ptr);
    }
};

template<typename Ty>
using unique_ptr = std::unique_ptr<Ty, deleter>;

template<typename Ty>
[[nodiscard]] auto alloc(size_t count = 1) {
    Ty* ptr{};

    cuda_try ::cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(Ty));

    return unique_ptr<Ty>{ptr};
}

template<typename DstTy, typename SrcTy>
DstTy* memcpy(DstTy* dst, const SrcTy& src, cudaMemcpyKind kind) {
    cuda_try ::cudaMemcpy(dst, &src, sizeof(SrcTy), kind);
    return reinterpret_cast<DstTy*>(reinterpret_cast<std::byte*>(dst) + sizeof(SrcTy));
}

template<typename DstTy, typename SrcTy>
DstTy* memcpy(DstTy* dst, const SrcTy& src, cudaMemcpyKind kind, CUstream stream) {
    cuda_try ::cudaMemcpyAsync(dst, &src, sizeof(SrcTy), kind, stream);
    return reinterpret_cast<DstTy*>(reinterpret_cast<std::byte*>(dst) + sizeof(SrcTy));
}

template<typename DstTy, typename SrcTy>
DstTy* memcpy(DstTy* dst, SrcTy* src, size_t count, cudaMemcpyKind kind) {
    cuda_try ::cudaMemcpy(dst, src, count * sizeof(SrcTy), kind);
    return reinterpret_cast<DstTy*>(reinterpret_cast<std::byte*>(dst) + count * sizeof(SrcTy));
}

template<typename DstTy, typename SrcTy>
DstTy* memcpy(DstTy* dst, SrcTy* src, size_t count, cudaMemcpyKind kind, CUstream stream) {
    cuda_try ::cudaMemcpyAsync(dst, src, count * sizeof(SrcTy), kind, stream);
    return reinterpret_cast<DstTy*>(reinterpret_cast<std::byte*>(dst) + sizeof(SrcTy));
}

template<typename DstTy, typename ContiguousContainer>
DstTy* memcpy(DstTy* dst, ContiguousContainer&& src, cudaMemcpyKind kind) {
    return memcpy(dst, std::data(src), std::size(src), kind);
}

template<typename Ty>
[[nodiscard]] auto device_create(const Ty& example) {
    auto ptr = alloc<Ty>();
    cui::memcpy(ptr.get(), example, cudaMemcpyHostToDevice);
    return std::move(ptr);
}

template<typename Ty>
[[nodiscard]] auto device_create(const Ty* values, size_t count) {
    auto ptr = alloc<Ty>(count);
    cui::memcpy(ptr.get(), values, count, cudaMemcpyHostToDevice);
    return std::move(ptr);
}

} // namespace cui
