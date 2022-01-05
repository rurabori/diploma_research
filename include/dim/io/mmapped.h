#ifndef INCLUDE_DIM_IO_MMAPPED
#define INCLUDE_DIM_IO_MMAPPED

// TODO: make more standard/portable.
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstddef>
#include <system_error>
#include <utility>

#include <span>

namespace dim::io {

class mmapped
{
    void* _memory{};
    size_t _size{};

public:
    mmapped(int fd, size_t size, int prototype = PROT_READ, int flags = MAP_SHARED | MAP_FILE | MAP_POPULATE,
            std::ptrdiff_t offset = 0)
      : _memory{::mmap(nullptr, size, prototype, flags, fd, offset)}, _size{size} {
        if (!_memory)
            throw std::system_error{errno, std::generic_category(), "mmap failed"};
    }

    static mmapped from_file(FILE* file) {
        int fd = fileno(file);
        struct stat stat = {};
        ::fstat(fd, &stat);

        return mmapped{fd, static_cast<size_t>(stat.st_size)};
    }

    mmapped(const mmapped&) = delete;
    mmapped(mmapped&& o) noexcept : _memory{std::exchange(o._memory, nullptr)}, _size{std::exchange(o._size, 0)} {}
    mmapped& operator=(const mmapped&) = delete;
    mmapped& operator=(mmapped&& o) noexcept {
        if (this == &o)
            return *this;

        std::swap(_memory, o._memory);
        std::swap(_size, o._size);
        return *this;
    }

    ~mmapped() { release(); }

    void release() {
        if (_memory)
            ::munmap(std::exchange(_memory, nullptr), std::exchange(_size, 0));
    }

    template<typename As>
    auto as() -> std::span<As> {
        return std::span{reinterpret_cast<As*>(_memory), _size / sizeof(As)};
    }
};

} // namespace dim::io

#endif /* INCLUDE_DIM_IO_MMAPPED */
