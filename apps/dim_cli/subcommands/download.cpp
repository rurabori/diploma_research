#include "download.h"
#include "arguments.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <span>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <curl/curl.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <archive.h>
#include <archive_entry.h>

#include <zconf.h>
#include <zlib.h>

#include <scn/scn.h>

#include <dim/io/file.h>

namespace {

struct curl_deleter_t
{
    void operator()(CURL* ptr) { ::curl_easy_cleanup(ptr); }
};
using curl_t = std::unique_ptr<CURL, curl_deleter_t>;

struct curl_multi_deleter_t
{
    void operator()(CURLM* ptr) { ::curl_multi_cleanup(ptr); }
};
using curlm_t = std::unique_ptr<CURLM, curl_multi_deleter_t>;

struct archive_deleter_t
{
    void operator()(archive* a) { ::archive_read_free(a); }
};
using archive_t = std::unique_ptr<archive, archive_deleter_t>;

auto to_mib(auto&& value) { return static_cast<double>(value) / 1'048'576; }

auto get_percentage(auto numerator, auto denominator) {
    return (static_cast<double>(numerator) / static_cast<double>(denominator)) * 100;
}

struct download_ctx
{
    using clock_t = std::chrono::steady_clock;

    std::vector<std::byte> buffer;
    size_t downloaded{0};
    size_t to_download{0};
    curl_t curl{::curl_easy_init()};
    curlm_t curlm{::curl_multi_init()};
    clock_t::time_point last_status{clock_t::now()};

    auto try_parse_content_length(std::string_view header_entry, std::string_view prefix) -> bool {
        if (!header_entry.starts_with(prefix))
            return false;

        header_entry.remove_prefix(prefix.size());

        if (auto result = scn::scan(header_entry, "{}", to_download); !result)
            spdlog::warn("couldnt parse Content-Length header, reason '{}'", result.error().msg());

        return true;
    }

    explicit download_ctx(const std::string& url) {
        ::curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());

        constexpr auto data_callback = [](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
            auto data = std::as_writable_bytes(std::span{ptr, size * nmemb});
            auto& self = *reinterpret_cast<download_ctx*>(userdata);
            try {
                self.buffer.insert(self.buffer.end(), data.begin(), data.end());
            } catch (...) {
                return 0;
            }

            return data.size();
        };
        ::curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, +data_callback);
        ::curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, this);

        ::curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);

        constexpr auto header_callback = [](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
            auto& self = *reinterpret_cast<download_ctx*>(userdata);

            auto entry = std::string_view{ptr, size * nmemb};

            // prefixes in {HTTP, FTP}.
            for (const auto& possible_prefix : {"Content-Length:", "213 "}) {
                if (self.try_parse_content_length(entry, possible_prefix))
                    break;
            }

            return size * nmemb;
        };
        ::curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, +header_callback);
        ::curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, this);

        ::curl_multi_add_handle(curlm.get(), curl.get());
    }

    [[nodiscard]] auto verify_curl_ok() const noexcept -> bool {
        int queue_size{0};
        auto* item = ::curl_multi_info_read(curlm.get(), &queue_size);
        if (item && (item->msg = CURLMSG_DONE)) {
            auto* handle = item->easy_handle;

            // NOLINTNEXTLINE - curl guarantees the member is active if msg is CURLMSG_DONE
            if (const auto errc = item->data.result; errc != CURLE_OK) {
                spdlog::error("curl failed with code {}", errc);
                return false;
            }

            long code{}; // NOLINT - signature required by libcurl
            ::curl_easy_getinfo(handle, CURLINFO_HTTP_CODE, &code);

            if (code < 200 || code > 299) {
                spdlog::error("request failed with http code {}", code);
                return false;
            }

            spdlog::trace("transfer ended with http code {}", code);
        }

        return true;
    }

    auto get_more_data() noexcept -> std::optional<std::span<std::byte>> {
        // clear out the buffer, the previous part was already handled.
        buffer.clear();

        int active{1};
        while (buffer.empty() && active != 0) {
            auto result = ::curl_multi_perform(curlm.get(), &active);
            if (result != CURLM_OK) {
                spdlog::error("curl_multi failed with error code {}", result);
                return std::nullopt;
            }
        }

        downloaded += buffer.size();

        constexpr auto interval = std::chrono::seconds{5};
        if (const auto now = clock_t::now(); now - last_status > interval) {
            spdlog::info("downloaded {:.3}MiB ({:.3}%)", to_mib(downloaded), get_percentage(downloaded, to_download));
            last_status = now;
        }

        if (active == 0 && !verify_curl_ok())
            return std::nullopt;

        return std::span{buffer};
    }
};

void log_download_stats(size_t downloaded_bytes, size_t written_bytes, std::chrono::duration<double> elapsed) {
    spdlog::info(
      "download took {}s: downloaded:{:.3}MiB, extracted:{:.3}MiB @ {:.3}MiB/s({:.3}MiB/s with decompression)",
      elapsed.count(), to_mib(downloaded_bytes), to_mib(written_bytes), to_mib(downloaded_bytes) / elapsed.count(),
      to_mib(written_bytes) / elapsed.count());
}

auto archive_extract_file(struct archive* archive, const std::filesystem::path& destination) -> size_t {
    std::filesystem::create_directories(destination.parent_path());

    auto output_file = dim::io::open(destination, "wb");

    size_t total_written{0};
    while (true) {
        const void* buffer{};
        size_t size{};
        int64_t pos{};
        auto read_result = archive_read_data_block(archive, &buffer, &size, &pos);

        if (read_result == ARCHIVE_EOF)
            break;

        if (read_result < ARCHIVE_OK)
            throw std::runtime_error{fmt::format("couldn't decompress because {}", archive_error_string(archive))};

        if (const auto written = ::fwrite(buffer, sizeof(std::byte), size, output_file.get());
            written != size && ferror(output_file.get()))
            throw std::system_error{errno, std::system_category(), "couldn't write file"};

        total_written += size;
    }

    return total_written;
}

auto cust_archive_read_callback(struct archive* /*archive*/, void* client_data, const void** buff) -> ssize_t {
    auto* ctx = reinterpret_cast<download_ctx*>(client_data);

    auto result = ctx->get_more_data();
    if (!result)
        return ARCHIVE_FAILED;

    *buff = result->data();
    return static_cast<ssize_t>(result->size());
}

auto download_archive(const dim_cli::download_t& args) {
    download_ctx ctx{args.url};

    auto archive = archive_t{archive_read_new()};
    archive_read_support_filter_gzip(archive.get());
    archive_read_support_format_tar(archive.get());

    spdlog::stopwatch sw;
    if (archive_read_open(archive.get(), &ctx, nullptr, cust_archive_read_callback, nullptr) != ARCHIVE_OK) {
        // EILSEQ is returned when archive format is not recognized, fallback to gzip for now.
        if (auto err = archive_errno(archive.get()) != EILSEQ)
            throw std::runtime_error{fmt::format("couldn't read archive: {}", archive_error_string(archive.get()))};
    }

    size_t decompressed_size{0};
    while (true) {
        archive_entry* header{};
        const auto arch_result = archive_read_next_header(archive.get(), &header);
        if (arch_result == ARCHIVE_EOF)
            break;

        if (arch_result != ARCHIVE_OK)
            throw std::runtime_error{
              fmt::format("couldn't read archive header: {}", archive_error_string(archive.get()))};

        const auto* name = archive_entry_pathname(header);
        if (std::string_view{name}.ends_with('/'))
            continue;
        const auto output = *args.destination_dir / name;
        spdlog::info("decompressing archive entry {} into {}", name, output.string());

        decompressed_size = archive_extract_file(archive.get(), output);
    }

    log_download_stats(ctx.downloaded, decompressed_size, sw.elapsed());
}

enum class gzip_flags : uint8_t
{
    crc = 0x02,
    ext_data = 0x04,
    orig_filename = 0x08,
    comment = 0x10,
    reserved = 0xe0
};

auto has_flags_set(std::byte value, gzip_flags flags) -> bool {
    return (value & static_cast<std::byte>(flags)) != std::byte{0};
}

struct gzip_header
{
    std::optional<std::string_view> orig_filename;
    std::span<std::byte> data;

    static auto parse(std::span<std::byte> data) -> gzip_header {
        constexpr auto gzip_magic = std::array{std::byte{0x1f}, std::byte{0x8b}};
        constexpr auto compression_deflate = std::byte{0x08};

        const auto get_null_terminated = [&] {
            auto it = std::find(data.begin(), data.end(), std::byte{0});
            if (it == data.end())
                throw std::runtime_error{"couldn't find null terminator"};

            const auto length = static_cast<size_t>(it - data.begin());
            auto retval = std::string_view{reinterpret_cast<const char*>(data.data()), length};

            data = data.subspan(length + 1);
            return retval;
        };

        gzip_header retval{};

        if (const auto start = data.first(2); !std::equal(start.begin(), start.end(), gzip_magic.begin()))
            throw std::runtime_error{"gzip magic not found"};

        data = data.subspan(2);

        if (data[0] != compression_deflate)
            throw std::runtime_error{fmt::format("only deflate allowed (0x08), got {}", data[0])};
        data = data.subspan(1);

        const auto header_flags = data[0];
        // 1 byte header flags + 4 byte timestamp + 1 byte compression flags + 1 byte OS ID.
        data = data.subspan(1 + 4 + 1 + 1);

        if (has_flags_set(header_flags, gzip_flags::reserved))
            throw std::runtime_error{"reserved flags set"};

        if (has_flags_set(header_flags, gzip_flags::ext_data)) {
            const auto to_skip
              = static_cast<size_t>(std::to_integer<uint16_t>(data[0]) | (std::to_integer<uint16_t>(data[1]) << 8));
            data = data.subspan(to_skip + 2);
        }

        if (has_flags_set(header_flags, gzip_flags::orig_filename))
            retval.orig_filename.emplace(get_null_terminated());

        if (has_flags_set(header_flags, gzip_flags::comment))
            (void)get_null_terminated(); // we ignore comments for now.

        if (has_flags_set(header_flags, gzip_flags::crc))
            data = data.subspan(2);

        retval.data = data;

        return retval;
    }
};

auto zlib_input_callback(void* userdata, z_const uint8_t** buffer) -> unsigned {
    auto& ctx = *reinterpret_cast<download_ctx*>(userdata);

    const auto maybe_data = ctx.get_more_data();
    if (!maybe_data) {
        *buffer = nullptr;
        return 0;
    }

    *buffer = reinterpret_cast<uint8_t*>(maybe_data->data());
    return static_cast<unsigned>(maybe_data->size());
}

struct zlib_out_data
{
    dim::io::file_t file;
    size_t written{};
};

auto zlib_output_callback(void* userdata, uint8_t* data, unsigned data_size) -> int {
    auto& ctx = *reinterpret_cast<zlib_out_data*>(userdata);

    ctx.written += data_size;
    return std::fwrite(data, 1, data_size, ctx.file.get()) == data_size ? 0 : 1;
};

void download_gzip(const dim_cli::download_t& args) {
    constexpr auto window_bits = 15;

    download_ctx ctx{args.url};

    auto maybe_data = ctx.get_more_data();
    if (!maybe_data)
        return;

    const auto header = gzip_header::parse(*maybe_data);
    const auto filename = *args.destination_dir / (header.orig_filename ? *header.orig_filename : "unknown.mtx");

    z_stream stream{
      .next_in = reinterpret_cast<Bytef*>(header.data.data()),
      .avail_in = static_cast<uInt>(header.data.size()),
    };

    std::byte window[2 << window_bits];
    if (auto err = ::inflateBackInit(&stream, window_bits, reinterpret_cast<uint8_t*>(std::data(window))); err != Z_OK)
        throw std::runtime_error{fmt::format("inflateInit failed: {} ({})", stream.msg, err)};

    spdlog::stopwatch sw;

    auto write_data = zlib_out_data{.file = dim::io::open(filename, "wb")};
    if (auto err = ::inflateBack(&stream, zlib_input_callback, &ctx, zlib_output_callback, &write_data);
        err != Z_STREAM_END && err != Z_OK)
        throw std::runtime_error{fmt::format("zlib inflateBack failed with: {} ({})", stream.msg, err)};

    ::inflateBackEnd(&stream);

    log_download_stats(ctx.downloaded, write_data.written, sw.elapsed());
}

auto download(const dim_cli::download_t& args, download_format format) -> int {
    switch (format) {
        case download_format::archive:
            download_archive(args);
            break;
        case download_format::gzip:
            download_gzip(args);
            break;
        case download_format::detect:
            throw std::logic_error("auto detect download format should never come to this implementation");
    }

    return 0;
}

auto guess_format(std::string_view url) -> download_format {
    // TODO: this needs better heuristic, but for now this will suffice.
    if (url.ends_with("tar.gz"))
        return download_format::archive;

    return download_format::gzip;
}

} // namespace

auto download(const dim_cli::download_t& args) -> int {
    return download(args, *args.format == download_format::detect ? guess_format(args.url) : *args.format);
}
