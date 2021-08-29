#include "download.h"

#include <chrono>
#include <ctime>
#include <memory>
#include <span>
#include <system_error>
#include <vector>

#include <curl/curl.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <archive.h>
#include <archive_entry.h>

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
            constexpr auto prefix = std::string_view{"Content-Length:"};

            auto entry = std::string_view{ptr, size * nmemb};
            if (!entry.starts_with("Content-Length:"))
                return size * nmemb;

            entry.remove_prefix(prefix.size());

            auto& self = *reinterpret_cast<download_ctx*>(userdata);
            if (auto result = scn::scan(entry, "{}", self.to_download); !result)
                spdlog::warn("couldnt parse Content-Length header, reason '{}'", result.error().msg());

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

} // namespace

auto download(const dim_cli::download_t& args) -> int {
    download_ctx ctx{args.url};

    auto archive = archive_t{archive_read_new()};
    archive_read_support_filter_gzip(archive.get());
    archive_read_support_format_tar(archive.get());

    spdlog::stopwatch sw;
    if (archive_read_open(archive.get(), &ctx, nullptr, cust_archive_read_callback, nullptr) != ARCHIVE_OK)
        throw std::runtime_error{fmt::format("couldn't read archive: {}", archive_error_string(archive.get()))};

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
    const auto elapsed = sw.elapsed();

    const auto download_size = to_mib(ctx.to_download);
    const auto decompressed_mib = to_mib(decompressed_size);
    spdlog::info(
      "download took {}s: downloaded:{:.3}MiB, extracted:{:.3}MiB @ {:.3}MiB/s({:.3}MiB/s with decompression)", sw,
      download_size, decompressed_mib, download_size / elapsed.count(), decompressed_mib / elapsed.count());

    return 0;
}
