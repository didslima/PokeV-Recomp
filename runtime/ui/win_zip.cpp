#include "win_zip.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unordered_set>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

static std::string to_lower_ascii(std::string s) {
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

static bool is_windows_reserved_name(const std::string &seg) {
    size_t dot = seg.find('.');
    std::string base = to_lower_ascii(dot == std::string::npos ? seg : seg.substr(0, dot));
    if (base == "con" || base == "prn" || base == "aux" || base == "nul") return true;
    if (base.size() == 4 && (base.rfind("com", 0) == 0 || base.rfind("lpt", 0) == 0) &&
        base[3] >= '1' && base[3] <= '9') {
        return true;
    }
    return false;
}

static bool is_forbidden_user_data_rel(const std::string &rel_low) {
    if (rel_low == "larecomp.ini" || rel_low == "data_root.txt" || rel_low == "gamecontrollerdb.txt")
        return true;
    static const char *const k_dirs[] = {
        "recompdata/", "save/", "extracted/", "mods/", "rando/", "cache/", "screenshots/"
    };
    for (const char *d : k_dirs) {
        size_t n = strlen(d);
        if (rel_low.rfind(d, 0) == 0 || rel_low == std::string(d, n - 1)) return true;
    }
    return false;
}

bool updater_validate_zip_entry_path(const std::string &raw_name,
                                     const std::string &strip_top_prefix,
                                     std::string &out_rel_path,
                                     bool &out_is_dir,
                                     std::string &out_error) {
    out_rel_path.clear();
    out_is_dir = false;
    out_error.clear();

    if (raw_name.empty()) {
        out_error = "Empty ZIP entry path.";
        return false;
    }
    for (unsigned char c : raw_name) {
        if (c == 0 || c < 0x20) {
            out_error = "ZIP entry contains control characters or NUL byte.";
            return false;
        }
    }

    std::string norm = raw_name;
    std::replace(norm.begin(), norm.end(), '\\', '/');

    if (norm[0] == '/') {
        out_error = "ZIP entry has absolute path: " + raw_name;
        return false;
    }
    if (norm.size() >= 2 && isalpha((unsigned char)norm[0]) && norm[1] == ':') {
        out_error = "ZIP entry has drive-letter path: " + raw_name;
        return false;
    }

    out_is_dir = (!norm.empty() && norm.back() == '/');

    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < norm.size()) {
        size_t sl = norm.find('/', pos);
        if (sl == std::string::npos) sl = norm.size();
        std::string seg = norm.substr(pos, sl - pos);
        pos = sl + 1;
        if (seg.empty()) {
            // Disallow empty segments like "a//b" (only trailing slash is allowed)
            if (pos <= norm.size()) {
                out_error = "ZIP entry has empty path segment: " + raw_name;
                return false;
            }
            continue;
        }
        if (seg == "." || seg == "..") {
            out_error = "ZIP entry contains path traversal segment: " + raw_name;
            return false;
        }
        if (seg.back() == ' ' || seg.back() == '.') {
            out_error = "ZIP entry has trailing space or dot in segment: " + raw_name;
            return false;
        }
        for (char c : seg) {
            if (c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
                out_error = "ZIP entry contains illegal Windows filename character: " + raw_name;
                return false;
            }
        }
        if (is_windows_reserved_name(seg)) {
            out_error = "ZIP entry uses Windows reserved device name: " + raw_name;
            return false;
        }
        parts.push_back(seg);
    }

    if (parts.empty()) {
        out_error = "ZIP entry resolves to empty path: " + raw_name;
        return false;
    }

    if (!strip_top_prefix.empty()) {
        std::string want = strip_top_prefix;
        while (!want.empty() && want.back() == '/') want.pop_back();
        if (!want.empty()) {
            if (to_lower_ascii(parts[0]) != to_lower_ascii(want)) {
                out_error = "ZIP entry outside expected root '" + want + "': " + raw_name;
                return false;
            }
            parts.erase(parts.begin());
            if (parts.empty()) {
                // Top-level directory entry itself ("PokeV-Recomp/")
                out_is_dir = true;
                out_rel_path.clear();
                return true;
            }
        }
    }

    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out_rel_path += '/';
        out_rel_path += parts[i];
    }

    if (is_forbidden_user_data_rel(to_lower_ascii(out_rel_path))) {
        out_error = "ZIP entry attempts to overwrite protected user data: " + out_rel_path;
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ CRC-32

static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc & 1) ? ((crc >> 1) ^ 0xEDB88320u) : (crc >> 1);
        }
    }
    return ~crc;
}

// ------------------------------------------------------------------ RFC 1951 DEFLATE

namespace {

struct BitStream {
    const uint8_t *src = nullptr;
    size_t len = 0;
    size_t pos = 0;
    uint32_t bitbuf = 0;
    int bitcnt = 0;

    bool ensure(int n) {
        while (bitcnt < n) {
            if (pos >= len) return false;
            bitbuf |= (uint32_t)src[pos++] << bitcnt;
            bitcnt += 8;
        }
        return true;
    }

    bool read_bits(int n, uint32_t &out) {
        if (n == 0) { out = 0; return true; }
        if (!ensure(n)) return false;
        out = bitbuf & ((1u << n) - 1u);
        bitbuf >>= n;
        bitcnt -= n;
        return true;
    }

    void align_byte() {
        int drop = bitcnt & 7;
        bitbuf >>= drop;
        bitcnt -= drop;
    }

    bool read_bytes(uint8_t *dst, size_t n) {
        align_byte();
        while (n > 0 && bitcnt >= 8) {
            *dst++ = (uint8_t)(bitbuf & 0xff);
            bitbuf >>= 8;
            bitcnt -= 8;
            --n;
        }
        if (pos + n > len) return false;
        if (n > 0) {
            memcpy(dst, src + pos, n);
            pos += n;
        }
        return true;
    }
};

struct HuffTable {
    static constexpr int MAX_BITS = 15;
    int count[MAX_BITS + 1] = {};
    int symbol[320] = {};

    bool build(const int *lengths, int num_syms) {
        memset(count, 0, sizeof count);
        for (int i = 0; i < num_syms; ++i) {
            if (lengths[i] < 0 || lengths[i] > MAX_BITS) return false;
            count[lengths[i]]++;
        }
        if (count[0] == num_syms) return true;
        int left = 1;
        for (int len = 1; len <= MAX_BITS; ++len) {
            left <<= 1;
            left -= count[len];
            if (left < 0) return false;
        }
        int offs[MAX_BITS + 1] = {};
        for (int len = 1; len < MAX_BITS; ++len) {
            offs[len + 1] = offs[len] + count[len];
        }
        for (int sym = 0; sym < num_syms; ++sym) {
            if (lengths[sym] != 0) {
                symbol[offs[lengths[sym]]++] = sym;
            }
        }
        return true;
    }

    bool decode(BitStream &bs, int &out_sym) const {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= MAX_BITS; ++len) {
            uint32_t bit = 0;
            if (!bs.read_bits(1, bit)) return false;
            code |= (int)bit;
            int cnt = count[len];
            if (code - cnt < first) {
                out_sym = symbol[index + (code - first)];
                return true;
            }
            index += cnt;
            first += cnt;
            first <<= 1;
            code <<= 1;
        }
        return false;
    }
};

}  // namespace

static constexpr uint64_t k_max_entry_uncomp = 1024ULL * 1024ULL * 1024ULL;       // 1 GiB per entry
static constexpr uint64_t k_max_total_uncomp = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2 GiB aggregate
static constexpr uint64_t k_max_cd_entries = 100000ULL;

static bool bounds_ok(uint64_t off, uint64_t len, uint64_t total) {
    return off <= total && len <= total - off;
}

static bool inflate_deflate(const uint8_t *comp, size_t comp_len,
                            uint64_t expected_out_len,
                            std::vector<uint8_t> &out) {
    out.clear();
    if (expected_out_len > k_max_entry_uncomp) return false;
    out.reserve((size_t)expected_out_len);

    BitStream bs;
    bs.src = comp;
    bs.len = comp_len;

    static const int k_len_base[29] = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
        35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
    };
    static const int k_len_extra[29] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
    };
    static const int k_dist_base[30] = {
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
        257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
        8193, 12289, 16385, 24577
    };
    static const int k_dist_extra[30] = {
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
        7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
    };
    static const int k_cl_order[19] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
    };

    uint32_t bfinal = 0;
    do {
        uint32_t btype = 0;
        if (!bs.read_bits(1, bfinal) || !bs.read_bits(2, btype)) return false;

        if (btype == 0) {
            uint8_t hdr[4] = {};
            if (!bs.read_bytes(hdr, 4)) return false;
            uint16_t len = (uint16_t)(hdr[0] | (hdr[1] << 8));
            uint16_t nlen = (uint16_t)(hdr[2] | (hdr[3] << 8));
            if ((uint16_t)(len ^ 0xffffu) != nlen) return false;
            if (out.size() + len > expected_out_len) return false;
            size_t old_sz = out.size();
            out.resize(old_sz + len);
            if (!bs.read_bytes(out.data() + old_sz, len)) return false;
            continue;
        }

        if (btype == 3) return false;

        HuffTable litlen_tab, dist_tab;
        if (btype == 1) {
            int ll[288], dl[32];
            for (int i = 0; i <= 143; ++i) ll[i] = 8;
            for (int i = 144; i <= 255; ++i) ll[i] = 9;
            for (int i = 256; i <= 279; ++i) ll[i] = 7;
            for (int i = 280; i <= 287; ++i) ll[i] = 8;
            for (int i = 0; i < 32; ++i) dl[i] = 5;
            if (!litlen_tab.build(ll, 288) || !dist_tab.build(dl, 32)) return false;
        } else {
            uint32_t hlit_r = 0, hdist_r = 0, hclen_r = 0;
            if (!bs.read_bits(5, hlit_r) || !bs.read_bits(5, hdist_r) || !bs.read_bits(4, hclen_r))
                return false;
            int hlit = (int)hlit_r + 257;
            int hdist = (int)hdist_r + 1;
            int hclen = (int)hclen_r + 4;
            if (hlit > 286 || hdist > 30) return false;

            int cl_lens[19] = {};
            for (int i = 0; i < hclen; ++i) {
                uint32_t v = 0;
                if (!bs.read_bits(3, v)) return false;
                cl_lens[k_cl_order[i]] = (int)v;
            }
            HuffTable cl_tab;
            if (!cl_tab.build(cl_lens, 19)) return false;

            int lengths[320] = {};
            int idx = 0;
            int total = hlit + hdist;
            while (idx < total) {
                int sym = 0;
                if (!cl_tab.decode(bs, sym)) return false;
                if (sym < 16) {
                    lengths[idx++] = sym;
                } else {
                    int rep = 0;
                    int val = 0;
                    uint32_t extra = 0;
                    if (sym == 16) {
                        if (idx == 0 || !bs.read_bits(2, extra)) return false;
                        val = lengths[idx - 1];
                        rep = 3 + (int)extra;
                    } else if (sym == 17) {
                        if (!bs.read_bits(3, extra)) return false;
                        rep = 3 + (int)extra;
                    } else {
                        if (!bs.read_bits(7, extra)) return false;
                        rep = 11 + (int)extra;
                    }
                    if (idx + rep > total) return false;
                    for (int k = 0; k < rep; ++k) lengths[idx++] = val;
                }
            }
            if (lengths[256] == 0) return false;
            if (!litlen_tab.build(lengths, hlit) || !dist_tab.build(lengths + hlit, hdist))
                return false;
        }

        for (;;) {
            int sym = 0;
            if (!litlen_tab.decode(bs, sym)) return false;
            if (sym < 256) {
                if (out.size() >= expected_out_len) return false;
                out.push_back((uint8_t)sym);
            } else if (sym == 256) {
                break;
            } else {
                int lidx = sym - 257;
                if (lidx < 0 || lidx >= 29) return false;
                uint32_t lextra = 0;
                if (!bs.read_bits(k_len_extra[lidx], lextra)) return false;
                size_t match_len = (size_t)(k_len_base[lidx] + (int)lextra);

                int dsym = 0;
                if (!dist_tab.decode(bs, dsym) || dsym < 0 || dsym >= 30) return false;
                uint32_t dextra = 0;
                if (!bs.read_bits(k_dist_extra[dsym], dextra)) return false;
                size_t dist = (size_t)(k_dist_base[dsym] + (int)dextra);
                if (dist == 0 || dist > out.size()) return false;
                if (out.size() + match_len > expected_out_len) return false;
                for (size_t k = 0; k < match_len; ++k) {
                    out.push_back(out[out.size() - dist]);
                }
            }
        }
    } while (!bfinal);

    return out.size() == expected_out_len;
}

// ------------------------------------------------------------------ Little-endian helpers

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

#ifdef _WIN32
static std::wstring utf8_to_wide_long(const std::string &u8) {
    if (u8.empty() || u8.size() > (size_t)INT_MAX) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, u8.data(), (int)u8.size(), nullptr, 0);
    if (n <= 0) return L"";
    std::wstring w((size_t)n, L'\0');
    int written = MultiByteToWideChar(CP_UTF8, 0, u8.data(), (int)u8.size(), w.data(), n);
    if (written <= 0) return L"";
    w.resize((size_t)written);
    std::replace(w.begin(), w.end(), L'/', L'\\');
    if (w.size() >= 240 && w.rfind(L"\\\\?\\", 0) != 0) {
        if (w.rfind(L"\\\\", 0) == 0) w = L"\\\\?\\UNC\\" + w.substr(2);
        else if (w.size() >= 2 && w[1] == L':') w = L"\\\\?\\" + w;
    }
    return w;
}

static bool mkdirs_win(const std::string &dir_u8) {
    std::error_code ec;
    fs::create_directories(fs::path(utf8_to_wide_long(dir_u8)), ec);
    return !ec;
}

static bool write_file_win(const std::string &path_u8, const uint8_t *data, size_t len) {
    std::wstring wp = utf8_to_wide_long(path_u8);
    HANDLE h = CreateFileW(wp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    size_t off = 0;
    bool ok = true;
    while (off < len) {
        DWORD chunk = (DWORD)std::min<size_t>(len - off, 1u << 20);
        DWORD written = 0;
        if (!WriteFile(h, data + off, chunk, &written, nullptr) || written != chunk) {
            ok = false;
            break;
        }
        off += written;
    }
    CloseHandle(h);
    return ok;
}
#endif

static fs::path make_fs_path(const std::string &u8) {
#ifdef _WIN32
    return fs::path(utf8_to_wide_long(u8));
#else
    return fs::path(u8);
#endif
}

static FILE *open_zip_file_rb(const std::string &u8) {
#ifdef _WIN32
    return _wfopen(utf8_to_wide_long(u8).c_str(), L"rb");
#else
    return fopen(u8.c_str(), "rb");
#endif
}

static bool seek_zip_u64(FILE *f, uint64_t off) {
#ifdef _WIN32
    if (off > (uint64_t)_I64_MAX) return false;
    return _fseeki64(f, (__int64)off, SEEK_SET) == 0;
#else
    if (off > (uint64_t)INT64_MAX) return false;
    return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
}

WinZipExtractResult updater_extract_zip_safe(const std::string &zip_path,
                                             const std::string &dest_dir,
                                             const std::string &strip_top_prefix) {
    WinZipExtractResult res;
    std::error_code ec;

    fs::path dest_p = make_fs_path(dest_dir);
    if (fs::is_symlink(dest_p, ec)) {
        res.error = "Destination staging directory cannot be a symbolic link.";
        return res;
    }
    fs::create_directories(dest_p, ec);
    if (ec) {
        res.error = "Failed to create staging extraction directory.";
        return res;
    }
    fs::path canonical_dest = fs::weakly_canonical(dest_p, ec);
    if (ec || canonical_dest.empty()) {
        res.error = "Failed to resolve staging extraction directory.";
        return res;
    }

    FILE *f = open_zip_file_rb(zip_path);
    if (!f) {
        res.error = "Unable to open ZIP archive.";
        return res;
    }
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) {
        fclose(f);
        res.error = "Unable to seek ZIP archive.";
        return res;
    }
    __int64 fsz_i64 = _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) {
        fclose(f);
        res.error = "Unable to seek ZIP archive.";
        return res;
    }
    off_t fsz_i64 = ftello(f);
#endif
    if (fsz_i64 < 22) {
        fclose(f);
        res.error = "ZIP archive is truncated or invalid.";
        return res;
    }
    uint64_t file_size = (uint64_t)fsz_i64;
    rewind(f);

    // Find EOCD (End of Central Directory) within last 65557 bytes
    size_t tail_len = (size_t)std::min<uint64_t>(file_size, 65557);
    std::vector<uint8_t> tail(tail_len);
    if (!seek_zip_u64(f, file_size - tail_len) ||
        fread(tail.data(), 1, tail_len, f) != tail_len) {
        fclose(f);
        res.error = "Failed to read ZIP End of Central Directory.";
        return res;
    }

    size_t eocd_off = SIZE_MAX;
    for (size_t i = tail_len - 22 + 1; i-- > 0;) {
        if (rd32(tail.data() + i) == 0x06054b50u) {
            uint16_t clen = rd16(tail.data() + i + 20);
            if (i + 22 + clen == tail_len) {
                eocd_off = i;
                break;
            }
        }
    }
    if (eocd_off == SIZE_MAX) {
        fclose(f);
        res.error = "Missing ZIP End of Central Directory signature.";
        return res;
    }

    uint16_t disk_no = rd16(tail.data() + eocd_off + 4);
    uint16_t cd_disk = rd16(tail.data() + eocd_off + 6);
    if (disk_no != 0 || cd_disk != 0) {
        fclose(f);
        res.error = "Multi-disk ZIP archives are not supported.";
        return res;
    }

    uint64_t entry_count = rd16(tail.data() + eocd_off + 10);
    uint64_t cd_size = rd32(tail.data() + eocd_off + 12);
    uint64_t cd_offset = rd32(tail.data() + eocd_off + 16);

    // Check ZIP64 EOCD locator (20 bytes immediately before EOCD)
    if (eocd_off >= 20 && rd32(tail.data() + eocd_off - 20) == 0x07064b50u) {
        uint64_t z64_off = rd64(tail.data() + eocd_off - 12);
        if (!bounds_ok(z64_off, 56, file_size)) {
            fclose(f);
            res.error = "ZIP64 End of Central Directory offset is out of bounds.";
            return res;
        }
        uint8_t z64[56];
        if (!seek_zip_u64(f, z64_off) || fread(z64, 1, 56, f) != 56 ||
            rd32(z64) != 0x06064b50u) {
            fclose(f);
            res.error = "Corrupt ZIP64 End of Central Directory.";
            return res;
        }
        entry_count = rd64(z64 + 32);
        cd_size = rd64(z64 + 40);
        cd_offset = rd64(z64 + 48);
    }

    if (entry_count > k_max_cd_entries || !bounds_ok(cd_offset, cd_size, file_size) ||
        cd_size > 64ULL * 1024ULL * 1024ULL) {
        fclose(f);
        res.error = "ZIP Central Directory bounds are invalid.";
        return res;
    }

    std::vector<uint8_t> cd((size_t)cd_size);
    if (!seek_zip_u64(f, cd_offset) ||
        fread(cd.data(), 1, (size_t)cd_size, f) != (size_t)cd_size) {
        fclose(f);
        res.error = "Failed to read ZIP Central Directory.";
        return res;
    }

    std::unordered_set<std::string> seen_lower;
    size_t cpos = 0;
    uint64_t parsed_entries = 0;
    uint64_t total_uncomp_bytes = 0;
    std::vector<uint8_t> comp_buf, uncomp_buf;

    auto abort_cleanup = [&](const std::string &err) {
        fclose(f);
        std::error_code rm_ec;
        fs::remove_all(dest_p, rm_ec);
        res.ok = false;
        res.files.clear();
        res.error = err;
        return res;
    };

    while (cpos < cd.size()) {
        if (cpos + 46 > cd.size() || rd32(cd.data() + cpos) != 0x02014b50u) {
            return abort_cleanup("Corrupt ZIP Central Directory header.");
        }
        const uint8_t *ch = cd.data() + cpos;
        uint16_t ver_made = rd16(ch + 4);
        uint16_t flags = rd16(ch + 8);
        uint16_t method = rd16(ch + 10);
        uint32_t crc_expected = rd32(ch + 16);
        uint64_t comp_size = rd32(ch + 20);
        uint64_t uncomp_size = rd32(ch + 24);
        uint16_t name_len = rd16(ch + 28);
        uint16_t extra_len = rd16(ch + 30);
        uint16_t comment_len = rd16(ch + 32);
        uint32_t ext_attr = rd32(ch + 38);
        uint64_t lfh_offset = rd32(ch + 42);

        size_t next_cpos = cpos + 46 + name_len + extra_len + comment_len;
        if (next_cpos > cd.size()) {
            return abort_cleanup("Truncated ZIP Central Directory entry.");
        }
        std::string raw_name((const char *)(ch + 46), name_len);

        // Parse ZIP64 extra field if needed
        if (comp_size == 0xffffffffu || uncomp_size == 0xffffffffu || lfh_offset == 0xffffffffu) {
            const uint8_t *ex = ch + 46 + name_len;
            size_t ex_pos = 0;
            while (ex_pos + 4 <= extra_len) {
                uint16_t tag = rd16(ex + ex_pos);
                uint16_t esz = rd16(ex + ex_pos + 2);
                ex_pos += 4;
                if (ex_pos + esz > extra_len) break;
                if (tag == 0x0001) {
                    size_t zpos = 0;
                    if (uncomp_size == 0xffffffffu && zpos + 8 <= esz) {
                        uncomp_size = rd64(ex + ex_pos + zpos); zpos += 8;
                    }
                    if (comp_size == 0xffffffffu && zpos + 8 <= esz) {
                        comp_size = rd64(ex + ex_pos + zpos); zpos += 8;
                    }
                    if (lfh_offset == 0xffffffffu && zpos + 8 <= esz) {
                        lfh_offset = rd64(ex + ex_pos + zpos); zpos += 8;
                    }
                    break;
                }
                ex_pos += esz;
            }
        }
        cpos = next_cpos;
        ++parsed_entries;

        if ((flags & 1u) != 0) {
            return abort_cleanup("Encrypted ZIP entries are not supported: " + raw_name);
        }

        // Reject Unix symlinks / special files and Windows reparse points
        uint8_t creator = (uint8_t)(ver_made >> 8);
        if (creator == 3 || creator == 19) {
            uint32_t ftype = (ext_attr >> 16) & 0170000u;
            if (ftype == 0120000u || ftype == 0060000u || ftype == 0020000u ||
                ftype == 0010000u || ftype == 0140000u) {
                return abort_cleanup("ZIP entry is a symbolic link or special file: " + raw_name);
            }
        }
        if ((ext_attr & 0x0400u) != 0) {
            return abort_cleanup("ZIP entry has Windows reparse point attribute: " + raw_name);
        }

        std::string rel_path, v_err;
        bool is_dir = false;
        if (!updater_validate_zip_entry_path(raw_name, strip_top_prefix, rel_path, is_dir, v_err)) {
            return abort_cleanup(v_err);
        }

        if (rel_path.empty()) {
            // Top-level directory entry itself ("PokeV-Recomp/")
            continue;
        }

        std::string low_key = to_lower_ascii(rel_path);
        if (!seen_lower.insert(low_key).second) {
            return abort_cleanup("Duplicate ZIP entry (case-insensitive collision): " + rel_path);
        }

        fs::path out_p = canonical_dest / fs::u8path(rel_path);
        fs::path norm_out = out_p.lexically_normal();
        auto dest_str = canonical_dest.generic_u8string();
        auto out_str = norm_out.generic_u8string();
        if (out_str.size() <= dest_str.size() ||
            out_str.rfind(dest_str, 0) != 0 ||
            out_str[dest_str.size()] != '/') {
            return abort_cleanup("Resolved ZIP path escapes destination directory: " + rel_path);
        }

        if (is_dir) {
#ifdef _WIN32
            if (!mkdirs_win(norm_out.u8string())) {
#else
            fs::create_directories(norm_out, ec);
            if (ec) {
#endif
                return abort_cleanup("Failed to create directory: " + rel_path);
            }
            continue;
        }

        // Read Local File Header with overflow-safe bounds checks
        if (!bounds_ok(lfh_offset, 30, file_size)) {
            return abort_cleanup("Invalid Local File Header offset for: " + rel_path);
        }
        uint8_t lfh[30];
        if (!seek_zip_u64(f, lfh_offset) || fread(lfh, 1, 30, f) != 30 ||
            rd32(lfh) != 0x04034b50u) {
            return abort_cleanup("Corrupt Local File Header for: " + rel_path);
        }
        uint16_t lfh_nlen = rd16(lfh + 26);
        uint16_t lfh_elen = rd16(lfh + 28);
        uint64_t hdr_total = 30ULL + (uint64_t)lfh_nlen + (uint64_t)lfh_elen;
        if (!bounds_ok(lfh_offset, hdr_total, file_size)) {
            return abort_cleanup("ZIP Local File Header exceeds archive bounds: " + rel_path);
        }
        uint64_t data_off = lfh_offset + hdr_total;
        if (!bounds_ok(data_off, comp_size, file_size) || comp_size > k_max_entry_uncomp) {
            return abort_cleanup("ZIP entry payload exceeds archive bounds: " + rel_path);
        }
        if (uncomp_size > k_max_entry_uncomp ||
            uncomp_size > k_max_total_uncomp - total_uncomp_bytes) {
            return abort_cleanup("ZIP uncompressed size exceeds safety limit: " + rel_path);
        }
        total_uncomp_bytes += uncomp_size;

        if (method != 0 && method != 8) {
            return abort_cleanup("Unsupported ZIP compression method for: " + rel_path);
        }

        comp_buf.resize((size_t)comp_size);
        if (comp_size > 0) {
            if (!seek_zip_u64(f, data_off) ||
                fread(comp_buf.data(), 1, (size_t)comp_size, f) != (size_t)comp_size) {
                return abort_cleanup("Failed to read compressed payload for: " + rel_path);
            }
        }

        const uint8_t *payload = comp_buf.data();
        size_t payload_len = (size_t)comp_size;
        if (method == 0) {
            if (comp_size != uncomp_size) {
                return abort_cleanup("Stored ZIP entry size mismatch for: " + rel_path);
            }
        } else {
            if (!inflate_deflate(comp_buf.data(), (size_t)comp_size, uncomp_size, uncomp_buf)) {
                return abort_cleanup("Failed to decompress DEFLATE entry (corrupt ZIP): " + rel_path);
            }
            payload = uncomp_buf.data();
            payload_len = uncomp_buf.size();
        }

        uint32_t actual_crc = crc32_update(0, payload, payload_len);
        if (actual_crc != crc_expected) {
            return abort_cleanup("CRC-32 checksum mismatch for ZIP entry: " + rel_path);
        }

        fs::path parent = norm_out.parent_path();
        if (fs::is_symlink(parent, ec)) {
            return abort_cleanup("Staging parent path is a symbolic link: " + rel_path);
        }
#ifdef _WIN32
        if (!mkdirs_win(parent.u8string())) {
            return abort_cleanup("Failed to create parent directory for: " + rel_path);
        }
        if (!write_file_win(norm_out.u8string(), payload, payload_len)) {
            return abort_cleanup("Failed to write extracted file: " + rel_path);
        }
#else
        fs::create_directories(parent, ec);
        if (ec) {
            return abort_cleanup("Failed to create parent directory for: " + rel_path);
        }
        FILE *out_f = fopen(norm_out.c_str(), "wb");
        if (!out_f) {
            return abort_cleanup("Failed to open extracted file for writing: " + rel_path);
        }
        if (payload_len > 0 && fwrite(payload, 1, payload_len, out_f) != payload_len) {
            fclose(out_f);
            return abort_cleanup("Failed to write extracted file (disk full?): " + rel_path);
        }
        if (fclose(out_f) != 0) {
            return abort_cleanup("Failed to flush extracted file: " + rel_path);
        }
#endif

        WinZipEntryInfo info;
        info.rel_path = rel_path;
        info.size = payload_len;
        info.crc32 = actual_crc;
        info.is_dir = false;
        res.files.push_back(info);
    }

    fclose(f);
    if (parsed_entries != entry_count) {
        std::error_code rm_ec;
        fs::remove_all(dest_p, rm_ec);
        res.error = "ZIP Central Directory entry count mismatch.";
        return res;
    }

    res.ok = true;
    return res;
}
