#include "updater.h"

#include "../cache_sha256.h"

#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <pthread.h>
#include <string>
#include <sys/stat.h>
#include <thread>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

extern "C" {
#include "../platform.h"
}

namespace fs = std::filesystem;

static const char *GITHUB_LATEST_URL =
    "https://api.github.com/repos/didslima/PokeV-Recomp/releases/latest";
static const char *GITHUB_RELEASES_WEB =
    "https://github.com/didslima/PokeV-Recomp/releases";
static const char *LINUX_ASSET_NAME = "PokeV-Recomp-linux-x86_64.AppImage";
static const char *SHA256SUMS_NAME = "SHA256SUMS";

static pthread_mutex_t g_upd_lock = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<bool> g_upd_running{false};
static std::atomic<bool> g_inst_running{false};
static std::atomic<bool> g_cancel_install{false};
static std::atomic<bool> g_shutting_down{false};
static pthread_t g_upd_thread{};
static bool g_thread_active = false;
static pthread_t g_inst_thread{};
static bool g_inst_thread_active = false;
static bool g_atexit_registered = false;
static UpdaterSnapshot g_snap;
static std::string g_data_root;
static bool g_flag_play = false;
static bool g_flag_prepare = false;

static std::string g_staged_path;
static std::string g_backup_path;
static std::string g_lock_path;

long long updater_tag_rank(const std::string &tag) {
    int n = 0;
    char end = 0;
    if (sscanf(tag.c_str(), "beta_%d%c", &n, &end) == 1 && n >= 0)
        return -1000000LL + n;

    const char *p = tag.c_str();
    if (*p == 'v' || *p == 'V') p++;
    int maj = 0, min = 0, pat = 0;
    if (sscanf(p, "%d.%d.%d%c", &maj, &min, &pat, &end) == 3 &&
        maj >= 0 && min >= 0 && min < 1000 && pat >= 0 && pat < 1000) {
        return (long long)maj * 1000000LL + (long long)min * 1000LL + pat;
    }
    return -2000000LL;
}

std::string updater_tag_label(const std::string &tag, bool with_word) {
    int n = 0;
    char end = 0;
    if (sscanf(tag.c_str(), "beta_%d%c", &n, &end) == 1)
        return "Beta " + std::to_string(n);
    std::string v = (tag.size() > 1 && (tag[0] == 'v' || tag[0] == 'V')) ? tag.substr(1) : tag;
    return with_word ? "Version " + v : v;
}

std::string updater_find_curl() {
    if (const char *override_curl = getenv("LARECOMP_CURL_BIN")) {
        if (*override_curl) {
#ifdef _WIN32
            std::error_code ec;
            return fs::exists(fs::u8path(override_curl), ec) ? std::string(override_curl) : std::string();
#else
            return access(override_curl, X_OK) == 0 ? std::string(override_curl) : std::string();
#endif
        }
        return "";
    }
#ifdef _WIN32
    const char *sr = getenv("SystemRoot");
    std::string sys_curl = std::string(sr ? sr : "C:\\Windows") + "\\System32\\curl.exe";
    std::error_code ec;
    if (fs::exists(fs::u8path(sys_curl), ec)) return sys_curl;
    return "";
#else
    if (!getenv("LARECOMP_CURL_NO_DEFAULT")) {
        static const char *const k_candidates[] = {
            "/usr/bin/curl",
            "/usr/local/bin/curl",
            "/opt/homebrew/bin/curl",
        };
        for (const char *c : k_candidates) {
            if (access(c, X_OK) == 0) return c;
        }
    }
    const char *path_env = getenv("PATH");
    if (path_env && *path_env) {
        const char *p = path_env;
        while (*p) {
            const char *colon = strchr(p, ':');
            std::string dir = colon ? std::string(p, (size_t)(colon - p)) : std::string(p);
            if (!dir.empty()) {
                std::string candidate = dir + "/curl";
                if (access(candidate.c_str(), X_OK) == 0) return candidate;
            }
            if (!colon) break;
            p = colon + 1;
        }
    }
    return "";
#endif
}

static size_t skip_ws(const std::string &s, size_t i) {
    while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
    return i;
}

static void append_utf8(std::string &out, unsigned int cp) {
    if (cp <= 0x7F) {
        out.push_back((char)cp);
    } else if (cp <= 0x7FF) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

static bool parse_hex4(const std::string &s, size_t pos, unsigned int &out) {
    if (pos + 4 > s.size()) return false;
    unsigned int cp = 0;
    for (int k = 0; k < 4; k++) {
        char h = s[pos + k];
        cp <<= 4;
        if (h >= '0' && h <= '9') cp |= (unsigned int)(h - '0');
        else if (h >= 'a' && h <= 'f') cp |= (unsigned int)(h - 'a' + 10);
        else if (h >= 'A' && h <= 'F') cp |= (unsigned int)(h - 'A' + 10);
        else return false;
    }
    out = cp;
    return true;
}

static bool parse_json_string(const std::string &s, size_t &pos, std::string &out) {
    out.clear();
    if (pos >= s.size() || s[pos] != '"') return false;
    pos++;
    while (pos < s.size()) {
        char c = s[pos++];
        if (c == '"') return true;
        if (c == '\\') {
            if (pos >= s.size()) return false;
            char esc = s[pos++];
            switch (esc) {
                case '"':  out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/'); break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    unsigned int cp = 0;
                    if (!parse_hex4(s, pos, cp)) return false;
                    pos += 4;
                    if (cp >= 0xD800u && cp <= 0xDBFFu) {
                        unsigned int low = 0;
                        if (pos + 6 <= s.size() && s[pos] == '\\' && s[pos + 1] == 'u' &&
                            parse_hex4(s, pos + 2, low) && low >= 0xDC00u && low <= 0xDFFFu) {
                            pos += 6;
                            unsigned int full = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                            append_utf8(out, full);
                        } else {
                            append_utf8(out, 0xFFFDu);
                        }
                    } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                        append_utf8(out, 0xFFFDu);
                    } else {
                        append_utf8(out, cp);
                    }
                    break;
                }
                default: out.push_back(esc); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

static size_t skip_json_value(const std::string &s, size_t pos) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return pos;
    char c = s[pos];
    if (c == '"') {
        std::string tmp;
        parse_json_string(s, pos, tmp);
        return pos;
    }
    if (c == '{' || c == '[') {
        char open = c, close = (c == '{') ? '}' : ']';
        int depth = 1;
        pos++;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '"') {
                std::string tmp;
                if (!parse_json_string(s, pos, tmp)) return s.size();
                continue;
            }
            if (s[pos] == open) depth++;
            else if (s[pos] == close) depth--;
            pos++;
        }
        return pos;
    }
    while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' && !std::isspace((unsigned char)s[pos]))
        pos++;
    return pos;
}

struct ParsedAsset {
    std::string name;
    std::string url;
    uint64_t size = 0;
};

static bool parse_asset_object(const std::string &s, size_t &pos, ParsedAsset &out) {
    pos = skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '{') return false;
    pos++;
    while (pos < s.size()) {
        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == '}') {
            pos++;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, pos, key)) return false;
        pos = skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        pos = skip_ws(s, pos + 1);

        if (key == "name" && pos < s.size() && s[pos] == '"') {
            if (!parse_json_string(s, pos, out.name)) return false;
        } else if (key == "browser_download_url" && pos < s.size() && s[pos] == '"') {
            if (!parse_json_string(s, pos, out.url)) return false;
        } else if (key == "size" && pos < s.size() && std::isdigit((unsigned char)s[pos])) {
            out.size = strtoull(s.c_str() + pos, NULL, 10);
            pos = skip_json_value(s, pos);
        } else {
            pos = skip_json_value(s, pos);
        }
        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
    }
    return false;
}

struct ParsedObj {
    bool valid = false;
    bool is_draft = false;
    bool is_prerelease = false;
    std::string tag_name;
    std::string html_url;
    std::string body;
    std::string linux_asset_url;
    uint64_t linux_asset_size = 0;
    std::string sha256sums_url;
};

static bool is_https_url(const std::string &u) {
    return u.compare(0, 8, "https://") == 0;
}

static ParsedObj parse_release_object(const std::string &s, size_t &pos) {
    ParsedObj obj;
    pos = skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '{') return obj;
    pos++;
    while (pos < s.size()) {
        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == '}') {
            pos++;
            obj.valid = !obj.tag_name.empty();
            return obj;
        }
        std::string key;
        if (!parse_json_string(s, pos, key)) return ParsedObj{};
        pos = skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return ParsedObj{};
        pos = skip_ws(s, pos + 1);

        if (key == "tag_name" && pos < s.size() && s[pos] == '"') {
            if (!parse_json_string(s, pos, obj.tag_name)) return ParsedObj{};
        } else if (key == "html_url" && pos < s.size() && s[pos] == '"') {
            if (!parse_json_string(s, pos, obj.html_url)) return ParsedObj{};
        } else if (key == "body" && pos < s.size() && s[pos] == '"') {
            if (!parse_json_string(s, pos, obj.body)) return ParsedObj{};
        } else if (key == "draft") {
            if (s.compare(pos, 4, "true") == 0) { obj.is_draft = true; pos += 4; }
            else if (s.compare(pos, 5, "false") == 0) { obj.is_draft = false; pos += 5; }
            else pos = skip_json_value(s, pos);
        } else if (key == "prerelease") {
            if (s.compare(pos, 4, "true") == 0) { obj.is_prerelease = true; pos += 4; }
            else if (s.compare(pos, 5, "false") == 0) { obj.is_prerelease = false; pos += 5; }
            else pos = skip_json_value(s, pos);
        } else if (key == "assets" && pos < s.size() && s[pos] == '[') {
            pos++;
            while (pos < s.size()) {
                pos = skip_ws(s, pos);
                if (pos < s.size() && s[pos] == ']') { pos++; break; }
                if (pos < s.size() && s[pos] == '{') {
                    ParsedAsset a;
                    if (!parse_asset_object(s, pos, a)) return ParsedObj{};
                    if (a.name == LINUX_ASSET_NAME && is_https_url(a.url)) {
                        obj.linux_asset_url = a.url;
                        obj.linux_asset_size = a.size;
                    } else if (a.name == SHA256SUMS_NAME && is_https_url(a.url)) {
                        obj.sha256sums_url = a.url;
                    }
                } else {
                    return ParsedObj{};
                }
                pos = skip_ws(s, pos);
                if (pos < s.size() && s[pos] == ',') pos++;
            }
        } else {
            pos = skip_json_value(s, pos);
        }

        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
    }
    return ParsedObj{};
}

static std::string sanitize_release_url(const std::string &url, const std::string &tag) {
    if (is_https_url(url)) return url;
    if (!tag.empty()) return std::string(GITHUB_RELEASES_WEB) + "/tag/" + tag;
    return GITHUB_RELEASES_WEB;
}

UpdaterParseResult updater_parse_release_json(const std::string &json) {
    UpdaterParseResult res;
    size_t pos = skip_ws(json, 0);
    if (pos >= json.size()) return res;

    auto accept_candidate = [&](const ParsedObj &o) {
        if (!o.valid || o.is_draft || o.is_prerelease) return;
        long long r = updater_tag_rank(o.tag_name);
        if (r <= -2000000LL) return;
        if (!res.found_stable || r > updater_tag_rank(res.tag_name)) {
            res.found_stable = true;
            res.tag_name = o.tag_name;
            res.html_url = sanitize_release_url(o.html_url, o.tag_name);
            res.body = o.body;
            res.linux_asset_url = o.linux_asset_url;
            res.linux_asset_size = o.linux_asset_size;
            res.sha256sums_url = o.sha256sums_url;
        }
    };

    if (json[pos] == '{') {
        ParsedObj o = parse_release_object(json, pos);
        if (!o.valid) return res;
        res.ok = true;
        accept_candidate(o);
        return res;
    }

    if (json[pos] == '[') {
        pos++;
        while (pos < json.size()) {
            pos = skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ']') {
                res.ok = true;
                return res;
            }
            if (pos < json.size() && json[pos] == '{') {
                ParsedObj o = parse_release_object(json, pos);
                if (!o.valid) return res;
                accept_candidate(o);
            } else {
                return res;
            }
            pos = skip_ws(json, pos);
            if (pos < json.size() && json[pos] == ',') pos++;
        }
        return res;
    }

    return res;
}

std::string updater_parse_sha256sums(const std::string &sums_text, const std::string &filename) {
    size_t i = 0;
    while (i < sums_text.size()) {
        size_t eol = sums_text.find('\n', i);
        if (eol == std::string::npos) eol = sums_text.size();
        std::string line = sums_text.substr(i, eol - i);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        i = (eol < sums_text.size()) ? eol + 1 : sums_text.size();

        if (line.size() < 66) continue;
        std::string hex;
        bool valid_hex = true;
        for (size_t k = 0; k < 64; k++) {
            unsigned char ch = (unsigned char)line[k];
            if (!std::isxdigit(ch)) { valid_hex = false; break; }
            hex.push_back((char)std::tolower(ch));
        }
        if (!valid_hex) continue;
        if (line[64] != ' ' && line[64] != '\t') continue;
        size_t name_pos = 65;
        while (name_pos < line.size() && (line[name_pos] == ' ' || line[name_pos] == '\t')) name_pos++;
        if (name_pos < line.size() && (line[name_pos] == '*' || line[name_pos] == ' ')) name_pos++;
        std::string entry_name = line.substr(name_pos);
        // Handle optional "./" prefix
        if (entry_name.compare(0, 2, "./") == 0) entry_name = entry_name.substr(2);
        if (entry_name == filename) return hex;
    }
    return "";
}

std::string updater_file_sha256(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return "";
    CacheSha256 sha;
    unsigned char buf[65536];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
        sha.add(buf, n);
    }
    bool err = ferror(f) != 0;
    fclose(f);
    if (err) return "";
    return sha.finish();
}

UpdaterInstallKind updater_detect_install_kind(std::string &out_target_path, std::string &out_reason) {
    out_target_path.clear();
    out_reason.clear();

    const char *rl = getenv("RECOMPLAUNCHER_DATA");
    const char *ldr = getenv("LARECOMP_DATA_ROOT");
    if ((rl && *rl) || (ldr && *ldr) || g_flag_play || g_flag_prepare) {
        out_reason = "Updates are managed by RecompLauncher.";
        return UPD_INSTALL_RECOMPLAUNCHER;
    }

#ifdef __linux__
    const char *appimage = getenv("APPIMAGE");
    if (!appimage || !*appimage) {
        out_reason = "Not running from an AppImage; please update manually from GitHub.";
        return UPD_INSTALL_UNSUPPORTED;
    }
    char resolved[4096];
    if (!realpath(appimage, resolved)) {
        out_reason = "Unable to resolve AppImage path on disk.";
        return UPD_INSTALL_UNSUPPORTED;
    }
    struct stat st{};
    if (stat(resolved, &st) != 0 || !S_ISREG(st.st_mode)) {
        out_reason = "AppImage target is not a regular file.";
        return UPD_INSTALL_UNSUPPORTED;
    }
    out_target_path = resolved;
    fs::path p = fs::path(resolved).parent_path();
    if (p.empty() || access(p.c_str(), W_OK) != 0) {
        out_reason = "AppImage folder is not writable; move it to a user folder or update manually.";
        return UPD_INSTALL_LINUX_APPIMAGE_RO;
    }
    return UPD_INSTALL_LINUX_APPIMAGE;
#else
    out_reason = "In-app installation is currently supported only for Linux AppImage.";
    return UPD_INSTALL_UNSUPPORTED;
#endif
}

bool updater_apply_staged_file(const std::string &target_path, const std::string &staged_path,
                               const std::string &backup_path, std::string &out_error) {
#ifdef _WIN32
    (void)target_path; (void)staged_path; (void)backup_path;
    out_error = "Not supported on Windows in Phase B.";
    return false;
#else
    struct stat st{};
    mode_t mode = 0755;
    if (stat(target_path.c_str(), &st) == 0) {
        mode = (st.st_mode & 0777) | 0111;
    }
    if (chmod(staged_path.c_str(), mode) != 0) {
        out_error = std::string("Failed to set executable permissions: ") + strerror(errno);
        return false;
    }

    unlink(backup_path.c_str());
    bool have_backup = false;
    if (stat(target_path.c_str(), &st) == 0) {
        if (link(target_path.c_str(), backup_path.c_str()) == 0) {
            have_backup = true;
        } else {
            // Fallback copy if hardlink is disallowed
            FILE *in = fopen(target_path.c_str(), "rb");
            FILE *out = in ? fopen(backup_path.c_str(), "wb") : NULL;
            if (in && out) {
                char buf[65536];
                size_t n;
                bool io_ok = true;
                while ((n = fread(buf, 1, sizeof buf, in)) > 0) {
                    if (fwrite(buf, 1, n, out) != n) { io_ok = false; break; }
                }
                if (ferror(in)) io_ok = false;
                fclose(in);
                fclose(out);
                if (io_ok) {
                    chmod(backup_path.c_str(), mode);
                    have_backup = true;
                } else {
                    unlink(backup_path.c_str());
                }
            } else {
                if (in) fclose(in);
                if (out) fclose(out);
            }
        }
        if (!have_backup) {
            out_error = "Failed to create backup of existing AppImage.";
            return false;
        }
    }

    if (rename(staged_path.c_str(), target_path.c_str()) != 0) {
        int saved_err = errno;
        if (have_backup && access(target_path.c_str(), F_OK) != 0) {
            rename(backup_path.c_str(), target_path.c_str());
        } else if (have_backup) {
            unlink(backup_path.c_str());
        }
        out_error = std::string("Failed to replace AppImage: ") + strerror(saved_err);
        return false;
    }

    if (have_backup) unlink(backup_path.c_str());
    return true;
#endif
}

static std::string read_file_capped(const std::string &path, size_t max_bytes) {
    std::string out;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return out;
    char buf[8192];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0 && out.size() < max_bytes) {
        size_t take = (out.size() + n > max_bytes) ? (max_bytes - out.size()) : n;
        out.append(buf, take);
    }
    fclose(f);
    return out;
}

static void cleanup_on_boot_locked(const std::string &data_root, const std::string &target_path) {
#ifndef _WIN32
    if (!target_path.empty()) {
        fs::path tp(target_path);
        fs::path dir = tp.parent_path();
        std::string bak = target_path + ".bak";
        if (access(target_path.c_str(), F_OK) != 0 && access(bak.c_str(), F_OK) == 0) {
            // Interrupted replacement: restore backup!
            rename(bak.c_str(), target_path.c_str());
        } else if (access(bak.c_str(), F_OK) == 0) {
            unlink(bak.c_str());
        }
        std::string lock = (dir / ".PokeV-Recomp.update.lock").string();
        if (FILE *lf = fopen(lock.c_str(), "rb")) {
            long pid = 0;
            bool stale = (fscanf(lf, "%ld", &pid) == 1 && pid > 0 &&
                          (pid == (long)getpid() || (kill((pid_t)pid, 0) != 0 && errno == ESRCH)));
            fclose(lf);
            if (stale) unlink(lock.c_str());
        }
        std::error_code ec;
        for (const auto &ent : fs::directory_iterator(dir, ec)) {
            std::string name = ent.path().filename().string();
            if (name.rfind(".PokeV-Recomp.", 0) == 0 &&
                (name.size() > 5 && (name.compare(name.size() - 5, 5, ".part") == 0 ||
                                     name == ".PokeV-Recomp.staged"))) {
                fs::remove(ent.path(), ec);
            }
        }
    }
#else
    (void)target_path;
#endif
    if (!data_root.empty()) {
        std::string note = data_root + "/cache/_updated.txt";
        std::string tag = read_file_capped(note, 64);
        while (!tag.empty() && (tag.back() == '\n' || tag.back() == '\r')) tag.pop_back();
        if (!tag.empty()) {
            g_snap.updated_from_tag = tag;
            remove(note.c_str());
        }
    }
}

void updater_configure(const char *current_version, const char *data_root,
                       bool launched_with_play, bool launched_with_prepare) {
    pthread_mutex_lock(&g_upd_lock);
    g_shutting_down = false;
    g_cancel_install = false;
    if (!g_atexit_registered) {
        atexit(updater_shutdown);
        g_atexit_registered = true;
    }
    g_snap.current_version = current_version ? current_version : "";
    g_data_root = data_root ? data_root : "";
    g_flag_play = launched_with_play;
    g_flag_prepare = launched_with_prepare;

    const char *rl = getenv("RECOMPLAUNCHER_DATA");
    const char *ldr = getenv("LARECOMP_DATA_ROOT");
    g_snap.suppressed_by_launcher = (rl && *rl) || (ldr && *ldr) || g_flag_play || g_flag_prepare;

    std::string tgt, reason;
    g_snap.install_kind = updater_detect_install_kind(tgt, reason);
    std::string recovery_tgt = tgt;
#ifdef __linux__
    if (recovery_tgt.empty()) {
        if (const char *ai = getenv("APPIMAGE")) {
            if (*ai) {
                fs::path ap(ai);
                char rdir[4096];
                if (!ap.parent_path().empty() && realpath(ap.parent_path().c_str(), rdir)) {
                    recovery_tgt = (fs::path(rdir) / ap.filename()).string();
                }
            }
        }
    }
#endif
    cleanup_on_boot_locked(g_data_root, recovery_tgt);
    if (tgt.empty() && !recovery_tgt.empty()) {
        // Re-detect in case boot recovery restored the AppImage from .bak
        g_snap.install_kind = updater_detect_install_kind(tgt, reason);
    }
    g_snap.install_target_path = tgt;
    g_snap.install_reason = reason;
    g_snap.install_state = UPD_INST_IDLE;
    g_snap.install_error.clear();
    g_snap.downloaded_bytes = 0;
    g_snap.total_bytes = 0;
    g_staged_path.clear();
    g_backup_path.clear();
    g_lock_path.clear();
    pthread_mutex_unlock(&g_upd_lock);
}

bool updater_is_suppressed_by_launcher() {
    pthread_mutex_lock(&g_upd_lock);
    bool s = g_snap.suppressed_by_launcher;
    pthread_mutex_unlock(&g_upd_lock);
    return s;
}

UpdaterSnapshot updater_get_snapshot() {
    pthread_mutex_lock(&g_upd_lock);
    UpdaterSnapshot copy = g_snap;
    pthread_mutex_unlock(&g_upd_lock);
    return copy;
}

static void set_check_result(UpdaterCheckState st, const std::string &tag,
                             const std::string &url, const std::string &notes,
                             const std::string &linux_url, uint64_t linux_size,
                             const std::string &sums_url, const std::string &err) {
    if (g_shutting_down.load()) return;
    pthread_mutex_lock(&g_upd_lock);
    if (!g_shutting_down.load()) {
        g_snap.state = st;
        g_snap.latest_tag = tag;
        g_snap.html_url = url;
        g_snap.notes = notes;
        g_snap.linux_asset_url = linux_url;
        g_snap.linux_asset_size = linux_size;
        g_snap.sha256sums_url = sums_url;
        g_snap.error = err;
        g_snap.has_checked = true;
    }
    pthread_mutex_unlock(&g_upd_lock);
}

static int parse_http_status_from_log(const std::string &raw) {
    const std::string marker = "__HTTP_CODE__:";
    size_t m = raw.rfind(marker);
    if (m == std::string::npos) return 0;
    int code = atoi(raw.c_str() + m + marker.size());
    return (code >= 100 && code <= 599) ? code : 0;
}

static bool run_curl_to_file(const std::string &curl, const std::string &ua,
                             const std::string &url, const std::string &out_path,
                             const std::string &status_path, int timeout_sec,
                             std::atomic<bool> *cancel_flag, int &out_http_code,
                             bool &out_cancelled, std::string &out_err) {
    out_http_code = 0;
    out_cancelled = false;
    remove(out_path.c_str());
    remove(status_path.c_str());

    if (url.compare(0, 5, "file:") == 0) {
        std::error_code ec;
        fs::copy_file(fs::u8path(url.substr(5)), fs::u8path(out_path),
                      fs::copy_options::overwrite_existing, ec);
        if (ec) {
            out_err = "Failed to copy local file: " + ec.message();
            return false;
        }
        out_http_code = 200;
        return true;
    }
    if (!is_https_url(url)) {
        out_err = "Refusing non-HTTPS URL.";
        return false;
    }

    std::string to_str = std::to_string(timeout_sec);
    const char *av[] = {
        curl.c_str(), "-sL", "--proto", "=https", "--tlsv1.2",
        "--max-time", to_str.c_str(),
        "-H", "Accept: application/vnd.github+json",
        "-A", ua.c_str(),
        "-o", out_path.c_str(),
        "-w", "\n__HTTP_CODE__:%{http_code}\n",
        url.c_str(), NULL
    };

    PlatProc p = plat_spawn(av, status_path.c_str(), false);
    if (!p) {
        out_err = "Failed to launch curl.";
        return false;
    }

    int rc = -1;
    while (!g_shutting_down.load() && (!cancel_flag || !cancel_flag->load())) {
        rc = plat_proc_wait(p, false);
        if (rc >= 0) break;
        if (cancel_flag) {
            struct stat st{};
            if (stat(out_path.c_str(), &st) == 0 && st.st_size >= 0) {
                pthread_mutex_lock(&g_upd_lock);
                g_snap.downloaded_bytes = (uint64_t)st.st_size;
                pthread_mutex_unlock(&g_upd_lock);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (rc < 0) {
        plat_proc_kill(p);
        out_cancelled = true;
        remove(out_path.c_str());
        remove(status_path.c_str());
        return false;
    }

    std::string status_raw = read_file_capped(status_path, 64u << 10);
    remove(status_path.c_str());
    out_http_code = parse_http_status_from_log(status_raw);
    if (rc != 0 || out_http_code <= 0) {
        out_err = "Offline or unable to reach GitHub.";
        return false;
    }
    return true;
}

static void *check_thread(void *) {
    pthread_mutex_lock(&g_upd_lock);
    std::string cur_ver = g_snap.current_version;
    std::string data_root = g_data_root;
    pthread_mutex_unlock(&g_upd_lock);

    std::string body;
    int http_code = 200;
    bool network_ok = true;
    std::string err_detail;

    if (const char *sim_code = getenv("LARECOMP_UPDATE_HTTP_STATUS")) {
        http_code = atoi(sim_code);
        if (http_code <= 0) network_ok = false;
    }

    if (const char *mock = getenv("LARECOMP_UPDATE_JSON")) {
        if (mock[0] == '@') body = read_file_capped(mock + 1, 2u << 20);
        else body = mock;
    } else if (network_ok && http_code == 200) {
        std::string curl = updater_find_curl();
        if (curl.empty()) {
            network_ok = false;
            http_code = 0;
            err_detail = "curl executable not found.";
        } else {
            std::string cache_dir = data_root.empty() ? "cache" : (data_root + "/cache");
            std::error_code ec;
            fs::create_directories(fs::u8path(cache_dir), ec);
            std::string body_path = cache_dir + "/release_latest.json";
            std::string status_path = cache_dir + "/release_latest.status";
            std::string ua = "PokeV-Recomp/" + (cur_ver.empty() ? std::string("unknown") : cur_ver);

            bool cancelled = false;
            if (!run_curl_to_file(curl, ua, GITHUB_LATEST_URL, body_path, status_path, 15,
                                  NULL, http_code, cancelled, err_detail)) {
                network_ok = false;
            } else {
                body = read_file_capped(body_path, 2u << 20);
            }
            remove(body_path.c_str());
            remove(status_path.c_str());
        }
    }

    if (g_shutting_down.load()) {
        g_upd_running = false;
        return NULL;
    }

    if (!network_ok) {
        set_check_result(UPD_CHK_OFFLINE, "", "", "", "", 0, "",
                         err_detail.empty() ? "Offline or unable to reach GitHub." : err_detail);
        g_upd_running = false;
        return NULL;
    }
    if (http_code == 403 || http_code == 429) {
        set_check_result(UPD_CHK_RATE_LIMITED, "", GITHUB_RELEASES_WEB, "", "", 0, "",
                         "GitHub API rate limit reached (HTTP " + std::to_string(http_code) + ").");
        g_upd_running = false;
        return NULL;
    }
    if (http_code < 200 || http_code >= 300) {
        set_check_result(UPD_CHK_HTTP_ERROR, "", GITHUB_RELEASES_WEB, "", "", 0, "",
                         "GitHub returned HTTP " + std::to_string(http_code) + ".");
        g_upd_running = false;
        return NULL;
    }

    UpdaterParseResult parsed = updater_parse_release_json(body);
    if (!parsed.ok) {
        set_check_result(UPD_CHK_HTTP_ERROR, "", GITHUB_RELEASES_WEB, "", "", 0, "",
                         "Invalid response from GitHub Releases API.");
        g_upd_running = false;
        return NULL;
    }
    if (!parsed.found_stable) {
        set_check_result(UPD_CHK_UP_TO_DATE, cur_ver, GITHUB_RELEASES_WEB, "", "", 0, "", "");
        g_upd_running = false;
        return NULL;
    }

    long long cur_rank = updater_tag_rank(cur_ver);
    long long lat_rank = updater_tag_rank(parsed.tag_name);
    if (lat_rank > cur_rank) {
        fprintf(stderr, "[updater] update available: %s (installed %s)\n",
                parsed.tag_name.c_str(), cur_ver.c_str());
        set_check_result(UPD_CHK_AVAILABLE, parsed.tag_name, parsed.html_url, parsed.body,
                         parsed.linux_asset_url, parsed.linux_asset_size, parsed.sha256sums_url, "");
    } else {
        set_check_result(UPD_CHK_UP_TO_DATE, parsed.tag_name, parsed.html_url, parsed.body,
                         parsed.linux_asset_url, parsed.linux_asset_size, parsed.sha256sums_url, "");
    }

    g_upd_running = false;
    return NULL;
}

void updater_start_check(bool manual) {
    if (g_shutting_down.load()) return;
    if (!manual) {
        const char *no_chk = getenv("LARECOMP_NO_UPDATE_CHECK");
        if (no_chk && strcmp(no_chk, "1") == 0) return;
        if (updater_is_suppressed_by_launcher()) return;
    }
    bool expected = false;
    if (!g_upd_running.compare_exchange_strong(expected, true))
        return;

    pthread_t old_t{};
    bool join_old = false;
    pthread_mutex_lock(&g_upd_lock);
    if (g_thread_active) {
        old_t = g_upd_thread;
        join_old = true;
        g_thread_active = false;
    }
    g_snap.state = UPD_CHK_CHECKING;
    g_snap.error.clear();
    pthread_mutex_unlock(&g_upd_lock);

    if (join_old) pthread_join(old_t, NULL);
    if (g_shutting_down.load()) {
        g_upd_running = false;
        return;
    }

    pthread_t t{};
    if (pthread_create(&t, NULL, check_thread, NULL) == 0) {
        pthread_mutex_lock(&g_upd_lock);
        g_upd_thread = t;
        g_thread_active = true;
        pthread_mutex_unlock(&g_upd_lock);
    } else {
        g_upd_running = false;
        set_check_result(UPD_CHK_OFFLINE, "", "", "", "", 0, "", "Failed to start update check thread.");
    }
}

static void set_install_failed(const std::string &err) {
    pthread_mutex_lock(&g_upd_lock);
    if (!g_shutting_down.load()) {
        g_snap.install_state = UPD_INST_FAILED;
        g_snap.install_error = err;
    }
    pthread_mutex_unlock(&g_upd_lock);
}

static void *install_thread(void *) {
#ifdef _WIN32
    set_install_failed("Linux AppImage updater is not supported on Windows.");
    g_inst_running = false;
    return NULL;
#else
    pthread_mutex_lock(&g_upd_lock);
    std::string cur_ver = g_snap.current_version;
    std::string tag = g_snap.latest_tag;
    std::string target = g_snap.install_target_path;
    std::string asset_url = g_snap.linux_asset_url;
    std::string sums_url = g_snap.sha256sums_url;
    uint64_t expected_size = g_snap.linux_asset_size;
    UpdaterInstallKind kind = g_snap.install_kind;
    pthread_mutex_unlock(&g_upd_lock);

    if (const char *ov_app = getenv("LARECOMP_UPDATE_APPIMAGE")) {
        asset_url = (strncmp(ov_app, "http", 4) == 0 || strncmp(ov_app, "file:", 5) == 0)
                        ? std::string(ov_app)
                        : std::string("file:") + ov_app;
    }
    if (const char *ov_sums = getenv("LARECOMP_UPDATE_SHA256SUMS")) {
        sums_url = (strncmp(ov_sums, "http", 4) == 0 || strncmp(ov_sums, "file:", 5) == 0)
                       ? std::string(ov_sums)
                       : std::string("file:") + ov_sums;
    }

    if (kind != UPD_INSTALL_LINUX_APPIMAGE || target.empty()) {
        set_install_failed("Installation is not a writable Linux AppImage.");
        g_inst_running = false;
        return NULL;
    }
    if (asset_url.empty() || sums_url.empty()) {
        set_install_failed("Release is missing Linux AppImage or SHA256SUMS asset.");
        g_inst_running = false;
        return NULL;
    }

    fs::path dir = fs::path(target).parent_path();
    std::string lock_path = (dir / ".PokeV-Recomp.update.lock").string();
    std::string part_path = (dir / (".PokeV-Recomp." + tag + ".part")).string();
    std::string sums_path = (dir / ".PokeV-Recomp.SHA256SUMS.part").string();
    std::string status_path = (dir / ".PokeV-Recomp.curl.status").string();
    std::string staged_path = (dir / ".PokeV-Recomp.staged").string();
    std::string backup_path = target + ".bak";

    // Prevent simultaneous updaters on the same AppImage
    int lock_fd = open(lock_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (lock_fd < 0) {
        // Check for stale PID
        if (FILE *lf = fopen(lock_path.c_str(), "rb")) {
            long pid = 0;
            bool dead = (fscanf(lf, "%ld", &pid) == 1 && pid > 0 && kill((pid_t)pid, 0) != 0 && errno == ESRCH);
            fclose(lf);
            if (dead) {
                unlink(lock_path.c_str());
                lock_fd = open(lock_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
            }
        }
        if (lock_fd < 0) {
            set_install_failed("Another update is already in progress for this AppImage.");
            g_inst_running = false;
            return NULL;
        }
    }
    char pid_buf[32];
    int plen = snprintf(pid_buf, sizeof pid_buf, "%ld\n", (long)getpid());
    if (plen > 0) (void)!write(lock_fd, pid_buf, (size_t)plen);
    close(lock_fd);

    auto cleanup_temp = [&](bool remove_lock) {
        unlink(part_path.c_str());
        unlink(sums_path.c_str());
        unlink(status_path.c_str());
        if (remove_lock) unlink(lock_path.c_str());
    };

    // Disk space check
    uint64_t avail_bytes = 0;
    if (const char *sim_free = getenv("LARECOMP_UPDATE_MIN_FREE_BYTES")) {
        avail_bytes = strtoull(sim_free, NULL, 10);
    } else {
        struct statvfs vfs{};
        if (statvfs(dir.c_str(), &vfs) == 0) {
            avail_bytes = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
        } else {
            cleanup_temp(true);
            set_install_failed("Unable to check available disk space.");
            g_inst_running = false;
            return NULL;
        }
    }
    uint64_t needed = expected_size > 0 ? (expected_size + (32ULL << 20)) : (256ULL << 20);
    if (avail_bytes < needed) {
        cleanup_temp(true);
        set_install_failed("Not enough free disk space in the AppImage folder.");
        g_inst_running = false;
        return NULL;
    }

    std::string curl = updater_find_curl();
    if (curl.empty() && (asset_url.compare(0, 5, "http:") == 0 || asset_url.compare(0, 6, "https:") == 0)) {
        cleanup_temp(true);
        set_install_failed("curl executable not found.");
        g_inst_running = false;
        return NULL;
    }
    std::string ua = "PokeV-Recomp/" + (cur_ver.empty() ? std::string("unknown") : cur_ver);

    // 1. Download SHA256SUMS
    int http_code = 0;
    bool cancelled = false;
    std::string err;
    if (!run_curl_to_file(curl, ua, sums_url, sums_path, status_path, 30,
                          &g_cancel_install, http_code, cancelled, err) ||
        http_code < 200 || http_code >= 300) {
        cleanup_temp(true);
        if (cancelled) {
            pthread_mutex_lock(&g_upd_lock);
            g_snap.install_state = UPD_INST_CANCELLED;
            pthread_mutex_unlock(&g_upd_lock);
        } else {
            set_install_failed(err.empty() ? ("Failed to download SHA256SUMS (HTTP " + std::to_string(http_code) + ").") : err);
        }
        g_inst_running = false;
        return NULL;
    }

    std::string sums_text = read_file_capped(sums_path, 64u << 10);
    std::string expected_sha = updater_parse_sha256sums(sums_text, LINUX_ASSET_NAME);
    unlink(sums_path.c_str());
    if (expected_sha.empty()) {
        cleanup_temp(true);
        set_install_failed("SHA256SUMS does not contain a valid checksum for the Linux AppImage.");
        g_inst_running = false;
        return NULL;
    }

    // 2. Download AppImage
    if (!run_curl_to_file(curl, ua, asset_url, part_path, status_path, 1800,
                          &g_cancel_install, http_code, cancelled, err) ||
        http_code < 200 || http_code >= 300) {
        cleanup_temp(true);
        if (cancelled) {
            pthread_mutex_lock(&g_upd_lock);
            g_snap.install_state = UPD_INST_CANCELLED;
            pthread_mutex_unlock(&g_upd_lock);
        } else {
            set_install_failed(err.empty() ? ("Download failed (HTTP " + std::to_string(http_code) + ").") : err);
        }
        g_inst_running = false;
        return NULL;
    }

    if (g_cancel_install.load() || g_shutting_down.load()) {
        cleanup_temp(true);
        pthread_mutex_lock(&g_upd_lock);
        g_snap.install_state = UPD_INST_CANCELLED;
        pthread_mutex_unlock(&g_upd_lock);
        g_inst_running = false;
        return NULL;
    }

    // 3. Verify size and SHA-256
    pthread_mutex_lock(&g_upd_lock);
    g_snap.install_state = UPD_INST_VERIFYING;
    pthread_mutex_unlock(&g_upd_lock);

    struct stat pst{};
    if (stat(part_path.c_str(), &pst) != 0 || pst.st_size <= 0 ||
        (expected_size > 0 && (uint64_t)pst.st_size != expected_size)) {
        cleanup_temp(true);
        set_install_failed("Downloaded AppImage size does not match release metadata.");
        g_inst_running = false;
        return NULL;
    }

    std::string got_sha = updater_file_sha256(part_path);
    if (got_sha.empty() || got_sha != expected_sha) {
        cleanup_temp(true);
        set_install_failed("SHA-256 checksum verification failed; installation aborted.");
        g_inst_running = false;
        return NULL;
    }

    // 4. Preserve executable permissions & stage
    struct stat ost{};
    mode_t mode = 0755;
    if (stat(target.c_str(), &ost) == 0) {
        mode = (ost.st_mode & 0777) | 0111;
    }
    if (chmod(part_path.c_str(), mode) != 0 || rename(part_path.c_str(), staged_path.c_str()) != 0) {
        cleanup_temp(true);
        set_install_failed("Failed to stage verified AppImage.");
        g_inst_running = false;
        return NULL;
    }

    pthread_mutex_lock(&g_upd_lock);
    g_staged_path = staged_path;
    g_backup_path = backup_path;
    g_lock_path = lock_path;
    g_snap.install_state = UPD_INST_STAGED;
    pthread_mutex_unlock(&g_upd_lock);

    g_inst_running = false;
    return NULL;
#endif
}

void updater_start_install() {
    if (g_shutting_down.load()) return;
    bool expected = false;
    if (!g_inst_running.compare_exchange_strong(expected, true))
        return;

    pthread_t old_t{};
    bool join_old = false;
    pthread_mutex_lock(&g_upd_lock);
    if (g_inst_thread_active) {
        old_t = g_inst_thread;
        join_old = true;
        g_inst_thread_active = false;
    }
    g_cancel_install = false;
    g_snap.install_state = UPD_INST_DOWNLOADING;
    g_snap.install_error.clear();
    g_snap.downloaded_bytes = 0;
    g_snap.total_bytes = g_snap.linux_asset_size;
    pthread_mutex_unlock(&g_upd_lock);

    if (join_old) pthread_join(old_t, NULL);
    if (g_shutting_down.load()) {
        g_inst_running = false;
        return;
    }

    pthread_t t{};
    if (pthread_create(&t, NULL, install_thread, NULL) == 0) {
        pthread_mutex_lock(&g_upd_lock);
        g_inst_thread = t;
        g_inst_thread_active = true;
        pthread_mutex_unlock(&g_upd_lock);
    } else {
        g_inst_running = false;
        set_install_failed("Failed to start update download thread.");
    }
}

void updater_cancel_install() {
    g_cancel_install = true;
}

bool updater_launch_external_apply_and_exit_ready() {
#ifdef _WIN32
    return false;
#else
    pthread_mutex_lock(&g_upd_lock);
    UpdaterInstallState st = g_snap.install_state;
    std::string target = g_snap.install_target_path;
    std::string staged = g_staged_path;
    std::string backup = g_backup_path;
    std::string lock = g_lock_path;
    std::string tag = g_snap.latest_tag;
    std::string note_path = g_data_root.empty() ? "" : (g_data_root + "/cache/_updated.txt");
    pthread_mutex_unlock(&g_upd_lock);

    if (st != UPD_INST_STAGED || target.empty() || staged.empty()) return false;

    pthread_mutex_lock(&g_upd_lock);
    g_snap.install_state = UPD_INST_APPLYING;
    pthread_mutex_unlock(&g_upd_lock);

    // Positional-arg POSIX /bin/sh helper: waits for this PID, backs up target, atomically moves staged -> target,
    // rolls back on failure, cleans temp files, writes _updated.txt, and relaunches target.
    static const char *k_helper =
        "PID=\"$1\"; TGT=\"$2\"; STG=\"$3\"; BAK=\"$4\"; LCK=\"$5\"; NOTE=\"$6\"; TAG=\"$7\"; "
        "i=0; while kill -0 \"$PID\" 2>/dev/null && [ $i -lt 300 ]; do sleep 0.05; i=$((i+1)); done; "
        "rm -f \"$BAK\"; "
        "if [ -e \"$TGT\" ]; then "
        "  if ! ln \"$TGT\" \"$BAK\" 2>/dev/null && ! cp -p \"$TGT\" \"$BAK\" 2>/dev/null; then "
        "    rm -f \"$STG\" \"$LCK\"; exit 1; "
        "  fi; "
        "fi; "
        "if ! mv -f \"$STG\" \"$TGT\"; then "
        "  if [ ! -e \"$TGT\" ] && [ -e \"$BAK\" ]; then mv -f \"$BAK\" \"$TGT\" 2>/dev/null || true; fi; "
        "  rm -f \"$BAK\" \"$STG\" \"$LCK\"; exit 1; "
        "fi; "
        "rm -f \"$BAK\" \"$LCK\"; "
        "if [ -n \"$NOTE\" ]; then mkdir -p \"$(dirname \"$NOTE\")\" 2>/dev/null || true; printf '%s' \"$TAG\" > \"$NOTE\" 2>/dev/null || true; fi; "
        "exec \"$TGT\" >/dev/null 2>&1";

    std::string pid_str = std::to_string((long)getpid());
    const char *av[] = {
        "/bin/sh", "-c", k_helper, "pokev-updater",
        pid_str.c_str(), target.c_str(), staged.c_str(), backup.c_str(),
        lock.c_str(), note_path.c_str(), tag.c_str(), NULL
    };

    // Spawn detached so closing this app does not kill the external updater
    pid_t child = fork();
    if (child < 0) {
        set_install_failed("Failed to launch external updater process.");
        return false;
    }
    if (child == 0) {
        setsid();
        execv("/bin/sh", (char *const *)av);
        _exit(127);
    }
    return true;
#endif
}

void updater_shutdown() {
    g_shutting_down = true;
    g_cancel_install = true;
    pthread_t t_chk{}, t_inst{};
    bool join_chk = false, join_inst = false;

    pthread_mutex_lock(&g_upd_lock);
    if (g_thread_active) {
        t_chk = g_upd_thread;
        join_chk = true;
        g_thread_active = false;
    }
    if (g_inst_thread_active) {
        t_inst = g_inst_thread;
        join_inst = true;
        g_inst_thread_active = false;
    }
    pthread_mutex_unlock(&g_upd_lock);

    if (join_chk) pthread_join(t_chk, NULL);
    if (join_inst) pthread_join(t_inst, NULL);
    g_upd_running = false;
    g_inst_running = false;
}
