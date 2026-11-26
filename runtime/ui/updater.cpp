#include "updater.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <pthread.h>
#include <string>
#include <thread>

#ifndef _WIN32
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

static pthread_mutex_t g_upd_lock = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<bool> g_upd_running{false};
static std::atomic<bool> g_shutting_down{false};
static pthread_t g_upd_thread{};
static bool g_thread_active = false;
static bool g_atexit_registered = false;
static UpdaterSnapshot g_snap;
static std::string g_data_root;
static bool g_flag_play = false;
static bool g_flag_prepare = false;

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
    static const char *const k_candidates[] = {
        "/usr/bin/curl",
        "/usr/local/bin/curl",
        "/opt/homebrew/bin/curl",
    };
    for (const char *c : k_candidates) {
        if (access(c, X_OK) == 0) return c;
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
                case 'r':  break;  // normalize CRLF -> LF for ImGui display
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
                            append_utf8(out, 0xFFFDu);  // unpaired high surrogate
                        }
                    } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                        append_utf8(out, 0xFFFDu);  // unpaired low surrogate
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

struct ParsedObj {
    bool valid = false;
    bool is_draft = false;
    bool is_prerelease = false;
    std::string tag_name;
    std::string html_url;
    std::string body;
};

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
        } else {
            pos = skip_json_value(s, pos);
        }

        pos = skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') pos++;
    }
    return ParsedObj{};
}

static std::string sanitize_release_url(const std::string &url, const std::string &tag) {
    if (url.compare(0, 8, "https://") == 0) return url;
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

void updater_configure(const char *current_version, const char *data_root,
                       bool launched_with_play, bool launched_with_prepare) {
    pthread_mutex_lock(&g_upd_lock);
    g_shutting_down = false;
    if (!g_atexit_registered) {
        atexit(updater_shutdown);
        g_atexit_registered = true;
    }
    g_snap.current_version = current_version ? current_version : "";
    g_data_root = data_root ? data_root : "";
    g_flag_play = launched_with_play;
    g_flag_prepare = launched_with_prepare;
    const char *rl = getenv("RECOMPLAUNCHER_DATA");
    g_snap.suppressed_by_launcher = (rl && *rl) || g_flag_play || g_flag_prepare;
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

static void set_result(UpdaterCheckState st, const std::string &tag,
                       const std::string &url, const std::string &notes,
                       const std::string &err) {
    if (g_shutting_down.load()) return;
    pthread_mutex_lock(&g_upd_lock);
    if (!g_shutting_down.load()) {
        g_snap.state = st;
        g_snap.latest_tag = tag;
        g_snap.html_url = url;
        g_snap.notes = notes;
        g_snap.error = err;
        g_snap.has_checked = true;
    }
    pthread_mutex_unlock(&g_upd_lock);
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

static int parse_http_status_from_log(const std::string &raw) {
    const std::string marker = "__HTTP_CODE__:";
    size_t m = raw.rfind(marker);
    if (m == std::string::npos) return 0;
    int code = atoi(raw.c_str() + m + marker.size());
    return (code >= 100 && code <= 599) ? code : 0;
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
            remove(body_path.c_str());
            remove(status_path.c_str());
            std::string ua = "PokeV-Recomp/" + (cur_ver.empty() ? std::string("unknown") : cur_ver);

            // -o writes only HTTP body to body_path; plat_spawn captures -w status (and any stderr) in status_path
            const char *av[] = {
                curl.c_str(), "-sL", "--proto", "=https", "--tlsv1.2",
                "--max-time", "15",
                "-H", "Accept: application/vnd.github+json",
                "-A", ua.c_str(),
                "-o", body_path.c_str(),
                "-w", "\n__HTTP_CODE__:%{http_code}\n",
                GITHUB_LATEST_URL, NULL
            };
            PlatProc p = plat_spawn(av, status_path.c_str(), false);
            int rc = -1;
            if (p) {
                while (!g_shutting_down.load()) {
                    rc = plat_proc_wait(p, false);
                    if (rc >= 0) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                if (rc < 0) plat_proc_kill(p);
            }

            if (g_shutting_down.load()) {
                remove(body_path.c_str());
                remove(status_path.c_str());
                g_upd_running = false;
                return NULL;
            }

            std::string status_raw = read_file_capped(status_path, 64u << 10);
            body = read_file_capped(body_path, 2u << 20);
            remove(body_path.c_str());
            remove(status_path.c_str());

            http_code = parse_http_status_from_log(status_raw);
            if (rc != 0 || http_code <= 0) {
                network_ok = false;
                http_code = 0;
            }
        }
    }

    if (g_shutting_down.load()) {
        g_upd_running = false;
        return NULL;
    }

    if (!network_ok) {
        set_result(UPD_CHK_OFFLINE, "", "", "",
                   err_detail.empty() ? "Offline or unable to reach GitHub." : err_detail);
        g_upd_running = false;
        return NULL;
    }
    if (http_code == 403 || http_code == 429) {
        set_result(UPD_CHK_RATE_LIMITED, "", GITHUB_RELEASES_WEB, "",
                   "GitHub API rate limit reached (HTTP " + std::to_string(http_code) + ").");
        g_upd_running = false;
        return NULL;
    }
    if (http_code < 200 || http_code >= 300) {
        set_result(UPD_CHK_HTTP_ERROR, "", GITHUB_RELEASES_WEB, "",
                   "GitHub returned HTTP " + std::to_string(http_code) + ".");
        g_upd_running = false;
        return NULL;
    }

    UpdaterParseResult parsed = updater_parse_release_json(body);
    if (!parsed.ok) {
        set_result(UPD_CHK_HTTP_ERROR, "", GITHUB_RELEASES_WEB, "",
                   "Invalid response from GitHub Releases API.");
        g_upd_running = false;
        return NULL;
    }
    if (!parsed.found_stable) {
        set_result(UPD_CHK_UP_TO_DATE, cur_ver, GITHUB_RELEASES_WEB, "", "");
        g_upd_running = false;
        return NULL;
    }

    long long cur_rank = updater_tag_rank(cur_ver);
    long long lat_rank = updater_tag_rank(parsed.tag_name);
    if (lat_rank > cur_rank) {
        fprintf(stderr, "[updater] update available: %s (installed %s)\n",
                parsed.tag_name.c_str(), cur_ver.c_str());
        set_result(UPD_CHK_AVAILABLE, parsed.tag_name, parsed.html_url, parsed.body, "");
    } else {
        set_result(UPD_CHK_UP_TO_DATE, parsed.tag_name, parsed.html_url, parsed.body, "");
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
        set_result(UPD_CHK_OFFLINE, "", "", "", "Failed to start update check thread.");
    }
}

void updater_shutdown() {
    g_shutting_down = true;
    pthread_t t{};
    bool join_t = false;

    pthread_mutex_lock(&g_upd_lock);
    if (g_thread_active) {
        t = g_upd_thread;
        join_t = true;
        g_thread_active = false;
    }
    pthread_mutex_unlock(&g_upd_lock);

    if (join_t) pthread_join(t, NULL);
    g_upd_running = false;
}
