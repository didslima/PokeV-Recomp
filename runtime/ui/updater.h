#pragma once

#include <string>

enum UpdaterCheckState {
    UPD_CHK_IDLE = 0,
    UPD_CHK_CHECKING,
    UPD_CHK_UP_TO_DATE,
    UPD_CHK_AVAILABLE,
    UPD_CHK_OFFLINE,
    UPD_CHK_RATE_LIMITED,
    UPD_CHK_HTTP_ERROR,
};

struct UpdaterSnapshot {
    UpdaterCheckState state = UPD_CHK_IDLE;
    std::string current_version;
    std::string latest_tag;
    std::string html_url;
    std::string notes;
    std::string error;
    bool suppressed_by_launcher = false;
    bool has_checked = false;
};

// Pure helpers (also used by unit tests)
long long updater_tag_rank(const std::string &tag);
std::string updater_tag_label(const std::string &tag, bool with_word);
std::string updater_find_curl();

struct UpdaterParseResult {
    bool ok = false;
    bool found_stable = false;
    std::string tag_name;
    std::string html_url;
    std::string body;
};

// Parse GitHub /releases/latest object (or /releases array), ignoring draft and prerelease entries.
UpdaterParseResult updater_parse_release_json(const std::string &json);

// Lifecycle & Welcome Screen API (never blocks the caller)
void updater_configure(const char *current_version, const char *data_root,
                       bool launched_with_play, bool launched_with_prepare);
bool updater_is_suppressed_by_launcher();
void updater_start_check(bool manual);
void updater_shutdown();
UpdaterSnapshot updater_get_snapshot();
