#pragma once

#include <cstdint>
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

enum UpdaterInstallKind {
    UPD_INSTALL_UNSUPPORTED = 0,
    UPD_INSTALL_RECOMPLAUNCHER,
    UPD_INSTALL_LINUX_APPIMAGE_RO,
    UPD_INSTALL_LINUX_APPIMAGE,
};

enum UpdaterInstallState {
    UPD_INST_IDLE = 0,
    UPD_INST_DOWNLOADING,
    UPD_INST_VERIFYING,
    UPD_INST_STAGED,
    UPD_INST_APPLYING,
    UPD_INST_CANCELLED,
    UPD_INST_FAILED,
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

    // Phase B: Linux AppImage install state
    UpdaterInstallKind install_kind = UPD_INSTALL_UNSUPPORTED;
    std::string install_target_path;
    std::string install_reason;
    std::string linux_asset_url;
    uint64_t linux_asset_size = 0;
    std::string sha256sums_url;

    UpdaterInstallState install_state = UPD_INST_IDLE;
    uint64_t downloaded_bytes = 0;
    uint64_t total_bytes = 0;
    std::string install_error;
    std::string updated_from_tag;  // non-empty when this launch follows a completed update
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
    std::string linux_asset_url;
    uint64_t linux_asset_size = 0;
    std::string sha256sums_url;
};

// Parse GitHub /releases/latest object (or /releases array), ignoring draft entries and (by default) prereleases.
UpdaterParseResult updater_parse_release_json(const std::string &json);
UpdaterParseResult updater_parse_release_json_opts(const std::string &json, bool include_prerelease);

// Parse SHA256SUMS content and return lowercase 64-hex sha256 for `filename`, or "" if missing/invalid.
std::string updater_parse_sha256sums(const std::string &sums_text, const std::string &filename);

// Compute lowercase hex SHA-256 of a file on disk using CacheSha256, or "" on I/O error.
std::string updater_file_sha256(const std::string &path);

// Detect current installation mode (AppImage writable/read-only, RecompLauncher, or unsupported).
UpdaterInstallKind updater_detect_install_kind(std::string &out_target_path, std::string &out_reason);

// Atomic file replacement with backup & rollback (used in tests and recovery).
bool updater_apply_staged_file(const std::string &target_path, const std::string &staged_path,
                               const std::string &backup_path, std::string &out_error);

// Lifecycle & Welcome Screen API (never blocks the caller)
void updater_configure(const char *current_version, const char *data_root,
                       bool launched_with_play, bool launched_with_prepare);
bool updater_is_suppressed_by_launcher();
void updater_start_check(bool manual);

// Phase B: Linux AppImage download, SHA-256 verification, cancel, and external apply+relaunch
void updater_start_install();
void updater_cancel_install();
bool updater_launch_external_apply_and_exit_ready();

void updater_shutdown();
UpdaterSnapshot updater_get_snapshot();
