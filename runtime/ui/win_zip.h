#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct WinZipEntryInfo {
    std::string rel_path;  // normalized forward-slash relative path
    uint64_t size = 0;
    uint32_t crc32 = 0;
    bool is_dir = false;
};

struct WinZipExtractResult {
    bool ok = false;
    std::string error;
    std::vector<WinZipEntryInfo> files;
};

// Validate a single ZIP entry path and normalize it (returns false if unsafe).
// Rejects absolute paths, drive letters, UNC paths, '..', control chars, backslashes escaping,
// Windows reserved device names, and user-data paths (larecomp.ini, save/, extracted/, mods/, etc.).
bool updater_validate_zip_entry_path(const std::string &raw_name,
                                     const std::string &strip_top_prefix,
                                     std::string &out_rel_path,
                                     bool &out_is_dir,
                                     std::string &out_error);

// Safely validate and extract a ZIP archive into `dest_dir`.
// Supports Stored (0) and Deflate (8), ZIP32/ZIP64 EOCD, CRC-32 verification,
// symlink/reparse-point rejection, case-insensitive duplicate detection,
// overflow-safe bounds checks, and decompression limits (1 GiB/entry, 2 GiB aggregate).
WinZipExtractResult updater_extract_zip_safe(const std::string &zip_path,
                                             const std::string &dest_dir,
                                             const std::string &strip_top_prefix);
