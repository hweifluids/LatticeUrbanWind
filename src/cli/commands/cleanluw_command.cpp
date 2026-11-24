#include "luw/cleanluw_command.hpp"

#include <QString>

#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

fs::path pathFromUtf8(const std::string& value) {
#ifdef _WIN32
    return fs::path(QString::fromUtf8(value).toStdWString());
#else
    return fs::path(value);
#endif
}

std::string pathToUtf8(const fs::path& value) {
#ifdef _WIN32
    return QString::fromStdWString(value.wstring()).toUtf8().toStdString();
#else
    return value.string();
#endif
}

fs::path normalizedAbsolute(const fs::path& value) {
    std::error_code error;
    fs::path result = fs::weakly_canonical(value, error);
    if (!error) {
        return result;
    }
    result = fs::absolute(value, error);
    return (error ? value : result).lexically_normal();
}

void removeProjectTemporaryFiles(const fs::path& parent, std::ostream& errorOutput) {
    const fs::path target = parent / "proj_temp";
    std::error_code statusError;
    const fs::file_status targetStatus = fs::status(target, statusError);
    if (statusError == std::errc::no_such_file_or_directory
        || targetStatus.type() == fs::file_type::not_found) {
        return;
    }
    if (statusError) {
        throw fs::filesystem_error("cannot inspect proj_temp", target, statusError);
    }
    if (!fs::is_directory(targetStatus)) {
        throw std::runtime_error(pathToUtf8(target) + " is not a directory");
    }

    std::error_code iteratorError;
    fs::recursive_directory_iterator iterator(
        target, fs::directory_options::skip_permission_denied, iteratorError);
    const fs::recursive_directory_iterator end;
    if (iteratorError) {
        throw fs::filesystem_error("cannot traverse proj_temp", target, iteratorError);
    }

    while (iterator != end) {
        const fs::directory_entry entry = *iterator;
        std::error_code linkError;
        const fs::file_status linkStatus = entry.symlink_status(linkError);
        if (linkError) {
            throw fs::filesystem_error("cannot inspect proj_temp entry", entry.path(), linkError);
        }

        bool removeEntry = fs::is_regular_file(linkStatus);
        if (fs::is_symlink(linkStatus)) {
            std::error_code targetError;
            const fs::file_status followedStatus = entry.status(targetError);
            removeEntry = targetError || !fs::is_directory(followedStatus);
            if (!removeEntry) {
                iterator.disable_recursion_pending();
            }
        }

        if (removeEntry) {
            std::error_code removeError;
            fs::remove(entry.path(), removeError);
            if (removeError == std::errc::no_such_file_or_directory) {
                removeError.clear();
            }
            if (removeError == std::errc::permission_denied) {
                errorOutput << "Warning: failed to delete file: "
                            << pathToUtf8(entry.path()) << " ("
                            << removeError.message() << ")\n";
            } else if (removeError) {
                throw fs::filesystem_error("failed to delete file", entry.path(), removeError);
            }
        }

        iterator.increment(iteratorError);
        if (iteratorError) {
            if (iteratorError == std::errc::permission_denied) {
                iteratorError.clear();
                continue;
            }
            throw fs::filesystem_error("cannot traverse proj_temp", target, iteratorError);
        }
    }
}

} // namespace

int run_cleanluw(const CommandArguments& arguments,
                 std::ostream& output,
                 std::ostream& error) {
    static_cast<void>(output);
    if (arguments.size() != 1U) {
        error << "Usage: python cleanluw.py <text file path>\n";
        return 1;
    }

    const fs::path deck = normalizedAbsolute(pathFromUtf8(arguments.front()));
    std::error_code fileError;
    if (!fs::is_regular_file(deck, fileError) || fileError) {
        error << "Error: " << pathToUtf8(deck) << " is not a valid file\n";
        return 1;
    }

    try {
        removeProjectTemporaryFiles(deck.parent_path(), error);
    } catch (const std::exception& exception) {
        error << "Failed to clean proj_temp: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}

} // namespace luw::cli
