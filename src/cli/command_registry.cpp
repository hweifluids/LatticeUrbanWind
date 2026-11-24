#include "luw/command_registry.hpp"

#include "luw/preprocess_compute_config.hpp"

#include <QString>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <filesystem>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <system_error>

namespace luw::cli {

namespace {

namespace fs = std::filesystem;

std::string lowerAscii(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return result;
}

fs::path pathFromUtf8(const std::string& value) {
#ifdef _WIN32
    return fs::path(QString::fromStdString(value).toStdWString());
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

bool isRegularFile(const fs::path& path) {
    std::error_code error;
    return fs::is_regular_file(path, error) && !error;
}

fs::path normalizedAbsolute(const fs::path& path) {
    std::error_code error;
    fs::path result = fs::absolute(path, error);
    if (error) result = path;
    return result.lexically_normal();
}

bool hasLuwExtension(const fs::path& path) {
    std::string extension = pathToUtf8(path.extension());
#ifdef _WIN32
    extension = lowerAscii(extension);
#endif
    return extension == ".luw";
}

bool optionValue(const CommandArguments& arguments,
                 std::size_t& index,
                 std::string_view option,
                 std::string& value,
                 std::ostream& error,
                 std::string_view commandName) {
    const std::string& argument = arguments[index];
    const std::string prefix = std::string(option) + '=';
    if (argument.rfind(prefix, 0) == 0) {
        value = argument.substr(prefix.size());
        if (!value.empty()) return true;
    } else if (argument == option && index + 1U < arguments.size()
               && arguments[index + 1U] != "--"
               && arguments[index + 1U].rfind("--", 0) != 0) {
        value = arguments[++index];
        return true;
    }
    error << commandName << ": error: argument " << option
          << ": expected one argument\n";
    return false;
}

std::optional<int> parseNonNegativeInt(std::string_view text) {
    while (!text.empty()
           && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1U);
    }
    while (!text.empty()
           && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1U);
    }
    std::string normalized;
    normalized.reserve(text.size());
    for (std::size_t index = 0; index < text.size(); ++index) {
        if (text[index] != '_') {
            normalized.push_back(text[index]);
            continue;
        }
        const bool betweenDigits = index > 0U && index + 1U < text.size()
            && std::isdigit(static_cast<unsigned char>(text[index - 1U])) != 0
            && std::isdigit(static_cast<unsigned char>(text[index + 1U])) != 0;
        if (!betweenDigits) return std::nullopt;
    }
    text = normalized;
    if (text.empty()) return std::nullopt;
    if (text.front() == '+') text.remove_prefix(1U);
    if (text.empty()) return std::nullopt;
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()
        || value < 0) {
        return std::nullopt;
    }
    return value;
}

std::optional<bool> parseBoolean(std::string_view text) {
    const std::string normalized = lowerAscii(text);
    if (normalized == "true" || normalized == "1" || normalized == "yes"
        || normalized == "on") {
        return true;
    }
    if (normalized == "false" || normalized == "0" || normalized == "no"
        || normalized == "off") {
        return false;
    }
    return std::nullopt;
}

bool extractPreprocessOptions(std::string_view commandName,
                              CommandArguments& arguments,
                              PreprocessComputeOverrides& overrides,
                              std::ostream& error) {
    constexpr std::string_view useGpuOption = "--preprocess-use-gpu";
    constexpr std::string_view gpuDeviceOption = "--preprocess-gpu-device";
    CommandArguments retained;
    retained.reserve(arguments.size());

    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "--") {
            retained.insert(retained.end(), arguments.begin() + static_cast<std::ptrdiff_t>(index),
                            arguments.end());
            break;
        }

        const bool useGpu = argument == useGpuOption
            || argument.rfind(std::string(useGpuOption) + '=', 0) == 0;
        const bool gpuDevice = argument == gpuDeviceOption
            || argument.rfind(std::string(gpuDeviceOption) + '=', 0) == 0;
        if (!useGpu && !gpuDevice) {
            retained.push_back(argument);
            continue;
        }

        std::string value;
        const std::string_view option = useGpu ? useGpuOption : gpuDeviceOption;
        if (!optionValue(arguments, index, option, value, error, commandName)) return false;
        if (useGpu) {
            const auto parsed = parseBoolean(value);
            if (!parsed) {
                error << commandName << ": error: argument " << useGpuOption
                      << ": invalid boolean value: '" << value
                      << "' (expected true or false)\n";
                return false;
            }
            overrides.useGpu = *parsed;
        } else {
            const auto parsed = parseNonNegativeInt(value);
            if (!parsed) {
                error << commandName << ": error: argument " << gpuDeviceOption
                      << ": invalid non-negative integer value: '" << value << "'\n";
                return false;
            }
            overrides.gpuDevice = *parsed;
        }
    }

    arguments = std::move(retained);
    return true;
}

} // namespace

void CommandRegistry::add(CommandSpec command) {
    if (command.name.empty()) throw std::invalid_argument("Command name must not be empty.");
    command.name = lowerAscii(command.name);
    if (!command.handler) throw std::invalid_argument("Command handler must not be empty.");
    if (find(command.name)) {
        throw std::invalid_argument("Duplicate command registration: " + command.name);
    }
    commands_.push_back(std::move(command));
}

const CommandSpec* CommandRegistry::find(std::string_view name) const noexcept {
    const std::string normalized = lowerAscii(name);
    const auto match = std::find_if(commands_.cbegin(), commands_.cend(), [&](const CommandSpec& item) {
        return item.name == normalized;
    });
    return match == commands_.cend() ? nullptr : &*match;
}

std::vector<std::string> CommandRegistry::commandNames() const {
    std::vector<std::string> names;
    names.reserve(commands_.size());
    for (const CommandSpec& command : commands_) names.push_back(command.name);
    std::sort(names.begin(), names.end());
    return names;
}

int CommandRegistry::invoke(std::string_view name,
                            CommandArguments arguments,
                            std::ostream& output,
                            std::ostream& error) const {
    const CommandSpec* command = find(name);
    if (!command) {
        error << "ERROR: Unknown command: " << name << '\n';
        printHelp(error);
        return 2;
    }

    PreprocessComputeOverrides preprocessOverrides;
    if (command->acceptsPreprocessOptions
        && !extractPreprocessOptions(command->name, arguments, preprocessOverrides, error)) {
        return 2;
    }

    const auto invokeHandler = [&]() {
        const bool helpWithoutDeck = !arguments.empty()
            && (arguments.front() == "--help" || arguments.front() == "-h");
        const bool shouldResolveDeck = !helpWithoutDeck
            && (command->deckArgument == DeckArgumentPolicy::Required
            || (command->deckArgument == DeckArgumentPolicy::AutoWhenMissing
                && (arguments.empty() || arguments.front().empty())));
        if (shouldResolveDeck
            && !resolveDeckArgument(command->name, arguments, error)) {
            return 2;
        }

        try {
            const int status = command->handler(arguments, output, error);
            if (status == 0 && helpWithoutDeck &&
                command->acceptsPreprocessOptions) {
                output
                    << "\nshared preprocessing compute options:\n"
                    << "  --preprocess-use-gpu BOOL\n"
                    << "                        Select CPU (false) or the OpenCL GPU backend (true).\n"
                    << "                        An explicit GPU request fails if the device/backend is unavailable.\n"
                    << "  --preprocess-gpu-device DEVICE\n"
                    << "                        Zero-based index in the global GPU-only OpenCL device list.\n";
            }
            return status;
        } catch (const std::exception& exception) {
            error << "ERROR: " << command->name << " failed: " << exception.what() << '\n';
        } catch (...) {
            error << "ERROR: " << command->name << " failed with an unknown exception.\n";
        }
        return 1;
    };

    if (command->acceptsPreprocessOptions) {
        PreprocessComputeOverrideScope scope(preprocessOverrides);
        return invokeHandler();
    }
    return invokeHandler();
}

void CommandRegistry::printHelp(std::ostream& output, std::string_view programName) const {
    output << "Usage: " << programName << " <command> [arguments]\n\n"
           << "Native LatticeUrbanWind commands:\n";
    std::vector<const CommandSpec*> sorted;
    sorted.reserve(commands_.size());
    for (const CommandSpec& command : commands_) sorted.push_back(&command);
    std::sort(sorted.begin(), sorted.end(), [](const CommandSpec* left, const CommandSpec* right) {
        return left->name < right->name;
    });
    for (const CommandSpec* command : sorted) {
        output << "  " << command->name;
        if (command->name.size() < 20) {
            output << std::string(20 - command->name.size(), ' ');
        } else {
            output << ' ';
        }
        output << command->summary << '\n';
    }
    output << "\nRun '" << programName << " <command> --help' for command-specific usage.\n";
}

bool resolveDeckArgument(std::string_view commandName,
                         CommandArguments& arguments,
                         std::ostream& error) {
    fs::path candidate;
    if (!arguments.empty() && !arguments.front().empty()) {
        candidate = pathFromUtf8(arguments.front());
    } else {
        std::error_code currentError;
        const fs::path current = fs::current_path(currentError);
        if (currentError) {
            error << "ERROR: Cannot inspect current directory: " << currentError.message() << '\n';
            return false;
        }
        const fs::path conventional = current / "conf.luw";
        if (isRegularFile(conventional)) {
            candidate = conventional;
        } else {
            std::vector<fs::path> matches;
            std::error_code iteratorError;
            for (fs::directory_iterator iterator(current, iteratorError), end;
                 !iteratorError && iterator != end;
                 iterator.increment(iteratorError)) {
                std::error_code typeError;
                if (iterator->is_regular_file(typeError) && !typeError
                    && hasLuwExtension(iterator->path())) {
                    matches.push_back(iterator->path());
                    if (matches.size() > 1) break;
                }
            }
            if (matches.size() == 1) candidate = matches.front();
        }

        if (candidate.empty()) {
            error << "Usage: " << commandName
                  << " <path-to-conf.luw>. No conf.luw in current directory and no unique *.luw found.\n";
            return false;
        }
    }

    const fs::path absolute = normalizedAbsolute(candidate);
    if (!isRegularFile(absolute)) {
        error << "ERROR: Conf file not found: " << pathToUtf8(absolute) << '\n';
        return false;
    }

    const std::string resolved = pathToUtf8(absolute);
    if (arguments.empty()) {
        arguments.push_back(resolved);
    } else {
        arguments.front() = resolved;
    }
    return true;
}

} // namespace luw::cli
