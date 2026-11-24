#pragma once

#include <functional>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace luw::cli {

using CommandArguments = std::vector<std::string>;
using CommandHandler = std::function<int(const CommandArguments&, std::ostream&, std::ostream&)>;

enum class DeckArgumentPolicy {
    None,
    // Preserve explicit command-specific inputs (for example .nc or .shp),
    // but apply wrapper-style deck discovery when no input was supplied.
    AutoWhenMissing,
    Required,
};

struct CommandSpec {
    std::string name;
    std::string summary;
    std::string usage;
    DeckArgumentPolicy deckArgument = DeckArgumentPolicy::Required;
    CommandHandler handler;
    bool acceptsPreprocessOptions = false;
};

class CommandRegistry final {
public:
    void add(CommandSpec command);

    [[nodiscard]] const CommandSpec* find(std::string_view name) const noexcept;
    [[nodiscard]] std::vector<std::string> commandNames() const;

    int invoke(std::string_view name,
               CommandArguments arguments,
               std::ostream& output,
               std::ostream& error) const;

    void printHelp(std::ostream& output, std::string_view programName = "luw") const;

private:
    std::vector<CommandSpec> commands_;
};

// Resolves the compatibility wrapper's first deck argument.  When arguments
// are empty, conf.luw wins; otherwise exactly one *.luw file is accepted.
// The resolved absolute path is placed in arguments[0].
bool resolveDeckArgument(std::string_view commandName,
                         CommandArguments& arguments,
                         std::ostream& error);

// Populates every built-in native command.  Kept separate from the generic
// registry so lightweight dispatcher tests can use fake handlers.
CommandRegistry makeCommandRegistry();

} // namespace luw::cli
