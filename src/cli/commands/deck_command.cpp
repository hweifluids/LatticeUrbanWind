#include "commands/deck_command.hpp"

#include "luw/deck_document.hpp"

#include <QString>

#include <ostream>

namespace luw::cli {

namespace {

void printUsage(std::ostream& output) {
    output << "Usage:\n"
           << "  luw deck check <deck>\n"
           << "  luw deck render <deck>\n"
           << "  luw deck get <deck> <key>\n"
           << "  luw deck set <deck> <key> <raw-value>\n";
}

QString fromUtf8(const std::string& value) {
    return QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
}

} // namespace

int runDeckCommand(const std::vector<std::string>& arguments,
                   std::ostream& output,
                   std::ostream& error) {
    if (arguments.empty() || arguments.front() == "--help" || arguments.front() == "-h") {
        printUsage(output);
        return arguments.empty() ? 2 : 0;
    }
    const std::string& operation = arguments.front();
    if (arguments.size() < 2) {
        printUsage(error);
        return 2;
    }

    try {
        luw::DeckDocument deck = luw::DeckDocument::load(fromUtf8(arguments.at(1)));
        if (operation == "check") {
            const QStringList duplicates = deck.duplicateKeys();
            if (duplicates.isEmpty()) {
                output << "Deck is valid.\n";
                return 0;
            }
            error << "Duplicate deck keys: " << duplicates.join(", ").toStdString() << '\n';
            return 1;
        }
        if (operation == "render") {
            output << deck.render().toStdString();
            return 0;
        }
        if (operation == "get") {
            if (arguments.size() != 3) {
                printUsage(error);
                return 2;
            }
            const auto value = deck.raw(fromUtf8(arguments.at(2)));
            if (!value) return 1;
            output << value->toStdString() << '\n';
            return 0;
        }
        if (operation == "set") {
            if (arguments.size() != 4) {
                printUsage(error);
                return 2;
            }
            deck.setRaw(fromUtf8(arguments.at(2)), fromUtf8(arguments.at(3)));
            deck.save();
            return 0;
        }
    } catch (const std::exception& exception) {
        error << "Error: " << exception.what() << '\n';
        return 1;
    }

    error << "Unknown deck operation: " << operation << '\n';
    printUsage(error);
    return 2;
}

} // namespace luw::cli

