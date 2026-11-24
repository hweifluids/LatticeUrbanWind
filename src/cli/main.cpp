#include "luw/command_registry.hpp"
#include "luw/runtime_paths.hpp"

#include <QCoreApplication>
#include <QFileInfo>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

namespace {

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string utf8(const QString& value) {
    return value.toUtf8().toStdString();
}

} // namespace

int main(int argc, char* argv[]) {
    QCoreApplication application(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("LatticeUrbanWind"));
    QCoreApplication::setApplicationVersion(QStringLiteral(LUW_VERSION));
    luw::configureBundledDataPaths();

    luw::cli::CommandRegistry registry = luw::cli::makeCommandRegistry();
    const QStringList rawArguments = QCoreApplication::arguments();
    std::string invokedAs = lowerAscii(
        utf8(QFileInfo(QCoreApplication::applicationFilePath()).completeBaseName()));

    luw::cli::CommandArguments arguments;
    arguments.reserve(rawArguments.size() > 1
                          ? static_cast<std::size_t>(rawArguments.size() - 1)
                          : 0U);
    for (qsizetype index = 1; index < rawArguments.size(); ++index) {
        arguments.push_back(utf8(rawArguments.at(index)));
    }

    std::string command;
    if (invokedAs == "luw") {
        if (arguments.empty()
            || arguments.front() == "--help"
            || arguments.front() == "-h"
            || arguments.front() == "help") {
            registry.printHelp(std::cout);
            return 0;
        }
        if (arguments.front() == "--version" || arguments.front() == "-V") {
            std::cout << "LatticeUrbanWind Native " << LUW_VERSION << '\n';
            return 0;
        }
        command = lowerAscii(arguments.front());
        arguments.erase(arguments.begin());
    } else {
        command = invokedAs;
    }

    return registry.invoke(command, std::move(arguments), std::cout, std::cerr);
}
