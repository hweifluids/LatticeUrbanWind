#include "luw/runluw_command.hpp"

#include "luw/deck_document.hpp"
#include "luw/runtime_paths.hpp"

#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStringList>

#include <exception>
#include <optional>
#include <ostream>

namespace luw::cli {

namespace {

enum class SolverBackend {
    Culbm,
    FluidX3D
};

std::optional<SolverBackend> selectSolver(const DeckDocument& deck,
                                          std::ostream& error) {
    SolverBackend backend = SolverBackend::Culbm;
    if (deck.has(QStringLiteral("cfd_solver"))) {
        const QString configured = deck.textOr(QStringLiteral("cfd_solver"))
                                       .trimmed()
                                       .toLower();
        if (configured.isEmpty() || configured == QStringLiteral("culbm")) {
            backend = SolverBackend::Culbm;
        } else if (configured == QStringLiteral("fluidx3d")) {
            backend = SolverBackend::FluidX3D;
        } else {
            error << "ERROR: invalid cfd_solver='"
                  << configured.toUtf8().toStdString()
                  << "' (expected 'culbm' or 'fluidx3d')\n";
            return std::nullopt;
        }
    }

    bool ibmEnabled = false;
    if (deck.has(QStringLiteral("ibm_enabler"))) {
        const std::optional<bool> configured = deck.boolean(QStringLiteral("ibm_enabler"));
        if (!configured) {
            error << "ERROR: invalid ibm_enabler='"
                  << deck.rawOr(QStringLiteral("ibm_enabler")).toUtf8().toStdString()
                  << "' (expected a boolean value)\n";
            return std::nullopt;
        }
        ibmEnabled = *configured;
    }

    if (ibmEnabled && backend == SolverBackend::FluidX3D) {
        error << "ERROR: ibm_enabler=true requires cfd_solver=culbm; "
                 "cfd_solver=fluidx3d is incompatible with surface IBM\n";
        return std::nullopt;
    }
    return backend;
}

QString backendName(SolverBackend backend) {
    return backend == SolverBackend::Culbm
        ? QStringLiteral("CULBM")
        : QStringLiteral("FluidX3D");
}

QString backendDirectory(SolverBackend backend) {
    return backend == SolverBackend::Culbm
        ? QStringLiteral("core/cfd_core/CULBM/bin")
        : QStringLiteral("core/cfd_core/FluidX3D/bin");
}

QString backendExecutableName(SolverBackend backend) {
    QString name = backendName(backend);
#ifdef _WIN32
    name += QStringLiteral(".exe");
#endif
    return name;
}

void forwardAvailable(QProcess& process, std::ostream& output, std::ostream& error) {
    const QByteArray standardOutput = process.readAllStandardOutput();
    if (!standardOutput.isEmpty()) {
        output.write(standardOutput.constData(), static_cast<std::streamsize>(standardOutput.size()));
        output.flush();
    }
    const QByteArray standardError = process.readAllStandardError();
    if (!standardError.isEmpty()) {
        error.write(standardError.constData(), static_cast<std::streamsize>(standardError.size()));
        error.flush();
    }
}

} // namespace

int run_runluw(const CommandArguments& arguments,
               std::ostream& output,
               std::ostream& error) {
    if (arguments.size() == 1U
        && (arguments.front() == "--help" || arguments.front() == "-h")) {
        output << "Usage: runluw <path-to-deck> [solver-arguments]\n\n"
                  "Selects the CFD backend from cfd_solver in the deck.\n"
                  "The default is CULBM; FluidX3D must be selected explicitly.\n"
                  "Surface IBM (ibm_enabler=true) requires CULBM.\n";
        return 0;
    }
    if (arguments.empty() || arguments.front().empty()) {
        error << "ERROR: runluw requires a deck path\n"
                 "Usage: runluw <path-to-deck> [solver-arguments]\n";
        return 2;
    }

    const QFileInfo deckInfo(QString::fromStdString(arguments.front()));
    const QString deckPath = deckInfo.absoluteFilePath();
    if (!QFileInfo(deckPath).isFile()) {
        error << "ERROR: deck not found at: " << deckPath.toUtf8().toStdString() << '\n';
        return 2;
    }

    std::optional<SolverBackend> backend;
    try {
        backend = selectSolver(DeckDocument::load(deckPath), error);
    } catch (const std::exception& exception) {
        error << "ERROR: cannot read solver configuration from "
              << deckPath.toUtf8().toStdString() << ": " << exception.what() << '\n';
        return 2;
    }
    if (!backend) return 2;

    const QString home = detectInstallRoot();
    const QString executableName = backendExecutableName(*backend);
    const QString executable = QDir(home).filePath(
        backendDirectory(*backend) + QLatin1Char('/') + executableName);
    if (!QFileInfo(executable).isFile()) {
        error << "ERROR: selected " << backendName(*backend).toStdString()
              << " executable not found at: "
              << executable.toUtf8().toStdString()
              << ". Please verify LUW_HOME and the installation layout.\n";
        return 1;
    }

    QStringList forwarded{deckPath};
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        forwarded.push_back(QString::fromStdString(arguments[index]));
    }

    QProcess process;
    process.setProgram(executable);
    process.setArguments(forwarded);
    process.setWorkingDirectory(QFileInfo(executable).absolutePath());
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("LUW_HOME"), home);
    process.setProcessEnvironment(environment);
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.setInputChannelMode(QProcess::ForwardedInputChannel);
    process.start();
    if (!process.waitForStarted(30000)) {
        error << "Failed to start " << executableName.toStdString() << ": "
              << process.errorString().toUtf8().toStdString() << '\n';
        return 1;
    }

    while (process.state() != QProcess::NotRunning) {
        process.waitForReadyRead(100);
        forwardAvailable(process, output, error);
    }
    forwardAvailable(process, output, error);
    return process.exitStatus() == QProcess::NormalExit ? process.exitCode() : 1;
}

} // namespace luw::cli
