#include "luw/runluw_command.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireContains(const std::string& text,
                     const std::string& expected,
                     const std::string& message) {
    require(text.find(expected) != std::string::npos, message + ": " + text);
}

void requireNotContains(const std::string& text,
                        const std::string& unexpected,
                        const std::string& message) {
    require(text.find(unexpected) == std::string::npos, message + ": " + text);
}

class EnvironmentGuard final {
public:
    explicit EnvironmentGuard(const char* name)
        : name_(name), existed_(qEnvironmentVariableIsSet(name)), value_(qgetenv(name)) {}
    ~EnvironmentGuard() {
        if (existed_) qputenv(name_.constData(), value_);
        else qunsetenv(name_.constData());
    }

private:
    QByteArray name_;
    bool existed_;
    QByteArray value_;
};

class CurrentDirectoryGuard final {
public:
    CurrentDirectoryGuard() : original_(QDir::currentPath()) {}
    ~CurrentDirectoryGuard() { QDir::setCurrent(original_); }

private:
    QString original_;
};

QString executableName(const QString& baseName) {
#ifdef _WIN32
    return baseName + QStringLiteral(".exe");
#else
    return baseName;
#endif
}

QString fakeSolverName() {
    const QString base = QFileInfo(QCoreApplication::applicationFilePath()).completeBaseName();
    if (base.compare(QStringLiteral("CULBM"), Qt::CaseInsensitive) == 0) {
        return QStringLiteral("CULBM");
    }
    if (base.compare(QStringLiteral("FluidX3D"), Qt::CaseInsensitive) == 0) {
        return QStringLiteral("FluidX3D");
    }
    return {};
}

int runFakeSolver(const QString& solverName, const QStringList& arguments) {
    std::cout << "FAKE_SOLVER=" << solverName.toUtf8().constData() << '\n';
    std::cout << "FAKE_SOLVER_CWD=" << QDir::currentPath().toUtf8().constData() << '\n';
    for (qsizetype index = 1; index < arguments.size(); ++index) {
        std::cout << "FAKE_SOLVER_ARG=" << arguments.at(index).toUtf8().constData() << '\n';
    }
    return solverName == QStringLiteral("CULBM") ? 41 : 42;
}

QString installFakeSolver(const QTemporaryDir& root, const QString& solverName) {
    const QString directory = root.filePath(
        QStringLiteral("core/cfd_core/") + solverName + QStringLiteral("/bin"));
    require(QDir().mkpath(directory), "failed to create fake solver directory");
    const QString target = QDir(directory).filePath(executableName(solverName));
    require(QFile::copy(QCoreApplication::applicationFilePath(), target),
            "failed to install fake solver executable");
    require(QFile::setPermissions(
                target, QFile::permissions(QCoreApplication::applicationFilePath())),
            "failed to set fake solver permissions");
    return target;
}

void installDeckSchema(const QTemporaryDir& root) {
    const QString source = QDir(QStringLiteral(LUW_SOURCE_ROOT))
                               .filePath(QStringLiteral("core/deck_schema.json"));
    const QString target = root.filePath(QStringLiteral("core/deck_schema.json"));
    require(QDir().mkpath(QFileInfo(target).absolutePath()),
            "failed to create temporary schema directory");
    require(QFile::copy(source, target), "failed to install temporary deck schema");
}

QString writeDeck(const QString& directory,
                  const QString& fileName,
                  const QString& contents) {
    const QString path = QDir(directory).filePath(fileName);
    QFile deck(path);
    require(deck.open(QIODevice::WriteOnly | QIODevice::Truncate),
            "failed to create temporary deck");
    const QByteArray encoded = contents.toUtf8();
    require(deck.write(encoded) == encoded.size(), "failed to write temporary deck");
    deck.close();
    return path;
}

struct InvocationResult {
    int code = -1;
    std::string output;
    std::string error;
};

InvocationResult invoke(const luw::cli::CommandArguments& arguments) {
    std::ostringstream output;
    std::ostringstream error;
    const int code = luw::cli::run_runluw(arguments, output, error);
    return InvocationResult{code, output.str(), error.str()};
}

} // namespace

int main(int argc, char* argv[]) {
    QCoreApplication application(argc, argv);
    const QString solverName = fakeSolverName();
    if (!solverName.isEmpty()) {
        return runFakeSolver(solverName, QCoreApplication::arguments());
    }

    try {
        QTemporaryDir root;
        require(root.isValid(), "failed to create temporary LUW_HOME");
        installDeckSchema(root);
        const QString culbmPath = installFakeSolver(root, QStringLiteral("CULBM"));
        const QString fluidPath = installFakeSolver(root, QStringLiteral("FluidX3D"));
        const QString culbmDirectory = QFileInfo(culbmPath).absolutePath();
        const QString fluidDirectory = QFileInfo(fluidPath).absolutePath();

        const QString caseDirectory = root.filePath(QStringLiteral("case"));
        require(QDir().mkpath(caseDirectory), "failed to create temporary case directory");

        const QString defaultDeck = writeDeck(
            caseDirectory, QStringLiteral("default.luw"), QStringLiteral("// defaults\n"));
        const QString ibmDeck = writeDeck(
            caseDirectory, QStringLiteral("ibm.luw"),
            QStringLiteral("cfd_solver = culbm\nibm_enabler = true\n"));
        writeDeck(caseDirectory, QStringLiteral("fluid.luw"),
                  QStringLiteral("cfd_solver = fluidx3d\nibm_enabler = false\n"));
        writeDeck(caseDirectory, QStringLiteral("conflict.luw"),
                  QStringLiteral("cfd_solver = fluidx3d\nibm_enabler = true\n"));
        writeDeck(caseDirectory, QStringLiteral("invalid-solver.luw"),
                  QStringLiteral("cfd_solver = unknown\n"));
        writeDeck(caseDirectory, QStringLiteral("blank-solver.luw"),
                  QStringLiteral("cfd_solver =\n"));
        writeDeck(caseDirectory, QStringLiteral("invalid-ibm.luw"),
                  QStringLiteral("ibm_enabler = perhaps\n"));

        EnvironmentGuard environment("LUW_HOME");
        CurrentDirectoryGuard currentDirectory;
        qputenv("LUW_HOME", root.path().toUtf8());
        require(QDir::setCurrent(caseDirectory), "failed to enter temporary case directory");

        const InvocationResult defaultResult = invoke({"default.luw", "--probe"});
        require(defaultResult.code == 41, "missing cfd_solver did not select CULBM");
        require(defaultResult.error.empty(), "default run unexpectedly wrote to stderr");
        requireContains(defaultResult.output, "FAKE_SOLVER=CULBM",
                        "default run did not launch CULBM");
        requireContains(defaultResult.output,
                        "FAKE_SOLVER_CWD=" + culbmDirectory.toStdString(),
                        "CULBM working directory was not preserved");
        requireContains(defaultResult.output,
                        QFileInfo(defaultDeck).absoluteFilePath().toStdString(),
                        "relative deck was not forwarded as an absolute path");
        requireContains(defaultResult.output, "FAKE_SOLVER_ARG=--probe",
                        "additional solver argument was not forwarded");

        const InvocationResult ibmResult = invoke({"ibm.luw"});
        require(ibmResult.code == 41, "IBM-enabled CULBM run failed");
        requireContains(ibmResult.output, "FAKE_SOLVER=CULBM",
                        "IBM-enabled run did not launch CULBM");
        requireContains(ibmResult.output, QFileInfo(ibmDeck).absoluteFilePath().toStdString(),
                        "IBM deck path was not forwarded");

        const InvocationResult fluidResult = invoke({"fluid.luw"});
        require(fluidResult.code == 42, "explicit FluidX3D run failed");
        require(fluidResult.error.empty(), "FluidX3D run unexpectedly wrote to stderr");
        requireContains(fluidResult.output, "FAKE_SOLVER=FluidX3D",
                        "explicit selection did not launch FluidX3D");
        requireContains(fluidResult.output,
                        "FAKE_SOLVER_CWD=" + fluidDirectory.toStdString(),
                        "FluidX3D working directory was not preserved");

        const InvocationResult conflictResult = invoke({"conflict.luw"});
        require(conflictResult.code == 2, "IBM/FluidX3D conflict was not rejected");
        requireContains(conflictResult.error, "ibm_enabler=true requires cfd_solver=culbm",
                        "IBM/FluidX3D conflict error was not actionable");
        requireNotContains(conflictResult.output, "FAKE_SOLVER=",
                           "conflicting configuration launched a solver");

        const InvocationResult invalidSolverResult = invoke({"invalid-solver.luw"});
        require(invalidSolverResult.code == 2, "invalid cfd_solver was not rejected");
        requireContains(invalidSolverResult.error, "invalid cfd_solver='unknown'",
                        "invalid solver error omitted the configured value");

        const InvocationResult blankSolverResult = invoke({"blank-solver.luw"});
        require(blankSolverResult.code == 41, "blank cfd_solver did not default to CULBM");
        require(blankSolverResult.error.empty(),
                "blank cfd_solver default unexpectedly wrote to stderr");
        requireContains(blankSolverResult.output, "FAKE_SOLVER=CULBM",
                        "blank cfd_solver did not launch CULBM");

        const InvocationResult invalidIbmResult = invoke({"invalid-ibm.luw"});
        require(invalidIbmResult.code == 2, "invalid ibm_enabler was not rejected");
        requireContains(invalidIbmResult.error, "invalid ibm_enabler='perhaps'",
                        "invalid IBM error omitted the configured value");

        const InvocationResult missingDeckResult = invoke({"missing.luw"});
        require(missingDeckResult.code == 2, "missing deck did not return a usage error");
        requireContains(missingDeckResult.error, "deck not found",
                        "missing deck error was not actionable");

        const InvocationResult noArgumentsResult = invoke({});
        require(noArgumentsResult.code == 2, "missing deck argument was not rejected");
        requireContains(noArgumentsResult.error, "requires a deck path",
                        "missing argument error was not actionable");

        const InvocationResult helpResult = invoke({"--help"});
        require(helpResult.code == 0, "runluw help failed");
        requireContains(helpResult.output, "The default is CULBM",
                        "runluw help omitted the default solver");

        require(QFile::remove(culbmPath), "failed to remove fake CULBM executable");
        const InvocationResult noFallbackResult = invoke({"default.luw"});
        require(noFallbackResult.code == 1,
                "missing default CULBM executable unexpectedly fell back to FluidX3D");
        requireContains(noFallbackResult.error, "selected CULBM executable not found",
                        "missing CULBM error did not identify the selected backend");
        requireNotContains(noFallbackResult.output, "FAKE_SOLVER=FluidX3D",
                           "missing CULBM executable fell back to FluidX3D");

        std::cout << "runluw solver-selection tests passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "runluw solver-selection tests failed: " << exception.what() << '\n';
        return 1;
    }
}
