#include "luwgui/StartupDiagnostics.h"

#include "luwgui/RuntimePaths.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStringList>

namespace luwgui {

namespace {

QString pythonCheckScript() {
    return QStringLiteral(R"PY(import json
import sys
from pathlib import Path

requirements_path = Path(sys.argv[1]).resolve()
repo_root = requirements_path.parent.parent
core_dir = repo_root / "core"
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

from accelerator_runtime import build_startup_report

print(json.dumps(build_startup_report(requirements_path), ensure_ascii=False))
)PY");
}

QString buildProcessErrorMessage(const StartupCheckResult& result) {
    QStringList lines;
    if (!result.processError.trimmed().isEmpty()) {
        lines << result.processError.trimmed();
    }
    if (!result.rawStderr.trimmed().isEmpty()) {
        lines << result.rawStderr.trimmed();
    }
    if (!result.rawStdout.trimmed().isEmpty()) {
        lines << result.rawStdout.trimmed();
    }
    return lines.join('\n').trimmed();
}

AcceleratorDeviceCheck parseDevice(const QJsonObject& object) {
    AcceleratorDeviceCheck device;
    device.name = object.value(QStringLiteral("name")).toString();
    device.vendor = object.value(QStringLiteral("vendor")).toString();
    device.version = object.value(QStringLiteral("version")).toString();
    device.driverVersion = object.value(QStringLiteral("driver_version")).toString();
    device.computeCapability = object.value(QStringLiteral("compute_capability")).toString();
    device.computeUnits = object.value(QStringLiteral("compute_units")).toInt(0);
    return device;
}

AcceleratorCheck parseAcceleratorCheck(const QJsonObject& object) {
    AcceleratorCheck check;
    check.summary = object.value(QStringLiteral("summary")).toString();
    check.errorText = object.value(QStringLiteral("error")).toString();
    check.provider = object.value(QStringLiteral("provider")).toString();
    check.sourceRoot = object.value(QStringLiteral("source_root")).toString();
    check.runtimeRoot = object.value(QStringLiteral("runtime_root")).toString();
    check.cudaHome = object.value(QStringLiteral("cuda_home")).toString();
    check.runtimeVersion = object.value(QStringLiteral("runtime_version")).toString();
    check.driverVersion = object.value(QStringLiteral("driver_version")).toString();
    check.numbaVersion = object.value(QStringLiteral("numba_version")).toString();
    check.version = object.value(QStringLiteral("version")).toString();
    check.configured = object.value(QStringLiteral("configured")).toBool(false);
    check.prepared = object.value(QStringLiteral("prepared")).toBool(false);
    check.available = object.value(QStringLiteral("available")).toBool(false);
    check.warning = object.value(QStringLiteral("warning")).toBool(false);

    const QJsonArray devices = object.value(QStringLiteral("devices")).toArray();
    check.devices.reserve(devices.size());
    for (const QJsonValue& value : devices) {
        if (!value.isObject()) {
            continue;
        }
        check.devices.push_back(parseDevice(value.toObject()));
    }
    return check;
}

} // namespace

bool StartupCheckResult::hasWarnings() const {
    if (!pythonResolved || !requirementsResolved || !processError.trimmed().isEmpty()) {
        return true;
    }
    for (const PythonImportCheck& check : packageChecks) {
        if (!check.success) {
            return true;
        }
    }
    return opencl.warning || cuda.warning;
}

QString StartupCheckResult::warningText() const {
    QStringList lines;
    lines << QStringLiteral("Startup finished, but the runtime environment self-check reported problems.");
    lines << QStringLiteral("The GUI continued to load, but preprocessing, solver orchestration, or post-processing commands may fail.");
    lines << "";
    lines << QStringLiteral("Repository root: ") + QDir::toNativeSeparators(repoRoot);
    lines << QStringLiteral("Python interpreter: ") + (pythonExecutable.isEmpty() ? QStringLiteral("Unresolved.") : QDir::toNativeSeparators(pythonExecutable));
    if (!pythonVersion.isEmpty()) {
        lines << QStringLiteral("Python version: ") + pythonVersion;
    }
    if (!hostPlatform.isEmpty()) {
        lines << QStringLiteral("Host platform: ") + hostPlatform;
    }
    if (!requirementsPath.isEmpty()) {
        lines << QStringLiteral("Requirements file: ") + QDir::toNativeSeparators(requirementsPath);
    }
    if (!pythonSummary.isEmpty()) {
        lines << QStringLiteral("Python environment: ") + pythonSummary;
    }

    const QString processMessage = buildProcessErrorMessage(*this);
    if (!processMessage.isEmpty()) {
        lines << "";
        lines << QStringLiteral("Self-check process details:");
        lines << processMessage;
    }

    QStringList failedPackages;
    for (const PythonImportCheck& check : packageChecks) {
        if (!check.success) {
            failedPackages << QStringLiteral("- %1 -> import %2 failed: %3")
                                  .arg(check.packageName,
                                       check.moduleName.isEmpty() ? QStringLiteral("?") : check.moduleName,
                                       check.errorText.isEmpty() ? QStringLiteral("Unknown error.") : check.errorText);
        }
    }
    if (!failedPackages.isEmpty()) {
        lines << "";
        lines << QStringLiteral("Packages that failed to import:");
        lines += failedPackages;
    }

    if (opencl.warning) {
        lines << "";
        lines << QStringLiteral("OpenCL:");
        lines << (opencl.summary.isEmpty() ? QStringLiteral("OpenCL is unavailable.") : opencl.summary);
        if (!opencl.errorText.isEmpty()) {
            lines << opencl.errorText;
        }
    }
    if (cuda.warning) {
        lines << "";
        lines << QStringLiteral("CUDA:");
        lines << (cuda.summary.isEmpty() ? QStringLiteral("CUDA is unavailable.") : cuda.summary);
        if (!cuda.errorText.isEmpty()) {
            lines << cuda.errorText;
        }
    }

    return lines.join('\n');
}

StartupCheckResult runPythonEnvironmentSelfCheck(const QString& repoRoot, int timeoutMs) {
    StartupCheckResult result;
    result.repoRoot = repoRoot;
    result.requirementsPath = resolveRepoFilePath(repoRoot, QStringLiteral("installer/requirements.txt"));
    result.requirementsResolved = QFileInfo::exists(result.requirementsPath);
    result.pythonExecutable = findPythonExecutable(repoRoot);
    result.pythonResolved = !result.pythonExecutable.trimmed().isEmpty();

    if (!result.requirementsResolved) {
        result.processError = QStringLiteral("Could not find installer/requirements.txt, so the startup dependency self-check could not run.");
        return result;
    }

    QProcess process;
    process.setWorkingDirectory(repoRoot);

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("LUW_HOME"), repoRoot);
    QString path = environment.value(QStringLiteral("PATH"));
    const QString binPath = resolveRepoFilePath(repoRoot, QStringLiteral("bin"));
    if (!path.contains(binPath, Qt::CaseInsensitive)) {
        path.prepend(binPath + QDir::listSeparator());
    }
    environment.insert(QStringLiteral("PATH"), path);
    process.setProcessEnvironment(environment);
    process.setProcessChannelMode(QProcess::SeparateChannels);

    process.start(result.pythonExecutable, {QStringLiteral("-"), result.requirementsPath});
    if (!process.waitForStarted(5000)) {
        result.processError = QStringLiteral("Failed to start the Python interpreter used for the self-check.");
        result.rawStderr = process.errorString();
        return result;
    }

    process.write(pythonCheckScript().toUtf8());
    process.closeWriteChannel();

    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        process.waitForFinished(3000);
        result.processError = QStringLiteral("The Python dependency import self-check timed out.");
        result.rawStdout = QString::fromUtf8(process.readAllStandardOutput());
        result.rawStderr = QString::fromUtf8(process.readAllStandardError());
        return result;
    }

    result.rawStdout = QString::fromUtf8(process.readAllStandardOutput());
    result.rawStderr = QString::fromUtf8(process.readAllStandardError());

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        result.processError = QStringLiteral("The Python dependency import self-check exited with a non-zero status.");
        return result;
    }

    QJsonParseError parseError{};
    const QJsonDocument document = QJsonDocument::fromJson(result.rawStdout.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        result.processError = QStringLiteral("Failed to parse the Python self-check result.");
        return result;
    }

    const QJsonObject root = document.object();
    result.pythonExecutable = root.value(QStringLiteral("python_executable")).toString(result.pythonExecutable);
    result.pythonVersion = root.value(QStringLiteral("python_version")).toString();
    result.hostPlatform = root.value(QStringLiteral("host_platform")).toString();

    const QJsonObject requirementsSummary = root.value(QStringLiteral("requirements_summary")).toObject();
    result.pythonSummary = requirementsSummary.value(QStringLiteral("text")).toString();

    const QJsonArray checks = root.value(QStringLiteral("checks")).toArray();
    result.packageChecks.reserve(checks.size());
    for (const QJsonValue& value : checks) {
        if (!value.isObject()) {
            continue;
        }
        const QJsonObject object = value.toObject();
        PythonImportCheck check;
        check.packageName = object.value(QStringLiteral("package")).toString();
        check.moduleName = object.value(QStringLiteral("module")).toString();
        check.success = object.value(QStringLiteral("success")).toBool(false);
        check.errorText = object.value(QStringLiteral("error")).toString();
        result.packageChecks.push_back(check);
    }

    result.cuda = parseAcceleratorCheck(root.value(QStringLiteral("cuda")).toObject());
    result.opencl = parseAcceleratorCheck(root.value(QStringLiteral("opencl")).toObject());
    return result;
}

} // namespace luwgui
