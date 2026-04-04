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
    return QStringLiteral(R"PY(import contextlib
import importlib
import io
import json
import re
import sys
from pathlib import Path

IMPORT_NAME_MAP = {
    "netcdf4": "netCDF4",
}


def sanitize_requirement(line: str) -> str | None:
    raw = line.split("#", 1)[0].strip()
    if not raw:
        return None
    return re.split(r"[<>=!~;\[]", raw, maxsplit=1)[0].strip()


def candidate_modules(package_name: str) -> list[str]:
    normalized = package_name.replace("-", "_")
    mapped = IMPORT_NAME_MAP.get(package_name.lower(), normalized)
    candidates = []
    for value in (mapped, normalized, package_name):
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def try_import(module_name: str) -> tuple[bool, str]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            importlib.import_module(module_name)
        return True, ""
    except Exception as exc:
        details = []
        std_out = stdout_buffer.getvalue().strip()
        std_err = stderr_buffer.getvalue().strip()
        if std_out:
            details.append(std_out)
        if std_err:
            details.append(std_err)
        details.append(str(exc))
        return False, " | ".join(part for part in details if part)


requirements_path = Path(sys.argv[1])
checks = []
for line in requirements_path.read_text(encoding="utf-8").splitlines():
    package_name = sanitize_requirement(line)
    if not package_name:
        continue

    chosen_module = ""
    errors = []
    success = False
    for module_name in candidate_modules(package_name):
        chosen_module = module_name
        ok, error_text = try_import(module_name)
        if ok:
            success = True
            errors = []
            break
        errors.append(f"{module_name}: {error_text}")

    checks.append(
        {
            "package": package_name,
            "module": chosen_module,
            "success": success,
            "error": " || ".join(errors),
        }
    )

payload = {
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "checks": checks,
}
print(json.dumps(payload, ensure_ascii=False))
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
    return false;
}

QString StartupCheckResult::warningText() const {
    QStringList lines;
    lines << QStringLiteral("Startup finished, but the Python environment self-check reported problems.");
    lines << QStringLiteral("The GUI continued to load, but preprocessing, solver orchestration, or post-processing commands may fail.");
    lines << "";
    lines << QStringLiteral("Repository root: ") + QDir::toNativeSeparators(repoRoot);
    lines << QStringLiteral("Python interpreter: ") + (pythonExecutable.isEmpty() ? QStringLiteral("Unresolved.") : QDir::toNativeSeparators(pythonExecutable));
    if (!pythonVersion.isEmpty()) {
        lines << QStringLiteral("Python version: ") + pythonVersion;
    }
    if (!requirementsPath.isEmpty()) {
        lines << QStringLiteral("Requirements file: ") + QDir::toNativeSeparators(requirementsPath);
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
        result.processError = QStringLiteral("Could not find installer/requirements.txt, so the dependency import self-check could not run.");
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

    return result;
}

}
