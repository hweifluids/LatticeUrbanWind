#include "luw/accelerator_report_command.hpp"

#include <QByteArray>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#ifndef LUW_HAS_OPENCL
#define LUW_HAS_OPENCL 0
#endif

#ifndef LUW_ACCELERATOR_REPORT_TEST_WORK_ROOT
#define LUW_ACCELERATOR_REPORT_TEST_WORK_ROOT "accelerator-report-retained-fixtures"
#endif

namespace {

int failures = 0;

constexpr std::string_view kLegacyHelp =
    "usage: accelerator_runtime.py [-h] [--prepare-cuda-runtime]\n"
    "                              [--startup-report STARTUP_REPORT] [--json]\n"
    "\n"
    "Prepare accelerator runtimes and print startup diagnostics.\n"
    "\n"
    "options:\n"
    "  -h, --help            show this help message and exit\n"
    "  --prepare-cuda-runtime\n"
    "                        Prepare the local CUDA runtime layout and print its\n"
    "                        summary.\n"
    "  --startup-report STARTUP_REPORT\n"
    "                        Requirements file used to build a startup diagnostics\n"
    "                        JSON report.\n"
    "  --json                Print the result as JSON.\n";

constexpr std::string_view kLegacyUsage =
    "usage: accelerator_runtime.py [-h] [--prepare-cuda-runtime]\n"
    "                              [--startup-report STARTUP_REPORT] [--json]\n";

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

QJsonObject parse_object(const std::string& text, const char* message) {
    QJsonParseError error{};
    const QJsonDocument document = QJsonDocument::fromJson(
        QByteArray::fromStdString(text), &error);
    expect(error.error == QJsonParseError::NoError, message);
    expect(document.isObject(), message);
    return document.object();
}

} // namespace

int main() {
    const QByteArray cudaHomeBefore = qgetenv("CUDA_HOME");
    const QByteArray cudaPathBefore = qgetenv("CUDA_PATH");
    const QByteArray pathBefore = qgetenv("PATH");
    const QString untouchedPath =
        QStringLiteral(LUW_ACCELERATOR_REPORT_TEST_WORK_ROOT) +
        QStringLiteral("/must-not-be-created/requirements.txt");
    expect(!QFileInfo::exists(untouchedPath), "fixture path starts absent");

    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--prepare-cuda-runtime", "--json"}, output, error);
        expect(status == 0, "read-only prepare compatibility status");
        expect(error.str().empty(), "read-only prepare compatibility stderr");
        const QJsonObject root = parse_object(output.str(), "prepare JSON parses");
        expect(!root.value(QStringLiteral("applicable")).toBool(true),
               "wheel preparation is explicitly not applicable");
        expect(!root.value(QStringLiteral("prepared")).toBool(true),
               "wheel preparation is not claimed");
        expect(!root.value(QStringLiteral("files_written")).toBool(true),
               "prepare mode declares no writes");
        expect(!root.value(QStringLiteral("environment_modified")).toBool(true),
               "prepare mode declares no environment mutation");
        expect(root.value(QStringLiteral("cuda")).isObject(),
               "prepare mode includes native CUDA driver report");
    }

    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--startup-report", untouchedPath.toUtf8().toStdString(), "--json"},
            output, error);
        expect(status == 0, "startup report status");
        expect(error.str().empty(), "startup report stderr");
        const QJsonObject root = parse_object(output.str(), "startup JSON parses");
        expect(root.value(QStringLiteral("schema_version")).toInt() == 1,
               "schema version");
        expect(!root.value(QStringLiteral("host_platform")).toString().isEmpty(),
               "host platform");
        expect(root.value(QStringLiteral("python_executable")).toString().isEmpty(),
               "Python executable is explicitly absent");
        expect(root.value(QStringLiteral("python_version")).toString().isEmpty(),
               "Python version is explicitly absent");
        expect(root.value(QStringLiteral("requirements_path")).toString() ==
                   QFileInfo(untouchedPath).absoluteFilePath(),
               "legacy requirements path is echoed without reading");
        expect(root.value(QStringLiteral("checks")).toArray().isEmpty(),
               "Python package checks are empty");
        const QJsonObject requirements =
            root.value(QStringLiteral("requirements_summary")).toObject();
        expect(requirements.value(QStringLiteral("total")).toInt(-1) == 0,
               "native requirements total");
        expect(requirements.value(QStringLiteral("text")).toString().contains(
                   QStringLiteral("not applicable"), Qt::CaseInsensitive),
               "native requirements explanation");
        const QJsonObject contract =
            root.value(QStringLiteral("native_runtime")).toObject();
        expect(contract.value(QStringLiteral("python_free")).toBool(false),
               "native report marks Python-free runtime");
        expect(!contract.value(
                   QStringLiteral("cuda_wheel_preparation_applicable")).toBool(true),
               "native report marks wheel preparation inapplicable");
        const QJsonObject effects =
            root.value(QStringLiteral("side_effects")).toObject();
        expect(!effects.value(QStringLiteral("files_written")).toBool(true),
               "startup report declares no writes");
        expect(!effects.value(QStringLiteral("environment_modified")).toBool(true),
               "startup report declares no environment mutation");

        const QJsonObject cuda = root.value(QStringLiteral("cuda")).toObject();
        expect(cuda.contains(QStringLiteral("available")), "CUDA availability field");
        expect(cuda.value(QStringLiteral("devices")).isArray(), "CUDA devices array");
        expect(!cuda.value(QStringLiteral("summary")).toString().isEmpty(),
               "CUDA summary");
        if (cuda.value(QStringLiteral("available")).toBool(false)) {
            const QJsonArray devices = cuda.value(QStringLiteral("devices")).toArray();
            expect(!devices.isEmpty(),
                   "available CUDA report has devices");
            for (const QJsonValue& value : devices) {
                const QJsonObject device = value.toObject();
                expect(!device.value(QStringLiteral("name")).toString().isEmpty(),
                       "legacy CUDA device name value");
                expect(device.value(QStringLiteral("vendor")).toString() ==
                           QStringLiteral("NVIDIA"),
                       "legacy CUDA device vendor value");
                expect(device.value(QStringLiteral("version")).toString() ==
                           QStringLiteral("CUDA"),
                       "legacy CUDA device version value");
                expect(device.value(QStringLiteral("driver_version")).toString() ==
                           cuda.value(QStringLiteral("driver_version")).toString(),
                       "legacy CUDA device driver value");
                expect(!device.value(QStringLiteral("compute_capability"))
                            .toString().isEmpty(),
                       "legacy CUDA compute capability value");
                expect(device.value(QStringLiteral("compute_units")).isNull(),
                       "legacy CUDA compute units remain null");
            }
        }

        const QJsonObject opencl = root.value(QStringLiteral("opencl")).toObject();
        expect(opencl.contains(QStringLiteral("available")), "OpenCL availability field");
        expect(opencl.value(QStringLiteral("platforms")).isArray(),
               "OpenCL platforms array");
        expect(opencl.value(QStringLiteral("devices")).isArray(),
               "OpenCL devices array");
        expect(!opencl.value(QStringLiteral("summary")).toString().isEmpty(),
               "OpenCL summary");
        if (opencl.value(QStringLiteral("available")).toBool(false)) {
            const QJsonArray platforms =
                opencl.value(QStringLiteral("platforms")).toArray();
            const QJsonArray devices = opencl.value(QStringLiteral("devices")).toArray();
            expect(!platforms.isEmpty(),
                   "available OpenCL report has platforms");
            expect(!devices.isEmpty(), "available OpenCL report has devices");
            int declaredDeviceCount = 0;
            for (const QJsonValue& value : platforms) {
                const QJsonObject platform = value.toObject();
                expect(!platform.value(QStringLiteral("name")).toString().isEmpty(),
                       "legacy OpenCL platform name value");
                expect(!platform.value(QStringLiteral("vendor")).toString().isEmpty(),
                       "legacy OpenCL platform vendor value");
                expect(!platform.value(QStringLiteral("version")).toString().isEmpty(),
                       "legacy OpenCL platform version value");
                expect(platform.value(QStringLiteral("device_count")).isDouble(),
                       "legacy OpenCL platform device count");
                declaredDeviceCount +=
                    platform.value(QStringLiteral("device_count")).toInt();
            }
            expect(declaredDeviceCount == devices.size(),
                   "legacy platform counts match flattened devices");
            const QJsonObject primary = devices.first().toObject();
            expect(primary.contains(QStringLiteral("type")), "OpenCL device type extension");
            expect(primary.contains(QStringLiteral("max_clock_mhz")),
                   "OpenCL device clock extension");
            expect(primary.contains(QStringLiteral("global_memory_bytes")),
                   "OpenCL device memory extension");
            for (const QJsonValue& value : devices) {
                const QJsonObject device = value.toObject();
                expect(device.value(QStringLiteral("name")).isString(),
                       "legacy OpenCL device name");
                expect(!device.value(QStringLiteral("name")).toString().isEmpty(),
                       "legacy OpenCL device name value");
                expect(device.value(QStringLiteral("vendor")).isString(),
                       "legacy OpenCL device vendor");
                expect(!device.value(QStringLiteral("vendor")).toString().isEmpty(),
                       "legacy OpenCL device vendor value");
                expect(device.value(QStringLiteral("version")).isString(),
                       "legacy OpenCL device version");
                expect(!device.value(QStringLiteral("version")).toString().isEmpty(),
                       "legacy OpenCL device version value");
                expect(device.value(QStringLiteral("driver_version")).isString(),
                       "legacy OpenCL device driver version");
                expect(!device.value(QStringLiteral("driver_version")).toString().isEmpty(),
                       "legacy OpenCL device driver version value");
                expect(device.value(QStringLiteral("compute_units")).isDouble(),
                       "legacy OpenCL device compute units");
                expect(device.value(QStringLiteral("compute_units")).toInt() > 0,
                       "legacy OpenCL device compute units value");
                expect(device.value(QStringLiteral("compute_capability")).isString(),
                       "legacy OpenCL device compute capability");
                expect(device.value(QStringLiteral("compute_capability"))
                           .toString().isEmpty(),
                       "legacy OpenCL compute capability remains empty");
            }
        }
#if !LUW_HAS_OPENCL
        expect(opencl.value(QStringLiteral("warning")).toBool(false),
               "no-SDK OpenCL warning");
        expect(opencl.value(QStringLiteral("summary")).toString().contains(
                   QStringLiteral("no OpenCL support")),
               "no-SDK OpenCL explanation");
#endif
    }

    expect(qgetenv("CUDA_HOME") == cudaHomeBefore, "CUDA_HOME remains unchanged");
    expect(qgetenv("CUDA_PATH") == cudaPathBefore, "CUDA_PATH remains unchanged");
    expect(qgetenv("PATH") == pathBefore, "PATH remains unchanged");
    expect(!QFileInfo::exists(untouchedPath), "report creates no fixture path");

    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--startup-report", untouchedPath.toUtf8().toStdString(),
             "--prepare-cuda-runtime", "--json"}, output, error);
        expect(status == 0, "legacy prepare precedence status");
        const QJsonObject root = parse_object(output.str(), "prepare precedence JSON parses");
        expect(root.contains(QStringLiteral("applicable")),
               "prepare mode takes precedence over startup report");
        expect(!root.contains(QStringLiteral("schema_version")),
               "prepare mode omits startup schema");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report({}, output, error);
        expect(status == 0, "default help status");
        expect(output.str() == kLegacyHelp, "default exact legacy help");
        expect(error.str().empty(), "default help stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status =
            luw::cli::run_accelerator_report({"--json"}, output, error);
        expect(status == 0, "lone JSON flag help status");
        expect(output.str() == kLegacyHelp, "lone JSON exact legacy help");
        expect(error.str().empty(), "lone JSON flag help stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--prepare-cuda-runtime"}, output, error);
        expect(status == 0, "silent prepare status");
        expect(output.str().empty(), "prepare without JSON is silent");
        expect(error.str().empty(), "silent prepare stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--startup-report", untouchedPath.toUtf8().toStdString()},
            output, error);
        expect(status == 0, "silent startup report status");
        expect(output.str().empty(), "startup report without JSON is silent");
        expect(error.str().empty(), "silent startup report stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--unknown"}, output, error);
        expect(status == 2, "unknown option status");
        expect(output.str().empty(), "unknown option stdout");
        expect(error.str() == std::string(kLegacyUsage) +
                   "accelerator_runtime.py: error: unrecognized arguments: --unknown\n",
               "unknown option exact argparse stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--startup-report"}, output, error);
        expect(status == 2, "missing startup-report path status");
        expect(output.str().empty(), "missing startup-report stdout");
        expect(error.str() == std::string(kLegacyUsage) +
                   "accelerator_runtime.py: error: argument --startup-report: "
                   "expected one argument\n",
               "missing startup-report exact argparse stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status =
            luw::cli::run_accelerator_report({"--help"}, output, error);
        expect(status == 0, "help status");
        expect(output.str() == kLegacyHelp, "explicit exact legacy help");
        expect(error.str().empty(), "help stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--unknown", "--help"}, output, error);
        expect(status == 0, "help action overrides prior unknown argument");
        expect(output.str() == kLegacyHelp, "mixed help exact legacy output");
        expect(error.str().empty(), "mixed help stderr");
    }
    {
        std::ostringstream output;
        std::ostringstream error;
        const int status = luw::cli::run_accelerator_report(
            {"--startup-report", "--help"}, output, error);
        expect(status == 2, "missing value before help retains argparse error");
        expect(output.str().empty(), "missing value before help stdout");
        expect(error.str() == std::string(kLegacyUsage) +
                   "accelerator_runtime.py: error: argument --startup-report: "
                   "expected one argument\n",
               "missing value before help exact argparse stderr");
    }

    expect(qgetenv("CUDA_HOME") == cudaHomeBefore,
           "CUDA_HOME remains unchanged after every mode");
    expect(qgetenv("CUDA_PATH") == cudaPathBefore,
           "CUDA_PATH remains unchanged after every mode");
    expect(qgetenv("PATH") == pathBefore,
           "PATH remains unchanged after every mode");
    expect(!QFileInfo::exists(untouchedPath),
           "no mode creates the requirements fixture path");

    if (failures == 0) {
        std::cout << "accelerator_report compatibility tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
