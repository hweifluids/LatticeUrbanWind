#include "luw/cleanluw_command.hpp"

#include <QString>
#include <QTemporaryDir>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

std::string utf8Path(const fs::path& value) {
#ifdef _WIN32
    return QString::fromStdWString(value.wstring()).toUtf8().toStdString();
#else
    return value.string();
#endif
}

fs::path nativePath(const QString& value) {
#ifdef _WIN32
    return fs::path(value.toStdWString());
#else
    return fs::path(value.toUtf8().constData());
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void cleanRemovesOnlyFilesBelowProjectTemporaryDirectory() {
    QTemporaryDir temporary;
    require(temporary.isValid(), "failed to create cleanluw fixture");
    const fs::path root = nativePath(temporary.path());
    const fs::path deck = root / "conf.luw";
    const fs::path projectTemporary = root / "proj_temp";
    const fs::path nested = projectTemporary / "nested";
    fs::create_directories(nested);
    std::ofstream(deck) << "flag = true\n";
    std::ofstream(projectTemporary / "root.tmp") << "root";
    std::ofstream(nested / "nested.tmp") << "nested";
    std::ofstream(root / "keep.txt") << "keep";

    std::ostringstream output;
    std::ostringstream error;
    const int exitCode = luw::cli::run_cleanluw({utf8Path(deck)}, output, error);

    require(exitCode == 0, "cleanluw should succeed");
    require(output.str().empty(), "cleanluw unexpectedly wrote stdout");
    require(error.str().empty(), "cleanluw unexpectedly wrote stderr");
    require(!fs::exists(projectTemporary / "root.tmp"), "root file was not removed");
    require(!fs::exists(nested / "nested.tmp"), "nested file was not removed");
    require(fs::is_directory(projectTemporary), "proj_temp directory was removed");
    require(fs::is_directory(nested), "nested directory was removed");
    require(fs::is_regular_file(root / "keep.txt"), "file outside proj_temp was removed");
}

void invalidDeckMatchesLegacyFailureShape() {
    QTemporaryDir temporary;
    require(temporary.isValid(), "failed to create invalid cleanluw fixture");
    const fs::path root = nativePath(temporary.path());
    std::ostringstream output;
    std::ostringstream error;
    const int exitCode = luw::cli::run_cleanluw(
        {utf8Path(root / "missing.luw")}, output, error);

    require(exitCode == 1, "missing deck should fail");
    require(output.str().empty(), "missing deck unexpectedly wrote stdout");
    require(error.str().find(" is not a valid file\n") != std::string::npos,
            "missing deck failure text changed");
}

} // namespace

int main() {
    try {
        cleanRemovesOnlyFilesBelowProjectTemporaryDirectory();
        invalidDeckMatchesLegacyFailureShape();
        std::cout << "cleanluw native tests passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "cleanluw native tests failed: " << exception.what() << '\n';
        return 1;
    }
}
