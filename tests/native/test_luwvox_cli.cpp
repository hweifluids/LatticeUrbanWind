#include "luw/luwvox_command.hpp"
#include "luw/luwvox_opencl.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Result {
    int status = -1;
    std::string output;
    std::string error;
};

Result run(const luw::cli::CommandArguments& arguments) {
    std::ostringstream output;
    std::ostringstream error;
    const int status = luw::cli::run_luwvox(arguments, output, error);
    return {status, output.str(), error.str()};
}

} // namespace

int main() {
    try {
        const Result help = run({"--help"});
        require(help.status == 0 && help.error.empty(),
                "--help did not exit cleanly");
        require(help.output.find("positional arguments:\n") != std::string::npos &&
                    help.output.find("--terrain-approach {idw,kriging}") !=
                        std::string::npos,
                "--help omitted argparse-compatible sections");
        require(help.output.find("--preprocess-use-gpu") == std::string::npos,
                "command-local help retained dispatcher-owned GPU options");

        const Result missing = run({});
        require(missing.status == 2 &&
                    missing.error.find("the following arguments are required: conf") !=
                        std::string::npos,
                "missing conf did not produce an argparse-compatible error");

        const Result choice = run({"--dem-source", "AUTO", "conf.luw"});
        require(choice.status == 2 &&
                    choice.error.find("invalid choice: 'AUTO'") != std::string::npos,
                "choice parsing became incorrectly case-insensitive");

        const Result removedLegacyChoice =
            run({"--terrain-approach", "kriging_gpu", "conf.luw"});
        require(removedLegacyChoice.status == 2 &&
                    removedLegacyChoice.error.find(
                        "choose from 'idw', 'kriging'") != std::string::npos,
                "removed kriging_gpu method remained accepted");

        const Result globalOptionBypass = run(
            {"--preprocess-use-gpu", "false", "conf.luw"});
        require(globalOptionBypass.status == 2 &&
                    globalOptionBypass.error.find(
                        "unrecognized arguments: --preprocess-use-gpu") !=
                        std::string::npos,
                "luwvox retained a second parser for dispatcher-owned GPU options");

        const Result emptyFloat =
            run({"--elevation-scale=", "conf.luw"});
        require(emptyFloat.status == 2 &&
                    emptyFloat.error.find("invalid float value: ''") !=
                        std::string::npos,
                "empty float option was not rejected");

        const Result unknown = run({"--unknown", "conf.luw"});
        require(unknown.status == 2 &&
                    unknown.error.find("unrecognized arguments: --unknown") !=
                        std::string::npos,
                "unknown option was not rejected");

        const Result plusInteger =
            run({"--idw-neighbors", "+4", "conf.luw"});
        require(plusInteger.status == 1 &&
                    plusInteger.error.find("requires GDAL/OGR with GEOS support") !=
                        std::string::npos,
                "Python-compatible signed integer was not accepted");

        const Result underscoredInteger =
            run({"--idw-neighbors", "1_000", "conf.luw"});
        require(underscoredInteger.status == 1 &&
                    underscoredInteger.error.find("requires GDAL/OGR") !=
                        std::string::npos,
                "Python-compatible underscored integer was not accepted");

        const Result missingStringValue =
            run({"--dem-path", "--unknown", "conf.luw"});
        require(missingStringValue.status == 2 &&
                    missingStringValue.error.find(
                        "argument --dem-path: expected one argument") !=
                        std::string::npos,
                "option-looking string value was incorrectly consumed");

        const Result abbreviated =
            run({"--dem-s", "auto", "conf.luw"});
        require(abbreviated.status == 1 &&
                    abbreviated.error.find("requires GDAL/OGR") !=
                        std::string::npos,
                "unique argparse long-option abbreviation was rejected");

        const Result ambiguous = run({"--dem", "auto", "conf.luw"});
        require(ambiguous.status == 2 &&
                    ambiguous.error.find(
                        "ambiguous option: --dem could match --dem-path, --dem-source") !=
                        std::string::npos,
                "ambiguous argparse long-option abbreviation was not diagnosed");

        const Result abbreviatedHelp = run({"--hel"});
        require(abbreviatedHelp.status == 0 &&
                    abbreviatedHelp.output.find("positional arguments:\n") !=
                        std::string::npos,
                "unique --help abbreviation did not display help");

        const Result explicitFlagValue = run({"--no=x", "conf.luw"});
        require(explicitFlagValue.status == 2 &&
                    explicitFlagValue.error.find(
                        "argument --no-reproject: ignored explicit argument 'x'") !=
                        std::string::npos,
                "explicit value on an argparse flag was not diagnosed");

        const Result delimiter = run({"--", "--help"});
        require(delimiter.status == 1 && delimiter.output.empty(),
                "-- delimiter did not suppress option parsing");

        bool unavailableRejected = false;
        try {
            (void)luw::cli::requireNativeOpenClKrigingBackend();
        } catch (const luw::cli::OpenClKrigingError& exception) {
            unavailableRejected =
                std::string(exception.what()) ==
                "OpenCL support is not enabled in this build";
        }
        require(unavailableRejected,
                "a build without OpenCL did not reject the requested GPU backend");

        std::cout << "luwvox CLI compatibility checks passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "luwvox CLI test failure: " << exception.what() << '\n';
        return 1;
    }
}
