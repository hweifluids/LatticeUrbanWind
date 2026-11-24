#include "launch.hpp"

#include <cstdlib>
#include <utility>

namespace culbm::app {

namespace {

std::vector<std::string>& stored_case_arguments() {
	static std::vector<std::string> arguments;
	return arguments;
}

} // namespace

LaunchOptions parse_launch_options(const std::vector<std::string>& words) {
	LaunchOptions options;
	const std::string progress_option = "--progress=";
	const std::string dump_option = "--dump-config=";
	bool literal = false; // after "--"
	for(std::size_t i = 0u; i < words.size(); ++i) {
		const std::string& word = words[i];
		if(literal) {
			options.case_arguments.push_back(word);
		} else if(word=="--") {
			literal = true;
		} else if(word=="--help" || word=="-h") {
			options.help = true;
		} else if(word=="--progress-json") {
			options.progress = ProgressOutput::json;
		} else if(word.compare(0u, progress_option.size(), progress_option)==0) {
			const std::string value = word.substr(progress_option.size());
			if(!parse_progress_output(value, options.progress) && options.error.empty())
				options.error = "unknown progress output '"+value+"' (expected line, json or off)";
		} else if(word=="--progress") {
			if(options.error.empty()) options.error = "--progress needs a value: --progress=line|json|off";
		} else if(word.compare(0u, dump_option.size(), dump_option)==0) {
			options.dump_config = word.substr(dump_option.size());
			if(options.dump_config.empty() && options.error.empty()) options.error = "--dump-config needs a file name";
		} else if(word=="--dump-config") {
			if(i+1u < words.size()) options.dump_config = words[++i];
			else if(options.error.empty()) options.error = "--dump-config needs a file name";
		} else {
			options.case_arguments.push_back(word);
		}
	}
	return options;
}

ProgressOutput progress_output_from_environment(std::string& invalid) {
	invalid.clear();
	const char* const value = std::getenv("CULBM_PROGRESS");
	if(value==nullptr || *value=='\0') return ProgressOutput::automatic;
	ProgressOutput output = ProgressOutput::automatic;
	if(!parse_progress_output(value, output)) invalid = value;
	return output;
}

std::string usage_text() {
	return
		"Usage: CULBM [options] <deck.luw|deck.luwdg|deck.luwpf>\n"
		"  --progress=line|json|off  console progress: status line (default), one JSON\n"
		"                            record per second, or none; also CULBM_PROGRESS\n"
		"  --progress-json           same as --progress=json\n"
		"  --dump-config <file>      write the parsed configuration to <file> and stop\n"
		"                            before a solver is allocated; also CULBM_DUMP_CONFIG\n"
		"  --help, -h                this text\n"
		"Without a deck path the deck is asked for on the console.\n";
}

void set_case_arguments(std::vector<std::string> arguments) {
	stored_case_arguments() = std::move(arguments);
}

const std::vector<std::string>& case_arguments() {
	return stored_case_arguments();
}

} // namespace culbm::app
