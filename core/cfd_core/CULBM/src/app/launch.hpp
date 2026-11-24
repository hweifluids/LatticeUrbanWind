#pragma once

// Command line of the CULBM executable.
//
//   CULBM [options] <deck.luw|deck.luwdg|deck.luwpf> [case arguments]
//
//   --progress=line|json|off  console progress (default: CULBM_PROGRESS, else line)
//   --progress-json           same as --progress=json
//   --dump-config <file>      write the parsed configuration of the deck to
//                             <file> and stop before a solver is allocated
//                             (also --dump-config=<file>; see case/config_dump.hpp)
//   --help, -h                usage
//   --                        everything after it is a case argument
//
// Options may appear anywhere; every other word is a case argument, in order.
// The first case argument is the deck path; without one the case setup asks
// for it on the console.

#include "progress.hpp"

#include <string>
#include <vector>

namespace culbm::app {

struct LaunchOptions {
	std::vector<std::string> case_arguments;
	ProgressOutput progress = ProgressOutput::automatic; // automatic: not on the command line
	std::string dump_config; // non-empty: configuration dump file
	bool help = false;
	std::string error; // non-empty: the command line is invalid
};

LaunchOptions parse_launch_options(const std::vector<std::string>& words);
// CULBM_PROGRESS; automatic when unset. An invalid value sets `invalid`.
ProgressOutput progress_output_from_environment(std::string& invalid);
std::string usage_text();

// Case arguments of this process, for the case setup.
void set_case_arguments(std::vector<std::string> arguments);
const std::vector<std::string>& case_arguments();

} // namespace culbm::app
