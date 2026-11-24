// CULBM executable: MPI start-up, the command line, the case thread that runs
// the deck, and the console loop that keeps the run progress current.
//
// Exit codes: 0 when the deck finished; 1 on a fatal error, including an
// exception that escaped the case; 2 for an invalid command line; 128+signal
// when interrupted (130 for Ctrl+C, 149 for Ctrl+Break on Windows). The case
// setup keeps its own exit codes for the errors it reports itself.

#include "launch.hpp"
#include "progress.hpp"
#include "../core/core.hpp"
#include "../cuda_mpi.hpp"
#include "../case/config_dump.hpp"
#include "../run/case_runner.hpp"

// core.hpp provides <chrono>, <iostream>, <mutex>, <string>, <thread>, <vector>.
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <exception>

namespace {

volatile std::sig_atomic_t pending_signal = 0;

void remember_signal(const int signal_number) {
	pending_signal = signal_number;
	std::signal(signal_number, SIG_DFL); // a second signal ends the process at once
}

[[noreturn]] void leave(const int code) {
	close_console_log_file();
	std::cout.flush();
	std::_Exit(code);
}

[[noreturn]] void stop_for_signal(const int signal_number) {
	string reason = "terminated by signal "+to_string(signal_number);
	if(signal_number==SIGINT) reason = "interrupted (Ctrl+C)";
#ifdef SIGBREAK
	if(signal_number==SIGBREAK) reason = "interrupted (Ctrl+Break)";
#endif
	culbm::app::run_progress().abandon(reason);
	if(culbm::mpi_is_distributed()) culbm::mpi_abort(reason.c_str());
	leave(128+signal_number);
}

// An exception that leaves the case thread ends the process through
// std::terminate, as it always did; it is reported here first.
[[noreturn]] void report_termination() {
	string what = "internal error: the program was terminated";
	if(const std::exception_ptr error = std::current_exception()) {
		try {
			std::rethrow_exception(error);
		} catch(const std::exception& e) {
			what = string("unhandled exception: ")+e.what();
		} catch(...) {
			what = "unhandled exception of unknown type";
		}
	}
	log_error(what);
	if(culbm::mpi_is_distributed()) culbm::mpi_abort(what.c_str());
	leave(1);
}

} // namespace

int main(int argc, char* argv[]) {
	culbm::mpi_initialize(&argc, &argv); // MPI may remove its own arguments
	std::set_terminate(report_termination);

	culbm::app::LaunchOptions options = culbm::app::parse_launch_options(
		std::vector<std::string>(argv+std::min(argc, 1), argv+argc));
	if(options.help || !options.error.empty()) {
		if(!options.error.empty()) print("CULBM: "+options.error+"\n");
		print(culbm::app::usage_text());
		return options.error.empty() ? 0 : 2;
	}
	string invalid;
	const culbm::app::ProgressOutput from_environment = culbm::app::progress_output_from_environment(invalid);
	if(!invalid.empty())
		log_warning("CULBM_PROGRESS="+invalid+" is not line, json or off; the default progress output is used.");
	culbm::app::run_progress().select_output(options.progress!=culbm::app::ProgressOutput::automatic
		? options.progress : from_environment);
	if(!options.dump_config.empty()) culbm::cases::ConfigDump::select(options.dump_config);
	culbm::app::set_case_arguments(std::move(options.case_arguments));
	std::signal(SIGINT, remember_signal);
	std::signal(SIGTERM, remember_signal);
#ifdef SIGBREAK
	std::signal(SIGBREAK, remember_signal);
#endif

	std::mutex finish_mutex;
	std::condition_variable finished;
	bool done = false;
	std::thread case_thread([&]() {
		culbm::run::run_deck_cases();
		if(culbm::mpi_is_distributed()) culbm::mpi_barrier();
		{
			std::lock_guard<std::mutex> lock(finish_mutex);
			done = true;
		}
		finished.notify_one();
	});

	// Console loop: progress every 50 ms until the deck is done.
	std::unique_lock<std::mutex> lock(finish_mutex);
	while(!done) {
		lock.unlock();
		culbm::app::run_progress().poll();
		if(pending_signal!=0) stop_for_signal((int)pending_signal);
		lock.lock();
		finished.wait_for(lock, std::chrono::milliseconds(50), [&]() { return done; });
	}
	lock.unlock();
	case_thread.join();
	return 0;
}
