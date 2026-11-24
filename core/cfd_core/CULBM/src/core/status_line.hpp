#pragma once

// Live status line: the last console line, rewritten in place by a reporter
// (the run progress) while ordinary rows scroll above it. A row printed over
// the line starts with '\r'. Code that prints a block of rows while the line
// may be shown holds a StatusPause: the reporter cannot redraw the line in the
// middle of the block, and the line is drawn again below the block when the
// pause ends. The line is never mirrored into the console log file.

#include "console.hpp" // also <iostream>, <mutex>, <string>

namespace culbm {
namespace core {

class StatusLine {
public:
	// Serializes drawing against StatusPause blocks.
	std::mutex& gate() { return gate_; }

	// The *_locked members expect gate() to be held by the caller.
	void draw_locked(const std::string& text) {
		text_ = text;
		shown_ = true;
		show();
	}
	void redraw_locked() {
		if(shown_) show();
	}
	// Leaves the last drawing on screen and moves to a fresh line.
	void close_locked() {
		if(!shown_) return;
		shown_ = false;
		if(!culbm::mpi_is_root()) return;
		std::lock_guard<std::mutex> lock(console_io_mutex());
		std::cout << '\n';
		std::cout.flush();
	}
	bool shown_locked() const { return shown_; }

private:
	void show() {
		reprint(text_);
		if(!culbm::mpi_is_root()) return;
		std::lock_guard<std::mutex> lock(console_io_mutex());
		std::cout.flush(); // piped consumers (launchers) see every update
	}

	std::mutex gate_;
	std::string text_;
	bool shown_ = false;
};

inline StatusLine& status_line() {
	static StatusLine line;
	return line;
}

// Holds the status line still while a block of rows is printed.
class StatusPause {
public:
	StatusPause() : hold_(status_line().gate()) {}
	~StatusPause() { status_line().redraw_locked(); }
	StatusPause(const StatusPause&) = delete;
	StatusPause& operator=(const StatusPause&) = delete;

private:
	std::lock_guard<std::mutex> hold_;
};

} // namespace core
} // namespace culbm
