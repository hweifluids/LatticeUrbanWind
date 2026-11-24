#pragma once

// Console output of the host code.
//  * print/println/reprint write on the MPI root only, under the console
//    lock, and mirror the text into the console log file. println and reprint
//    pass the text through the console table layout first.
//  * log_info/log_warning/log_error/fatal_error print a framed message: the
//    keyword (coloured on a terminal) followed by the message re-flowed into
//    lines of the console width. fatal_error terminates the process.

#include "console_log.hpp"
#include "console_table.hpp"
#include "platform.hpp"
#include "text.hpp"
#include "../cuda_mpi.hpp"

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace culbm {
namespace core {

inline void print(const std::string& s="") {
	if(!culbm::mpi_is_root()) return;
	std::lock_guard<std::mutex> lock(console_io_mutex());
	std::cout << s;
	write_console_log(s);
}
inline void println(const std::string& s="") {
	if(!culbm::mpi_is_root()) return;
	const std::string line = format_console_table_block(s, true)+"\n";
	std::lock_guard<std::mutex> lock(console_io_mutex());
	std::cout << line;
	write_console_log(line);
}
template<typename T> inline void println(const T& value) { println(to_string(value)); }
// Overwrites the current console line (no line feed); logged only on request.
inline void reprint(const std::string& s="", const bool write_log=false) {
	if(!culbm::mpi_is_root()) return;
	const std::string line = format_console_table_line(s, false);
	std::lock_guard<std::mutex> lock(console_io_mutex());
	std::cout << "\r"+line;
	if(write_log) write_console_log(line+"\n");
}
// Blocks until the user presses Enter (MPI root only).
inline void wait() {
	if(culbm::mpi_is_root()) (void)std::cin.get();
}

namespace detail {

enum class MessageTone { plain, info, warning, error };

// Writes a message keyword in its tone colour. Like the former helper this
// write is not restricted to the MPI root.
inline void write_toned(const std::string& text, const MessageTone tone) {
	std::lock_guard<std::mutex> lock(console_io_mutex());
#if defined(_WIN32)
	const WORD attribute = tone==MessageTone::info ? 10 : tone==MessageTone::warning ? 6 : tone==MessageTone::error ? 12 : 7;
	static const HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(console, attribute);
	std::cout << text;
	SetConsoleTextAttribute(console, 7); // default grey on black
#else
	const char* code = tone==MessageTone::info ? "92" : tone==MessageTone::warning ? "33" : tone==MessageTone::error ? "91" : "37";
	std::cout << "\033["+std::string(code)+"m";
	std::cout << text;
	std::cout << "\033[0m";
#endif
	write_console_log(text);
}

inline bool is_word_separator(const char c) {
	return c==' '||c=='\t'||c=='\n'||c=='\v'||c=='\f'||c=='\r';
}
// Words of a message (maximal runs of non-separator characters). Leading
// separators, and an empty message, produce one empty first word; trailing
// separators produce none.
inline std::vector<std::string> message_words(const std::string& message) {
	std::vector<std::string> words;
	const size_t n = message.size();
	if(n==0u||is_word_separator(message[0])) words.emplace_back();
	size_t i = 0u;
	while(i<n) {
		while(i<n&&is_word_separator(message[i])) i++;
		if(i==n) break;
		const size_t first = i;
		while(i<n&&!is_word_separator(message[i])) i++;
		words.emplace_back(message, first, i-first);
	}
	return words;
}

// Prints "| <keyword>: <message ...> |" re-flowed over as many lines as needed.
// Continuation lines are indented by the keyword width; a word longer than a
// whole line is split across lines.
inline void print_framed(const std::string& message, const std::string& keyword, const MessageTone tone, const bool colons) {
	const uint indent = (uint)keyword.length()+2u;
	const uint width = (uint)CONSOLE_WIDTH-4u-indent; // text columns per line
	const std::string line_break = "|\n| "+std::string(indent, ' ');
	std::string body = colons ? ": " : "  ";
	uint used = 0u; // columns used on the current line, each word followed by one blank
	const auto pad_line = [&]() { if(used<=width) body.append(width-used+1u, ' '); };
	std::vector<std::string> words = message_words(message);
	for(size_t i=0u; i<words.size(); ) {
		const uint word_length = (uint)words[i].length();
		if(used+word_length<=width) { // fits on the current line
			body += words[i]+" ";
			used += word_length+1u;
			i++;
		} else if(word_length>width&&used<=width) { // fill the line with the head of an overlong word
			const uint head = width-used;
			body += words[i].substr(0u, head)+" "+line_break;
			words[i] = words[i].substr(head);
			used = 0u;
		} else { // continue on the next line
			pad_line();
			body += line_break;
			used = 0u;
		}
	}
	pad_line();
	const std::string frame_start = "\r| ";
	if(tone==MessageTone::plain) { // keyword without colour in the same write
		print(frame_start+keyword);
	} else {
		print(frame_start);
		write_toned(keyword, tone);
	}
	println(body+"|");
}

} // namespace detail

inline void log_info(const std::string& s) {
	detail::print_framed(s, "Info", detail::MessageTone::info, true);
}
inline void log_warning(const std::string& s) {
	detail::print_framed(s, "Warning", detail::MessageTone::warning, true);
}
// Reports an error and returns; the caller decides how the process ends.
inline void log_error(const std::string& s) {
	detail::print_framed(s, "Error", detail::MessageTone::error, true);
}
// Prints the error and ends the process: through MPI_Abort in distributed
// runs, otherwise with exit code 1 (on Windows after the user presses Enter).
[[noreturn]] inline void fatal_error(const std::string& s) {
	detail::print_framed(s, "Error", detail::MessageTone::error, true);
	if(culbm::mpi_is_distributed()) culbm::mpi_abort(s.c_str());
#if defined(_WIN32)
	detail::print_framed("Press Enter to exit.", "     ", detail::MessageTone::plain, false);
#endif
	println("'"+std::string((size_t)CONSOLE_WIDTH-2u, '-')+"'");
#if defined(_WIN32)
	wait();
#endif
	std::exit(1);
}

} // namespace core
} // namespace culbm
