#pragma once

// Console table layout: re-wraps "| key | value |" rows and border rows to the
// fixed console width.
// LatticeUrbanWind-owned code, carried over unchanged from the former
// solver utilities (it was written for LUW and has no upstream origin).

#include "text.hpp"

#include <string>
#include <vector>

namespace culbm {
namespace core {

using std::string;
using std::vector;

inline string truncate_with_ellipsis(const string& s, const uint width) {
	if(width==0u) return "";
	if((uint)s.size()<=width) return s;
	if(width<=3u) return s.substr(0u, width);
	return s.substr(0u, width-3u)+"...";
}
inline string pad_right_exact(const string& s, const uint width) {
	string out = truncate_with_ellipsis(s, width);
	if((uint)out.size()<width) out += string(width-(uint)out.size(), ' ');
	return out;
}
inline vector<string> wrap_console_cell(const string& cell, const uint width, const bool allow_wrap, const bool preserve_spacing) {
	vector<string> lines;
	if(width==0u) {
		lines.push_back("");
		return lines;
	}
	string remaining = preserve_spacing ? cell : trim(cell);
	if(remaining.empty()) {
		lines.push_back("");
		return lines;
	}
	if(!allow_wrap) {
		lines.push_back(truncate_with_ellipsis(remaining, width));
		return lines;
	}
	while(!remaining.empty()) {
		if((uint)remaining.size()<=width) {
			lines.push_back(remaining);
			break;
		}
		uint cut = width;
		const size_t space = remaining.rfind(' ', width);
		if(space!=string::npos && space>0u) cut = (uint)space;
		if(cut==0u) cut = width;
		string part = remaining.substr(0u, cut);
		if(!preserve_spacing) part = trim(part);
		if(part.empty()) {
			part = remaining.substr(0u, width);
			cut = width;
		}
		lines.push_back(truncate_with_ellipsis(part, width));
		remaining = remaining.substr(cut);
		if(!preserve_spacing) remaining = trim(remaining);
		else while(!remaining.empty() && remaining.front()==' ') remaining.erase(remaining.begin());
	}
	if(lines.empty()) lines.push_back("");
	return lines;
}
inline bool is_border_like_row(const string& inner) {
	bool has_border_char = false;
	for(size_t i=0u; i<inner.size(); i++) {
		const char c = inner[i];
		if(c==' ' || c=='\t') continue;
		if(c=='-' || c=='=' || c=='\'' || c=='.' || c=='_' || c=='+') {
			has_border_char = true;
			continue;
		}
		return false;
	}
	return has_border_char;
}
inline string rtrim_console_copy(const string& s) {
	if(s.empty()) return s;
	size_t e = s.size();
	while(e>0u && (s[e-1u]==' ' || s[e-1u]=='\t' || s[e-1u]=='\r')) e--;
	return s.substr(0u, e);
}
inline string resize_border_inner(const string& inner, const uint target) {
	string out = rtrim_console_copy(inner);
	if((uint)out.size()>target) return out.substr(0u, target);
	char fill = '-';
	for(size_t i=out.size(); i>0u; i--) {
		const char c = out[i-1u];
		if(c=='-' || c=='=' || c=='_' || c=='+' || c=='.') {
			fill = c=='.' ? '-' : c;
			break;
		}
	}
	if((uint)out.size()<target) out += string(target-(uint)out.size(), fill);
	return out;
}
inline uint count_char(const string& s, const char c) {
	uint n = 0u;
	for(size_t i=0u; i<s.size(); i++) if(s[i]==c) n++;
	return n;
}
inline vector<string> split_keep_empty(const string& s, const char c) {
	vector<string> out;
	size_t start = 0u;
	while(start<=s.size()) {
		const size_t pos = s.find(c, start);
		if(pos==string::npos) {
			out.push_back(s.substr(start));
			break;
		}
		out.push_back(s.substr(start, pos-start));
		start = pos+1u;
	}
	return out;
}
inline string format_console_table_line(const string& line, const bool allow_wrap) {
	constexpr uint key_col_width = 17u; // keeps separator aligned with |-----------------.----------------...|
	string normalized = rtrim_console_copy(line);
	if(!normalized.empty() && normalized.front()=='|' && normalized.back()!='|') normalized += "|";
	if(normalized.size()<2u || normalized.front()!='|' || normalized.back()!='|') return normalized;
	const string inner = normalized.substr(1u, normalized.size()-2u);
	const uint table_width = (uint)CONSOLE_WIDTH;
	if(table_width<3u) return normalized;
	const uint inner_target = table_width-2u;
	if(is_border_like_row(inner)) {
		return "|"+resize_border_inner(inner, inner_target)+"|";
	}
	const uint separator_count = count_char(inner, '|');
	if(separator_count>=2u) { // already multi-column formatted line; keep original separators/spacing
		return "|"+pad_right_exact(inner, inner_target)+"|";
	}
	vector<string> cells = separator_count==1u ? split_keep_empty(inner, '|') : vector<string>{inner};
	if(cells.empty()) return "|"+pad_right_exact("", inner_target)+"|";
	const uint cols = (uint)cells.size();
	if(table_width<=cols+1u) return normalized;
	const uint usable = table_width-(cols+1u);
	vector<uint> widths(cols, usable);
	if(cols==2u) {
		const uint min_right = 8u;
		uint left = key_col_width;
		if(left>usable-min_right) left = usable-min_right;
		if(left<4u) left = min(usable, 4u);
		widths[0] = left;
		widths[1] = usable-left;
	}
	vector<vector<string>> wrapped(cols, vector<string>{""});
	uint row_count = 1u;
	for(uint i=0u; i<cols; i++) {
		if(cols==2u) {
			const uint text_width = widths[i]>0u ? widths[i]-1u : 0u; // keep one leading space in each cell
			wrapped[i] = wrap_console_cell(trim(cells[i]), text_width, allow_wrap, false);
		} else {
			wrapped[i] = wrap_console_cell(cells[i], widths[i], allow_wrap, true);
		}
		row_count = max(row_count, (uint)wrapped[i].size());
	}
	string out = "";
	for(uint r=0u; r<row_count; r++) {
		string row = "|";
		for(uint c=0u; c<cols; c++) {
			string text = r<(uint)wrapped[c].size() ? wrapped[c][r] : "";
			if(cols==2u) text = " "+text; // requirement: one-space inset from vertical separators
			row += pad_right_exact(text, widths[c])+"|";
		}
		if(r>0u) out += "\n";
		out += row;
	}
	return out;
}
inline string format_console_table_block(const string& s, const bool allow_wrap) {
	string out = "";
	size_t start = 0u;
	while(start<=s.size()) {
		const size_t pos = s.find('\n', start);
		const string line = pos==string::npos ? s.substr(start) : s.substr(start, pos-start);
		if(!out.empty()) out += "\n";
		out += format_console_table_line(line, allow_wrap);
		if(pos==string::npos) break;
		start = pos+1u;
	}
	return out;
}

} // namespace core
} // namespace culbm
