#pragma once

// Parsed-configuration dump of the case setup.
//
// Started with the command-line option --dump-config <file> (or the
// environment variable CULBM_DUMP_CONFIG=<file>), the case setup reads the
// deck, performs every parsing, defaulting and derivation step it performs
// before a solver is allocated, writes each effective parameter in a fixed
// order and returns without allocating a solver or running a time step.
// Two builds that read and derive a deck identically write identical files;
// the dump is the regression check for changes in deck handling.
//
// Format: "[section]" lines followed by "key = value" lines. Floats print with
// 9 and doubles with 17 significant digits (both round-trip exactly); arrays
// print their length and a 64-bit FNV-1a checksum of their bytes, followed by
// the values when there are at most 16 of them. Lines are flushed as they are
// written, so a deck that stops with an error leaves the part it reached.

#include "../core/core.hpp"
#include "../core/lattice_units.hpp"

#include <cstddef>
#include <cstdint>

struct Mesh;

namespace culbm::cases {

// Incremental FNV-1a over raw bytes.
class Checksum {
public:
	Checksum& bytes(const void* data, std::size_t count);
	template<typename T> Checksum& value(const T& v) { return bytes(&v, sizeof(T)); }
	std::uint64_t result() const { return state_; }
	std::string hex() const;

private:
	std::uint64_t state_ = 14695981039346656037ull;
};

class ConfigDump {
public:
	// The file named on the command line or by CULBM_DUMP_CONFIG; empty: off.
	static void select(const std::string& path);
	static bool enabled();
	static ConfigDump& get(); // opens the file on first use

	void section(const std::string& name);
	void put(const std::string& key, const std::string& value);
	void put(const std::string& key, const char* value) { put(key, std::string(value)); }
	void put(const std::string& key, bool value) { put(key, std::string(value ? "true" : "false")); }
	void put(const std::string& key, float value);
	void put(const std::string& key, double value);
	template<typename T, typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
	void put(const std::string& key, const T value) {
		if constexpr(std::is_signed_v<T>) put(key, std::to_string((long long)value));
		else put(key, std::to_string((unsigned long long)value));
	}
	void put(const std::string& key, const float3& value);
	void put(const std::string& key, const uint3& value);
	void put(const std::string& key, const std::vector<float>& values);
	void put(const std::string& key, const std::vector<double>& values);
	void put(const std::string& key, const std::vector<uint>& values);
	// Length and checksum only.
	void put_checksum(const std::string& key, std::size_t count, const Checksum& checksum);
	// Triangle count, bounds and a checksum of every corner.
	void put_mesh(const std::string& key, const Mesh* mesh);
	// The SI value of one lattice unit of each scale.
	void put_units(const std::string& prefix, const Units& units);

	static std::string text(float value);
	static std::string text(double value);

private:
	void line(const std::string& text);
	std::ofstream file_;
};

} // namespace culbm::cases
