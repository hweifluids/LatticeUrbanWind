#include "config_dump.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace culbm::cases {

namespace {

std::string& selected_path() {
	static std::string path;
	return path;
}

template<typename T> std::string list_text(const std::vector<T>& values) {
	Checksum sum;
	if(!values.empty()) sum.bytes(values.data(), values.size()*sizeof(T));
	std::string out = "n=" + std::to_string(values.size()) + " fnv=" + sum.hex();
	if(values.size() <= 16u) {
		out += " [";
		for(std::size_t i = 0u; i < values.size(); ++i) {
			if(i) out += ", ";
			if constexpr(std::is_floating_point_v<T>) out += ConfigDump::text(values[i]);
			else out += std::to_string(values[i]);
		}
		out += "]";
	}
	return out;
}

} // namespace

Checksum& Checksum::bytes(const void* data, const std::size_t count) {
	const unsigned char* p = static_cast<const unsigned char*>(data);
	for(std::size_t i = 0u; i < count; ++i) {
		state_ ^= (std::uint64_t)p[i];
		state_ *= 1099511628211ull;
	}
	return *this;
}

std::string Checksum::hex() const {
	char buffer[24];
	std::snprintf(buffer, sizeof(buffer), "%016llx", (unsigned long long)state_);
	return buffer;
}

void ConfigDump::select(const std::string& path) {
	selected_path() = path;
}

bool ConfigDump::enabled() {
	if(selected_path().empty()) {
		const char* const value = std::getenv("CULBM_DUMP_CONFIG");
		if(value != nullptr && *value != '\0') selected_path() = value;
	}
	return !selected_path().empty();
}

ConfigDump& ConfigDump::get() {
	static std::unique_ptr<ConfigDump> dump;
	if(!dump) {
		dump = std::make_unique<ConfigDump>();
		if(enabled()) {
			dump->file_.open(selected_path(), std::ios::out | std::ios::trunc | std::ios::binary);
			if(!dump->file_.is_open()) fatal_error("cannot create the configuration dump " + selected_path());
			dump->line("# CULBM parsed configuration");
		}
	}
	return *dump;
}

std::string ConfigDump::text(const float value) {
	char buffer[48];
	std::snprintf(buffer, sizeof(buffer), "%.9g", (double)value);
	return buffer;
}

std::string ConfigDump::text(const double value) {
	char buffer[48];
	std::snprintf(buffer, sizeof(buffer), "%.17g", value);
	return buffer;
}

void ConfigDump::line(const std::string& text) {
	if(!file_.is_open()) return;
	file_ << text << '\n';
	file_.flush();
}

void ConfigDump::section(const std::string& name) {
	line("[" + name + "]");
}

void ConfigDump::put(const std::string& key, const std::string& value) {
	line(key + " = " + value);
}

void ConfigDump::put(const std::string& key, const float value) {
	put(key, text(value));
}

void ConfigDump::put(const std::string& key, const double value) {
	put(key, text(value));
}

void ConfigDump::put(const std::string& key, const float3& value) {
	put(key, text(value.x) + ", " + text(value.y) + ", " + text(value.z));
}

void ConfigDump::put(const std::string& key, const uint3& value) {
	put(key, std::to_string(value.x) + ", " + std::to_string(value.y) + ", " + std::to_string(value.z));
}

void ConfigDump::put(const std::string& key, const std::vector<float>& values) {
	put(key, list_text(values));
}

void ConfigDump::put(const std::string& key, const std::vector<double>& values) {
	put(key, list_text(values));
}

void ConfigDump::put(const std::string& key, const std::vector<uint>& values) {
	put(key, list_text(values));
}

void ConfigDump::put_checksum(const std::string& key, const std::size_t count, const Checksum& checksum) {
	put(key, "n=" + std::to_string(count) + " fnv=" + checksum.hex());
}

void ConfigDump::put_mesh(const std::string& key, const Mesh* mesh) {
	if(mesh == nullptr) {
		put(key, std::string("none"));
		return;
	}
	Checksum sum;
	for(uint i = 0u; i < mesh->triangle_number; ++i) {
		sum.value(mesh->p0[i]).value(mesh->p1[i]).value(mesh->p2[i]);
	}
	put_checksum(key + ".triangles", (std::size_t)mesh->triangle_number, sum);
	put(key + ".pmin", mesh->pmin);
	put(key + ".pmax", mesh->pmax);
	put(key + ".center", mesh->center);
}

void ConfigDump::put_units(const std::string& prefix, const Units& u) {
	put(prefix + "metre", u.si_x(1.0f));
	put(prefix + "second", u.si_t((ulong)1u));
	put(prefix + "velocity", u.si_u(1.0f));
	put(prefix + "density", u.si_rho(1.0f));
	put(prefix + "viscosity", u.si_nu(1.0f));
	put(prefix + "kelvin", u.si_dT(1.0f));
	put(prefix + "kelvin_offset", u.si_T(0.0f));
}

} // namespace culbm::cases
