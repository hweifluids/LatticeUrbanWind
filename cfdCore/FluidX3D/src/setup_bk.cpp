#include "setup.hpp"


void main_setup() { // urban canopy airflow over STL city model (non‑interactive screenshot mode)
	// required extensions in defines.hpp: SUBGRID, EQUILIBRIUM_BOUNDARIES, GRAPHICS
	// ################################################################## simulation parameters ##################################################################
	const uint  memory = 4000u;               // target VRAM usage in MB (adjust to your GPU)
	const float lbm_u = 0.05f;              // inlet velocity in LBM units (maps to 2 m/s)
	const float si_u = 2.0f;               // wind speed [m/s]
	const float si_nu = 1.48E-5f;           // kinematic viscosity of air [m^2/s]
	const float si_rho = 1.225f;             // density of air   [kg/m^3]

	// ------------------------------------------------------------------ physical domain (SI) ---------------------------------------------------------------
	const float3 si_size = float3(5379.31348f, 4510.8149f, 550.0f); // Lx, Ly, Lz in metres

	// Determine lattice resolution that fits into the available memory
	const uint3 lbm_N = resolution(si_size, memory); // returns grid resolution Nx, Ny, Nz (multiples of 4)

	// Map SI units → LBM units.  We choose the y‑dimension (flow direction) as reference length.
	units.set_m_kg_s((float)lbm_N.y, lbm_u, 1.0f, si_size.y, si_u, si_rho);
	const float lbm_nu = units.nu(si_nu);

	// ------------------------------------------------------------------- create solver ---------------------------------------------------------------------
	LBM lbm(lbm_N, lbm_nu); // compile with GRAPHICS (without INTERACTIVE_GRAPHICS)

	// ------------------------------------------------------------------- import & scale geometry -----------------------------------------------------------
	Mesh* mesh = read_stl(get_exe_path() + "../stl/buildings_base.stl");

	// Scale STL so its x‑extent matches 5379.31348 m (converted to LBM units)
	const float target_lbm_x = units.x(si_size.x);
	const float scale = target_lbm_x / mesh->get_bounding_box_size().x;
	mesh->scale(scale);

	// Position model: leave a 1‑cell buffer to the negative boundaries, sit on z = 1
	mesh->translate(float3(1.0f - mesh->pmin.x,
		1.0f - mesh->pmin.y,
		1.0f - mesh->pmin.z));

	lbm.voxelize_mesh_on_device(mesh); // buildings become TYPE_S cells automatically

	// ------------------------------------------------------------------- boundary & initial conditions -----------------------------------------------------
	const uint Nx = lbm.get_Nx(), Ny = lbm.get_Ny(), Nz = lbm.get_Nz();
	parallel_for(lbm.get_N(), [&](ulong n) {
		uint x = 0u, y = 0u, z = 0u; lbm.coordinates(n, x, y, z);

		// Ground
		if (z == 0u) lbm.flags[n] = TYPE_S;

		// Initialise background flow everywhere that is not solid
		if (lbm.flags[n] != TYPE_S) lbm.u.y[n] = lbm_u;

		// ----- Pressure outlet (+y) -----
		if (y == Ny - 1u) {
			lbm.flags[n] = TYPE_E;  // equilibrium outflow
		}
		// ----- Velocity inlets (+x, -x, -y, +z) -----
		else if ((x == 0u || x == Nx - 1u || y == 0u || z == Nz - 1u)) {
			lbm.flags[n] = TYPE_E;  // equilibrium inflow
			lbm.u.y[n] = lbm_u;     // prescribe 2 m/s in +y (mapped to lbm_u)
		}
		});

	// ------------------------------------------------------------------- graphics & run --------------------------------------------------------------------
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE | VIS_Q_CRITERION;

	const ulong lbm_T = 20000ull;   // total simulation time steps (~steady‑state)
	const uint  vtk_dt = 1000u;        // export VTK every 20 time steps
	const string vtk_dir = get_exe_path() + "vtk/"; // ensure this directory exists beforehand

#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
	// Camera: inspired by Ahmed‑body case (slightly upstream, high elevation, moderate tilt)
	lbm.graphics.set_camera_free(
		float3(0.6f * Nx, -0.7f * Ny, 2.2f * Nz), // upstream & elevated
		-45.0f,                               // yaw
		30.0f,                                // pitch
		80.0f);                               // FOV

	lbm.run(0u, lbm_T); // initialize fields and graphics buffers
	while (lbm.get_t() < lbm_T) {
		// ------------------ off‑screen PNG rendering (optional video) ------------------
		if (lbm.graphics.next_frame(lbm_T, 10.0f)) {
			lbm.graphics.write_frame(); // default path: exe_dir/export/frame_#####.png
		}

		// ------------------ synchronous VTK output ------------------
		if (lbm.get_t() % vtk_dt == 0u) {
			const string ts = to_string(lbm.get_t());
			// velocity (vector field)
			lbm.u.write_device_to_vtk(vtk_dir + "u_" + ts + ".vtk", true);
			// density as proxy for pressure (p = c_s^2 * (rho-1) can be computed in ParaView)
			lbm.rho.write_device_to_vtk(vtk_dir + "rho_" + ts + ".vtk", true);
		}

		lbm.run(1u, lbm_T);
	}
#else // GRAPHICS + INTERACTIVE or pure CLI
	while (lbm.get_t() < lbm_T) {
		if (lbm.get_t() % vtk_dt == 0u) {
			const string vtk_file = vtk_dir + "step_" + to_string(lbm.get_t(), 5u) + ".vtk";
			lbm.write_vtk(vtk_file, true, true);
		}
		lbm.run(1u, lbm_T);
	}
#endif
}



