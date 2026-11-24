#include "cuda_nested_solver.hpp"
#include "solver/solver.hpp"
#include "surface_ibm.hpp"

#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

float coriolis_Omegax_lbmu=0.0f, coriolis_Omegay_lbmu=0.0f, coriolis_Omegaz_lbmu=0.0f;
bool buffer_nudging_active=false, top_sponge_active=false;
int buffer_n_cells=1, buffer_nudge_vertical=0, buffer_downstream_face_id=0;
int sponge_n_cells=1, sponge_ref_mode=0;
float buffer_inv_tau_lbmu=0.0f, sponge_inv_tau_lbmu=0.0f;

int main(int argc, char** argv) {
    try {
        bool enabled=true, reference=false;
        unsigned domains=1;
        std::string dump;
        for(int i=1; i<argc; ++i) {
            const std::string arg=argv[i];
            if(arg=="--off") enabled=false;
            else if(arg=="--host") reference=true;
            else if(arg=="--dual") domains=2;
            else dump=arg;
        }
        constexpr unsigned nx=28, ny=20, nz=15;
        auto plan=culbm::make_vertical_nesting_plan({5.0,10.0},2.5,32.5,nx,ny,domains,1,1);
        culbm::NestedSeedState seed;
        seed.nx=nx; seed.ny=ny; seed.nz=nz;
        const size_t n=seed.cell_count();
        seed.rho.assign(n,1.0f); seed.ux.assign(n,0.025f);
        seed.uy.assign(n,0.0f); seed.uz.assign(n,0.0f); seed.flags.assign(n,0);
        for(unsigned i=0; i<nx*ny; ++i) { seed.flags[i]=CELL_SOLID; seed.ux[i]=0.0f; }
        // A skewed prism crosses both slab planes and the two-device x seam.
        // Its flat base lies on the sealed physical wall.
        const float3 a(9.25f,5.25f,0.5f), x(6.0f,1.5f,0.0f), y(-0.75f,3.0f,0.0f), z(0.0f,0.0f,6.0f);
        const std::array<float3,8> v={a,a+x,a+x+y,a+y,a+z,a+x+z,a+x+y+z,a+y+z};
        const unsigned f[12][3]={{0,2,1},{0,3,2},{4,5,6},{4,6,7},{0,1,5},{0,5,4},
                                 {1,2,6},{1,6,5},{2,3,7},{2,7,6},{3,0,4},{3,4,7}};
        Mesh mesh(12,float3(0.0f));
        for(unsigned t=0;t<12;++t){mesh.p0[t]=v[f[t][0]];mesh.p1[t]=v[f[t][1]];mesh.p2[t]=v[f[t][2]];}
        mesh.find_bounds();
        culbm::surface_ibm::Config config; config.enabled=enabled;
        culbm::surface_ibm::set_config(config);
        culbm::NestedLbmPhysics physics; physics.coarse_nu=0.01f; physics.coarse_dt_si=2.5/78.;
        culbm::NestedVerticalLbm hierarchy;
        std::string error;
        if(!culbm::build_nested_vertical_lbm(plan,seed,mesh,domains,1,1,physics,hierarchy,error))
            throw std::runtime_error(error);
        culbm::NestedLbmScheduler schedule(hierarchy.make_scheduler_levels());
        schedule.set_interface_hook([&](auto& coarse,auto& fine,auto){
            if(reference) culbm::NestedMacroPlaneBridge::exchange_host(coarse,fine);
            else culbm::NestedMacroPlaneBridge::exchange(coarse,fine);
        });
        schedule.initialize();
        for(unsigned i=0;i<4;++i)schedule.advance_one_coarse_step();
        schedule.finish();
        const auto start=std::chrono::steady_clock::now();
        for(unsigned i=0;i<32;++i)schedule.advance_one_coarse_step();
        schedule.finish();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream binary;
        if(!dump.empty()) binary.open(dump,std::ios::binary);
        const double perimeter=2.0*(std::hypot(6.0,1.5)+std::hypot(0.75,3.0))*2.5;
        for(auto& level:hierarchy.levels) {
            auto& lbm=*level.solver; auto& p=level.plan;
            if(lbm.time()!=36u*p.ratio_to_coarse)throw std::runtime_error("incorrect native step");
            if(enabled) {
                const auto stats=culbm::surface_ibm::statistics(lbm);
                const double area=perimeter*5.0+(p.ratio_to_coarse==1 ? (18.0+1.125)*6.25:0.0);
                if(!stats.valid || !stats.markers || !stats.active_cells ||
                   std::fabs(level.surface_ibm_area_si-area)>0.005 ||
                   !std::isfinite(stats.max_slip_lu) || stats.max_slip_lu>0.025f)
                    throw std::runtime_error("wall area, coupling or slip check failed");
                std::cout<<"WALL ratio="<<p.ratio_to_coarse<<" area="<<level.surface_ibm_area_si
                         <<" expected="<<area<<" markers="<<stats.markers<<" cells="<<stats.active_cells
                         <<" slip="<<stats.max_slip_lu<<" reaction_x="<<stats.reaction_force_lu.x<<"\n";
                // Verify the force actually stored on both ghost planes and the
                // equal-and-opposite Eulerian/Lagrangian resultant in native units.
                double sums[3]{};
                for(unsigned d=0;d<lbm.local_count();++d) {
                    auto& domain=lbm.local(d);
                    if(domains==2 && d==1 && domain.device().ordinal()==lbm.local(0).device().ordinal())
                        throw std::runtime_error("dual test requires distinct physical devices");
                    const auto nlocal=domain.cells();
                    std::vector<float> force(3*nlocal);
                    gpu::make_current(domain.device().context());
                    gpu::check(cuMemcpyDtoH(force.data(),domain.fields().ibm_force.address(),
                                            force.size()*sizeof(float)),"test: IBM force");
                    const unsigned hx=domains>1 ? 1u:0u;
                    for(unsigned k=0;k<domain.extent(2u);++k)for(unsigned j=0;j<domain.extent(1u);++j)
                        for(unsigned i=hx;i<domain.extent(0u)-hx;++i)for(unsigned q=0;q<3;++q){
                            const float value=force[q*nlocal+i+domain.extent(0u)*(j+domain.extent(1u)*k)];
                            if((k==0 || k+1==domain.extent(2u)) && value!=0.0f)
                                throw std::runtime_error("IBM wrote to a ghost row");
                            sums[q]+=value;
                        }
                }
                const double reaction[3]={stats.reaction_force_lu.x,stats.reaction_force_lu.y,stats.reaction_force_lu.z};
                for(unsigned q=0;q<3;++q) if(std::abs(reaction[q]+sums[q])>2e-5*(1+std::abs(reaction[q])))
                    throw std::runtime_error("Eulerian and marker forces do not balance");
            }
            lbm.velocity().download(); lbm.density().download(); lbm.flags().download();
            for(ulong i=0;i<lbm.grid().cells();++i) {
                const float fields[]={lbm.density()[i],lbm.velocity().x[i],lbm.velocity().y[i],lbm.velocity().z[i]};
                for(float value:fields) if(!std::isfinite(value)) throw std::runtime_error("non-finite field");
                if(binary.is_open()) binary.write((const char*)fields,sizeof fields);
            }
        }
        std::cout<<std::setprecision(9)<<"NESTED_PASS ibm="<<enabled<<" domains="<<domains
                 <<" host_bridge="<<reference<<" timed_steps=32 seconds="<<seconds<<std::endl;
    } catch(const std::exception& err) { std::cerr<<"NESTED_FAIL "<<err.what()<<std::endl; return 2; }
    return 0;
}
