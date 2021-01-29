#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <limits>
#include <omp.h>

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
std::random_device rd;
//std::mt19937 mt(rd());
std::uniform_real_distribution<float> dist(0.0, 1.0);

const float eps = std::numeric_limits<float>::epsilon();
const float ds = 1e-3; 
const int ZC = 666;
const int SC = 667;
const int AB = 668;
const int BD = 669;

float rayleigh(std::mt19937& mt)
{
    const float q = 4.f*dist(mt)-2.f;
    const float d = 1.f+q*q;
    const float u = pow(-q+sqrt(d), 1.f/3.f);
    return u-1.f/u;
}

float henyey(float g, std::mt19937& mt)
{
    const float a = pow(1.f-pow(g,2.f), 2.f);
    const float b = 2.f*g*pow(2*dist(mt)*g+1.f-g, 2.f);
    const float c = -g/2.f-1.f/(2.f*g);
    return -1.f*(a/b)-c;
}

float sample_tau(std::mt19937& mt)
{
    return -1.f*log(-dist(mt)+1.f);
}

void move_photon(const float* tau, const float* ssa, const float k_null, const int* size,
                float* position, float* direction, int& event, std::mt19937& mt)
{
    const float s = sample_tau(mt) / k_null;
    const float s_max = std::min((float(size[0])*(direction[0]>0)-position[0])/direction[0],
                                 (float(size[1])*(direction[1]>0)-position[1])/direction[1]);
    if (s+ds >= s_max)
    {
        position[0] += direction[0]*(s_max+ds);
        position[1] += direction[1]*(s_max+ds);
        
        if (position[1] >= size[1])
        {
            position[1] = ds;
        }
        else if (position[1] <= 0)
        {
            position[1] = size[1] - ds;
        }   
        event = BD;

    }
    else
    {
        position[0] += direction[0]*s;
        position[1] += direction[1]*s;
        const float r = dist(mt);
        const int idx = int(position[1]) + int(position[0]) * size[1];
        if (r*k_null >= tau[idx])
            event = ZC;
        else if (r*k_null <= tau[idx]*ssa[idx])
            event = SC;
        else
            event = AB;
    }
}

void hit_event(const int event, const float* ssa, const int* cld_mask, const float cloud_clear_frac, 
               const int* size, const float albedo, const float g,
               float* position, float* direction, bool& f_direct, bool& f_alive, std::mt19937& mt)
{
    if (event == SC)
    {
        f_direct = false;
        f_alive = true;
        const int idx = int(position[1]) + int(position[0]) * size[1];
        if (cld_mask[idx]==1 && dist(mt) < cloud_clear_frac) // cloud scattering
        {
            const float mu_scat  = henyey(g, mt);
            const float angle = acos(std::min(std::max(-1.f+eps,mu_scat),1.f-eps)) + atan2(direction[0], direction[1]) * int(-1+2*(dist(mt) > .5f)); 

            direction[0] = sin(angle);
            direction[1] = cos(angle);
        }
        else // gas scattering
        {
            const float mu_scat = rayleigh(mt);   
            const float angle= acos(mu_scat) + atan2(direction[0], direction[1]) * int(-1+2*(dist(mt) > .5f)); 
            direction[0] = sin(angle);
            direction[1] = cos(angle);
        }
    }
    else if (event == AB) // absorption event
    {
        f_alive = false;
    }
    else if (event == ZC) // zero collision event
    {
        f_alive = true;
    }
    else if (event == BD) // boundary hit (surface or top-of-domain
    {
        if (position[0] >= size[0]) // left top of domain
        {
            f_alive = false;
        }
        else if (position[0] <= 0.) // surface interaction
        {
            if (dist(mt) > albedo) // absorption by surface
            {
                f_alive = false;
            }
            else // scattering by surface
            {
                f_direct = false;
                position[0] = ds;
                
                const float mu_sfc = sqrt(dist(mt));
                direction[0] = mu_sfc;
                direction[1] = sin(acos(mu_sfc) * int(-1+2*(dist(mt) > .5f)));
                
                f_alive = true;
            }
        }
        else
        {
            f_alive = true;
        }
    
    }
    else
    {
        std::cout<<"oh oh, big problems here"<<std::endl;
    }
}

void trace_ray(const float* tau, const float* ssa, const float g, const int* cld_mask, 
               const int* size, const float albedo, const float sza_rad,
               const float cloud_clear_frac, const float k_null,
               const int n_photon, int* sfc_dir, int* sfc_dif, int nseed)
{
    int n_threads = omp_get_max_threads();
    int photons_per_block = int(n_photon/n_threads);
    
    #pragma omp parallel for
    for (int ithread = 0; ithread < n_threads; ++ithread)
    {
        std::mt19937 mt(ithread+100*nseed);
        #pragma omp critical
        std::cout<<"#thread "<<ithread<<std::endl;
        std::vector<int> sfc_dir_tmp(size[1], 0);      
        std::vector<int> sfc_dif_tmp(size[1], 0);      
        for (int iphoton = 0; iphoton < photons_per_block; ++iphoton)
        {
            int nhreads = omp_get_num_threads();
            float direction[2] = {-float(cos(sza_rad)), float(sin(sza_rad))};       
            float position[2]  = {size[0]-ds, dist(mt)*size[1]};       
            bool f_alive  = true;
            bool f_direct = true;
            int event     = 0;
            while (f_alive)
            {
                move_photon(tau, ssa, k_null, size, position, direction, event, mt);
                hit_event(event, ssa, cld_mask, cloud_clear_frac, size, albedo, g, position, direction, f_direct, f_alive, mt);
            }
            if (position[0] <= 0)
            {
                if (f_direct)
                    sfc_dir_tmp[int(position[1])] += 1;
                else
                    sfc_dif_tmp[int(position[1])] += 1;
            }
        }
        #pragma omp critical
        for (int ix = 0; ix < size[1]; ++ix)
        {
            sfc_dir[ix] += sfc_dir_tmp[ix];
            sfc_dif[ix] += sfc_dif_tmp[ix];
        }
    }      
}

int main()
{
    const int size[2] = {47, 82};
    const float k_null = .8f + 3e-4f;
    const float g = .85f;
    const float albedo = .0f;
    const float sza = 40.f/180.f * (4.f*atan(1.f));
    const float cld_clr_f = 0.99962515f; 
    int sfc_dir[82] = {};
    int sfc_dif[82] = {};
    float tau[47*82];
    float ssa[47*82];

    int  cld_mask[47*82];
    int n_photon = int(10000000);
    // read tau
    std::ifstream tau_in;
    tau_in.open("tau.txt");
    int i=0;
    for (std::string line; std::getline(tau_in, line); )
    {
        std::istringstream in(line); 
        in >> tau[i];
        i += 1;
    }
    
    std::ifstream ssa_in;
    ssa_in.open("ssa.txt");
    i=0;
    for (std::string line; std::getline(ssa_in, line); )
    {
        std::istringstream in(line); 
        in >> ssa[i];
        i += 1;
    }
    
    for (int i=0; i<47*82; ++i)
    {
        if (tau[i] == 3e-4)
            cld_mask [i] = false;
        else 
            cld_mask [i] = true;
    }
    
    
    std::cout<<"starting tracing rays"<<std::endl;
    
    double starttime = get_wall_time();
    trace_ray(tau, ssa, g, cld_mask, size, albedo, sza, cld_clr_f, k_null, n_photon, sfc_dir, sfc_dif, 1);
    double endtime = get_wall_time();
    
    std::cout<<n_photon<<" photons, runtime: "<<endtime-starttime<<" s"<<std::endl;
    std::cout<<"finished tracing rays"<<std::endl;


    std::ofstream fdir ("sfc_flux_dir.txt");
    std::ofstream fdif ("sfc_flux_dif.txt");
    for (int i=0; i<82; ++i)
    {
        fdir << float(sfc_dir[i]) / float(n_photon) << "\n";
        fdif << float(sfc_dif[i]) / float(n_photon) << "\n";
    }
    fdir.close();
    fdif.close();
    
    return 0;
}
















