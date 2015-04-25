#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(x) x
#endif
OCL_EXTERNAL_INCLUDE(

kernel void ocl(global double4 x[], global double4 v[], const global double4 f[], const double m, const double dt, const ulong n)
{
	const int i = get_global_id(0);

	const double4 a = f[i]/m;

	const double4 xx = x[i];
	const double4 vv = v[i];

	x[i] = xx + vv*dt + a*dt*dt/2;
	v[i] = vv + a*dt;
}

)