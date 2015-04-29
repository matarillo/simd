#define _SCL_SECURE_NO_WARNINGS
#include <intrin.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <chrono>
#include <omp.h>

class Timer
{
	typedef std::chrono::time_point<std::chrono::system_clock> time_point;

	time_point begin;

public:
	void Start()
	{
		this->begin = std::chrono::system_clock::now();
	}

	std::chrono::milliseconds Time()
	{
		const auto end = std::chrono::system_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	}
};

struct Vector4
{
public:
	double data[4];

	Vector4()
	{
		data[0] = 0;
		data[1] = 0;
		data[2] = 0;
		data[3] = 0;
	}

	Vector4(const double x, const double y, const double z)
	{
		data[0] = x;
		data[1] = y;
		data[2] = z;
		data[3] = 0;
	}
};

// なにもしない
static void Normal(Vector4 x[], Vector4 v[], const Vector4 f[], const double m, const double dt, const std::size_t n)
{
	const double tmp = dt*dt / 2;
	const double rm = 1.0 / m;
	for (int i = 0; i < n; i++)
	{
		// a = f/m
		Vector4 a;
		for (int j = 0; j < 4; j++)
		{
			a.data[j] = f[i].data[j] * rm;
		}

		// x += v*dt + a*dt*dt/2
		for (int j = 0; j < 4; j++)
		{
			const double dxv = v[i].data[j] * dt;
			const double dxa = a.data[j] * tmp;
			const double dx = dxv + dxa;
			x[i].data[j] += dx;
		}

		// v += a*dt
		for (int j = 0; j < 4; j++)
		{
			const double dv = a.data[j] * dt;
			v[i].data[j] += dv;
		}
	}
}

// なにもしない+OpenMP
static void NormalOmp(Vector4 x[], Vector4 v[], const Vector4 f[], const double m, const double dt, const std::size_t n)
{
	const double tmp = dt*dt / 2;
	const double rm = 1.0 / m;

#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		// a = f/m
		Vector4 a;
		for (int j = 0; j < 4; j++)
		{
			a.data[j] = f[i].data[j] * rm;
		}

		// x += v*dt + a*dt*dt/2
		for (int j = 0; j < 4; j++)
		{
			const double dxv = v[i].data[j] * dt;
			const double dxa = a.data[j] * tmp;
			const double dx = dxv + dxa;
			x[i].data[j] += dx;
		}

		// v += a*dt
		for (int j = 0; j < 4; j++)
		{
			const double dv = a.data[j] * dt;
			v[i].data[j] += dv;
		}
	}
}

typedef __declspec(align(32)) Vector4 Vector4Aligned;

// SIMD(AVX)
static void Simd(Vector4 x[], Vector4 v[], const Vector4 f[], const double m, const double dt, const std::size_t n)
{
	const __m256d dt2 = _mm256_broadcast_sd(&dt);
	const double tmp = dt*dt / 2;
	const __m256d tmp2 = _mm256_broadcast_sd(&tmp);
	const double rm = 1.0 / m;
	const __m256d rm2 = _mm256_broadcast_sd(&rm);

	for (int i = 0; i < n; i++)
	{
		// a = f/m
		const Vector4Aligned fi(f[i]);
		const __m256d f2 = _mm256_load_pd(fi.data);
		const __m256d a = _mm256_mul_pd(f2, rm2);

		// x += v*dt + a*dt*dt/2
		Vector4Aligned vi(v[i]);
		Vector4Aligned xi(x[i]);
		const __m256d v2 = _mm256_load_pd(vi.data);
		const __m256d x2 = _mm256_load_pd(xi.data);
		const __m256d x3 = _mm256_fmadd_pd(v2, dt2, x2);
		const __m256d x4 = _mm256_fmadd_pd(a, tmp2, x3);
		_mm256_store_pd(xi.data, x4);
		x[i] = xi;

		// v += a*dt
		const __m256d v3 = _mm256_fmadd_pd(a, dt2, v2);
		_mm256_store_pd(vi.data, v3);
		v[i] = vi;
	}
}

// SIMD(AVX)+OpenMP
static void SimdOmp(Vector4 x[], Vector4 v[], const Vector4 f[], const double m, const double dt, const std::size_t n)
{
	const __m256d dt2 = _mm256_broadcast_sd(&dt);
	const double tmp = dt*dt / 2;
	const __m256d tmp2 = _mm256_broadcast_sd(&tmp);
	const double rm = 1.0 / m;
	const __m256d rm2 = _mm256_broadcast_sd(&rm);

#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		// a = f/m
		const Vector4Aligned fi(f[i]);
		const __m256d f2 = _mm256_load_pd(fi.data);
		const __m256d a = _mm256_mul_pd(f2, rm2);

		// x += v*dt + a*dt*dt/2
		Vector4Aligned vi(v[i]);
		Vector4Aligned xi(x[i]);
		const __m256d v2 = _mm256_load_pd(vi.data);
		const __m256d x2 = _mm256_load_pd(xi.data);
		const __m256d x3 = _mm256_fmadd_pd(v2, dt2, x2);
		const __m256d x4 = _mm256_fmadd_pd(a, tmp2, x3);
		_mm256_store_pd(xi.data, x4);
		x[i] = xi;

		// v += a*dt
		const __m256d v3 = _mm256_fmadd_pd(a, dt2, v2);
		_mm256_store_pd(vi.data, v3);
		v[i] = vi;
	}
}

int main()
{
	const std::size_t n = 100000;
	const int loop = 1000;

	std::unique_ptr<Vector4[]> f(new Vector4[n]);
	std::unique_ptr<Vector4[]> v(new Vector4[n]);
	std::unique_ptr<Vector4[]> x(new Vector4[n]);
	auto generator = [](){return static_cast<double>(1 + std::rand()) / std::rand(); };
	auto generator4 = [generator](){return Vector4(generator(), generator(), generator()); };
	std::generate_n(f.get(), n, generator4);
	std::generate_n(v.get(), n, generator4);
	std::generate_n(x.get(), n, generator4);

	Timer timer;

	const double dt = 0.1;
	const double m = 2.5;

	// なにもしない
	std::unique_ptr<Vector4[]> vNormal(new Vector4[n]);
	std::unique_ptr<Vector4[]> xNormal(new Vector4[n]);
	{
		std::copy_n(v.get(), n, vNormal.get());
		std::copy_n(x.get(), n, xNormal.get());

		std::cout << "Normal: ";
		timer.Start();
		for (int i = 0; i < loop; i++)
		{
			Normal(xNormal.get(), vNormal.get(), f.get(), m, dt, n);
		}
		const auto normalTime = timer.Time();
		std::cout << normalTime.count() << "[ms]" << std::endl;
	}

	// SIMD
	std::unique_ptr<Vector4[]> vSimd(new Vector4[n]);
	std::unique_ptr<Vector4[]> xSimd(new Vector4[n]);
	{
		std::copy_n(v.get(), n, vSimd.get());
		std::copy_n(x.get(), n, xSimd.get());

		std::cout << "Simd  : ";
		timer.Start();
		for (int i = 0; i < loop; i++)
		{
			Simd(xSimd.get(), vSimd.get(), f.get(), m, dt, n);
		}
		const auto normalTime = timer.Time();
		std::cout << normalTime.count() << "[ms]" << std::endl;
	}

#pragma omp parallel
#pragma omp master
	{
		std::cout << "OpenMP: " << omp_get_num_threads() << "threads" << std::endl;
	}

	// なにもしない
	std::unique_ptr<Vector4[]> vNormalOmp(new Vector4[n]);
	std::unique_ptr<Vector4[]> xNormalOmp(new Vector4[n]);
	{
		std::copy_n(v.get(), n, vNormalOmp.get());
		std::copy_n(x.get(), n, xNormalOmp.get());

		std::cout << "Normal: ";
		timer.Start();
		for (int i = 0; i < loop; i++)
		{
			NormalOmp(xNormalOmp.get(), vNormalOmp.get(), f.get(), m, dt, n);
		}
		const auto normalOmpTime = timer.Time();
		std::cout << normalOmpTime.count() << "[ms]" << std::endl;
	}

	// SIMD
	std::unique_ptr<Vector4[]> vSimdOmp(new Vector4[n]);
	std::unique_ptr<Vector4[]> xSimdOmp(new Vector4[n]);
	{
		std::copy_n(v.get(), n, vSimdOmp.get());
		std::copy_n(x.get(), n, xSimdOmp.get());

		std::cout << "Simd  : ";
		timer.Start();
		for (int i = 0; i < loop; i++)
		{
			SimdOmp(xSimdOmp.get(), vSimdOmp.get(), f.get(), m, dt, n);
		}
		const auto normalTime = timer.Time();
		std::cout << normalTime.count() << "[ms]" << std::endl;
	}


	// エラーチェック
	const double eps = 1e-8;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			const double errorV = std::abs(vNormal[i].data[j] - vSimd[i].data[j]) / vNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error V[" << i << "][" << j << "]: " << errorV << std::endl;
			}

			const double errorX = std::abs(xNormal[i].data[j] - xSimd[i].data[j]) / xNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error X[" << i << "][" << j << "]" << errorX << std::endl;
			}
		}
	}

	system("pause");
	return 0;
}