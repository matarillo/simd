#define __CL_ENABLE_EXCEPTIONS

#pragma warning(push, 1)
#pragma warning(disable: 4996)
#include <iostream>
#include <memory>
#include <algorithm>
#include <chrono>
#include <CL/cl.hpp>
#pragma warning(pop)

#define OCL_EXTERNAL_INCLUDE(x) #x
const char srcStr[] =
#include "ocl.cl"
;

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

	Vector4& operator=(const Vector4& src)
	{
		data[0] = src.data[0];
		data[1] = src.data[1];
		data[2] = src.data[2];
		data[3] = src.data[3];

		return *this;
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

int main()
{
	// バッファーを作成＆引数設定
	const std::size_t n = 100000;
	const int loop = 1000;

	const std::size_t size = sizeof(Vector4) * n;

	std::unique_ptr<Vector4[]> f(new Vector4[n]);
	std::unique_ptr<Vector4[]> v(new Vector4[n]);
	std::unique_ptr<Vector4[]> x(new Vector4[n]);
	auto generator = [](){return static_cast<double>(1 + std::rand()) / std::rand(); };
	auto generator4 = [generator](){return Vector4(generator(), generator(), generator()); };
	std::generate_n(f.get(), n, generator4);
	std::generate_n(v.get(), n, generator4);
	std::generate_n(x.get(), n, generator4);

	const double dt = 0.1;
	const double m = 2.5;

	Timer timer;

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

	std::unique_ptr<Vector4[]> vOcl(new Vector4[n]);
	std::unique_ptr<Vector4[]> xOcl(new Vector4[n]);
	{
		std::copy_n(v.get(), n, vOcl.get());
		std::copy_n(x.get(), n, xOcl.get());

		std::cout << "OpenCL: ";
		timer.Start();

		// プラットフォーム取得（複数ある場合は一番最後）
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		const auto& platform = *(platforms.rbegin());

		// デバイスを取得（複数ある場合は一番最後）
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		const auto& device = *(devices.rbegin());

		// コンテキスト作成
		const cl::Context context(device);

		// プログラムの作成＆ビルド
		cl::Program::Sources sources;
		sources.push_back(std::make_pair(srcStr, std::strlen(srcStr)));
		cl::Program program(context, sources);
		try
		{
			program.build(devices);
		}
		// OpenCL例外があった場合
		catch (cl::Error error)
		{
			// ビルドエラーなら
			if (error.err() == CL_BUILD_PROGRAM_FAILURE)
			{
				// ビルドログを表示
				std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			}
			else
			{
				std::cout << "Unknown error #" << error.err() << " @ " << error.what() << std::endl;
			}
			system("pause");
			return -1;
		}

		// カーネルを作成
		const std::string KERNEL_FUNCTION_NAME = "ocl";
		cl::Kernel kernel(program, KERNEL_FUNCTION_NAME.c_str());

		cl::Buffer bufferF(context, CL_MEM_READ_ONLY, size);
		cl::Buffer bufferX(context, CL_MEM_READ_WRITE, size);
		cl::Buffer bufferV(context, CL_MEM_READ_WRITE, size);
		kernel.setArg(0, bufferX);
		kernel.setArg(1, bufferV);
		kernel.setArg(2, bufferF);
		kernel.setArg(3, static_cast<cl_double>(m));
		kernel.setArg(4, static_cast<cl_double>(dt));
		kernel.setArg(5, static_cast<cl_ulong>(n));

		// キュー作成
		const cl::CommandQueue queue(context, device);
		// ホスト->デバイス
		queue.enqueueWriteBuffer(bufferF, CL_FALSE, 0, size, f.get());
		queue.enqueueWriteBuffer(bufferX, CL_FALSE, 0, size, xOcl.get());
		queue.enqueueWriteBuffer(bufferV, CL_FALSE, 0, size, vOcl.get());

		for (int i = 0; i < loop; i++)
		{
			// 実行
			cl::Event kernelEvent;
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange, NULL, &kernelEvent);
			kernelEvent.wait();
		}

		// デバイス->ホスト
		queue.enqueueReadBuffer(bufferX, CL_TRUE, 0, size, xOcl.get());
		queue.enqueueReadBuffer(bufferV, CL_TRUE, 0, size, vOcl.get());

		const auto oclTime = timer.Time();
		std::cout << oclTime.count() << "[ms]" << std::endl;

		std::cout << "== Platform : " << platform() << " ==" << std::endl;
		std::cout <<
			"Name    : " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl <<
			"Vendor  : " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl <<
			"Version : " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

		std::cout << "== Device : " << device() << " ==" << std::endl;
		std::cout <<
			"Name                                             : " << device.getInfo<CL_DEVICE_NAME>() << std::endl <<
			"Vendor                                           : " << device.getInfo<CL_DEVICE_VENDOR>() << " (ID:" << device.getInfo<CL_DEVICE_VENDOR_ID>() << ")" << std::endl <<
			"Version                                          : " << device.getInfo<CL_DEVICE_VERSION>() << std::endl <<
			"Driver version                                   : " << device.getInfo<CL_DRIVER_VERSION>() << std::endl <<
			"OpenCL C version                                 : " << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
	}

	// エラーチェック
	const double eps = 1e-8;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			const double errorV = std::abs(vNormal[i].data[j] - vOcl[i].data[j]) / vNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error V[" << i << "][" << j << "]: " << errorV << std::endl;
			}

			const double errorX = std::abs(xNormal[i].data[j] - xOcl[i].data[j]) / xNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error X[" << i << "][" << j << "]" << errorX << std::endl;
			}
		}
	}

	system("pause");
	return 0;
}