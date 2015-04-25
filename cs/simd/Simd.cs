﻿namespace LWisteria.Simd
{
	using System.Linq;

	class Vector4
	{
		public double[] data = new double[4];

		public Vector4()
		{
			data[0] = 0;
			data[1] = 0;
			data[2] = 0;
			data[3] = 0;
		}

		public Vector4(double x, double y, double z)
		{
			data[0] = x;
			data[1] = y;
			data[2] = z;
			data[3] = 0;
		}
	}

	static class SimdMain
	{
		// なにもしない
		static void Normal(Vector4[] x, Vector4[] v, Vector4[] f, double m, double dt, int n)
		{
			double tmp = dt * dt / 2;
			double rm = 1.0 / m;

			for (int i = 0; i < n; i++)
			{
				// a = f/m
				var a = new Vector4();
				for (int j = 0; j < 4; j++)
				{
					a.data[j] = f[i].data[j] * rm;
				}

				// x += v*dt + a*dt*dt/2
				for (int j = 0; j < 4; j++)
				{
					double dxv = v[i].data[j] * dt;
					double dxa = a.data[j] * tmp;
					double dx = dxv + dxa;
					x[i].data[j] += dx;
				}

				// v += a*dt
				for (int j = 0; j < 4; j++)
				{
					double dv = a.data[j] * dt;
					v[i].data[j] += dv;
				}
			}
		}

		static int Main()
		{
			const int n = 100000;
			const int loop = 1000;

			var rand = new System.Random();
			var f = new Vector4[n];
			var v = new Vector4[n];
			var x = new Vector4[n];
			System.Func<double> generator = () => (rand.NextDouble());
			System.Func<Vector4> generator4 = () => (new Vector4(generator(), generator(), generator()));
			f = f.Select(val => generator4()).ToArray();
			v = v.Select(val => generator4()).ToArray();
			x = x.Select(val => generator4()).ToArray();

			var stopwatch = new System.Diagnostics.Stopwatch();

			const double dt = 0.1;
			const double m = 2.5;

			// なにもしない
			var vNormal = new Vector4[n];
			var xNormal = new Vector4[n];
			{
				x.CopyTo(xNormal, 0);
				v.CopyTo(vNormal, 0);

				System.Console.Write("Normal: ");
				stopwatch.Restart();
				for (int i = 0; i < loop; i++)
				{
					Normal(xNormal, vNormal, f, m, dt, n);
				}
				var time = stopwatch.ElapsedMilliseconds;
				System.Console.WriteLine("{0} [ms]", time);
			}

			System.Console.WriteLine("Press Any key...");
			System.Console.ReadKey();
			return System.Environment.ExitCode;
		}
	}
}