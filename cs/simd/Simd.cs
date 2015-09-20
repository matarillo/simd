namespace LWisteria.Simd
{
	using System.Linq;

	struct Vector4
	{
		public double[] data;

		public const int Count = 4;

		public static Vector4 Create()
		{
			Vector4 vec = new Vector4();
			vec.data = new double[Count] { 0, 0, 0, 0 };
			return vec;
		}

		public Vector4(double x, double y, double z)
		{
			data = new double[Count];
			data[0] = x;
			data[1] = y;
			data[2] = z;
			data[3] = 0;
		}

		public Vector4(Vector4 v)
		{
			data = new double[Count];
			data[0] = v.data[0];
			data[1] = v.data[1];
			data[2] = v.data[2];
			data[3] = v.data[3];
		}
	}

	struct Vector4Simd
	{
		public System.Numerics.Vector<double>[] data;

		public static readonly int Count = (int)System.Math.Floor((double)Vector4.Count / System.Numerics.Vector<double>.Count);

		public static Vector4Simd Create()
		{
			var v = new Vector4Simd();
            v.data = new System.Numerics.Vector<double>[Count];
			return v;
		}

		public Vector4Simd(Vector4 v)
		{
			data = new System.Numerics.Vector<double>[Count];

			for (int i = 0; i < Count; i++)
			{
				data[i] = new System.Numerics.Vector<double>(v.data, i * System.Numerics.Vector<double>.Count);
			}
		}

		public double this[int i]
		{
			get
			{
				int ii = i / System.Numerics.Vector<double>.Count;
				int j = i % System.Numerics.Vector<double>.Count;
				return data[ii][j];
            }
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
				var a = Vector4.Create();
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

		// SIMD
		static void Simd(Vector4Simd[] x, Vector4Simd[] v, Vector4Simd[] f, double m, double dt, int n)
		{
			double tmp = dt * dt / 2;
			double rm = 1.0 / m;

			for (int i = 0; i < n; i++)
			{
				// a = f/m
				var a = Vector4Simd.Create();
				for (int j = 0; j < Vector4Simd.Count; j++)
				{
					a.data[j] = f[i].data[j] * rm;
				}

				// x += v*dt + a*dt*dt/2
				for (int j = 0; j < Vector4Simd.Count; j++)
				{
					x[i].data[j] += v[i].data[j] * dt + a.data[j] * tmp;
				}

				// v += a*dt
				for (int j = 0; j < Vector4Simd.Count; j++)
				{
					v[i].data[j] += a.data[j] * dt;
				}
			}
		}

		static int Main()
		{
			const int n = 100000;
			const int loop = 1000;

			var rand = new System.Random(713);
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
			Vector4[] vNormal;
			Vector4[] xNormal;
			{
				xNormal = x.Select(val => new Vector4(val)).ToArray();
				vNormal = v.Select(val => new Vector4(val)).ToArray();

				System.Console.Write("Normal: ");
				stopwatch.Restart();

				for (int i = 0; i < loop; i++)
				{
					Normal(xNormal, vNormal, f, m, dt, n);
				}
				var time = stopwatch.ElapsedMilliseconds;
				System.Console.WriteLine("{0} [ms]", time);
			}

			// SIMD
			var vSimd = new Vector4Simd[n];
			var xSimd = new Vector4Simd[n];
			{
				xSimd = x.Select(val => new Vector4Simd(val)).ToArray();
				vSimd = v.Select(val => new Vector4Simd(val)).ToArray();

				var fSimd = f.Select(val => new Vector4Simd(val)).ToArray();

				System.Console.Write("Simd: ");
				stopwatch.Restart();

				for (int i = 0; i < loop; i++)
				{
					Simd(xSimd, vSimd, fSimd, m, dt, n);
				}

				var time = stopwatch.ElapsedMilliseconds;
				System.Console.WriteLine("{0} [ms]", time);
			}

			// エラーチェック
			const double eps = 1e-8;
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					double errorX = System.Math.Abs((xNormal[i].data[j] - xSimd[i][j]) / xNormal[i].data[j]);
					if (errorX > eps)
					{
						System.Console.WriteLine("errorX[{0}][{1}] = {2}", i, j, errorX);
					}

					double errorV = System.Math.Abs((vNormal[i].data[j] - vSimd[i][j]) / vNormal[i].data[j]);
					if (errorV > eps)
					{
						System.Console.WriteLine("errorV[{0}][{1}] = {2}", i, j, errorV);
					}
				}
			}

			return System.Environment.ExitCode;
		}
	}
}