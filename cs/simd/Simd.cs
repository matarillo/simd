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

	struct Vector4Simd128
	{
		public System.Numerics.Vector<double> data0;
		public System.Numerics.Vector<double> data1;

		public Vector4Simd128(System.Numerics.Vector<double> data0, System.Numerics.Vector<double> data1)
		{
			this.data0 = data0;
			this.data1 = data1;
		}

		private delegate void Creator(ref Vector4Simd128 simd, Vector4 v);
		private static readonly Creator creator =
		(ref Vector4Simd128 v, Vector4 val) =>
		{
			// 1SIMD = 2要素
			v.data0 = new System.Numerics.Vector<double>(val.data, 0);
			v.data1 = new System.Numerics.Vector<double>(val.data, 2);
		};

		public static Vector4Simd128 Create(Vector4 v)
		{
			var simd = new Vector4Simd128();
			creator(ref simd, v);
			return simd;
		}

		private static readonly System.Func<Vector4Simd128, double>[] getter =
		{
			/* 0番目の要素 */ v => v.data0[0],
			/* 1番目の要素 */ v => v.data0[1],
			/* 2番目の要素 */ v => v.data1[0],
			/* 3番目の要素 */ v => v.data1[1],
		};

		public double this[int i]
		{
			get
			{
				return getter[i](this);
			}
		}
	}

	static class SimdMain
	{
		// なにもしない
		static void Normal(Vector4[] x, Vector4[] v, Vector4[] f, double m, double dt, int n)
		{
			double halfDt2 = dt * dt / 2;
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
					double dxa = a.data[j] * halfDt2;
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
		static void Simd(Vector4Simd128[] x, Vector4Simd128[] v, Vector4Simd128[] f, double m, double dt, int n)
		{
			double halfDt2 = dt * dt / 2;
			double rm = 1.0 / m;

			for (int i = 0; i < n; i++)
			{
				// a = f/m
				var a = new Vector4Simd128(
					f[i].data0 * rm,
					f[i].data1 * rm);

				// x += v*dt + a*dt*dt/2
				x[i] = new Vector4Simd128(
					x[i].data0 + v[i].data0 * dt + a.data0 * halfDt2,
					x[i].data1 + v[i].data1 * dt + a.data1 * halfDt2);

				// v += a*dt
				v[i] = new Vector4Simd128(
					v[i].data0 + a.data0 * dt,
					v[i].data1 + a.data1 * dt);
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

			// SIMD (128bit)
			if (System.Numerics.Vector<double>.Count == 2)
			{
				var vSimd = new Vector4Simd128[n];
				var xSimd = new Vector4Simd128[n];
				{
					xSimd = x.Select(val => Vector4Simd128.Create(val)).ToArray();
					vSimd = v.Select(val => Vector4Simd128.Create(val)).ToArray();

					var fSimd = f.Select(val => Vector4Simd128.Create(val)).ToArray();

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
			}

			return System.Environment.ExitCode;
		}
	}
}
