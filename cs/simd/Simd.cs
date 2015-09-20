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
		public System.Numerics.Vector<double> data0;
		public System.Numerics.Vector<double> data1;
		public System.Numerics.Vector<double> data2;
		public System.Numerics.Vector<double> data3;
		// 最小でも1SIMDは1要素を計算するはず

		public static readonly int Count = (int)System.Math.Floor((double)Vector4.Count / System.Numerics.Vector<double>.Count);

		private delegate void Creator(ref Vector4Simd simd, Vector4 v);
		private static readonly Creator creator =
		(Count == 4) ?
		((ref Vector4Simd v, Vector4 val) =>
		{
			// 1SIMD = 1要素
			v.data0 = new System.Numerics.Vector<double>(val.data, 0);
			v.data1 = new System.Numerics.Vector<double>(val.data, 1);
			v.data2 = new System.Numerics.Vector<double>(val.data, 2);
			v.data3 = new System.Numerics.Vector<double>(val.data, 3);
		}) : ((Count == 2) ?
		((ref Vector4Simd v, Vector4 val) =>
		{
			// 1SIMD = 2要素
			v.data0 = new System.Numerics.Vector<double>(val.data, 0);
			v.data1 = new System.Numerics.Vector<double>(val.data, 2);
		}) :
		// 3要素は多分ないはずなので無視
		((Creator)((ref Vector4Simd v, Vector4 val) =>
		{
			// 1SIMD = 4要素以上
			v.data0 = new System.Numerics.Vector<double>(val.data, 0);
		})));
		public static Vector4Simd Create(Vector4 v)
		{
			var simd = new Vector4Simd();
			creator(ref simd, v);
			return simd;
		}

		private static readonly System.Func<Vector4Simd, double>[] getter =
		{
			/* 0番目の要素 */ (Count == 4) ? (v => v.data0[0]) : ((Count == 2) ? (v => v.data0[0]) : (System.Func<Vector4Simd, double>)(v => v.data0[0])),
			/* 1番目の要素 */ (Count == 4) ? (v => v.data1[0]) : ((Count == 2) ? (v => v.data0[1]) : (System.Func<Vector4Simd, double>)(v => v.data0[1])),
			/* 2番目の要素 */ (Count == 4) ? (v => v.data2[0]) : ((Count == 2) ? (v => v.data1[0]) : (System.Func<Vector4Simd, double>)(v => v.data0[2])),
			/* 3番目の要素 */ (Count == 4) ? (v => v.data3[0]) : ((Count == 2) ? (v => v.data1[1]) : (System.Func<Vector4Simd, double>)(v => v.data0[3])),
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

		private static readonly System.Func<Vector4Simd, double, Vector4Simd> force =
		(Vector4Simd.Count == 4) ?
		((Vector4Simd f, double rm) =>
		{
			// 1SIMD = 1要素
			var a = new Vector4Simd();
			a.data0 = f.data0 * rm;
			a.data1 = f.data1 * rm;
			a.data2 = f.data2 * rm;
			a.data3 = f.data3 * rm;
			return a;
		}) : ((Vector4Simd.Count == 2) ?
		((Vector4Simd f, double rm) =>
		{
			// 1SIMD = 2要素
			var a = new Vector4Simd();
			a.data0 = f.data0 * rm;
			a.data1 = f.data1 * rm;
			return a;
		}) :
		// 3要素は多分ないはずなので無視
		((System.Func<Vector4Simd, double, Vector4Simd>)((Vector4Simd f, double rm) =>
		{
			// 1SIMD = 4要素以上
			var a = new Vector4Simd();
			a.data0 = f.data0 * rm;
			return a;
		})));

		private delegate void Move(ref Vector4Simd x, Vector4Simd v, Vector4Simd a, double dt, double halfDt2);
		private static readonly Move move =
		(Vector4Simd.Count == 4) ?
		((ref Vector4Simd x, Vector4Simd v, Vector4Simd a, double dt, double halfDt2) =>
		{
			// 1SIMD = 1要素
			x.data0 += v.data0 * dt + a.data0 * halfDt2;
			x.data1 += v.data1 * dt + a.data1 * halfDt2;
			x.data2 += v.data2 * dt + a.data2 * halfDt2;
			x.data3 += v.data3 * dt + a.data3 * halfDt2;
		}) : ((Vector4Simd.Count == 2) ?
		((ref Vector4Simd x, Vector4Simd v, Vector4Simd a, double dt, double halfDt2) =>
		{
			// 1SIMD = 2要素
			x.data0 += v.data0 * dt + a.data0 * halfDt2;
			x.data1 += v.data1 * dt + a.data1 * halfDt2;
		}) :
		// 3要素は多分ないはずなので無視
		((Move)((ref Vector4Simd x, Vector4Simd v, Vector4Simd a, double dt, double halfDt2) =>
		{
			// 1SIMD = 4要素以上
			x.data0 += v.data0 * dt + a.data0 * halfDt2;
		})));

		private delegate void Accelerate(ref Vector4Simd v, Vector4Simd a, double dt);
		private static readonly Accelerate accelerate =
		(Vector4Simd.Count == 4) ?
		((ref Vector4Simd v, Vector4Simd a, double dt) =>
		{
			// 1SIMD = 1要素
			v.data0 += a.data0 * dt;
			v.data1 += a.data1 * dt;
			v.data2 += a.data2 * dt;
			v.data3 += a.data3 * dt;
		}) : ((Vector4Simd.Count == 2) ?
		((ref Vector4Simd v, Vector4Simd a, double dt) =>
		{
			// 1SIMD = 2要素
			v.data0 += a.data0 * dt;
			v.data1 += a.data1 * dt;
		}) :
		// 3要素は多分ないはずなので無視
		((Accelerate)((ref Vector4Simd v, Vector4Simd a, double dt) =>
		{
			// 1SIMD = 4要素以上
			v.data0 += a.data0 * dt;
		})));

		// SIMD
		static void Simd(Vector4Simd[] x, Vector4Simd[] v, Vector4Simd[] f, double m, double dt, int n)
		{
			double halfDt2 = dt * dt / 2;
			double rm = 1.0 / m;

			for (int i = 0; i < n; i++)
			{
				// a = f/m
				var a = force(f[i], rm);

				// x += v*dt + a*dt*dt/2
				move(ref x[i], v[i], a, dt, halfDt2);

				// v += a*dt
				accelerate(ref v[i], a, dt);
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
				xSimd = x.Select(val => Vector4Simd.Create(val)).ToArray();
				vSimd = v.Select(val => Vector4Simd.Create(val)).ToArray();

				var fSimd = f.Select(val => Vector4Simd.Create(val)).ToArray();

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