using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Numerics;
using MathNet.Numerics.Data.Text;

namespace PFCalc
{
    class Program
    {
        private static Complex i = Complex.ImaginaryOne;

        static void Main(string[] args)
        {
            test3();
        }

        private static void test3()
        {
            var p = DelimitedReader.Read<double>(@"c:\Users\liuzh\desktop\p").Column(0);
            var q = DelimitedReader.Read<double>(@"c:\Users\liuzh\desktop\q").Column(0);
            var u = DelimitedReader.Read<double>(@"c:\Users\liuzh\desktop\u").Column(0);
            var y = DelimitedReader.Read<Complex>(@"c:\Users\liuzh\desktop\y");
            Complex r = 0.982;
            var solver = new RectangularSolver();
            solver.Init(p, q, u, r, y);
            solver.Solve();
            var solver2 = new PolarSolver();
            solver2.Init(p, q, u, r, y);
            solver2.Solve();
        }

        private static void test2()
        {
            var problem = DataIO.ReadJson("data.json");
            var s = problem.GetSolver<RectangularSolver>();
            s.Solve();
        }

        static void test1()
        {
            var p = CreateVector.DenseOfArray(new[] { -0.3, -0.55, 0.5 });
            var q = CreateVector.DenseOfArray(new[] { -0.18, -0.13 });
            var u = CreateVector.DenseOfArray(new[] { 1.1 });
            Complex r = 1.05;
            var y = CreateMatrix.DenseOfArray(new[,]
            {
                { 1.0421 - i * 8.2429, -0.5882 + i * 2.3529, i * 3.6666, -0.4539 + i * 1.8911 },
                { 0, 1.0690 - i * 4.7274, 0, -0.4808 + i * 2.4038 },
                { 0, 0, -i * 3.3333, 0 },
                { 0, 0, 0, 0.9346 - i * 4.2616 }
            });
            y = y + y.StrictlyUpperTriangle().Transpose();
            var solver = new RectangularSolver();
            solver.Init(p, q, u, r, y);
            solver.Solve();
            var solver2 = new PolarSolver();
            solver2.Init(p, q, u, r, y);
            solver2.Solve();
        }
    }
}
