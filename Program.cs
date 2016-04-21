using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Numerics;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics;
using System.Diagnostics;

namespace PFCalc
{
    class Program
    {
        private static Complex i = Complex.ImaginaryOne;

        static void Main(string[] args)
        {
            //test1();
            var files = new List<string>();
            var param = new List<string>();
            foreach(var item in args)
            {
                if(item.StartsWith("-"))
                    param.Add(item.Substring(1).Trim().ToLower());
                else
                    files.Add(item.Trim());
            }
            Control.UseManaged();
            bool? rect = null;
            bool? dense = null;
            foreach(var item in param)
            {
                switch(item)
                {
                case "kml":
                    Control.UseNativeMKL();
                    break;
                case "rect":
                    if(rect == null)
                    {
                        rect = true;
                    }
                    else
                    {
                        Console.WriteLine($"参数重复：{item}");
                        Console.WriteLine();
                        return;
                    }
                    break;
                case "polar":
                    if(rect == null)
                    {
                        rect = false;
                    }
                    else
                    {
                        Console.WriteLine($"参数重复：{item}");
                        Console.WriteLine();
                        return;
                    }
                    break;
                case "dense":
                    if(dense == null)
                    {
                        dense = true;
                    }
                    else
                    {
                        Console.WriteLine($"参数重复：{item}");
                        Console.WriteLine();
                        return;
                    }
                    break;
                case "sparse":
                    if(dense == null)
                    {
                        dense = false;
                    }
                    else
                    {
                        Console.WriteLine($"参数重复：{item}");
                        Console.WriteLine();
                        return;
                    }
                    break;
                default:
                    Console.WriteLine($"未知参数：{item}");
                    Console.WriteLine();
                    return;
                }
            }
            var Dense = dense ?? true;
            var Rect = rect ?? true;
            Solver solver;
            var s = Stopwatch.StartNew();
            if(files.Count == 2)
            {
                if(Dense)
                {
                    if(Rect)
                        solver = solve<RectangularSolver>(files[0], files[1]);
                    else
                        solver = solve<PolarSolver>(files[0], files[1]);
                }
                else
                {
                    if(Rect)
                        solver = solve<SparseRectSolver>(files[0], files[1]);
                    else
                        solver = solve<SparsePolarSolver>(files[0], files[1]);
                }
            }
            else if(files.Count == 6)
            {
                if(Dense)
                {
                    if(Rect)
                        solver = solve<RectangularSolver>(files[0], files[1], files[2], files[3], files[4], files[5]);
                    else
                        solver = solve<PolarSolver>(files[0], files[1], files[2], files[3], files[4], files[5]);
                }
                else
                {
                    if(Rect)
                        solver = solve<SparseRectSolver>(files[0], files[1], files[2], files[3], files[4], files[5]);
                    else
                        solver = solve<SparsePolarSolver>(files[0], files[1], files[2], files[3], files[4], files[5]);
                }
            }
            else
            {
                Console.WriteLine($"参数数目不正确");
                Console.WriteLine();
                return;
            }
            s.Stop();
            Console.WriteLine("求解结束");
            Console.WriteLine($"求解器：{solver.GetType()}");
            Console.WriteLine($"运算器：{Control.LinearAlgebraProvider.GetType()}");
            Console.WriteLine($"结果：{(solver.NodeResult.Any(c => c.IsNaN()) ? "失败" : "成功")}");
            Console.WriteLine($"迭代次数：{solver.IterCount}");
            Console.WriteLine($"用时：{s.Elapsed}");
            Console.WriteLine();
        }

        private static T solve<T>(string pFile, string qFile, string vFile, string rFile, string yFile, string soluFile)
            where T : Solver, new()
        {
            var p = DelimitedReader.Read<double>(pFile).Column(0);
            var q = DelimitedReader.Read<double>(qFile).Column(0);
            var u = DelimitedReader.Read<double>(vFile).Column(0);
            var y = DelimitedReader.Read<Complex>(yFile);
            Complex r = DelimitedReader.Read<Complex>(rFile)[0, 0];
            T s = new T();
            s.Init(p, q, u, r, y);
            s.Solve();
            var solu = new Solution(s);
            DataIO.WriteJson(soluFile, solu);
            return s;
        }

        private static T solve<T>(string dataFile, string soluFile)
            where T : Solver, new()
        {
            var problem = DataIO.ReadJson(dataFile);
            var s = problem.GetSolver<T>();
            s.Solve();
            var solu = new Solution(problem, s);
            DataIO.WriteJson(soluFile, solu);
            return s;
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
            var solver = new SparsePolarSolver();
            solver.Init(p, q, u, r, y);
            solver.Solve();
            var solver2 = new RectangularSolver();
            solver2.Init(p, q, u, r, y);
            solver2.Solve();
            var solver3 = new PolarSolver();
            solver3.Init(p, q, u, r, y);
            solver3.Solve();
        }
    }
}
