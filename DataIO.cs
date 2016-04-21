using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;

namespace PFCalc
{
    public static class DataIO
    {
        public static Problem ReadJson(string filePath)
        {
            var value = File.ReadAllText(filePath);
            return JsonConvert.DeserializeObject<Problem>(value, compC);
        }

        public static void WriteJson(string filePath, Solution solu)
        {
            var st = JsonConvert.SerializeObject(solu, Formatting.Indented, compC, compE);
            File.WriteAllText(filePath, st);
        }

        private static readonly JsonConverter compC = new complexConverter();
        private static readonly JsonConverter compE = new Newtonsoft.Json.Converters.StringEnumConverter();

        private class complexConverter : JsonConverter
        {
            public override bool CanConvert(Type objectType)
            {
                return objectType == typeof(Complex);
            }

            public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
            {
                var real = reader.ReadAsDouble() ?? 0;
                var imaginary = reader.ReadAsDouble() ?? 0;
                reader.Read();
                return new Complex(real, imaginary);
            }

            public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
            {
                var v = (Complex)value;
                writer.WriteStartArray();
                writer.WriteValue(v.Real);
                writer.WriteValue(v.Imaginary);
                writer.WriteEndArray();
            }
        }
    }

    public class Problem
    {
        public ICollection<PVNode> PVNode
        {
            get;
            set;
        }

        public ICollection<PQNode> PQNode
        {
            get;
            set;
        }

        public RelaxNode RelaxNode
        {
            get;
            set;
        }

        public ICollection<GroundingBranch> Grounding
        {
            get;
            set;
        }

        public ICollection<Transformer> Transformer
        {
            get;
            set;
        }

        public ICollection<Transmission> Transmission
        {
            get;
            set;
        }

        public T GetSolver<T>() where T : Solver, new()
        {
            var i = 0;
            var p = CreateVector.Dense<double>(PQNode.Count + PVNode.Count);
            var q = CreateVector.Dense<double>(PQNode.Count);
            var u = CreateVector.Dense<double>(PVNode.Count);
            var r = RelaxNode.ToComplex();
            var y = CreateMatrix.Dense<Complex>(PQNode.Count + PVNode.Count + 1, PQNode.Count + PVNode.Count + 1);
            foreach(var item in PQNode)
            {
                p[i] = item.Pg;
                q[i] = item.Qg;
                Mapper.Add(item.Name, i++);
            }
            var j = 0;
            foreach(var item in PVNode)
            {
                p[i] = item.Pg;
                u[j++] = item.U;
                Mapper.Add(item.Name, i++);
            }
            Mapper.Add(RelaxNode.Name, i);

            foreach(var item in Grounding.Concat<Branch>(Transformer).Concat(Transmission))
            {
                item.IndexMapper = s => Mapper[s];
                item.Apply(y);
            }

            T a = new T();
            a.Init(p, q, u, r, y);
            return a;
        }

        [JsonIgnore]
        public IEnumerable<Node> Node
        {
            get
            {
                foreach(var item in PQNode)
                {
                    yield return item;
                }
                foreach(var item in PVNode)
                {
                    yield return item;
                }
                yield return RelaxNode;
            }
        }

        [JsonIgnore]
        public IDictionary<string, int> Mapper
        {
            get;
            private set;
        } = new Dictionary<string, int>();
    }

    public abstract class Node
    {
        public string Name
        {
            get;
            set;
        }
    }

    public class PVNode : Node
    {
        public double Pg
        {
            get;
            set;
        }

        public double U
        {
            get;
            set;
        }
    }

    public class PQNode : Node
    {
        public double Pg
        {
            get;
            set;
        }

        public double Qg
        {
            get;
            set;
        }
    }

    public class RelaxNode : Node
    {
        public double U
        {
            get;
            set;
        }

        public double Delta
        {
            get;
            set;
        }

        public Complex ToComplex()
        {
            return Complex.FromPolarCoordinates(U, Delta * Math.PI / 180);
        }
    }

    public class ResultNode : Node
    {
        public double P
        {
            get;
            set;
        }

        public double Q
        {
            get;
            set;
        }

        public double U
        {
            get;
            set;
        }

        public double Delta
        {
            get;
            set;
        }
    }

    public abstract class Branch
    {
        public abstract void Apply(Matrix<Complex> yMatrix);

        [JsonIgnore]
        public Func<string, int> IndexMapper
        {
            get;
            set;
        }
    }

    public class GroundingBranch : Branch
    {
        public string Node
        {
            get;
            set;
        }

        public Complex Y
        {
            get;
            set;
        }

        [JsonIgnore]
        public int NodeIndex => IndexMapper(Node);

        public override void Apply(Matrix<Complex> yMatrix)
        {
            var n = NodeIndex;
            yMatrix[n, n] += Y;
        }
    }

    public abstract class ConnectionBranch : Branch
    {
        public string LeftNode
        {
            get;
            set;
        }

        public string RightNode
        {
            get;
            set;
        }

        [JsonIgnore]
        public int LeftNodeIndex => IndexMapper(LeftNode);

        [JsonIgnore]
        public int RightNodeIndex => IndexMapper(RightNode);
    }

    public class Transmission : ConnectionBranch
    {
        public Complex Y
        {
            get; set;
        }

        public Complex Z
        {
            get; set;
        }

        public TransmissionModule Module
        {
            get; set;
        } = TransmissionModule.Middle;

        public override void Apply(Matrix<Complex> yMatrix)
        {
            Complex y_2 = 0, z = 0;
            switch(Module)
            {
            case TransmissionModule.Middle:
                z = Z;
                y_2 = Y / 2;
                break;
            case TransmissionModule.Long:
                var sqrtZY = Complex.Sqrt(Z * Y);
                z = Z * Complex.Sinh(sqrtZY) / sqrtZY;
                y_2 = Y * Complex.Tanh(sqrtZY / 2) / sqrtZY;
                break;
            case TransmissionModule.MidLong:
                var ZY = Z * Y;
                z = Z * (1 + ZY / 6);
                y_2 = Y * (1 - ZY / 12) / 2;
                break;
            case TransmissionModule.Short:
                z = Z;
                y_2 = 0;
                break;
            }
            var ys = Complex.Reciprocal(z);
            var l = LeftNodeIndex;
            var r = RightNodeIndex;
            yMatrix[l, l] += y_2 + ys;
            yMatrix[r, r] += y_2 + ys;
            yMatrix[l, r] -= ys;
            yMatrix[r, l] -= ys;
        }
    }

    public enum TransmissionModule
    {
        Middle,
        Long,
        MidLong,
        Short
    }

    public class Transformer : ConnectionBranch
    {
        public Complex Z
        {
            get;
            set;
        }

        public double K
        {
            get;
            set;
        }

        public override void Apply(Matrix<Complex> yMatrix)
        {
            var l = LeftNodeIndex;
            var r = RightNodeIndex;
            var Y = Complex.Reciprocal(Z);
            yMatrix[l, l] += Y;
            var ky = K * Y;
            yMatrix[l, r] -= ky;
            yMatrix[r, l] -= ky;
            yMatrix[r, r] += K * ky;
        }
    }

    public class PowerFlow
    {
        public string From
        {
            get;
            set;
        }

        public string To
        {
            get; set;
        }

        public double P
        {
            get; set;
        }

        public double Q
        {
            get; set;
        }
    }

    public class Solution
    {
        public Solution(Problem problem, Solver solver)
        {
            OriginalProblem = problem;
            var y = solver.Y;
            var u = solver.NodeResult;
            var i = y * u;
            var s = u.PointwiseMultiply(i.Conjugate());
            var ii = 0;
            var query = u.Zip(problem.Node, (re, no) => Tuple.Create(re, no)).Zip(s, (tuple, ss) =>
            {
                this.DeMapper.Add(ii++, tuple.Item2.Name);
                return new ResultNode
                {
                    Name = tuple.Item2.Name,
                    U = tuple.Item1.Magnitude,
                    Delta = tuple.Item1.Phase * 180 / Math.PI,
                    P = ss.Real,
                    Q = ss.Imaginary
                };
            });
            Nodes = query.ToArray();
            caculatePowerFlow(y, u);
        }

        private void caculatePowerFlow(Matrix<Complex> y, Vector<Complex> u)
        {
            PowerFlows = new List<PowerFlow>();
            for(int row = 0; row < y.RowCount; row++)
            {
                for(int col = 0; col < y.ColumnCount; col++)
                {
                    if(row == col || y[row, col] == 0)
                        continue;
                    var il = y[row, col] * (u[col] - u[row]);
                    var power = u[row] * Complex.Conjugate(il);
                    PowerFlows.Add(new PowerFlow
                    {
                        From = DeMapper[row],
                        To = DeMapper[col],
                        P = power.Real,
                        Q = power.Imaginary
                    });
                }
            }
        }

        public Solution(Solver solver)
        {
            var y = solver.Y;
            var u = solver.NodeResult;
            var i = y * u;
            var s = u.PointwiseMultiply(i.Conjugate());
            var nodes = new ResultNode[u.Count];
            Nodes = nodes;
            for(int ii = 0; ii < u.Count; ii++)
            {
                if(ii < solver.PQNodeCount)
                {
                    nodes[ii] = new ResultNode
                    {
                        Name = $"PQ Node {ii + 1}",
                        U = u[ii].Magnitude,
                        Delta = u[ii].Phase * 180 / Math.PI,
                        P = s[ii].Real,
                        Q = s[ii].Imaginary
                    };
                    DeMapper.Add(ii, $"PQ Node {ii + 1}");
                }
                else if(ii < solver.PQVNodeCount)
                {
                    nodes[ii] = new ResultNode
                    {
                        Name = $"PV Node {ii + 1 - solver.PQNodeCount}",
                        U = u[ii].Magnitude,
                        Delta = u[ii].Phase * 180 / Math.PI,
                        P = s[ii].Real,
                        Q = s[ii].Imaginary
                    };
                    DeMapper.Add(ii, $"PV Node {ii + 1 - solver.PQNodeCount}");
                }
                else
                {
                    nodes[ii] = new ResultNode
                    {
                        Name = $"Relax Node",
                        U = u[ii].Magnitude,
                        Delta = u[ii].Phase * 180 / Math.PI,
                        P = s[ii].Real,
                        Q = s[ii].Imaginary
                    };
                    DeMapper.Add(ii, $"Relax Node");
                }
            }
            caculatePowerFlow(y, u);
        }

        public Problem OriginalProblem
        {
            get;
            set;
        }

        [JsonIgnore]
        public IDictionary<int, string> DeMapper
        {
            get;
            private set;
        } = new Dictionary<int, string>();

        public ICollection<ResultNode> Nodes
        {
            get;
            private set;
        }

        public ICollection<PowerFlow> PowerFlows
        {
            get;
            private set;
        }
    }
}
