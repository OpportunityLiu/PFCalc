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

        public static string WriteJson(Problem prob)
        {

            return JsonConvert.SerializeObject(prob, Formatting.Indented, compC);
        }

        private static readonly complexConverter compC = new complexConverter();

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
            var mapper = new Dictionary<string, int>();
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
                mapper.Add(item.Name, i++);
            }
            var j = 0;
            foreach(var item in PVNode)
            {
                p[i] = item.Pg;
                u[j++] = item.U;
                mapper.Add(item.Name, i++);
            }
            mapper.Add(RelaxNode.Name, i);

            foreach(var item in Grounding.Concat<Branch>(Transformer).Concat(Transmission))
            {
                item.IndexMapper = s => mapper[s];
                item.Apply(y);
            }

            T a = new T();
            a.Init(p, q, u, r, y);
            return a;
        }
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

    public abstract class ConnectionBranch: Branch
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
}
