using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace PFCalc
{
    public abstract class Solver
    {
        public void Init(Vector<double> pData, Vector<double> qData, Vector<double> uData, Complex relaxData, Matrix<Complex> yMatrix)
        {
            IterCount = 0;
            Pg = pData;
            Qg = qData;
            U = uData;
            Y = yMatrix;
            G = yMatrix.Real();
            B = yMatrix.Imaginary();
            RelaxNode = relaxData;
            NodeCount = yMatrix.RowCount;
            PQNodeCount = qData.Count;
            PVNodeCount = uData.Count;
            PQVNodeCount = PQNodeCount + PVNodeCount;
            InitOverride(pData, qData, uData, relaxData, yMatrix);
        }

        protected virtual void InitOverride(Vector<double> pData, Vector<double> qData, Vector<double> uData, Complex relaxData, Matrix<Complex> yMatrix)
        {
        }

        public double Error
        {
            get;
            set;
        } = 1e-4;

        public virtual void Solve()
        {
            while(Difference().AbsoluteMaximum() > Error)
            {
                Iter();
                IterCount++;
            }
        }

        public int IterCount
        {
            get;
            private set;
        }

        protected abstract Vector<double> Difference();
        protected abstract void Iter();

        public Vector<double> Pg
        {
            get;
            private set;
        }

        public int NodeCount
        {
            get;
            private set;
        }

        public int PQVNodeCount
        {
            get;
            private set;
        }

        public int PVNodeCount
        {
            get;
            private set;
        }

        public int PQNodeCount
        {
            get;
            private set;
        }

        public abstract Vector<Complex> NodeResult
        {
            get;
        }

        public Vector<double> Qg
        {
            get;
            private set;
        }

        public Vector<double> U
        {
            get;
            private set;
        }

        public Matrix<Complex> Y
        {
            get;
            private set;
        }

        public Matrix<double> G
        {
            get;
            private set;
        }

        public Matrix<double> B
        {
            get;
            private set;
        }

        public Complex RelaxNode
        {
            get;
            private set;
        }
    }

    public class RectangularSolver : Solver
    {
        public override Vector<Complex> NodeResult => e.ToComplex() + Complex.ImaginaryOne * f.ToComplex();

        protected override void InitOverride(Vector<double> pData, Vector<double> qData, Vector<double> uData, Complex relaxData, Matrix<Complex> yMatrix)
        {
            a = CreateVector.Dense<double>(NodeCount);
            b = CreateVector.Dense<double>(NodeCount);

            e = CreateVector.Dense<double>(NodeCount);
            e[PQVNodeCount] = RelaxNode.Real;
            f = CreateVector.Dense<double>(NodeCount);
            f[PQVNodeCount] = RelaxNode.Imaginary;

            x = CreateVector.Dense(2 * PQVNodeCount, i => i < PQVNodeCount ? 1.0 : 0);
            dltx = CreateVector.Dense<double>(2 * PQVNodeCount);
            dlt = CreateVector.Dense<double>(2 * PQVNodeCount);
            u2 = U.PointwisePower(2);
            jcb = CreateMatrix.Dense<double>(2 * PQVNodeCount, 2 * PQVNodeCount);

            tempN = CreateVector.Dense<double>(NodeCount);
            tempPQ1 = CreateVector.Dense<double>(PQNodeCount);
            tempPQ2 = CreateVector.Dense<double>(PQNodeCount);
            tempPV1 = CreateVector.Dense<double>(PVNodeCount);
            tempPV2 = CreateVector.Dense<double>(PVNodeCount);
            tempPQV1 = CreateVector.Dense<double>(PQVNodeCount);
            tempPQV2 = CreateVector.Dense<double>(PQVNodeCount);

            subG = G.SubMatrix(0, PQVNodeCount, 0, PQVNodeCount);
            subB = B.SubMatrix(0, PQVNodeCount, 0, PQVNodeCount);

            ge = CreateMatrix.Dense<double>(PQVNodeCount, PQVNodeCount);
            bf = CreateMatrix.Dense<double>(PQVNodeCount, PQVNodeCount);
            pge_pbf = CreateMatrix.Dense<double>(PQVNodeCount, PQVNodeCount);
            be = CreateMatrix.Dense<double>(PQVNodeCount, PQVNodeCount);
            gf = CreateMatrix.Dense<double>(PQVNodeCount, PQVNodeCount);
            pbe_ngf = CreateMatrix.Dense<double>(PQVNodeCount, PQVNodeCount);

            tempPQV_PQV = CreateMatrix.Dense<double>(PQVNodeCount, PQVNodeCount);
            ml = CreateMatrix.Dense<double>(PQNodeCount, PQVNodeCount);
            rs = new DiagonalMatrix(PVNodeCount);

            ndiaga = new DiagonalMatrix(PQVNodeCount);
            diagb = new DiagonalMatrix(PQVNodeCount);
        }

        protected Vector<double> a, b, e, f, x, dltx, dlt, u2, tempN, tempPQ1, tempPQ2, tempPV1, tempPV2, tempPQV1, tempPQV2;
        protected Matrix<double> jcb, ge, bf, pge_pbf, be, gf, pbe_ngf, tempPQV_PQV, ml, subG, subB;
        protected DiagonalMatrix ndiaga, diagb, rs;

        protected override Vector<double> Difference()
        {
            x.CopySubVectorTo(e, 0, 0, PQVNodeCount);
            x.CopySubVectorTo(f, PQVNodeCount, 0, PQVNodeCount);
            G.Multiply(e, a);
            B.Multiply(f, tempN);
            a.Subtract(tempN, a);//a=G*e-B*f
            G.Multiply(f, b);
            B.Multiply(e, tempN);
            b.Add(tempN, b);//b=G*f+B*e

            //e.*a
            e.PointwiseMultiply(a, tempN);
            tempN.CopySubVectorTo(tempPQV1, 0, 0, PQVNodeCount);
            //f.*b
            f.PointwiseMultiply(b, tempN);
            tempN.CopySubVectorTo(tempPQV2, 0, 0, PQVNodeCount);
            //Pg-e.*a-f.*b=>dlt[]
            Pg.Subtract(tempPQV1, tempPQV1);
            tempPQV1.Subtract(tempPQV2, tempPQV1);
            tempPQV1.CopySubVectorTo(dlt, 0, 0, PQVNodeCount);

            //f.*a
            f.PointwiseMultiply(a, tempN);
            tempN.CopySubVectorTo(tempPQ1, 0, 0, PQNodeCount);
            //e.*b
            e.PointwiseMultiply(b, tempN);
            tempN.CopySubVectorTo(tempPQ2, 0, 0, PQNodeCount);
            //Qg-f.*a+e.*b=>dlt[]
            Qg.Subtract(tempPQ1, tempPQ1);
            tempPQ1.Add(tempPQ2, tempPQ1);
            tempPQ1.CopySubVectorTo(dlt, 0, PQVNodeCount, PQNodeCount);

            //e.*e
            e.CopySubVectorTo(tempPV1, PQNodeCount, 0, PVNodeCount);
            tempPV1.PointwiseMultiply(tempPV1, tempPV1);
            //f.*f
            f.CopySubVectorTo(tempPV2, PQNodeCount, 0, PVNodeCount);
            tempPV2.PointwiseMultiply(tempPV2, tempPV2);
            //u.*u-e.*e-f.*f=>dlt[]
            u2.Subtract(tempPV1, tempPV1);
            tempPV1.Subtract(tempPV2, tempPV1);
            tempPV1.CopySubVectorTo(dlt, 0, PQNodeCount + PQVNodeCount, PVNodeCount);
            return dlt;
        }

        private void jacobi()
        {
            subG.MapIndexed((row, col, va) => va * e[row], ge);
            subB.MapIndexed((row, col, va) => va * f[row], bf);
            ge.Add(bf, pge_pbf);
            subB.MapIndexed((row, col, va) => va * e[row], be);
            subG.MapIndexed((row, col, va) => va * f[row], gf);
            be.Subtract(gf, pbe_ngf);

            a.CopySubVectorTo(tempPQV1, 0, 0, PQVNodeCount);
            ndiaga.SetDiagonal(tempPQV1);
            ndiaga.Negate(ndiaga);
            b.CopySubVectorTo(tempPQV2, 0, 0, PQVNodeCount);
            diagb.SetDiagonal(tempPQV2);
            //H
            ndiaga.Subtract(pge_pbf, tempPQV_PQV);
            jcb.SetSubMatrix(0, NodeCount - 1, 0, NodeCount - 1, tempPQV_PQV);
            //N
            pbe_ngf.Subtract(diagb, tempPQV_PQV);
            jcb.SetSubMatrix(0, PQVNodeCount, PQVNodeCount, PQVNodeCount, tempPQV_PQV);
            //M
            diagb.Add(pbe_ngf, tempPQV_PQV);
            jcb.SetSubMatrix(PQVNodeCount, PQNodeCount, 0, PQVNodeCount, tempPQV_PQV);
            //L
            ndiaga.Add(pge_pbf, tempPQV_PQV);
            jcb.SetSubMatrix(PQVNodeCount, PQNodeCount, PQVNodeCount, PQVNodeCount, tempPQV_PQV);
            //R
            e.CopySubVectorTo(tempPV1, PQNodeCount, 0, PVNodeCount);
            tempPV1.Multiply(-2, tempPV1);
            rs.SetDiagonal(tempPV1);
            jcb.SetSubMatrix(PQNodeCount * 2 + PVNodeCount, PQVNodeCount - PVNodeCount, rs);
            //S
            f.CopySubVectorTo(tempPV1, PQNodeCount, 0, PVNodeCount);
            tempPV1.Multiply(-2, tempPV1);
            rs.SetDiagonal(tempPV1);
            jcb.SetSubMatrix(PQNodeCount * 2 + PVNodeCount, PQVNodeCount * 2 - PVNodeCount, rs);
        }

        protected override void Iter()
        {
            jacobi();
            jcb.Solve(dlt, dltx);
            x.Subtract(dltx, x);
        }
    }

    public class SparseRectSolver : RectangularSolver
    {
        protected override void InitOverride(Vector<double> pData, Vector<double> qData, Vector<double> uData, Complex relaxData, Matrix<Complex> yMatrix)
        {
            a = CreateVector.Dense<double>(NodeCount);
            b = CreateVector.Dense<double>(NodeCount);

            e = CreateVector.Dense<double>(NodeCount);
            e[PQVNodeCount] = RelaxNode.Real;
            f = CreateVector.Dense<double>(NodeCount);
            f[PQVNodeCount] = RelaxNode.Imaginary;

            x = CreateVector.Dense(2 * PQVNodeCount, i => i < PQVNodeCount ? 1.0 : 0);
            dltx = CreateVector.Dense<double>(2 * PQVNodeCount);
            dlt = CreateVector.Dense<double>(2 * PQVNodeCount);
            u2 = U.PointwisePower(2);
            jcb = CreateMatrix.Sparse<double>(2 * PQVNodeCount, 2 * PQVNodeCount);

            tempN = CreateVector.Dense<double>(NodeCount);
            tempPQ1 = CreateVector.Dense<double>(PQNodeCount);
            tempPQ2 = CreateVector.Dense<double>(PQNodeCount);
            tempPV1 = CreateVector.Dense<double>(PVNodeCount);
            tempPV2 = CreateVector.Dense<double>(PVNodeCount);
            tempPQV1 = CreateVector.Dense<double>(PQVNodeCount);
            tempPQV2 = CreateVector.Dense<double>(PQVNodeCount);

            subG = G.SubMatrix(0, PQVNodeCount, 0, PQVNodeCount);
            subB = B.SubMatrix(0, PQVNodeCount, 0, PQVNodeCount);

            ge = CreateMatrix.Sparse<double>(PQVNodeCount, PQVNodeCount);
            bf = CreateMatrix.Sparse<double>(PQVNodeCount, PQVNodeCount);
            pge_pbf = CreateMatrix.Sparse<double>(PQVNodeCount, PQVNodeCount);
            be = CreateMatrix.Sparse<double>(PQVNodeCount, PQVNodeCount);
            gf = CreateMatrix.Sparse<double>(PQVNodeCount, PQVNodeCount);
            pbe_ngf = CreateMatrix.Sparse<double>(PQVNodeCount, PQVNodeCount);

            tempPQV_PQV = CreateMatrix.Sparse<double>(PQVNodeCount, PQVNodeCount);
            ml = CreateMatrix.Sparse<double>(PQNodeCount, PQVNodeCount);
            rs = new DiagonalMatrix(PVNodeCount);

            ndiaga = new DiagonalMatrix(PQVNodeCount);
            diagb = new DiagonalMatrix(PQVNodeCount);
        }
    }

    public class PolarSolver : Solver
    {
        public override Vector<Complex> NodeResult => u.MapIndexed((i, va) => Complex.FromPolarCoordinates(va, delta[i]));

        protected override void InitOverride(Vector<double> pData, Vector<double> qData, Vector<double> uData, Complex relaxData, Matrix<Complex> yMatrix)
        {
            x = CreateVector.Dense(PQVNodeCount + PQNodeCount, i => i < PQVNodeCount ? 0 : 1.0);
            u = CreateVector.Dense<double>(NodeCount);
            u[NodeCount - 1] = RelaxNode.Magnitude;
            u.SetSubVector(PQNodeCount, PVNodeCount, U);
            delta = CreateVector.Dense<double>(NodeCount);
            delta[NodeCount - 1] = RelaxNode.Phase;

            dlt = CreateVector.Dense<double>(PQVNodeCount + PQNodeCount);
            dltx = CreateVector.Dense<double>(PQVNodeCount + PQNodeCount);

            jcb = CreateMatrix.Dense<double>(PQVNodeCount + PQNodeCount, PQVNodeCount + PQNodeCount);

            ngcnbs = CreateMatrix.Dense<double>(NodeCount, NodeCount);
            pgsnbc = CreateMatrix.Dense<double>(NodeCount, NodeCount);
            U_ngcnbs = CreateMatrix.Dense<double>(NodeCount, NodeCount);
            U_pgsnbc = CreateMatrix.Dense<double>(NodeCount, NodeCount);
            sinDelta = CreateMatrix.Dense<double>(NodeCount, NodeCount);
            ncosDelta = CreateMatrix.Dense<double>(NodeCount, NodeCount);

            tempNM = CreateMatrix.Dense<double>(NodeCount, NodeCount);

            tempNV = CreateVector.Dense<double>(NodeCount);

            u2 = CreateVector.Dense<double>(NodeCount);
            u2B = CreateVector.Dense<double>(NodeCount);
            u2G = CreateVector.Dense<double>(NodeCount);

            hl = CreateMatrix.Dense<double>(NodeCount, NodeCount);
            nm = CreateMatrix.Dense<double>(NodeCount, NodeCount);
            uc_ur = CreateMatrix.Dense<double>(NodeCount, NodeCount);

            u_sum_u_ngcnbs = CreateVector.Dense<double>(NodeCount);
            u_sum_u_pgsnbc = CreateVector.Dense<double>(NodeCount);
            cosdelta = CreateVector.Dense<double>(NodeCount);
            sindelta = CreateVector.Dense<double>(NodeCount);
        }

        protected Vector<double> x, dltx, dlt, u, delta, cosdelta, sindelta, u_sum_u_ngcnbs, u_sum_u_pgsnbc;
        protected Vector<double> u2, u2B, u2G, tempNV;
        protected Matrix<double> ncosDelta, sinDelta, jcb, ngcnbs, pgsnbc, U_ngcnbs, U_pgsnbc;
        protected Matrix<double> tempNM;
        protected Matrix<double> hl, nm, uc_ur;

        protected override Vector<double> Difference()
        {
            x.CopySubVectorTo(delta, 0, 0, PQVNodeCount);
            x.CopySubVectorTo(u, PQVNodeCount, 0, PQNodeCount);

            delta.Map(i => Trig.Sin(i), sindelta, Zeros.Include);
            delta.Map(i => Trig.Cos(i), cosdelta, Zeros.Include);
            var sdc = sindelta.ToColumnMatrix();
            var nsdr = -sindelta.ToRowMatrix();
            var cdc = cosdelta.ToColumnMatrix();
            var cdr = cosdelta.ToRowMatrix();

            sdc.Multiply(cdr, sinDelta);
            cdc.Multiply(nsdr, tempNM);
            sinDelta.Add(tempNM, sinDelta);
            //sinDelta = sdc * cdr + cdc * nsdr;
            sdc.Multiply(nsdr, ncosDelta);
            cdc.Multiply(cdr, tempNM);
            ncosDelta.Subtract(tempNM, ncosDelta);
            //ncosDelta = -cdc * cdr + sdc * nsdr;
            G.PointwiseMultiply(ncosDelta, ngcnbs);
            B.PointwiseMultiply(sinDelta, tempNM);
            ngcnbs.Subtract(tempNM, ngcnbs);
            //ngcnbs = G.PointwiseMultiply(ncosDelta) - B.PointwiseMultiply(sinDelta);

            ngcnbs.MapIndexed((row, col, va) => va * u[col], U_ngcnbs);
            u.PointwiseMultiply(U_ngcnbs.RowSums(), u_sum_u_ngcnbs);
            (Pg + u_sum_u_ngcnbs.SubVector(0, PQVNodeCount)).CopySubVectorTo(dlt, 0, 0, PQVNodeCount);

            G.PointwiseMultiply(sinDelta, pgsnbc);
            B.PointwiseMultiply(ncosDelta, tempNM);
            pgsnbc.Add(tempNM, pgsnbc);
            //pgsnbc = G.PointwiseMultiply(sinDelta) + B.PointwiseMultiply(ncosDelta);

            pgsnbc.MapIndexed((row, col, va) => va * u[col], U_pgsnbc);
            u.PointwiseMultiply(U_pgsnbc.RowSums(), u_sum_u_pgsnbc);
            (Qg - u_sum_u_pgsnbc.SubVector(0, PQNodeCount)).CopySubVectorTo(dlt, 0, PQVNodeCount, PQNodeCount);

            return dlt;
        }

        protected override void Iter()
        {
            jacobi();
            jcb.Solve(dlt, dltx);
            x.Subtract(dltx, x);
        }

        private void jacobi()
        {
            var uc = u.ToColumnMatrix();
            var ur = u.ToRowMatrix();
            uc.Multiply(ur, uc_ur);
            u.PointwiseMultiply(u, u2);

            uc_ur.Negate(tempNM);
            pgsnbc.PointwiseMultiply(tempNM, hl);

            u2.PointwiseMultiply(B.Diagonal(), u2B);
            u_sum_u_pgsnbc.Add(u2B, tempNV);
            hl.SetDiagonal(tempNV);
            jcb.SetSubMatrix(0, PQVNodeCount, 0, PQVNodeCount, hl);

            u2B.Subtract(u_sum_u_pgsnbc, tempNV);
            hl.SetDiagonal(tempNV);
            jcb.SetSubMatrix(PQVNodeCount, PQNodeCount, PQVNodeCount, PQNodeCount, hl);

            ngcnbs.PointwiseMultiply(uc_ur, nm);
            u2.PointwiseMultiply(G.Diagonal(), u2G);

            u_sum_u_ngcnbs.Subtract(u2G, tempNV);
            nm.SetDiagonal(tempNV);
            jcb.SetSubMatrix(0, PQVNodeCount, PQVNodeCount, PQNodeCount, nm);

            nm.Negate(nm);
            u_sum_u_ngcnbs.Add(u2G, tempNV);
            nm.SetDiagonal(tempNV);
            jcb.SetSubMatrix(PQVNodeCount, PQNodeCount, 0, PQVNodeCount, nm);
        }
    }

    public class SparsePolarSolver : PolarSolver
    {
        protected override void InitOverride(Vector<double> pData, Vector<double> qData, Vector<double> uData, Complex relaxData, Matrix<Complex> yMatrix)
        {
            x = CreateVector.Dense(PQVNodeCount + PQNodeCount, i => i < PQVNodeCount ? 0 : 1.0);
            u = CreateVector.Dense<double>(NodeCount);
            u[NodeCount - 1] = RelaxNode.Magnitude;
            u.SetSubVector(PQNodeCount, PVNodeCount, U);
            delta = CreateVector.Dense<double>(NodeCount);
            delta[NodeCount - 1] = RelaxNode.Phase;

            dlt = CreateVector.Dense<double>(PQVNodeCount + PQNodeCount);
            dltx = CreateVector.Dense<double>(PQVNodeCount + PQNodeCount);

            jcb = CreateMatrix.Sparse<double>(PQVNodeCount + PQNodeCount, PQVNodeCount + PQNodeCount);

            ngcnbs = CreateMatrix.Sparse<double>(NodeCount, NodeCount);
            pgsnbc = CreateMatrix.Sparse<double>(NodeCount, NodeCount);
            U_ngcnbs = CreateMatrix.Sparse<double>(NodeCount, NodeCount);
            U_pgsnbc = CreateMatrix.Sparse<double>(NodeCount, NodeCount);
            sinDelta = CreateMatrix.Sparse<double>(NodeCount, NodeCount);
            ncosDelta = CreateMatrix.Sparse<double>(NodeCount, NodeCount);

            tempNM = CreateMatrix.Sparse<double>(NodeCount, NodeCount);

            tempNV = CreateVector.Sparse<double>(NodeCount);

            u2 = CreateVector.Dense<double>(NodeCount);
            u2B = CreateVector.Sparse<double>(NodeCount);
            u2G = CreateVector.Sparse<double>(NodeCount);

            hl = CreateMatrix.Sparse<double>(NodeCount, NodeCount);
            nm = CreateMatrix.Sparse<double>(NodeCount, NodeCount);
            uc_ur = CreateMatrix.Sparse<double>(NodeCount, NodeCount);

            u_sum_u_ngcnbs = CreateVector.Sparse<double>(NodeCount);
            u_sum_u_pgsnbc = CreateVector.Sparse<double>(NodeCount);
            cosdelta = CreateVector.Sparse<double>(NodeCount);
            sindelta = CreateVector.Sparse<double>(NodeCount);
        }
    }
}
