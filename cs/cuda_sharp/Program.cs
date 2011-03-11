using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.IO;

namespace cudasharp
{
    [Serializable]
    class PythonError : Exception
    {
        public PythonError(string serr)
            : base(serr)
        {
        }
    }

    class Python
    {
        public string outfilename = Path.GetTempFileName();
        StreamReader pystream;
        const string pydll = "python26.DLL";

        [DllImport(pydll)]
        static public extern int Py_Initialize();

        [DllImport(pydll)]
        static public extern int Py_Main(int argc, string[] argv);

        [DllImport(pydll)]
        static public extern int PyRun_SimpleString(string s);

        public Python()
        {
            int err;
            err = Py_Initialize();
            err = PyRun_SimpleString("import sys");
            string scmd = String.Format("sys.stdout = sys.stderr = open(r'{0}','w')", outfilename);
            err = PyRun_SimpleString(scmd);
            err = PyRun_SimpleString("sys.argv = ['cudasharp', 'w']");

            FileStream fs = new FileStream(outfilename, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            pystream = new StreamReader(fs);
        }

        public void exec(string s)
        {
            Console.WriteLine(String.Format(">>> {0}", s));
            PyRun_SimpleString(s);
            PyRun_SimpleString("sys.stdout.flush()");
            string sout = pystream.ReadToEnd();
            Console.Write(sout);
        }

    }

    [Serializable]
    class CUDAError : Exception
    {
        public CUDAError(string serr)
            : base(serr)
        {
        }
    }

    class cu
    {
        public enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
        
        static public void SAFECALL(int errno)
        {
            if (errno != 0)
            {
                throw new CUDAError(String.Format("CUDA error {0}: {1}", errno, cu.cudaGetErrorString(errno)));
            }
        }

        const string cudart = "cudart32_32_7.dll";
        [DllImport(cudart)]
        static public extern string cudaGetErrorString(int n);

        [DllImport(cudart)]
        static public extern int cudaGetDeviceCount(out int n);

        [DllImport(cudart)]
        static public extern int cudaMalloc(out UInt32 p, int size);

        [DllImport(cudart)]
        static public extern int cudaMallocHost(out UInt32 p, int size);

        [DllImport(cudart)]
        static public extern int cudaFree(UInt32 p);

        [DllImport(cudart)]
        static public extern int cudaFreeHost(UInt32 p);

        [DllImport(cudart)]
        static public extern int cudaMemcpy(UInt32 pdst, UInt32 psrc, int count, cudaMemcpyKind kind);
    }

    [StructLayout(LayoutKind.Sequential)]
    class MappedBuffer<T> // data is array of T
    {
        public UInt32 p;
        public UInt32 pcu;
        public int nsize; // size in bytes

        bool own_p;  // always false for C# allocations
        bool own_pcu; // true if device memory is allocated by the constructor

        GCHandle gch;

        public MappedBuffer(
            T[] host_array,
            [Optional, DefaultParameterValue((UInt32)0)] UInt32 _pcu)
        {
            own_p = false;
            nsize = System.Runtime.InteropServices.Marshal.SizeOf(host_array[0]) * host_array.Length;

            gch = GCHandle.Alloc(host_array, GCHandleType.Pinned);
            p = (UInt32) gch.AddrOfPinnedObject();

            if (_pcu != 0)
            {
                own_pcu = false;
                pcu = _pcu;
            }
            else
            {
                own_pcu = true;
                cu.SAFECALL(cu.cudaMalloc(out pcu,nsize));
            }
        }

        ~MappedBuffer()
        {
            gch.Free();
            if (own_pcu)
            {
                try
                {
                    cu.SAFECALL(cu.cudaFree(pcu));
                }
                catch (CUDAError e)
                {
                    Console.WriteLine("Caught {0} during garbage collection.  Perhaps CUDA already freed the buffer.", e.Message);
                }
            }
        }

        public void update_host()
        {
            cu.SAFECALL(cu.cudaMemcpy(p, pcu, nsize, cu.cudaMemcpyKind.cudaMemcpyDeviceToHost));
        }

        public void update_device()
        {
            cu.SAFECALL(cu.cudaMemcpy(pcu, p, nsize, cu.cudaMemcpyKind.cudaMemcpyHostToDevice));
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello.");
            int n;
            cu.cudaGetDeviceCount(out n);
            Console.WriteLine(String.Format("{0} CUDA devices", n));

            UInt32[] host_array = new UInt32[10];
            for (UInt32 i = 0; i < host_array.Length; i++)
            {
                host_array[i] = i * i;
            }
            

            int nsize = System.Runtime.InteropServices.Marshal.SizeOf(host_array[0]) * host_array.Length;
            Console.WriteLine(String.Format("sizeof(UInt32[10]) = {0}", nsize));

            MappedBuffer<UInt32> mb = new MappedBuffer<UInt32>(host_array, 0);
            Console.WriteLine(String.Format("device pointer = {0}", mb.pcu));
            Console.WriteLine(String.Format("host_array[5] = {0}", host_array[5]));

            mb.update_device();

            Console.WriteLine("Starting python");
            Python py = new Python();
            py.exec("print 'hello from python'");
            py.exec("from aeropath import *");
            py.exec(String.Format("mb = MappedBuffer(c_uint32*10, pcu={0})", mb.pcu));
            py.exec("print list(mb.data)");
            py.exec("mb.update_host()");
            py.exec("print list(mb.data)");
            py.exec("mb.data[3]=123");
            py.exec("mb.update_device()");

            mb.update_host();
            Console.WriteLine("{0}", host_array[3]);
            Console.WriteLine("Hit enter to launch showcuda...");
            Console.ReadLine();
            py.exec("import aerodisplay");
            py.exec("aerodisplay.init()");
            py.exec("aerodisplay.randomize_flights()");
            py.exec("aerodisplay.run()");
            Console.WriteLine("showcuda lauched...");
            Console.WriteLine("Hit enter to end C# app...");
            Console.ReadLine();
        }
    }
}
