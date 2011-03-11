// cudabuffer.h
//
// allocation and data transfer for host/device buffer pairs

#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// buffer over detection
#ifdef _DEBUG
#define MALLOC_SUFFIX "deadbeef"
const int MALLOC_SUFFIX_SIZE = 8;
#define ASSERT_SUFFIX(p, t, n, name) { if (strncmp((char *)p+n*sizeof(t), MALLOC_SUFFIX, MALLOC_SUFFIX_SIZE)) CBUF_ERROR1(__FILE__ ":" TOSTRING(__LINE__) " CUDA buffer write overflow in %s.", name); } 
#else
const int MALLOC_SUFFIX_SIZE = 0;
#define ASSERT_SUFFIX(p, t, n, name) { } 
#endif

#define CBUF_ERROR(s) { fprintf(stderr, "%s\n", s); fflush(stderr); throw "cuda error"; }
#define CBUF_ERROR1(s, x1) { fprintf(stderr, s, x1); fprintf(stderr, "\n"); fflush(stderr); throw "cuda error"; }

// CUDO: Assert that expr returns cudaSuccess
#define CUDO(expr) { if (expr!=cudaSuccess) throw cudaGetErrorString(cudaGetLastError()); }

// CUDO_MALLOC_WITH: Assert that p==NULL then assert that expr returns cudaSuccess
#define CUDO_MALLOC_WITH(p, expr) { if (p==NULL) CUDO(expr) else CBUF_ERROR(#p" != NULL, "__FILE__ ":" TOSTRING(__LINE__) ", " #expr); }

// CUDO_FREE_WITH: if p!=NULL do expr and assert that expr returns cudaSuccess and set p=NULL
#define CUDO_FREE_WITH(p, expr) { if (p!=NULL) { CUDO(expr); p=NULL; } }

#define MallocHost(p, t, n) CUDO_MALLOC_WITH(p, cudaMallocHost((void **)&p, n*sizeof(t)))
#define MallocDevice(p, t, n) CUDO_MALLOC_WITH(p, cudaMalloc((void **)&p, n*sizeof(t)))
#define FreeHost(p) CUDO_FREE_WITH(p, cudaFreeHost(p))
#define FreeDevice(p) CUDO_FREE_WITH(p, cudaFree(p))


#if USE_MAPPED_PINNED==1
    #define MallocMapped(p, pcu, t, n) { CUDO_MALLOC_WITH(p, cudaHostAlloc((void **)&p, n*sizeof(t), cudaHostAllocMapped)); CUDO(cudaHostGetDevicePointer(&pcu, p, 0)); }
    #define UpdateHost(p, pcu, t, n) {}    // Data transfer is not required for mapped pinned memory
    #define UpdateDevice(pcu, p, t, n) {}  // Data transfer is not required for mapped pinned memory
#else
    #define UpdateHost(p, pcu, t, n) { CUDO(cudaMemcpy(p, pcu, n*sizeof(t)+MALLOC_SUFFIX_SIZE, cudaMemcpyDeviceToHost)); ASSERT_SUFFIX(p, t, n, name); }
    #define UpdateDevice(pcu, p, t, n) { CUDO(cudaMemcpy(pcu, p, n*sizeof(t)+MALLOC_SUFFIX_SIZE, cudaMemcpyHostToDevice)); }
	
    #ifdef _DEBUG
        #define MallocMapped(p, pcu, t, n) { CUDO_MALLOC_WITH(p, cudaMallocHost((void **)&p, n*sizeof(t)+MALLOC_SUFFIX_SIZE)); \
		cudaMalloc((void **)&pcu, n*sizeof(t)+MALLOC_SUFFIX_SIZE); strncpy((char *)p+n*sizeof(t), MALLOC_SUFFIX, MALLOC_SUFFIX_SIZE); \
		UpdateDevice(pcu, p, t, n); } // copy to device when debugging to make suffix validation work, and to clear device memory content from previous runs
    #else
        #define MallocMapped(p, pcu, t, n) { CUDO_MALLOC_WITH(p, cudaMallocHost((void **)&p, n*sizeof(t))); cudaMalloc((void **)&pcu, n*sizeof(t)); }
    #endif
    
#endif

#define VERBOSE if(verbose)printf

// MappedBuffer: template for mapped host/device buffer pair
// usage example:
//   MappedBuffer<float> foo; // create a MappedBuffer of type float, with pointers initialized to NULL
//   foo.alloc(640*480); // allocate 640*480 floats (do not multiply by sizeof(float))
//   get_data_from_somewhere(foo.p); // put data into host buffer foo.p
//   foo.update_device(); // transfer from host to device
//   invoke_kernel(foo.pcu); // Invoke a kernel on GPU, using device buffer foo.pcu
//   foo.update_host(); // transfer from device to host



extern int graphics_enabled;

template <typename T>
class MappedBuffer
{
public:
    //! host data
    T *p;
    //! device data
    T *pcu;
    
    //! number of elements
    int n;
    //! size of element
    int tsize;
    //! number of bytes (n*tsize)
    int size; 
    
    //! visual width
    int w;
    //! visual height
    int h;
    //! visual number of color channels
    int c;
    //! true if not the owner of the host buffer
    bool mirrored;

#ifdef IMAGING    
    //! image for testing (w,h,c)
    IplImage* img;
#endif

    //! name for introspection when debugging
	char const *name;
    //! id number used in fullname
    int id;
    //! name for introspection when debugging
	char fullname[20];
    
    
    MappedBuffer<T>() { p = pcu = NULL; n=0; tsize=sizeof(T); size=0; name="unnamed"; id=-1; 
#ifdef IMAGING
    img=NULL;
#endif
    }

    ~MappedBuffer<T>() { assert_suffix_host(); assert_suffix_device(); if (!mirrored) FreeHost(p); FreeDevice(pcu); } //if (img) cvReleaseImage(&img); }
    // FIXME: this cvReleaseImage breaks the window displaying the image
    // (too bad we don't have garbage collection) but then we only leak memory when testing->..

    //! If in debug mode, assert that corruption is not detected in host buffer
    void assert_suffix_host()
    {
        #ifdef _DEBUG
        if (p) ASSERT_SUFFIX(p,T,n,name);
        #endif
    }
        
    //! If in debug mode, assert that corruption is not detected in device buffer
    void assert_suffix_device()
    {
        // FIXME: do this directly on the device to avoid side effect of copy
        #ifdef _DEBUG
        if (p) update_host(); // update_host calls ASSERT_SUFFIX after copy
        #endif
    }
    
    //! Allocate w*h*d elements, suitable for visualization tools (or just call with width only)
    void alloc(int width, int height=1, int channels=1) { w=width; h=height; c=channels; n=w*h*c; MallocMapped(p, pcu, T, n); size = n*tsize; }

    void alloc_device_mirror(T *phost, int n) {
        MallocDevice(pcu, T, n);
        p = phost;
        mirrored = true;
    }

    //! Copy data from device to host
    void update_host() { UpdateHost(p, pcu, T, n); }
    
    //! Copy data from host to device
    void update_device() { UpdateDevice(pcu, p, T, n); }
    
#ifdef IMAGING
    //! Create a visual image of the host buffer.
    IplImage *get_image(double gamma=1.0) {
        if (h==1) {
            throw "display requires 2D allocation";
        }
        
        if (img) return img;
        
        int IPL_D = -1;
        if (typeid(T)==typeid(unsigned char)) IPL_D = IPL_DEPTH_8U;
        else if (typeid(T)==typeid(char)) IPL_D = IPL_DEPTH_8S;
        else if (typeid(T)==typeid(unsigned short)) IPL_D = IPL_DEPTH_16U;
        else if (typeid(T)==typeid(short)) IPL_D = IPL_DEPTH_16S;
        else if (typeid(T)==typeid(unsigned int)) IPL_D = IPL_DEPTH_32S;
        else if (typeid(T)==typeid(int)) IPL_D = IPL_DEPTH_32S;
        else if (typeid(T)==typeid(float)) IPL_D = IPL_DEPTH_32F;
        else if (typeid(T)==typeid(double)) IPL_D = IPL_DEPTH_64F;
        else throw "unsupported buffer type for visualization";
        
        img = cvCreateImage(cvSize(w,h), IPL_D, c);
        cvSetData(img, p, tsize*c*w);
        
        if (w>800 || h>800) {
            if (w>800) w=800;
            if (h>800) h=800;
            cvSetImageROI(img, cvRect(0,0,w,h));
            IplImage *cropped = cvCreateImage( cvSize(w, h), IPL_D, c);
            cvCopy(img, cropped);
            //cvReleaseImage(&img);
            img = cropped;
        }
        
        if (gamma != 1.0 && IPL_D == IPL_DEPTH_8U) {
            // FIXME: Gamma only works for type unsigned char
            uchar lut[256*4];
            CvMat* lutmat = cvCreateMatHeader( 1, 256, CV_8UC1 );
            
            for( int i = 0; i < 256; i++ )
            {
                lut[i] = (char)(255.0*pow((double)i/255.0, gamma));
            }
            cvSetData( lutmat, lut, 1);
            cvLUT(img, img, lutmat);
        }

        return img;
    }
    
    //! Display a visual image of the host buffer in a window.
    void display(int x, int y, double gamma=1.0) {
        if (graphics_enabled) {
            IplImage* img_out = get_image(gamma);

            if (id>=0)
            {
                sprintf(fullname, "%s%d", name, id);
            }
            else
            {
                sprintf(fullname, "%s", name);
            }
            
            cvNamedWindow(fullname, 1); 
            cvMoveWindow(fullname, x, y);
            cvShowImage(fullname, img_out);
        }
    }

    //! Load the host buffer from an image file.
    void load(const char *fname, CvRect *rect=NULL)
    {
        IplImage* img = cvLoadImage(fname, (c==3) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE );
        
        if (!img) {
            fprintf(stderr, "Failed to load %s\n", fname);
            throw fname;
        }
        
        if (rect) {
            cvSetImageROI(img, *rect);
            IplImage *cropped = cvCreateImage( cvSize(rect->width, rect->height), IPL_DEPTH_8U, c);
            cvCopy(img, cropped);
            cvReleaseImage(&img);
            img = cropped;
        }
       
        if (img->imageSize != n) throw "load size mismatch";
        
        memcpy(p, img->imageData, img->imageSize);
        cvReleaseImage(&img);
    }

    //! Save the host buffer to an image file.
    void save(const char *fname, double gamma=1.0) {
        IplImage* img_out = get_image(gamma);
        
        char sfname[100];
        
        if (fname==NULL) {
            sprintf(sfname, "output/%s.png", name);
            fname = sfname;
        }

        // hack for cvSaveImage bug
        IplImage *img_out2 = cvCreateImage( cvSize(img_out->width, img_out->height), IPL_DEPTH_8U, 3);
		cvConvertImage(img_out, img_out2, 0);

        cvSaveImage(fname, img_out2);
    }
    
    //! Save the host buffer to an image file.
    void save(double gamma=1.0) {
        save(NULL, gamma);
    }
#endif IMAGING

};


class Timer {
    // implements high resolution timers
    // only active in debug mode
    cudaEvent_t estart, estop;
public:
    float t;
//const char* name;
    
    Timer()
    {
        init();
    }

    void init()
    {
     //   name = "unnamed";
        cudaEventCreate(&estart);
        cudaEventCreate(&estop);
        t = 0.0f;
    }
    
    inline void start()
    {
        if (this) {
            cudaEventRecord(estart, 0);
        }
    }
    
    inline void stop()
    {
        if (this) {
            cudaEventRecord(estop, 0);
            cudaEventSynchronize(estop);
            cudaEventElapsedTime(&t, estart, estop);
        }
    }
    
};
