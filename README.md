ZArray is a C++ class (v1.1) that allows SIMD vector arrays to be indexed by
integer equivalent in vector size vectors such as int8, int4, int2

LICENSE: FREE for commercial and non-commercial use,
it is an ENJOYWARE and hope it speeds up your project for you. Credit mention
always welcome, as usual.

NEW (v1.1): Can it be further optimized? I am sure, if you have an idea write me
to: subband@protonmail.com. In the meantime I added a nicer way identify T vector 
size using special C++ macros (before included in const1.h in other projects, 
but now specifically added to this project)

    Example usage: 

    ZArray<double4v> arr[1024], aVec = 0.0;
    for (int i=0; i<1024; i++) {
        arr[i] = double4{ i, -i,  M_PI, -M_PI };
    }
    int4v idx = { 0, 1, 100, 500 };
    
    aVec = arr[idx];  
    
    // aVec now contains: { 0.0, -1.0, 3.1415, -3.1415 }

    // lets change indexes to 10, 20, 30 and 40
    
    idx = int4{ 10, 20, 30, 40 };
    
    arr[idx] = aVec; 

    // now the arr contains:
    // arr[10][0] = 0.0; arr[20][1] = -1.0;  
    // arr[30][2] = 3.1415; arr[40][3] = -3.1415

    well you get the idea... 

DMT <subband@protonmail.com>


This is the real 100% honest truth.
