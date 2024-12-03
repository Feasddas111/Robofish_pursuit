/*   Copyright 2005-2008 The MathWorks, Inc. */
/*   Written by Peter Lindskog. */

/* Include libraries. */
#include "mex.h"
#include <math.h>

/* Specify the number of outputs here. */
#define NY 6

/* State equations. */
void compute_dx(double *dx, double *x, double *u, double **p)
{
    /* Declaration of model parameters and intermediate variables. */
    double *Mx, *Mz, *Ly, *Kx, *Kz, *Ktau, *Kt, *KD, *KL, *rbp, *mrbm;
    double pectz;   /* Intermediate variables. */
    
    /* Retrieve model parameters. */
    Mx   = p[0];    /* X-axis mass */
    Mz   = p[1];    /* Z-axis mass */
    Ly   = p[2];    /* Y-axis inertial moment */
    Kx   = p[3];    /* X-axis drag coefficient */
    Kz   = p[4];    /* Z-axis drag coefficient */
    Ktau = p[5];    /* Y-axis damping torque coefficient */
    Kt   = p[6];    /* Tail force coefficient */
    KD   = p[7];    /* Pectoral fin darg force coefficient */
    KL   = p[8];    /* Pectoral fin lift force coefficient */
    rbp  = p[9];    /* Distance between COG and pectoral fin */
    mrbm = p[10]; /* Mass plus the distance between COG and COB */
    
    /* Determine intermediate variables. */
    pectz = abs(x[2])*x[2]*(-KD[0]*sin(u[2])-KL[0]*u[2]*cos(u[2]));
    
    /* x[0]: Depth */
    /* x[1]: Pitch angle */
    /* x[2]: X-axis velocity */
    /* x[3]: Z-axis velocity. */
    /* x[4]: Picth angular velocity */
    /* x[5]: Mass of ballast system */

    dx[0] = sin(x[1])*x[2]-cos(x[1])*x[3];
    dx[1] = x[4];
    dx[2] = -x[3]*x[4]+1/Mx[0]*(-9.8*x[5]*sin(x[1])+abs(x[2])*x[2]*(-Kx[0]-KD[0]*cos(u[2])+KL[0]*u[2]*sin(u[2]))-Kt[0]*u[1]*sin(u[0]));
    dx[3] = x[2]*x[4]+1/Mz[0]*(9.8*x[5]*cos(x[1])-abs(x[3])*x[3]*Kz[0]+pectz);
    dx[4] = 1/Ly[0]*((Mx[0]-Mz[0])*x[2]*x[3]-Ktau[0]*abs(x[4])*x[4]+pectz*rbp[0]-mrbm[0]*9.8*sin(x[1]));
    dx[5] = u[3];

}

/* Output equations. */
void compute_y(double *y, double *x)
{
    y[0] = x[0];
    y[1] = x[1];
    y[2] = x[2];
    y[3] = x[3];
    y[4] = x[4];
    y[5] = x[5];
}



/*----------------------------------------------------------------------- *
   DO NOT MODIFY THE CODE BELOW UNLESS YOU NEED TO PASS ADDITIONAL
   INFORMATION TO COMPUTE_DX AND COMPUTE_Y
 
   To add extra arguments to compute_dx and compute_y (e.g., size
   information), modify the definitions above and calls below.
 *-----------------------------------------------------------------------*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* Declaration of input and output arguments. */
    double *x, *u, **p, *dx, *y, *t;
    int     i, np;
    size_t  nx;
    const mxArray *auxvar = NULL; /* Cell array of additional data. */
    
    if (nrhs < 3) {
        mexErrMsgIdAndTxt("IDNLGREY:ODE_FILE:InvalidSyntax",
        "At least 3 inputs expected (t, u, x).");
    }
    
    /* Determine if auxiliary variables were passed as last input.  */
    if ((nrhs > 3) && (mxIsCell(prhs[nrhs-1]))) {
        /* Auxiliary variables were passed as input. */
        auxvar = prhs[nrhs-1];
        np = nrhs - 4; /* Number of parameters (could be 0). */
    } else {
        /* Auxiliary variables were not passed. */
        np = nrhs - 3; /* Number of parameters. */
    }
    
    /* Determine number of states. */
    nx = mxGetNumberOfElements(prhs[1]); /* Number of states. */
    
    /* Obtain double data pointers from mxArrays. */
    t = mxGetPr(prhs[0]);  /* Current time value (scalar). */
    x = mxGetPr(prhs[1]);  /* States at time t. */
    u = mxGetPr(prhs[2]);  /* Inputs at time t. */
    
    p = mxCalloc(np, sizeof(double*));
    for (i = 0; i < np; i++) {
        p[i] = mxGetPr(prhs[3+i]); /* Parameter arrays. */
    }
    
    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateDoubleMatrix(nx, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(NY, 1, mxREAL);
    dx      = mxGetPr(plhs[0]); /* State derivative values. */
    y       = mxGetPr(plhs[1]); /* Output values. */
    
    /*
      Call the state and output update functions.
      
      Note: You may also pass other inputs that you might need,
      such as number of states (nx) and number of parameters (np).
      You may also omit unused inputs (such as auxvar).
      
      For example, you may want to use orders nx and nu, but not time (t)
      or auxiliary data (auxvar). You may write these functions as:
          compute_dx(dx, nx, nu, x, u, p);
          compute_y(y, nx, nu, x, u, p);
    */
    
    /* Call function for state derivative update. */
    compute_dx(dx, x, u, p);
    
    /* Call function for output update. */
    compute_y(y, x);
    
    /* Clean up. */
    mxFree(p);
}