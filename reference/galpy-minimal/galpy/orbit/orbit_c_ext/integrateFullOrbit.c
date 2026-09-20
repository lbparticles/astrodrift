/*
  Wrappers around the C integration code for Full Orbits
*/
#ifdef _WIN32
#include <Python.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <bovy_coords.h>
#include <leung_dop853.h>
#include <integrateFullOrbit.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef ORBITS_CHUNKSIZE
#define ORBITS_CHUNKSIZE 1
#endif
//Macros to export functions in DLL on different OS
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
// Just do nothing?
#define EXPORT
#endif
#ifdef _WIN32
// On Windows, *need* to define this function to allow the package to be imported
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_libgalpy(void) { // Python 3
  return NULL;
}
#else
PyMODINIT_FUNC initlibgalpy(void) {} // Python 2
#endif
#endif
/*
  Function Declarations
*/
void evalRectDeriv(double, double *, double *,
			 int, struct potentialArg *);
void initMovingObjectSplines(struct potentialArg *, double ** pot_args);
/*
  Actual functions
*/
void parse_leapFuncArgs_Full(int npot,
			     struct potentialArg * potentialArgs,
			     int ** pot_type,
			     double ** pot_args,
           tfuncs_type_arr * pot_tfuncs){
  int ii,jj,kk;
  int nR, nz, nr;
  double * Rgrid, * zgrid, * potGrid_splinecoeffs;
  init_potentialArgs(npot,potentialArgs);
  for (ii=0; ii < npot; ii++){
    switch ( *(*pot_type)++ ) {
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->potentialEval= &MiyamotoNagaiPotentialEval;
      potentialArgs->Rforce= &MiyamotoNagaiPotentialRforce;
      potentialArgs->zforce= &MiyamotoNagaiPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &MiyamotoNagaiPotentialDens;
      //potentialArgs->R2deriv= &MiyamotoNagaiPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->potentialEval= &NFWPotentialEval;
      potentialArgs->Rforce= &NFWPotentialRforce;
      potentialArgs->zforce= &NFWPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &NFWPotentialDens;
      //potentialArgs->R2deriv= &NFWPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 15: //PowerSphericalwCutoffPotential, 5 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialwCutoffEval;
      potentialArgs->Rforce= &PowerSphericalPotentialwCutoffRforce;
      potentialArgs->zforce= &PowerSphericalPotentialwCutoffzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &PowerSphericalPotentialwCutoffDens;
      //potentialArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 5;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 17: //PlummerPotential, 2 arguments
      potentialArgs->potentialEval= &PlummerPotentialEval;
      potentialArgs->Rforce= &PlummerPotentialRforce;
      potentialArgs->zforce= &PlummerPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &PlummerPotentialDens;
      //potentialArgs->R2deriv= &PlummerPotentialR2deriv;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -6: //MovingObjectPotential
      potentialArgs->Rforce= &MovingObjectPotentialRforce;
      potentialArgs->zforce= &MovingObjectPotentialzforce;
      potentialArgs->phitorque= &MovingObjectPotentialphitorque;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    }
    int setupMovingObjectSplines = *(*pot_type-1) == -6 ? 1 : 0;
    if ( *(*pot_type-1) < 0 ) { // Parse wrapped potential for wrappers
      potentialArgs->nwrapped= (int) *(*pot_args)++;
      potentialArgs->wrappedPotentialArg= \
	(struct potentialArg *) malloc ( potentialArgs->nwrapped	\
					 * sizeof (struct potentialArg) );
      parse_leapFuncArgs_Full(potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg,
			      pot_type,pot_args,pot_tfuncs);
    }
    if (setupMovingObjectSplines)
      initMovingObjectSplines(potentialArgs, pot_args);
    // Now load each potential's parameters
    potentialArgs->args= (double *) malloc( potentialArgs->nargs * sizeof(double));
    for (jj=0; jj < potentialArgs->nargs; jj++){
      *(potentialArgs->args)= *(*pot_args)++;
      potentialArgs->args++;
    }
    potentialArgs->args-= potentialArgs->nargs;
    // and load each potential's time functions
    if ( potentialArgs->ntfuncs > 0 ) {
      potentialArgs->tfuncs= (*pot_tfuncs);
      (*pot_tfuncs)+= potentialArgs->ntfuncs;
    }
    potentialArgs++;
  }
  potentialArgs-= npot;
}
EXPORT void integrateFullOrbit(int nobj,
			       double *yo,
			       int nt,
			       double *t,
			       int npot,
			       int * pot_type,
			       double * pot_args,
             tfuncs_type_arr pot_tfuncs,
			       double dt,
			       double rtol,
			       double atol,
			       double *result,
			       int * err,
			       int odeint_type){
  //Set up the forces, first count
  int ii,jj;
  int dim;
  int max_threads;
  int * thread_pot_type;
  double * thread_pot_args;
  tfuncs_type_arr thread_pot_tfuncs;
  max_threads= ( nobj < omp_get_max_threads() ) ? nobj : omp_get_max_threads();
  // Because potentialArgs may cache, safest to have one / thread
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( max_threads * npot * sizeof (struct potentialArg) );
#pragma omp parallel for schedule(static,1) private(ii,thread_pot_type,thread_pot_args,thread_pot_tfuncs) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++) {
    thread_pot_type= pot_type; // need to make thread-private pointers, bc
    thread_pot_args= pot_args; // these pointers are changed in parse_...
    thread_pot_tfuncs= pot_tfuncs; // ...
    parse_leapFuncArgs_Full(npot,potentialArgs+ii*npot,
			    &thread_pot_type,&thread_pot_args,&thread_pot_tfuncs);
  }
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  switch ( odeint_type ) {
  case 6: //DOP853
    odeint_func= &dop853;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  }
#pragma omp parallel for schedule(dynamic,ORBITS_CHUNKSIZE) private(ii,jj) num_threads(max_threads)
  for (ii=0; ii < nobj; ii++) {
    cyl_to_rect_galpy(yo+6*ii);
    odeint_func(odeint_deriv_func,dim,yo+6*ii,nt,dt,t,
		npot,potentialArgs+omp_get_thread_num()*npot,rtol,atol,
		result+6*nt*ii,err+ii);
    for (jj=0; jj < nt; jj++)
      rect_to_cyl_galpy(result+6*jj+6*nt*ii);
  }
  //Free allocated memory
#pragma omp parallel for schedule(static,1) private(ii) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++)
    free_potentialArgs(npot,potentialArgs+ii*npot);
  free(potentialArgs);
  //Done!
}
// LCOV_EXCL_START
// LCOV_EXCL_STOP
void evalRectDeriv(double t, double *q, double *a,
		   int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phitorque,z,vR,vT;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //Rest is force
  //q is rectangular so calculate R and phi, vR and vT (for dissipative)
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  vR=  *(q+3) * cosphi + *(q+4) * sinphi;
  vT= -*(q+3) * sinphi + *(q+4) * cosphi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs,vR,vT,*(q+5));
  phitorque= calcphitorque(R,z,phi,t,nargs,potentialArgs,vR,vT,*(q+5));
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a++= sinphi*Rforce+1./R*cosphi*phitorque;
  *a= calczforce(R,z,phi,t,nargs,potentialArgs,vR,vT,*(q+5));;
}


void initMovingObjectSplines(struct potentialArg * potentialArgs,
			     double ** pot_args){
  gsl_interp_accel *x_accel_ptr = gsl_interp_accel_alloc();
  gsl_interp_accel *y_accel_ptr = gsl_interp_accel_alloc();
  gsl_interp_accel *z_accel_ptr = gsl_interp_accel_alloc();
  int nPts = (int) **pot_args;

  gsl_spline *x_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);
  gsl_spline *y_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);
  gsl_spline *z_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);

  double * t_arr = *pot_args+1;
  double * x_arr = t_arr+1*nPts;
  double * y_arr = t_arr+2*nPts;
  double * z_arr = t_arr+3*nPts;

  double * t= (double *) malloc ( nPts * sizeof (double) );
  double tf = *(t_arr+4*nPts+2);
  double to = *(t_arr+4*nPts+1);

  int ii;
  for (ii=0; ii < nPts; ii++)
    *(t+ii) = (t_arr[ii]-to)/(tf-to);

  gsl_spline_init(x_spline, t, x_arr, nPts);
  gsl_spline_init(y_spline, t, y_arr, nPts);
  gsl_spline_init(z_spline, t, z_arr, nPts);

  potentialArgs->nspline1d= 3;
  potentialArgs->spline1d= (gsl_spline **) malloc ( 3*sizeof ( gsl_spline *) );
  potentialArgs->acc1d= (gsl_interp_accel **) \
    malloc ( 3 * sizeof ( gsl_interp_accel * ) );
  *potentialArgs->spline1d = x_spline;
  *potentialArgs->acc1d = x_accel_ptr;
  *(potentialArgs->spline1d+1)= y_spline;
  *(potentialArgs->acc1d+1)= y_accel_ptr;
  *(potentialArgs->spline1d+2)= z_spline;
  *(potentialArgs->acc1d+2)= z_accel_ptr;

  *pot_args = *pot_args + (int) (1+4*nPts);
  free(t);
}


// LCOV_EXCL_START

// LCOV_EXCL_STOP
