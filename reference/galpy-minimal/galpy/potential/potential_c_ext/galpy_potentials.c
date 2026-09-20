#include <galpy_potentials.h>
void init_potentialArgs(int npot, struct potentialArg * potentialArgs){
  int ii;
  for (ii=0; ii < npot; ii++) {
    (potentialArgs+ii)->wrappedPotentialArg= NULL;
    (potentialArgs+ii)->spline1d= NULL;
    (potentialArgs+ii)->acc1d= NULL;
    (potentialArgs+ii)->tfuncs= NULL;
    (potentialArgs+ii)->pot_data= NULL;
    (potentialArgs+ii)->free_pot_data= NULL;
  }
}
void free_potentialArgs(int npot, struct potentialArg * potentialArgs){
  int ii, jj;
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->wrappedPotentialArg ) {
      free_potentialArgs((potentialArgs+ii)->nwrapped,
			 (potentialArgs+ii)->wrappedPotentialArg);
      free((potentialArgs+ii)->wrappedPotentialArg);
    }
    if ( (potentialArgs+ii)->spline1d ) {
      for (jj=0; jj < (potentialArgs+ii)->nspline1d; jj++)
	gsl_spline_free(*((potentialArgs+ii)->spline1d+jj));
      free((potentialArgs+ii)->spline1d);
    }
    if ( (potentialArgs+ii)->acc1d ) {
      for (jj=0; jj < (potentialArgs+ii)->nspline1d; jj++)
	gsl_interp_accel_free (*((potentialArgs+ii)->acc1d+jj));
      free((potentialArgs+ii)->acc1d);
    }
    if ( (potentialArgs+ii)->free_pot_data )
      (potentialArgs+ii)->free_pot_data((potentialArgs+ii)->pot_data);
    free((potentialArgs+ii)->args);
  }
}
// function name in parentheses, because actual function defined by macro
// in galpy_potentials.h and parentheses are necessary to avoid macro expansion
double (calcRforce)(double R, double Z, double phi, double t,
		    int nargs, struct potentialArg * potentialArgs,
		    double vR, double vT, double vZ){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
      Rforce+= potentialArgs->RforceVelocity(R,Z,phi,t,potentialArgs,vR,vT,vZ);
    else
      Rforce+= potentialArgs->Rforce(R,Z,phi,t,
				     potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return Rforce;
}
double (calczforce)(double R, double Z, double phi, double t,
		    int nargs, struct potentialArg * potentialArgs,
		    double vR, double vT, double vZ){
  int ii;
  double zforce= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
      zforce+= potentialArgs->zforceVelocity(R,Z,phi,t,potentialArgs,vR,vT,vZ);
    else
      zforce+= potentialArgs->zforce(R,Z,phi,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return zforce;
}
double (calcphitorque)(double R, double Z, double phi, double t,
		      int nargs, struct potentialArg * potentialArgs,
		      double vR, double vT, double vZ){
  int ii;
  double phitorque= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
      phitorque+= potentialArgs->phitorqueVelocity(R,Z,phi,t,potentialArgs,
						 vR,vT,vZ);
    else
      phitorque+= potentialArgs->phitorque(R,Z,phi,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return phitorque;
}

// LCOV_EXCL_START

// LCOV_EXCL_STOP

