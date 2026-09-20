/* C implementations of galpy potentials */
/*
  Structure declarations
*/
#ifndef __GALPY_POTENTIALS_H__
#define __GALPY_POTENTIALS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <gsl/gsl_spline.h>
#ifndef M_1_PI
#define M_1_PI 0.31830988618379069122
#endif
typedef double (**tfuncs_type_arr)(double t); // array of functions of time
struct potentialArg{
  double (*potentialEval)(double R, double Z, double phi, double t,
			  struct potentialArg *);
  double (*Rforce)(double R, double Z, double phi, double t,
		   struct potentialArg *);
  double (*zforce)(double R, double Z, double phi, double t,
		   struct potentialArg *);
  double (*phitorque)(double R, double Z, double phi, double t,
		     struct potentialArg *);
  double (*planarRforce)(double R,double phi, double t,
			 struct potentialArg *);
  double (*planarphitorque)(double R,double phi, double t,
			   struct potentialArg *);
  double (*R2deriv)(double R,double Z,double phi, double t,
		    struct potentialArg *);
  double (*phi2deriv)(double R,double Z,double phi, double t,
		      struct potentialArg *);
  double (*Rphideriv)(double R,double Z,double phi, double t,
		      struct potentialArg *);
  double (*planarR2deriv)(double R,double phi, double t,
			  struct potentialArg *);
  double (*planarphi2deriv)(double R,double phi, double t,
			    struct potentialArg *);
  double (*planarRphideriv)(double R,double phi, double t,
			    struct potentialArg *);
  double (*linearForce)(double x, double t,
			 struct potentialArg *);
  double (*dens)(double R, double Z, double phi, double t,
		 struct potentialArg *);
  // For forces that require velocity input (e.g., dynam fric)
  bool requiresVelocity;
  double (*RforceVelocity)(double R, double Z, double phi, double t,
			    struct potentialArg *,double,double,double);
  double (*zforceVelocity)(double R, double Z, double phi, double t,
			   struct potentialArg *,double,double,double);
  double (*phitorqueVelocity)(double R, double Z, double phi, double t,
			     struct potentialArg *,double,double,double);
  double (*planarRforceVelocity)(double R,double phi, double t,
			 struct potentialArg *,double,double);
  double (*planarphitorqueVelocity)(double R,double phi, double t,
			   struct potentialArg *,double,double);

  int nargs;
  double * args;
  // To allow 1D interpolation for an arbitrary number of splines
  int nspline1d;
  gsl_interp_accel ** acc1d;
  gsl_spline ** spline1d;
  // To allow an arbitrary number of functions of time
  int ntfuncs;
  tfuncs_type_arr tfuncs; // see typedef above
  // Wrappers
  int nwrapped;
  struct potentialArg * wrappedPotentialArg;
  // For EllipsoidalPotentials
  double (*psi)(double m,double * args);
  double (*mdens)(double m,double * args);
  double (*mdensDeriv)(double m,double * args);
  // For SphericalPotentials
  double (*revaluate)(double r,double t,struct potentialArg *);
  double (*rforce)(double r,double t,struct potentialArg *);
  double (*r2deriv)(double r,double t,struct potentialArg *);
  double (*rdens)(double r,double t,struct potentialArg *);
  // Potential-specific pre-computed data and workspace
  void *pot_data;
  void (*free_pot_data)(void *);
};
/*
  Function declarations
*/
//Dealing with potentialArg
void init_potentialArgs(int,struct potentialArg *);
void free_potentialArgs(int,struct potentialArg *);
//Potential and force evaluation
// Hack to allow optional velocity for dissipative forces
// https://stackoverflow.com/a/52610204/10195320
// Reason to use ##__VA_ARGS__ is that when no optional velocity is supplied,
// there is a comma left in the argument list and ## absorbs that for gcc/icc
// MSVC supposedly does this automatically for regular __VA_ARGS__, at least
// for now, but I can't get this to work locally or on AppVeyor
// https://docs.microsoft.com/en-us/cpp/preprocessor/preprocessor-experimental-overview?view=vs-2019#comma-elision-in-variadic-macros
// Therefore, I use an alternative where all arguments are variadic, such
// that there should always be many. Final subtlety is that we have to define
// the EXPAND macro to expand the __VA_ARGS__ into multiple arguments,
// otherwise it's treated as a single argument in the next function call
// (e.g., CALCRFORCE would get R = __VA_ARGS = R,Z,phi,...)
#ifdef _MSC_VER
#define EXPAND(x) x
#define calcRforce(...)   EXPAND(CALCRFORCE(__VA_ARGS__,0.,0.,0.))
#define calczforce(...)   EXPAND(CALCZFORCE(__VA_ARGS__,0.,0.,0.))
#define calcphitorque(...) EXPAND(CALCPHITORQUE(__VA_ARGS__,0.,0.,0.))
#else
#define calcRforce(R,Z,phi,t,nargs,potentialArgs,...) CALCRFORCE(R,Z,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.,0.)
#define calczforce(R,Z,phi,t,nargs,potentialArgs,...) CALCZFORCE(R,Z,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.,0.)
#define calcphitorque(R,Z,phi,t,nargs,potentialArgs,...) CALCPHITORQUE(R,Z,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.,0.)
#endif
#define CALCRFORCE(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ,...) calcRforce(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ)
#define CALCZFORCE(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ,...) calczforce(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ)
#define CALCPHITORQUE(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ,...) calcphitorque(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ)
double (calcRforce)(double,double,double,double,int,struct potentialArg *,
		    double,double,double);
double (calczforce)(double,double,double,double,int,struct potentialArg *,
		      double,double,double);
double (calcphitorque)(double, double,double, double,
		      int, struct potentialArg *,
		      double,double,double);
// end hack
double ZeroForce(double,double,double,double,
		 struct potentialArg *);
//Miyamoto-Nagai Potential
double MiyamotoNagaiPotentialEval(double ,double , double, double,
				  struct potentialArg *);
double MiyamotoNagaiPotentialRforce(double ,double , double, double,
				    struct potentialArg *);
double MiyamotoNagaiPotentialzforce(double,double,double,double,
				    struct potentialArg *);
double MiyamotoNagaiPotentialDens(double ,double , double, double,
				  struct potentialArg *);
//NFWPotential
double NFWPotentialEval(double ,double , double, double,
			struct potentialArg *);
double NFWPotentialRforce(double ,double , double, double,
			  struct potentialArg *);
double NFWPotentialzforce(double,double,double,double,
			  struct potentialArg *);
double NFWPotentialDens(double ,double , double, double,
			 struct potentialArg *);
//PowerSphericalPotentialwCutoff
double PowerSphericalPotentialwCutoffEval(double ,double , double, double,
					  struct potentialArg *);
double PowerSphericalPotentialwCutoffRforce(double ,double , double, double,
					    struct potentialArg *);
double PowerSphericalPotentialwCutoffzforce(double,double,double,double,
					    struct potentialArg *);
double PowerSphericalPotentialwCutoffDens(double ,double , double, double,
					  struct potentialArg *);

//PlummerPotential
double PlummerPotentialEval(double,double,double,double,
                        struct potentialArg *);
double PlummerPotentialRforce(double,double,double,double,
                        struct potentialArg *);
double PlummerPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double PlummerPotentialDens(double,double,double,double,
			    struct potentialArg *);

//DiskSCFPotential

// SpiralArmsPotential





//NonInertialFrameForce, takes vR,vT,vZ

//////////////////////////////// WRAPPERS /////////////////////////////////////
//MovingObjectPotential
double MovingObjectPotentialRforce(double,double,double,double,
					struct potentialArg *);
double MovingObjectPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double MovingObjectPotentialzforce(double,double,double,double,
				        struct potentialArg *);
//CylindricallySep

#ifdef __cplusplus
}
#endif
#endif /* galpy_potentials.h */
