#include <math.h>
#include <bovy_coords.h>
/*
NAME: cyl_to_rect_galpy
PURPOSE: convert (R,vR,vT,z,vz,phi) to (x,y,z,vx,vy,vz)
INPUT:
   double * vxvv - (R,vR,vT,z,vz,phi)
OUTPUT:
   performed in-place
HISTORY: 2012-12-24 - Written - Bovy (UofT)
 */
void cyl_to_rect_galpy(double *vxvv){
  double R,phi,cp,sp,vR,vT;
  R  = *vxvv;
  phi= *(vxvv+5);
  cp = cos ( phi );
  sp = sin ( phi );
  vR = *(vxvv+1);
  vT = *(vxvv+2);
  *vxvv    = R * cp;
  *(vxvv+1)= R * sp;
  *(vxvv+2)= *(vxvv+3);
  *(vxvv+5)= *(vxvv+4);
  *(vxvv+3)= vR * cp - vT * sp;
  *(vxvv+4)= vR * sp + vT * cp;
}
/*
NAME: rect_to_cyl_galpy
PURPOSE: convert (x,y,z,vx,vy,vz) to (R,vR,vT,z,vzphi)
INPUT:
   double * vxvv - (x,y,z,vx,vy,vz)
OUTPUT (as arguments):
   performed in-place
HISTORY: 2012-12-24 - Written - Bovy (UofT)
 */
void rect_to_cyl_galpy(double *vxvv){
  double x,y,vx,vy,cp,sp;
  x = *vxvv;
  y = *(vxvv+1);
  vx= *(vxvv+3);
  vy= *(vxvv+4);
  *(vxvv+3)= *(vxvv+2);
  *(vxvv+4)= *(vxvv+5);
  *(vxvv+5)= atan2( y , x );
  cp = cos ( *(vxvv+5) );
  sp = sin ( *(vxvv+5) );
  *vxvv    = sqrt ( x * x + y * y );
  *(vxvv+1)=  vx * cp + vy * sp;
  *(vxvv+2)= -vx * sp + vy * cp;
}
