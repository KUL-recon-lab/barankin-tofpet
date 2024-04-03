#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nrutil.h"

#define     PI        3.14159265359

/* December 2021 */

double erfcc(x)

/* From C recipes, see book page 221. Fractional error everywhere smaller than 1.2 10^-7 */
/* For some reason does not link as an external .c file so I copied it here */

double x;
{
    double t,z,ans;
    
    z=fabs(x);
    t=1.0/(1.0+0.5*z);
    ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
                                                             t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
                                                                                                            t*(-0.82215223+t*0.17087277)))))))));
    return x >= 0.0 ? ans : 2.0-ans;
}
/*___________________________________________________*/

double pdf_purebiexp(t, taud, taur)

/* pdf of detection time of scintillation photons, bi-exponential model without optical transport */

double t;
double taud;    /* decay characteristic time */
double taur;    /* rise characteristic time */

{
    double pdf;
    
    pdf = (exp(-t/taud)-exp(-t/taur))/(taud-taur);
    return pdf;
}

/*___________________________________________________*/
       
double cdf_purebiexp(t, taud, taur)
       
/* cdf of detection time of scintillation photons, bi-exponential model without optical transport */
       
double t;
double taud;    /* decay characteristic time */
double taur;    /* rise characteristic time */
       
{
    double cdf;
    
    cdf = 1 - (taud*exp(-t/taud) - taur*exp(-t/taur))/(taud-taur);
    return cdf;
}
           
/*___________________________________________________*/
double abiexp(t, tau,trans,sigtrans)

/* function defined by equation (5) in Johan's note */

double t;
double tau;         /* characteristic time */
double trans;       /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double a;
    double x1,x2,a1,a2;
    double xmax = 26.0; /* threshold for using the asymptotic form of erfc(x), relative error O(1/2x*x) */
    x1 = (trans+sigtrans*sigtrans/tau)/(sqrt(2)*sigtrans);
    x2 = (t-trans-sigtrans*sigtrans/tau)/(sqrt(2)*sigtrans);
    if((x1 > xmax) && (x2 < -xmax)) { /* use asymptotic form of erfc(x) to avoid 0-0=nan */
        a1 = sigtrans*exp(-trans*trans/(2*sigtrans*sigtrans) - t/tau)/(sqrt(2*PI)*(trans+ sigtrans*sigtrans/tau));
        a2 = sigtrans*exp(-(trans-t)*(trans-t)/(2*sigtrans*sigtrans))/(sqrt(2*PI)*(trans+ sigtrans*sigtrans/tau -t));
        a = a2 - a1;
    }
    /* previous versions used erf(x) but same problem with large x values :
    a = 0.5*exp(-((t-trans)/tau) + 0.5*sigtrans*sigtrans/(tau*tau))*(erff(x2)+erff(x1));
    */
    else /* equation (5) in Johan's note */
        a = 0.5*exp(-((t-trans)/tau) + 0.5*sigtrans*sigtrans/(tau*tau))*(erfcc(-x2)-erfcc(x1));
    
    return a;
}
/*___________________________________________________*/

double Dabiexp(t, tau,trans,sigtrans)

/* t derivative of abiexp */

double t;
double tau;    /* characteristic time */
double trans;   /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double da;
    da = -abiexp(t, tau,trans,sigtrans)/tau + (1/(sigtrans*sqrt(2*PI)))*exp(-(t-trans)*(t-trans)/(2*sigtrans*sigtrans));
    return da;
}

    /*___________________________________________________*/

double D2abiexp(t, tau,trans,sigtrans)
                                               
/* second t derivative of abiexp */
                                               
 double t;
 double tau;    /* characteristic time */
 double trans;   /* mean optical transfer time */
 double sigtrans;    /* standard dev. optical transfer time */
                                               
{
    double d2a;
    double x1;
    x1 = (1/(sigtrans*sigtrans*sigtrans*sqrt(2*PI))*(sigtrans*sigtrans/tau + t - trans));
    d2a = abiexp(t, tau,trans,sigtrans)/(tau*tau) - x1*exp(-(t-trans)*(t-trans)/(2*sigtrans*sigtrans));
    return d2a;
}
                                           

/*___________________________________________________*/

double pdf_biexp(t, taud, taur,trans,sigtrans)

/* pdf of detection time of scintillation photons, bi-exponential model */
/* Equation (2) in Johan's note with a single component */


double t;
double taud;    /* decay characteristic time */
double taur;    /* rise characteristic time */
double trans;   /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double pdf;
    double C;
    
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    pdf = C*(abiexp(t, taud, trans,sigtrans)-abiexp(t, taur, trans,sigtrans))/(taud-taur);
    return pdf;
}
/*___________________________________________________*/

double cdf_biexp(t, taud, taur,trans,sigtrans)

/* Cumulative distribution of detection time of scintillation photons, bi-exponential model */

double t;
double taud;    /* decay characteristic time */
double taur;    /* rise characteristic time */
double trans;   /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double cdf;
    double C, x1, x2;
    
    x1 = trans/(sqrt(2)*sigtrans);
    x2 = (t-trans)/(sqrt(2)*sigtrans);
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    cdf = C*(- taud*abiexp(t, taud, trans,sigtrans) + taur*abiexp(t, taur, trans,sigtrans))/(taud-taur);
    cdf += 0.5*C*(erfcc(-x2)-erfcc(x1));
    return cdf;
}
/*___________________________________________________*/
double Dpdf_biexp(t, taud, taur,trans,sigtrans)

/* t derivative of the pdf of detection time of scintillation photons, bi-exponential model */

double t;
double taud;    /* decay characteristic time */
double taur;    /* rise characteristic time */
double trans;   /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double pdf;
    double C;
    
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    pdf = C*(Dabiexp(t, taud, trans,sigtrans)-Dabiexp(t, taur, trans,sigtrans))/(taud-taur);
    return pdf;
}

/*___________________________________________________*/
double D2pdf_biexp(t, taud, taur,trans,sigtrans)

/* second t derivative of the pdf of detection time of scintillation photons, bi-exponential model */

double t;
double taud;    /* decay characteristic time */
double taur;    /* rise characteristic time */
double trans;   /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double pdf;
    double C;
    
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    pdf = C*(D2abiexp(t, taud, trans,sigtrans)-D2abiexp(t, taur, trans,sigtrans))/(taud-taur);
    return pdf;
}

/*___________________________________________________*/

double pdf_delta(t,trans,sigtrans)

/* pdf of detection time of prompt emission as Dirac delta */
/* the delta emission is convolved with a gaussian */

double t;
double trans;       /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double pdf;
    double C;
    
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    pdf = C*exp(-(t-trans)*(t-trans)/(2*sigtrans*sigtrans))/(sqrt(2*PI)*sigtrans);
    return pdf;
}
/*___________________________________________________*/

double cdf_delta(t, trans,sigtrans)

/* Cumulative distribution of prompt emission as Dirac delta */
/* the delta emission is convolved with a gaussian */

double t;
double trans;   /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double cdf;
    double C, x1, x2;
    
    x1 = trans/(sqrt(2)*sigtrans);
    x2 = (t-trans)/(sqrt(2)*sigtrans);
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    cdf = 0.5*C*(erf(x2)+erf(x1));
    return cdf;
}

/*___________________________________________________*/
double Dpdf_delta(t,trans,sigtrans)

/* t derivative of the pdf of prompt emission as Dirac delta */
/* the delta emission is convolved with a gaussian */

double t;
double trans;   /* mean optical transfer time */
double sigtrans;    /* standard dev. optical transfer time */

{
    double pdf;
    double C;
    
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    pdf = C*(trans-t)*exp(-(t-trans)*(t-trans)/(2*sigtrans*sigtrans))/(sqrt(2*PI)*sigtrans*sigtrans*sigtrans);
    return pdf;
}

/*___________________________________________________*/
