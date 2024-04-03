#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nrutil.h"

#define     PI        3.14159265359
#define     BIG       4.5e+21
#define     TINY      1.0e-25
#define     EIGMIN    0.00000001
/* eigenvalues smaller than EIGMIN are ignored when using the pseudo-inverse */
float ran1(long *idum);
extern void     jacobi(double **a, int n, double d[], double **v, int *nrot);

/* functions defined in BiExp.c */
double abiexp(double t, double tau,double trans,double sigtrans);
double abiexpW(double t, double tau,double trans,double sigtrans);
double Dabiexp(double t, double tau,double trans,double sigtrans);
double D2abiexp(double t, double tau,double trans,double sigtrans);
double pdf_biexp(double t, double taud,double taur, double trans,double sigtrans);
double cdf_biexp(double t, double taud,double taur, double trans,double sigtrans);
double Dpdf_biexp(double t, double taud,double taur, double trans,double sigtrans);
double D2pdf_biexp(double t, double taud,double taur, double trans,double sigtrans);
double pdf_delta(double t, double trans,double sigtrans);
double Dpdf_delta(double t, double trans,double sigtrans);
double cdf_delta(double t, double trans,double sigtrans);

double erfcc(double x);

int main(argc,argv)


int argc;
char *argv[];

/* January 2022 */
/* Estimate Barankin bounds of order J for the standard deviation of an unbiased estimator */
/* of the arrival time for the combined prompt and bi-exponential model. */
/* For the prompt photons one can chose an impulse or a different bi-exponential model */
/* The two contributions are convolved with a gaussian model of the optical transfer time. */
/* The J shifts are selected randomly, and one tests Ntest different sets of J shifts */

/* compile as: 
 BIEXPDELTABARANKINORDERJ: BiExpDeltaBarankinOrderJ.o BiExp.o ran1.o nrutil.o jacobi.o $(HEADERS)
 gcc -o code/BIEXPDELTABARANKINORDERJ BiExpDeltaBarankinOrderJ.o BiExp.o ran1.o nrutil.o jacobi.o -I.
 
 */

{
int J;
double alpha;       /* prompt photon fraction */
double taudp;       /* decay characteristic time of the prompt photons, 0 = delta */
double taurp;       /* rise characteristic time of the prompt photons, 0 = delta */
double taud;        /* decay characteristic time of the scintillator */
double taur;        /* rise characteristic time of the scintillator*/
double trans;       /* mean optical transfer time */
double sigtrans;    /* standard dev. of the optical transfer time */
double dt;          /* discretization step for the t integrals */
double tmax;        /* truncate integrals at tmax */
long Np;            /* total number of detected photons */
double t;           /* detection time of a foton with t=0 set at the arrival time of the gamma */
double norm;        /* integral of pdf, should be 1 */
double deriv;       /* two points derivative (for checking only) */
double *pdf, *dpdf, *numdpdf, *cdf; /* samples of the pdf, its derivative, its two point derivative, and the cumulative distribution */
long n, it;             /* number of time discretization samples, and sample index */
double sum2, sumerr2;   /* checking analytic derivative by comparing with the numerical derivative */
double meant, vart;     /* mean and variance of the random variable t */
double meanprompt, varprompt;   /* mean and variance of prompt emission */
double meanbiexp, varbiexp;     /* mean and variance of scintillation emission */
double FisherI;

double tbarankin;           /* A Barankin shift in ns ("delta" in paper TRMPS_Barankin) */
long   itshift1;            /* A Barankin shift in number of time samples dt */
unsigned long   *itshift;   /* List of the J Barankin shifts in number of time samples dt */

double Bound, Bestbound;                    /* The Barankin bound for 1 photon */
double *Besttbarankin, *BesttbarankinNp;    /* The J Barankin shifts yielding the highest bound */
double BoundNp, BestboundNp;                /* The Barankin bound for Np photons */
double tbarankinmin;  /* min and max search values for Barankin shift */
double tbarankinmax;
int steps;      /* number of values of the Barankin shift among which J are chosen (set to 80 in paper TRMPS_Barankin) */
double fact;    /* tbarankin are sampled with multiplicative ratio fact */
double *delta;  /* list of tested values of the Barankin shift (in ns) */
double *deltashuffle; /* permutation of vector delta used to generate random combinations of J shifts */
int Ntest, test;      /* number of tested random sets of J shifts ("V_delta" in paper TRMPS_Barankin) and index of a set */
int NtestNp;          /* number of tested random sets for Np photons */
double v, C;

int istep;
double **U ; /* JxJ matrices for the Jnd order Barankin bound */
double **eigvector, *eigvalues;
double *Besteigvalues, *BesteigvaluesNp;
double normerror, rmseerror; /* to monitor the accuracy of the diagonalisation of U */
double normerrorNp, rmseerrorNp;
    
double **A; /* JxJ matrix to check the diagonalisation of U */
double *V;  /* vector with the J shift values delta */
double pp1, pp2;
int l1,l2,l3, j ;
int nrot;   /* argument of jacobi routine, unused here */
    
long idum;  /* used to initialize the random number generator */
int init;
    
if(argc < 17) {
    printf("EXPBARANKINORDERJ order Nphotons alpha taudprompt taurprompt taud taur trans sigtrans dt tmax t_barankin_min t_barankin_max steps Ntest init \n");
    printf("Order of the Barankin bound \n");
    printf("Number of photons \n");
    printf("Prompt photon fraction \n");
    printf("Decay time of the prompt photon fraction ns (0  -> delta) \n");
    printf("Rise time of the prompt photon fraction ns (0  -> delta) \n");
    printf("Decay time ns \n");
    printf("Rise time ns \n");
    printf("Optical mean transit ns \n");
    printf("Optical st. dev. transit ns \n");
    printf("Time sampling step ns \n");
    printf("Stop integration at tmax ns \n");
    printf("Minimum Barankin time shift > 0 \n");
    printf("Maximum Barankin time shift > 0 \n");
    printf("Number of steps \n");
    printf("Number of tested sets \n");
    printf("Random seed \n\n");
    return -1;
}
J = (int) atoi(argv[1]);
Np = (long) atoi(argv[2]);
alpha = (double) atof(argv[3]);
taudp = (double) atof(argv[4]);
taurp = (double) atof(argv[5]);
taud = (double) atof(argv[6]);
taur = (double) atof(argv[7]);
trans = (double) atof(argv[8]);
sigtrans = (double) atof(argv[9]);
dt =(double) atof(argv[10]);
tmax = (double) atof(argv[11]);
tbarankinmin =(double) atof(argv[12]);
tbarankinmax =(double) atof(argv[13]);
steps = (int) atoi(argv[14]); /* Barankin bound will be computed for steps values of the Barankin shift */
fact = exp(log(tbarankinmax/tbarankinmin)/steps); /* geometric progression sampling of the Barankin shift */
Ntest = (int) atoi(argv[15]);
init = atoi(argv[16]);

if((J ==1) || (J > 100) || (J > steps)) {
printf("Order must be > 1 and <= 100 (to prevent too large matrices) and <= steps. Check parameters ! \n");
return -1;
}

if((alpha < 0) || (alpha > 1)){
    printf("One must have prompt photon fraction 0 <= alpha <= 1 Check parameters ! \n");
    return -1;
}
if(taudp < taurp){
    printf("One must have taudp > taurp for the prompt photons. Check parameters ! \n");
    return -1;
}
if((taurp == 0) && (taudp > 0)){
    printf("One must have taudp = taurp = 0 for impulsive prompt photons. Check parameters ! \n");
    return -1;
}
if(taud < taudp){
    printf("One must have taud > taudp (prompt photons are decay faster). Check parameters ! \n");
    return -1;
}
if(taud <= taur){
    printf("One must have taud > taur. Check parameters ! \n");
    return -1;
}
if(tmax <= tbarankinmax){
    printf("Integration range too small wrt maximum shift value. Check parameters ! \n");
    return -1;
}
if(tbarankinmax <= tbarankinmin){
    printf("One must have tbarankinmax > tbarankinmin. Check parameters ! \n");
    return -1;
}
if(tbarankinmin < 5*dt){
    printf("Decrease integration step (less than 5 samples for first shift value. Check parameters ! \n");
    return -1;
}
    
printf("Barankin bound of order %d for the smoothed bi-exponential distribution\n",J);
printf("%ld photons detected\n",Np);
printf("Prompt photon fraction %lf \n", alpha);
if(taurp > 0) {
    printf("      Decay time %lf ns \n", taudp);
    printf("      Rise time %lf ns \n", taurp);}
else printf("      Prompt photons have impulse pdf ns \n");
printf("Scintillation photon fraction %lf \n", 1-alpha);
printf("      Decay time %lf ns \n", taud);
printf("      Rise time %lf ns \n", taur);
printf("Optical mean transit %lf ns \n", trans);
printf("Optical st. dev. transit %lf ns \n", sigtrans);
printf("dt = %lf ns , tmax = %lf ns \n", dt, tmax);
printf("Barankin time shift range: %20.10lf --> %20.10lf ns  %d steps rate %20.10lf \n\n", tbarankinmin, tbarankinmax, steps, fact);
printf("%d random sets of shifts are tested (seed: %d) \n",Ntest, init);

/* BUILD THE PDF AND CDF AND DO A FEW CHECKS */
    
n = (long) tmax/dt;    /* number of time discretization steps */
pdf = dvector(0,n);
dpdf = dvector(1,n);
cdf = dvector(0,n);
norm = 0.0;
    /* Store the analytic values of the pdf and its derivatives */
    if(taurp == 0) for(it=0;it <= n;++it) { /* prompt photons have an impulse pdf */
        t = it*dt;
        pdf[it] = alpha*pdf_delta(t,trans,sigtrans)+(1-alpha)*pdf_biexp(t, taud, taur,trans,sigtrans);
        dpdf[it] = alpha*Dpdf_delta(t,trans,sigtrans)+(1-alpha)*Dpdf_biexp(t, taud, taur,trans,sigtrans);
        cdf[it] = alpha*cdf_delta(t,trans,sigtrans)+(1-alpha)*cdf_biexp(t, taud, taur,trans,sigtrans);
        if(it==0) norm += 0.5*dt*pdf[it]; /* trapezoid quadrature */
        else norm += dt*pdf[it];
    }
    else for(it=0;it <= n;++it) {
        t = it*dt;
        pdf[it] = alpha*pdf_biexp(t, taudp, taurp,trans,sigtrans)+(1-alpha)*pdf_biexp(t, taud, taur,trans,sigtrans);
        dpdf[it] = alpha*Dpdf_biexp(t, taudp, taurp,trans,sigtrans)+(1-alpha)*Dpdf_biexp(t, taud, taur,trans,sigtrans);
        cdf[it] = alpha*cdf_biexp(t, taudp, taurp,trans,sigtrans)+(1-alpha)*cdf_biexp(t, taud, taur,trans,sigtrans);
        if(it==0) norm += 0.5*dt*pdf[it]; /* trapezoid quadrature */
        else norm += dt*pdf[it];
    }
    printf("Norm = %20.10lf , cdf = %20.10lf (both should be equal and equal to 1)\n\n",norm, cdf[n]);
    
    /* Store the two point estimate of the derivative (for checking) */
    numdpdf = dvector(0,n);
    numdpdf[0] = 0;
    for(it=1;it < n;++it) numdpdf[it] = (pdf[it+1]-pdf[it-1])/(2*dt);
    numdpdf[n] = 0;
        
    /* Check correspondance between analytic and numerical derivatives of the pdf */
    sum2 = 0;
    sumerr2 = 0;
    for(it=0;it <= n;++it) {
        sum2 +=dpdf[it]*dpdf[it];
        sumerr2 += (dpdf[it]-numdpdf[it])*(dpdf[it]-numdpdf[it]);
    }
    printf("\n Relative RMSE on derivative: %20.6lf\n",sqrt(sumerr2/sum2));
    
    /* Calculate mean and standard deviation of t by integrating the pdf */
    meant = 0;
    for(it=0;it <= n;++it) meant += it*dt*pdf[it]*dt;
    vart = 0;
    for(it=0;it <= n;++it) {
        if(it==0) vart += 0.5*(it*dt-meant)*(it*dt-meant)*pdf[it]*dt;
        else vart += (it*dt-meant)*(it*dt-meant)*pdf[it]*dt;
    }
    
    printf("\n <t> = %20.6lf   St. dev. = %20.6lf \n",meant, sqrt(vart));
    
    /* check by calculating the analytic expressions of meant and vart */
    
    C = 2/(1+ erf(trans/(sqrt(2)*sigtrans)));
    meanbiexp = taud+taur+trans+ exp(-trans*trans/(2*sigtrans*sigtrans))*sigtrans*C/(sqrt(2*PI));
    varbiexp = taud*taud+taur*taur+sigtrans*sigtrans - exp(-trans*trans/(2*sigtrans*sigtrans))*sigtrans*trans*C/(sqrt(2*PI));
    varbiexp -= C*C*exp(-trans*trans/(sigtrans*sigtrans))*sigtrans*sigtrans/(2*PI);
    
    if(taurp == 0) { /* prompt photons have an impulse pdf */
        meanprompt = trans + exp(-trans*trans/(2*sigtrans*sigtrans))*sigtrans*C/(sqrt(2*PI));
        varprompt = sigtrans*sigtrans - C*trans*sigtrans*exp(-trans*trans/(2*sigtrans*sigtrans))/(sqrt(2*PI));
        varprompt -= C*C*exp(-trans*trans/(sigtrans*sigtrans))*sigtrans*sigtrans/(2*PI);
    }
    else {
        meanprompt = taudp+taurp+trans+ exp(-trans*trans/(2*sigtrans*sigtrans))*sigtrans*C/(sqrt(2*PI));
        varprompt = taudp*taudp+taurp*taurp+sigtrans*sigtrans - exp(-trans*trans/(2*sigtrans*sigtrans))*sigtrans*trans*C/(sqrt(2*PI));
        varprompt -= C*C*exp(-trans*trans/(sigtrans*sigtrans))*sigtrans*sigtrans/(2*PI);
    }
    /* combine the two fractions */
    meant = alpha*meanprompt+(1-alpha)*meanbiexp;
    vart = alpha*varprompt+(1-alpha)*varbiexp+alpha*(1-alpha)*(meanprompt-meanbiexp)*(meanprompt-meanbiexp);
    if(vart >= 0) printf("\n Theory: <t> = %20.6lf  St. dev. = %20.6lf \n",meant,sqrt(vart));
        else printf("\n Theory: <t> = %20.6lf  WARNING: variance = %20.6lf \n",meant,vart);
            
            
    /* Calculate mean and standard deviation of first photon time t_(1), and norm of its pdf */
        meant = 0;
        norm = 0.0;
        for(it=0;it <= n;++it) if(cdf[it] < 1) {
            meant += it*dt*(Np*exp((Np-1)*log(1-cdf[it]))*pdf[it])*dt;
            if(it==0) norm += 0.5*(Np*exp((Np-1)*log(1-cdf[it]))*pdf[it])*dt;
            else norm += (Np*exp((Np-1)*log(1-cdf[it]))*pdf[it])*dt;
        }
    vart = 0;
    for(it=0;it <= n;++it) if(cdf[it] < 1) {
        if(it==0) vart += 0.5*(it*dt-meant)*(it*dt-meant)*(Np*exp((Np-1.0)*log(1.0-cdf[it]))*pdf[it])*dt;
        else vart += (it*dt-meant)*(it*dt-meant)*(Np*exp((Np-1.0)*log(1.0-cdf[it]))*pdf[it])*dt;
    }
    printf("\n Norm of pdf(t_(1)) = %20.6lf,  <t_(1)> = %20.6lf   St. dev. t_(1) = %20.6lf \n",norm, meant, sqrt(vart));
    
    /* Calculate mean and standard deviation of second photon time t_(2), and norm of its pdf */
    meant = 0;
    norm = 0.0;
    for(it=0;it <= n;++it) if(cdf[it] < 1) {
        meant += it*dt*(Np*(Np-1)*exp((Np-2)*log(1-cdf[it]))*cdf[it]*pdf[it])*dt;
        if(it==0) norm += 0.5*(Np*(Np-1)*exp((Np-2)*log(1-cdf[it]))*cdf[it]*pdf[it])*dt;
        else norm += (Np*(Np-1)*exp((Np-2)*log(1-cdf[it]))*cdf[it]*pdf[it])*dt;
    }
    vart = 0;
    for(it=0;it <= n;++it) if(cdf[it] < 1) {
        if(it==0) vart += 0.5*(it*dt-meant)*(it*dt-meant)*(Np*(Np-1)*exp((Np-2)*log(1-cdf[it]))*cdf[it]*pdf[it])*dt;
        else vart += (it*dt-meant)*(it*dt-meant)*(Np*(Np-1)*exp((Np-2)*log(1-cdf[it]))*cdf[it]*pdf[it])*dt;
    }
    printf("\n Norm of pdf(t_(2)) = %20.6lf,  <t_(2)> = %20.6lf   St. dev. t_(2) = %20.6lf \n",norm, meant, sqrt(vart));
    
    /* Calculate mean and standard deviation of third photon time t_(3), and norm of its pdf */
    meant = 0;
    norm = 0.0;
    for(it=0;it <= n;++it) if(cdf[it] < 1) {
        meant += it*dt*(0.5*Np*(Np-1)*(Np-2)*exp((Np-3)*log(1-cdf[it]))*cdf[it]*cdf[it]*pdf[it])*dt;
        if(it==0) norm += 0.5*(0.5*Np*(Np-1)*(Np-2)*exp((Np-3)*log(1-cdf[it]))*cdf[it]*cdf[it]*pdf[it])*dt;
        else norm += (0.5*Np*(Np-1)*(Np-2)*exp((Np-3)*log(1-cdf[it]))*cdf[it]*cdf[it]*pdf[it])*dt;
    }
    vart = 0;
    for(it=0;it <= n;++it) if(cdf[it] < 1) {
        if(it==0) vart += 0.5*(it*dt-meant)*(it*dt-meant)*(0.5*Np*(Np-1)*(Np-2)*exp((Np-3)*log(1-cdf[it]))*cdf[it]*cdf[it]*pdf[it])*dt;
        else vart += (it*dt-meant)*(it*dt-meant)*(0.5*Np*(Np-1)*(Np-2)*exp((Np-3)*log(1-cdf[it]))*cdf[it]*cdf[it]*pdf[it])*dt;
    }
    printf("\n Norm of pdf(t_(3)) = %20.6lf,  <t_(3)> = %20.6lf   St. dev. t_(3) = %20.6lf \n",norm, meant, sqrt(vart));
    
    
    FisherI = 0;
    it = 0;
    if(pdf[it] > 0) FisherI += 0.5*dt*dpdf[it]*dpdf[it]/pdf[it];
        for(it=1;it <= n;++it) if(pdf[it] > 0) FisherI += dt*dpdf[it]*dpdf[it]/pdf[it];
            printf("\n\n I = %20.10lf   Cramer Rao Bound (1 photon) = %20.10lf ns (%ld photons) = %20.10lf ns \n",FisherI, sqrt(1/FisherI), Np, sqrt(1/(Np*FisherI)));
            
            
/* CALCULATE THE OPTIMAL Jth ORDER BARANKIN BOUND FOR THE STANDARD DEVIATION OF AN UNBIASED ESTIMATOR OF THE ARRIVAL TIME */

itshift = lvector(1,J);
U = dmatrix(1,J, 1, J);
V = dvector(1,J);
A = dmatrix(1,J, 1, J);
eigvector = dmatrix(1,J, 1, J);
eigvalues = dvector(1,J);
Besteigvalues = dvector(1,J);
BesteigvaluesNp = dvector(1,J);
Besttbarankin = dvector(1,J);
BesttbarankinNp = dvector(1,J);
NtestNp = 0;
/* Build the list of the Barankin shifts values from which sets of J will be selected */
delta = dvector(1,steps);
deltashuffle = dvector(1,steps);
tbarankin=tbarankinmax;
printf("\n List of the shift values from which %d are chosen randomly (ns): \n", J);
for(istep=steps; istep >= 1; --istep ) {
    itshift1 = (long) floor(tbarankin/dt);
    tbarankin = itshift1*dt; /* make sure the shift is an integer number of sampling steps */
    delta[istep] = tbarankin;
    printf("%14.6lf ",tbarankin);
    tbarankin /= fact;
    if((istep % 8) == 0) printf("\n");
}
printf("\n\n");
Bestbound = 0;
BestboundNp = 0;
    
normerror = 0;
rmseerror = 0;
normerrorNp = 0;
rmseerrorNp = 0;
    
/* Calculate the Barankin bound for Ntest random sets of J shifts  */
for(test=1; test <= Ntest; ++test )  {
     /* random choice of J different shifts using Durstenfeld method */
    for(istep=1; istep <= steps; ++istep ) deltashuffle[istep]= delta[istep];
    for(j=1; j <= J; ++j) {
        istep = (int) (j+(steps-j+1)*ran1(&idum)); /* select one index from j to steps */
        if((istep < j) || (istep > steps)){
            printf("?? istep out of range %d (%d %d)\n", istep, j, steps);
            return -1;
        }
        tbarankin = deltashuffle[istep];
        /* printf("%20.12lf",tbarankin); */
        itshift[j] = (long) floor(tbarankin/dt);
        deltashuffle[istep] = deltashuffle[j]; /* replace the value at istep by the yet unused value at j */
    }
    /* BOUND FOR 1 PHOTON */
    /* Build and diagonalise the JxJ symmetrix matrix U, build vector V */
    for(j=1; j <= J; ++j) V[j] = itshift[j]*dt;
    

    for(l1=1; l1 <= J; ++l1) for(l2=l1; l2 <= J; ++l2) {
        U[l1][l2] = 0;
        for(it=0; it <= n;++it) if(pdf[it] > 0) {
            if(it >= itshift[l1]) pp1 = pdf[it-itshift[l1]]; else pp1 = 0;
            if(it >= itshift[l2]) pp2 = pdf[it-itshift[l2]]; else pp2 = 0;
            U[l1][l2] += dt*(pp1-pdf[it])*(pp2-pdf[it])/pdf[it];
        }
        if(l1 < l2) U[l2][l1] = U[l1][l2];
     }

/*
    for(l1=1; l1 <= J; ++l1) for(l2=l1; l2 <= J; ++l2) {
        U[l1][l2] = -1;
        itshiftmax = itshift[l1];
        if(itshiftmax < itshift[l2]) itshiftmax = itshift[l2];
        if(pdf[itshiftmax] >0) U[l1][l2] += 0.50*dt*pdf[itshiftmax-itshift[l1]]*pdf[itshiftmax-itshift[l2]]/pdf[itshiftmax];
        for(it=itshiftmax+1; it <= n;++it) if(pdf[it] > 0) {
            pp1 = pdf[it-itshift[l1]];
            pp2 = pdf[it-itshift[l2]];
            U[l1][l2] += dt*pp1*pp2/pdf[it];
        }
        if(l1 < l2) U[l2][l1] = U[l1][l2];
    }
 */

     /* diagonalise U */
    for(l1=1; l1 <= J; ++l1) for(l2=1; l2 <= J; ++l2) A[l1][l2] = U[l1][l2]; /* copy because jacobi modifies input matrix */
    jacobi(A,J,eigvalues,eigvector, &nrot);

    /* check the diagonalisation */
    pp1 = 0;
    pp2 = 0;
    for(l1=1; l1 <= J; ++l1) for(l2=1; l2 <= J; ++l2) {
        A[l1][l2] = 0;
        for(l3=1; l3 <= J; ++l3) A[l1][l2] += eigvector[l1][l3]*eigvalues[l3]*eigvector[l2][l3];
        pp1 += U[l1][l2]*U[l1][l2];
        pp2 +=(U[l1][l2]-A[l1][l2])*(U[l1][l2]-A[l1][l2]);
    }
    if(rmseerror < sqrt(pp2/pp1)) rmseerror = sqrt(pp2/pp1);
    /* check normalisation of the eigenvectors */
    pp1 = 0;
    pp2 = 0;
    for(l2=1; l2 <= J; ++l2) {
        pp1 = 0.0;
        for(l1=1; l1 <= J; ++l1) pp1 += eigvector[l1][l2]*eigvector[l1][l2];
        pp2 +=(pp1-1.0)*(pp1-1.0);
    }
    if(normerror < sqrt(pp2/pp1)) normerror = pp2;
    
    /* Calculate the Barankin bound for this set of shifts (equation (7) in paper TRMPS_Barankin)*/
    Bound = 0;
    for(l3=1; l3 <= J; ++l3) if(eigvalues[l3] > EIGMIN) {
        /* omit small eigenvalues to avoid potential numerical errors, effect is anyway to reduce the bound */
        pp1 = 0;
        for(l1=1; l1 <= J; ++l1) pp1 += V[l1]*eigvector[l1][l3];
        Bound += pp1*pp1/eigvalues[l3];
    }
    if(Bound > Bestbound) { /* record the highest hence sharpest bound among all pairs of tested shifts */
        Bestbound = Bound;
        for(j=1;j<=J;++j) {
            Besttbarankin[j] = dt*itshift[j]; /* monitor the shifts yielding the highest bound */
            Besteigvalues[j] = eigvalues[j];
        }
    }
    /* BOUND FOR Np PHOTONS: same procedure as for 1 photon, only U must be modified */
    /* Build the JxJ symmetrix matrix U and vector V */
    v = U[1][1];
    for(j=2; j <= J; ++j) if(U[j][j] > v) v = U[j][j];
    if(Np*log(1+v) < log(BIG)) {
        NtestNp += 1;
        for(l1=1; l1 <= J; ++l1) for(l2=1; l2 <= J; ++l2) U[l1][l2] = exp(Np*log(1+U[l1][l2]))-1;
        /* diagonalise U */
        for(l1=1; l1 <= J; ++l1) for(l2=1; l2 <= J; ++l2) A[l1][l2] = U[l1][l2];
        jacobi(A,J,eigvalues,eigvector, &nrot);
        
        /* check the diagonalisation of U */
        pp1 = 0;
        pp2 = 0;
        for(l1=1; l1 <= J; ++l1) for(l2=1; l2 <= J; ++l2) {
            A[l1][l2] = 0;
            for(l3=1; l3 <= J; ++l3) A[l1][l2] += eigvector[l1][l3]*eigvalues[l3]*eigvector[l2][l3];
            pp1 += U[l1][l2]*U[l1][l2];
            pp2 +=(U[l1][l2]-A[l1][l2])*(U[l1][l2]-A[l1][l2]);
        }
        if(rmseerrorNp < sqrt(pp2/pp1)) rmseerrorNp = sqrt(pp2/pp1);

        /* check normalisation of the eigenvectors */
        pp1 = 0;
        pp2 = 0;
        for(l2=1; l2 <= J; ++l2) {
            pp1 = 0.0;
            for(l1=1; l1 <= J; ++l1) pp1 += eigvector[l1][l2]*eigvector[l1][l2];
            pp2 +=(pp1-1.0)*(pp1-1.0);
        }
        if(normerrorNp < sqrt(pp2/pp1)) normerrorNp = pp2;
            
        /* Calculate the Barankin bound for this set of shifts */
        BoundNp = 0;
        for(l3=1; l3 <= J; ++l3) if(eigvalues[l3] > EIGMIN) {
            /* omit small eigenvalues to avoid potential numerical errors, effect is anyway to reduce the bound */
            pp1 = 0;
            for(l1=1; l1 <= J; ++l1) pp1 += V[l1]*eigvector[l1][l3];
            BoundNp += pp1*pp1/eigvalues[l3];
        }
        if(BoundNp > BestboundNp) { /* record the highest hence sharpest bound among all pairs of tested shifts */
            BestboundNp = BoundNp;
            for(j=1;j<=J;++j) {
                BesttbarankinNp[j] = dt*itshift[j]; /* monitor the shifts yielding the highest bound */
                BesteigvaluesNp[j] = eigvalues[j];
            }
        }
    }

 } /* next set of shifts */

/* Output results */
printf("\n\n Maximum bound (%d-th order, 1 photon): %20.10lf ns \n", J, sqrt(Bestbound));
printf("Shifts (ns) : ");
for(j=1;j<=J;++j) printf("%14.8lf  ",Besttbarankin[j]);
printf("\n");
printf("Eigenvalues (ns) : ");
for(j=1;j<=J;++j) printf("%14.8lf  ",Besteigvalues[j]);
printf("\n");
printf("\n\n Maximum bound (%d-th order, %ld photons, %d trials): %20.10lf ns \n", J, Np, NtestNp, sqrt(BestboundNp));
printf("Shifts (ns) : ");
for(j=1;j<=J;++j) printf("%14.8lf  ",BesttbarankinNp[j]);
printf("\n");
printf("Eigenvalues (ns) : ");
for(j=1;j<=J;++j) printf("%14.8lf  ",BesteigvaluesNp[j]);
printf("\n\n");
printf("Maximum RMSE Error SVD = %20.10lf \n", rmseerror);
printf("Maximum normalisation Error SVD = %20.10lf \n", normerror);
printf("%ld photons: Maximum RMSE Error SVD = %20.10lf \n", Np,rmseerrorNp);
printf("%ld photons: Maximum normalisation Error SVD = %20.10lf \n", Np, normerrorNp);
    
free_dvector(pdf,0,n);
free_dvector(dpdf,1,n);
free_dvector(numdpdf,0,n);
free_dvector(cdf,0,n);
free_dvector(delta,1,steps);
free_dvector(deltashuffle,1,steps);
free_dmatrix(U,1,J,1,J);
free_dvector(V,1,J);
free_dmatrix(A,1,J,1,J);
free_dmatrix(eigvector,1,J,1,J);
free_lvector(itshift,1,J);
free_dvector(eigvalues,1,J);
free_dvector(Besteigvalues,1,J);
free_dvector(BesteigvaluesNp,1,J);
free_dvector(Besttbarankin,1,J);
free_dvector(BesttbarankinNp,1,J);
}

