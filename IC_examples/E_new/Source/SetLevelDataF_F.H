#ifndef _SETLEVELDATAF_F_H_
#define _SETLEVELDATAF_F_H_

#include "CH_Timer.H"
#include "FORT_PROTO.H"
#include "REAL.H"

extern "C"
{

#ifndef GUARDGETLAPLACIANPSIF
#define GUARDGETLAPLACIANPSIF
    // Prototype for Fortran procedure GETLAPLACIANPSIF ...
    //
    void FORTRAN_NAME(GETLAPLACIANPSIF, getlaplacianpsif)(CHFp_FRA1(l_of_psi),
                                                          CHFp_CONST_FRA1(psi),
                                                          CHFp_CONST_REAL(dx),
                                                          CHFp_BOX(box));

#define FORT_GETLAPLACIANPSIF                                                  \
    FORTRAN_NAME(inlineGETLAPLACIANPSIF, inlineGETLAPLACIANPSIF)
#define FORTNT_GETLAPLACIANPSIF FORTRAN_NAME(GETLAPLACIANPSIF, getlaplacianpsif)

    inline void FORTRAN_NAME(inlineGETLAPLACIANPSIF,
                             inlineGETLAPLACIANPSIF)(CHFp_FRA1(l_of_psi),
                                                     CHFp_CONST_FRA1(psi),
                                                     CHFp_CONST_REAL(dx),
                                                     CHFp_BOX(box))
    {
        CH_TIMELEAF("FORT_GETLAPLACIANPSIF");
        FORTRAN_NAME(GETLAPLACIANPSIF, getlaplacianpsif)
        (CHFt_FRA1(l_of_psi), CHFt_CONST_FRA1(psi), CHFt_CONST_REAL(dx),
         CHFt_BOX(box));
    }
#endif // GUARDGETLAPLACIANPSIF

#ifndef GUARDGETRHOGRADPHIF
#define GUARDGETRHOGRADPHIF
    // Prototype for Fortran procedure GETRHOGRADPHIF ...
    //
    void FORTRAN_NAME(GETRHOGRADPHIF, getrhogradphif)(CHFp_FRA1(rho_grad_phi),
                                                      CHFp_CONST_FRA1(phi),
                                                      CHFp_CONST_REAL(dx),
                                                      CHFp_BOX(box));

#define FORT_GETRHOGRADPHIF                                                    \
    FORTRAN_NAME(inlineGETRHOGRADPHIF, inlineGETRHOGRADPHIF)
#define FORTNT_GETRHOGRADPHIF FORTRAN_NAME(GETRHOGRADPHIF, getrhogradphif)

    inline void FORTRAN_NAME(inlineGETRHOGRADPHIF,
                             inlineGETRHOGRADPHIF)(CHFp_FRA1(rho_grad_phi),
                                                   CHFp_CONST_FRA1(phi),
                                                   CHFp_CONST_REAL(dx),
                                                   CHFp_BOX(box))
    {
        CH_TIMELEAF("FORT_GETRHOGRADPHIF");
        FORTRAN_NAME(GETRHOGRADPHIF, getrhogradphif)
        (CHFt_FRA1(rho_grad_phi), CHFt_CONST_FRA1(phi), CHFt_CONST_REAL(dx),
         CHFt_BOX(box));
    }
#endif // GUARDGETRHOGRADPHIF
}

#endif
