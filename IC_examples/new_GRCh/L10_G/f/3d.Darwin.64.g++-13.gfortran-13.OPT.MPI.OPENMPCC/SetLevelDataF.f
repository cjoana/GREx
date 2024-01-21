      subroutine GETLAPLACIANPSIF(
     &           l_of_psi
     &           ,il_of_psilo0,il_of_psilo1,il_of_psilo2
     &           ,il_of_psihi0,il_of_psihi1,il_of_psihi2
     &           ,psi
     &           ,ipsilo0,ipsilo1,ipsilo2
     &           ,ipsihi0,ipsihi1,ipsihi2
     &           ,dx
     &           ,iboxlo0,iboxlo1,iboxlo2
     &           ,iboxhi0,iboxhi1,iboxhi2
     &           )
      implicit none
      integer*8 ch_flops
      COMMON/ch_timer/ ch_flops
      integer CHF_ID(0:5,0:5)
      data CHF_ID/ 1,0,0,0,0,0 ,0,1,0,0,0,0 ,0,0,1,0,0,0 ,0,0,0,1,0,0 ,0
     &,0,0,0,1,0 ,0,0,0,0,0,1 /
      integer il_of_psilo0,il_of_psilo1,il_of_psilo2
      integer il_of_psihi0,il_of_psihi1,il_of_psihi2
      REAL*8 l_of_psi(
     &           il_of_psilo0:il_of_psihi0,
     &           il_of_psilo1:il_of_psihi1,
     &           il_of_psilo2:il_of_psihi2)
      integer ipsilo0,ipsilo1,ipsilo2
      integer ipsihi0,ipsihi1,ipsihi2
      REAL*8 psi(
     &           ipsilo0:ipsihi0,
     &           ipsilo1:ipsihi1,
     &           ipsilo2:ipsihi2)
      REAL*8 dx
      integer iboxlo0,iboxlo1,iboxlo2
      integer iboxhi0,iboxhi1,iboxhi2
      integer d0, i0,i1,i2, ii0,ii1,ii2
      REAL*8 dpsidxdx
      do i2 = iboxlo2,iboxhi2
      do i1 = iboxlo1,iboxhi1
      do i0 = iboxlo0,iboxhi0
         l_of_psi(i0,i1,i2) = 0.0
         do d0 = 0,3 -1
             ii0 = CHF_ID(d0,0)
             ii1 = CHF_ID(d0,1)
             ii2 = CHF_ID(d0,2)
           dpsidxdx = 1.0/dx/dx * (
     &     +1.0 * psi(i0  -ii0,i1  -ii1,i2  -ii2  )
     &     -2.0 * psi(  i0,  i1,  i2        )
     &     +1.0 * psi(i0  +ii0,i1  +ii1,i2  +ii2  )
     &     )
           l_of_psi(i0,i1,i2) = l_of_psi(i0,i1,i2) + dpsidxdx
         enddo
      enddo
      enddo
      enddo
      return
      end
      subroutine GETRHOGRADPHIF(
     &           rho_grad_phi
     &           ,irho_grad_philo0,irho_grad_philo1,irho_grad_philo2
     &           ,irho_grad_phihi0,irho_grad_phihi1,irho_grad_phihi2
     &           ,phi
     &           ,iphilo0,iphilo1,iphilo2
     &           ,iphihi0,iphihi1,iphihi2
     &           ,dx
     &           ,iboxlo0,iboxlo1,iboxlo2
     &           ,iboxhi0,iboxhi1,iboxhi2
     &           )
      implicit none
      integer*8 ch_flops
      COMMON/ch_timer/ ch_flops
      integer CHF_ID(0:5,0:5)
      data CHF_ID/ 1,0,0,0,0,0 ,0,1,0,0,0,0 ,0,0,1,0,0,0 ,0,0,0,1,0,0 ,0
     &,0,0,0,1,0 ,0,0,0,0,0,1 /
      integer irho_grad_philo0,irho_grad_philo1,irho_grad_philo2
      integer irho_grad_phihi0,irho_grad_phihi1,irho_grad_phihi2
      REAL*8 rho_grad_phi(
     &           irho_grad_philo0:irho_grad_phihi0,
     &           irho_grad_philo1:irho_grad_phihi1,
     &           irho_grad_philo2:irho_grad_phihi2)
      integer iphilo0,iphilo1,iphilo2
      integer iphihi0,iphihi1,iphihi2
      REAL*8 phi(
     &           iphilo0:iphihi0,
     &           iphilo1:iphihi1,
     &           iphilo2:iphihi2)
      REAL*8 dx
      integer iboxlo0,iboxlo1,iboxlo2
      integer iboxhi0,iboxhi1,iboxhi2
      integer d0, i0,i1,i2, ii0,ii1,ii2
      REAL*8 dphidx
      do i2 = iboxlo2,iboxhi2
      do i1 = iboxlo1,iboxhi1
      do i0 = iboxlo0,iboxhi0
          rho_grad_phi(i0,i1,i2) = 0.0
          do d0 = 0,3 -1
              ii0 = CHF_ID(d0,0)
              ii1 = CHF_ID(d0,1)
              ii2 = CHF_ID(d0,2)
             dphidx = 0.5/dx * (
     &     + phi(i0  +ii0,i1  +ii1,i2  +ii2)
     &     - phi(i0  -ii0,i1  -ii1,i2  -ii2)
     &       )
            rho_grad_phi(i0,i1,i2) = rho_grad_phi(i0,i1,i2) + 0.5*dphidx
     &*dphidx
          enddo
      enddo
      enddo
      enddo
      return
      end
