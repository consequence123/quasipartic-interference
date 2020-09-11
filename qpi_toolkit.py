from numpy import*
import matplotlib.pyplot as plt
import pdb
from matplotlib import cm
from raman_toolkit import general_kron,sigma0,sigma1,sigma2,sigma3
from MFTtoolkit import*

sigma_up = diag([1.,0])
sigma_dn = diag([0.,1])

class qpi():
    def get_Gk0(self,w,eta = 1e-3):
        kmesh = self.get_kmesh()
        Hk = self.get_Hk(kmesh)
        Gk0inv = (w+1j*eta)*eye(Hk.shape[-1]) - Hk
        Gk0 = linalg.inv(Gk0inv)
        return Gk0
        #danger
    def get_T(self,Vs,Vm,g):
        #Vs = Vs + 1e-10*eye(Vs.shape[0])
        G0 = g.sum(axis = (0,1))/prod(g.shape[:2])
        Hk = self.get_Hk(kx, ky, delta, Vs, Vm)
        Hk = self.get_Hk(kx, ky, delta, Vs, Vm)
        Ek, Uk = linalg.eigh(Hk)
        Ek, Uk = linalg.eigh(Hk)
        V = kron(sigma3,Vs) + kron(sigma0,Vm)
#        pdb.set_trace()
        T = linalg.inv(eye(V.shape[0]) - V.dot(G0))
        T = T.dot(V)
        # print T
        return T
    def get_deltaGr(self,gk0,T):
        gr = fft.ifftn(gk0,axes = (0,1))
        gr_ = roll(roll(gr,-1,axis = 0),-1,axis = 1)[::-1,::-1]
        delta_Gr = matmul(matmul(gr_,T),gr)
#        print delta_Gr[10,100]
        return delta_Gr

    def get_nr_electron(self,Vs,Vm,w):
        Gk0 = self.get_Gk0(w)
        T = self.get_T(Vs,Vm,Gk0)
        delta_Gr = self.get_deltaGr(Gk0,T)
        part = arange(Gk0.shape[-1]/2)
        # part = arange(1)
        nr = delta_Gr[...,part,part].imag.sum(axis = -1)
        return nr
    def get_nr_hole(self,Vs,Vm,w):
        Gk0 = self.get_Gk0(-w)
        T = self.get_T(Vs,Vm,Gk0)
        delta_Gr = self.get_deltaGr(Gk0,T)
        part = arange(Gk0.shape[-1]/2,Gk0.shape[-1])
        # part = arange(Gk0.shape[-1]/2)
        nr = delta_Gr[...,part,part].imag.sum(axis = -1)
        return nr
    def get_nq_electron(self,Vs,Vm,w):
        Nx = self.Nx; Ny = self.Ny
        nr= self.get_nr_electron(Vs,Vm,w)
        nq = fft.fftn(nr,axes=(0,1))
        return nq
    def get_nq_hole(self,Vs,Vm,w):
        Nx = self.Nx; Ny = self.Ny
        nr = self.get_nr_hole(Vs,Vm,w)
        nq = fft.fftn(nr,axes = (0,1))
        return nq
    def get_nq(self,Vs,Vm,w,hole = 1):
        Nx = self.Nx; Ny = self.Ny
        nq_e = self.get_nq_electron(Vs,Vm,w)
        if hole == 1:
            nq_h = self.get_nq_hole(Vs,Vm,w)
        else:
            nq_h = 0
        nq = nq_e + nq_h
#        pdb.set_trace()
        nq[0,0] = 0
        nq = roll(roll(nq,Nx/2,axis=0),Ny/2,axis=1)
        savetxt('nq_w_'+str(w)+'.dat',nq)
        qmesh = self.get_kmesh()
        plt.pcolormesh(qmesh[0],qmesh[1],abs(nq),vmin=0.,vmax=abs(nq).max()*0.8)
        plt.axis('equal')
        plt.axis('off')
        plt.colorbar()
        plt.show()
    def get_zq(self,Vs,Vm,w):
        Nx = self.Nx; Ny = self.Ny 
        Gk0 = self.get_Gk0(w)
        G0 = Gk0.sum(axis = (0,1))/Nx/Ny
        s = G0.shape[-1]; block = arange(s/2)
        nr0 = G0[block, block].imag.sum()
        nr_pw = self.get_nr_electron(Vs,Vm,w) + nr0
        nr_nw = self.get_nr_electron(Vs,Vm,-w) + nr0
        zr = nr_pw/nr_nw
        pdb.set_trace()
        # zr = arctan(zr**0.5)-pi/4
        zq = fft.fftn(zr,axes = (0,1))
        zq[0,0] = 0.
        qmesh = self.get_kmesh()
        zq = roll(roll(zq,Nx/2,axis=0),Ny/2,axis=1)
        plt.pcolormesh(qmesh[0],qmesh[1],abs(zq),vmin=0.,vmax=abs(zq).max())
        plt.colorbar()
        plt.axis('equal')
        plt.axis('off')
        plt.show()
    def get_dos_e(self,Hk,Vs,Vm,w,eta = 1e-3):
        Gk0inv = (w+1j*eta)*eye(Hk.shape[-1]) - Hk
        Gk0 = linalg.inv(Gk0inv)
        Gr = fft.ifftn(Gk0,axes = (0,1))
        Gr00 = Gr[0,0]
        Gr0 = Gr[0,0]  # [0,0] for onsite and [0,1] for nearest neighbour
#        pdb.set_trace()
        part = arange(Gk0.shape[-1]/2)
        # part = [0]
        nr0 = Gr00[part,part].imag.sum()
        # pdb.set_trace()

        # T = self.get_T(Vs,Vm,Gk0)
        # delta_Gr = matmul(matmul(Gr0,T),Gr0)
        # nr = delta_Gr[part,part].imag.sum()
        # nr0 = nr0 + nr #  this line can be annotated or not for sc without/with impurity
        nr = -nr0
        print w
        return nr
    def get_dos_h(self,Hk,Vs,Vm,w,eta = 1e-3):
        Gk0inv = (-w+1j*eta)*eye(Hk.shape[-1]) - Hk
        Gk0 = linalg.inv(Gk0inv)
        Gr = fft.ifftn(Gk0,axes = (0,1))
        Gr00 = Gr[0,0]
        Gr0 = Gr[0,0]  # [0,0] for onsite and [0,1] for nearest neighbour
#        pdb.set_trace()
        part = arange(Gk0.shape[-1]/2,Gk0.shape[-1])
        nr0 = Gr00[part,part].imag.sum()

        # T = self.get_T(Vs,Vm,Gk0)
        # delta_Gr = matmul(matmul(Gr0,T),Gr0)
        # nr = delta_Gr[part,part].imag.sum()
        # nr0 = nr0 + nr #  this line can be annotated or not for sc without/with impurity
        nr = -nr0
#        pdb.set_trace()
        return nr
    def get_dos(self,Vs,Vm,wm):
        kmesh = self.get_kmesh()
        omega = linspace(-wm,wm,81)
        Hk = self.get_Hk(kmesh)
        # pdb.set_trace()
        # dos = [self.get_dos_e(Hk,Vs,Vm,w)+self.get_dos_h(Hk,Vs,Vm,w) for w in omega]
        dos = [self.get_dos_e(Hk,Vs,Vm,w) for w in omega]
        # pdb.set_trace()
        savetxt('dos.txt', dos)
        plt.plot(omega,dos)
        plt.savefig('dos.png')
        plt.show()
        return dos
    def get_nq_normal(self,Vs,Vm,w,eta = 1e-3):
        Nx = self.Nx; Ny = self.Ny
        V = Vs + Vm
        kmesh = self.get_kmesh()
        Hk0 = self.get_Hk0(kmesh)

        # s = Hk0.shape[-1]
        # Gk0inv = (w + 1j*eta)*eye(s) - Hk0
        Gk0inv = w + 1j*eta - Hk0
        Gk0 = linalg.inv(Gk0inv)
        G0 = sum(Gk0,axis=(0,1))/prod(Gk0.shape[:2])

        # Tinv = linalg.inv(V) - G0
        # T = linalg.inv(Tinv)
        Tinv = 1/V - G0
        T = 1/Tinv

#        pdb.set_trace()
        Gr0 = fft.ifftn(Gk0,axes = (0,1))
        Gr0_ = roll(roll(Gr0,-1,axis = 0),-1,axis = 1)[::-1,::-1]
        delta_Gr = Gr0_*T*Gr0
        # delta_Gr = matmul(matmul(Gr0,T),Gr0)
        # nr = delta_Gr[...,arange(s),arange(s)].imag.sum(axis = -1)
        nr = delta_Gr[...].imag
        nr = -nr/pi
        nq = fft.fftn(nr,axes = (0,1)); nq[0,0] = 0.
        nq = roll(roll(nq,Nx/2,axis=0),Ny/2,axis=1)
        qmesh = self.get_kmesh()
#        pdb.set_trace()
        plt.subplot(1,1,1)
        plt.pcolormesh(qmesh[0],qmesh[1],abs(nq),vmin=0,vmax=abs(nq).max()*0.8)
        plt.colorbar()
        plt.show()
    def get_Tinv(self,Vs,Vm,g):
        V = kron(sigma3,Vs) + kron(sigma0,Vm)
        G0 = sum(g,axis=(0,1))/prod(g.shape[:2])
        pdb.set_trace()
        Tinv = linalg.inv(V) - G0
#        print T
        return Tinv
    def get_resonance_dwave_Vs(self, eta = 1e-3):
        kmesh = self.get_kmesh()
        Hk0 = self.get_Hk(kmesh)
        s = Hk0.shape[-1]
        x = 0.1
        omega = linspace(-x,x,51)
        resonance = zeros([omega.shape[0],5],dtype = 'float64')
        resonance[:,0] = omega
        for i,w in enumerate(omega):
            Gk0inv = (w + 1j*eta)*eye(s) - Hk0
            Gk0 = linalg.inv(Gk0inv)
            G0 = sum(Gk0,axis=(0,1))/prod(Gk0.shape[:2])
#            pdb.set_trace()
            resonance[i,1] = +G0[0,0].real
            resonance[i,2] = abs(G0[0,0].imag)
            resonance[i,3] = -G0[1,1].real
            resonance[i,4] = abs(G0[1,1].imag)
#            if G0[0,0].real >= 0.:
#                resonance[i,1] = +G0[0,0].real
#                resonance[i,2] = abs(G0[0,0].imag)
#            else:
#                resonance[i,1] = -G0[1,1].real
#                resonance[i,2] = abs(G0[1,1].imag)
        pdb.set_trace()
        ydown = -40; yup = 40
        devide = 1.
        savetxt('resonance.txt',resonance)
        plt.plot(resonance[:,0],resonance[:,1],color = 'green',linewidth = 1.5)
#        plt.plot(resonance[:,0],resonance[:,3],color = 'green',linewidth = 1.5)
        plt.plot([-x,x],[0,0],linewidth = 1.,linestyle = '--',color = 'b')
        plt.plot([-0.046,-0.046],[ydown,yup],linewidth = 1.,linestyle = '--',color = 'b')
        plt.plot([0.046,0.046],[ydown,yup],linewidth = 1.,linestyle = '--',color = 'b')
        plt.errorbar(resonance[:,0],resonance[:,1],yerr = resonance[:,2]/devide,fmt='o',ecolor='r',color='b',elinewidth=1.5,capsize=4)
#        plt.errorbar(resonance[:,0],resonance[:,3],yerr = resonance[:,4]/devide,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
        plt.xlim(-x-1e-3,x+1e-3)
        plt.ylim(ydown,yup)
        plt.xlabel('$\omega/\\rm{eV}$',size = 38,labelpad = -2)
        plt.ylabel('$V^{-1}/(\\rm{eV})^{-1}$',size = 28,labelpad = -6)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.minorticks_on()
        plt.savefig('resonance_dwave.png')
        plt.show()
    def get_resonance_swave_Vm(self, eta = 1e-3):
        # only for particle-hole symmetrical case, which is commonly satisfied for s-wave sc
        kmesh = self.get_kmesh()        
        Hk0 = self.get_Hk(kmesh)
        s = Hk0.shape[-1]
        x = 0.15
        omega = linspace(-x,x,101)
        resonance = zeros([omega.shape[0],5],dtype = 'float64')
        resonance[:,0] = omega
        for i,w in enumerate(omega):
            Gk0inv = (w + 1j*eta)*eye(s) - Hk0
            Gk0 = linalg.inv(Gk0inv)
            G0 = Gk0.sum(axis=(0,1))/prod(Gk0.shape[:2])
#            pdb.set_trace()
            a = G0[0,0] + G0[0,1]; b = G0[0,0] - G0[0,1]
            resonance[i,1] = a.real
            resonance[i,2] = abs(a.imag)
            resonance[i,3] = b.real
            resonance[i,4] = abs(b.imag)

        plt.plot([-x,x],[0,0],linestyle='--')
        plt.plot(resonance[:,0],resonance[:,1],color = 'green',linewidth = 1.5)
        plt.plot(resonance[:,0],resonance[:,3],color = 'green',linewidth = 1.5)
        plt.errorbar(resonance[:,0],resonance[:,1],yerr = resonance[:,2]/5.,fmt='o',ecolor='r',color='b',elinewidth=1.5,capsize=4)
        plt.errorbar(resonance[:,0],resonance[:,3],yerr = resonance[:,4]/5.,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
        plt.xlim(-x-1e-3,x+1e-3)
        plt.xlabel('$\omega$',size = 30,labelpad = -5)
        plt.ylabel('$V^{-1}$',size = 25,labelpad = -5)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.minorticks_on()
        plt.savefig('resonance_swave_Vm.png')
        plt.show()

    def get_resonance_swave_Vs(self, eta = 1e-3):
        # only for particle-hole symmetrical case, which is commonly satisfied for s-wave sc
        kmesh = self.get_kmesh()        
        Hk0 = self.get_Hk(kmesh)
        s = Hk0.shape[-1]
        x = 0.1
        omega = linspace(-x,x,51)
        resonance = zeros([omega.shape[0],5],dtype = 'float64')
        resonance[:,0] = omega
        for i,w in enumerate(omega):
            Gk0inv = (w + 1j*eta)*eye(s) - Hk0
            Gk0 = linalg.inv(Gk0inv)
            G0 = Gk0.sum(axis=(0,1))/prod(Gk0.shape[:2])
#            pdb.set_trace()
            a = sqrt(G0[0,0]**2 - G0[0,1]**2)
            resonance[i,1] = +a.real
            resonance[i,2] = abs(a.imag)
            resonance[i,3] = -a.real
            resonance[i,4] = abs(a.imag)

        plt.plot(resonance[:,0],resonance[:,1],color = 'green',linewidth = 1.5)
        plt.plot(resonance[:,0],resonance[:,3],color = 'green',linewidth = 1.5)
        plt.errorbar(resonance[:,0],resonance[:,1],yerr = resonance[:,2]/5.,fmt='o',ecolor='r',color='b',elinewidth=1.5,capsize=4)
        plt.errorbar(resonance[:,0],resonance[:,3],yerr = resonance[:,4]/5.,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
        plt.xlim(-x-1e-3,x+1e-3)
        plt.xlabel('$\omega$',size = 30,labelpad = -5)
        plt.ylabel('$V^{-1}$',size = 25,labelpad = -5)
        plt.xticks(size = 15)
        plt.yticks(size = 15)
        plt.minorticks_on()
        plt.savefig('resonance_swave_Vs.png')
        plt.show()

    def get_A(self,k1,k2,Vs,Vm,w,eta = 1e-3):
        Gk0 = self.get_Gk0(w)
        T = self.get_T(Vs,Vm,Gk0)
        s = T.shape[0]

        Hk1 = self.get_Hk(k1)
        Gk1inv = (w+1j*eta)*eye(s) - Hk1
        Gk1 = linalg.inv(Gk1inv)

        Hk2 = self.get_Hk(k2)
        Gk2inv = (w+1j*eta)*eye(s) - Hk2
        Gk2 = linalg.inv(Gk2inv)

        Aq = matmul(matmul(Gk1, T), Gk2)
        A_q = matmul(matmul(Gk2, T), Gk1)
        Aq = Aq[arange(s/2),arange(s/2)].sum(axis = -1)
        A_q = A_q[arange(s/2),arange(s/2)].sum(axis = -1)
        A = -(Aq - A_q.conj())/pi/2j
#        pdb.set_trace()
        return A
    def get_A_main(self,Vs,Vm,w,filename):
#        a = random.random([10,2]) - 1.
#        savetxt('test.dat',a)
#        filename = 'test.dat'
        fs = loadtxt(filename)
        pdb.set_trace()
        A = zeros([fs.shape[0],fs.shape[0]],dtype = 'complex128')
        for i,k1 in enumerate(fs):
            for j,k2 in enumerate(fs):
#                pdb.set_trace()
                A[i,j] = self.get_A(k1,k2,Vs,Vm,w)
        A.shape = [-1]
        savetxt('A.dat',A)


class dwave():
    def __init__(self, J):
        self.Nx = 200; self.Ny = 200; self.J = J
    def get_kmesh(self):
        Nx = self.Nx; Ny = self.Ny
        kx = arange(-pi,pi,2*pi/Nx)
        ky = arange(-pi,pi,2*pi/Ny)
        kmesh=meshgrid(kx,ky,indexing='ij')
        return kmesh

    def get_RBZ(self):
        Nx = self.Nx; Ny = self.Ny
        kalpha = array([1,1])*pi/Nx
        kbetha = array([-1,1])*pi/Nx
        X=arange(0,Nx)
        Y=arange(0,Ny)
        kmesh=X.reshape(-1,1,1)*kalpha.reshape(1,1,-1)+Y.reshape(1,-1,1)*kbetha.reshape(1,1,-1)
        kx = kmesh[...,0]; ky = kmesh[...,1] - pi
        return kx, ky

    def get_Hk0(self, kx, ky):
        # t = array([0, -595.1,163.6,-51.9,-111.7,51.])/1000.
        t = array([0, -595.1, 0, 0, 0, 0])/1000.
        t = t/(-t[1])
        Ek=t[0]+t[1]*(cos(kx)+cos(ky))/2.+t[2]*cos(kx)*cos(ky)+t[3]*(cos(2*kx)+cos(2*ky))/2.\
+t[4]*(cos(2*kx)*cos(ky)+cos(kx)*cos(2*ky))/2.+t[5]*cos(2*kx)*cos(2*ky)
        return Ek

    def get_MFThk0(self, kx, ky, doping):
        Ek = self.get_Hk0(kx, ky)
        kx_Q = kx + pi
        ky_Q = ky + pi
        Ek_Q = self.get_Hk0(kx_Q, ky_Q)
        hk0 = general_kron(Ek, sigma_up) + general_kron(Ek_Q, sigma_dn)
        # pdb.set_trace()
        hk0 = hk0*doping
        hk0_nambu = kron(sigma3, hk0)
        return hk0_nambu

    def get_MFTmu(self, mu):
        Mu_nambu = diag([mu, mu, -mu, -mu])
        return Mu_nambu

    def get_MFTmoment(self, mA):
        J = self.J
        m = -mA*J*2
        M_nambu = zeros([4,4], dtype = 'complex128')
        M_nambu[0,1] = m; M_nambu[2,3] = m
        M_nambu = M_nambu + M_nambu.T.conj()
        return M_nambu

    def get_MFTchi(self, kx, ky, chi):
        J = self.J
        chi_k = -0.75*chi*(cos(kx) + cos(ky))*J
        chi_kQ = -chi_k
        chi_normal = general_kron(chi_k, sigma_up) + general_kron(chi_kQ, sigma_dn)
        chi_nambu = kron(sigma3, chi_normal)
        return chi_nambu
    
    def get_MFTdelta(self, kx, ky, delta):
        J = self.J
        sigma_c = (sigma1 + 1j*sigma2)/2
        sigma_a = (sigma1 - 1j*sigma2)/2
        delta_k = -0.75*delta*(cos(kx) - cos(ky))*J
        delta_kQ = -delta_k
        delta_normal = general_kron(delta_k, sigma_up) + general_kron(delta_kQ, sigma_dn)
        delta_nambu = kron(sigma_c, delta_normal) + kron(sigma_a, delta_normal.conj())
        # pdb.set_trace()
        return delta_nambu

    def get_MFTHk(self,kx,ky,Hk0_nambu,mu,mA,chi,delta):
        Mu_nambu = self.get_MFTmu(mu)
        M_nambu = self.get_MFTmoment(mA)
        Chi_nambu = self.get_MFTchi(kx,ky,chi)
        Delta_nambu = self.get_MFTdelta(kx,ky,delta)
        Hk = Hk0_nambu - Mu_nambu + M_nambu + Chi_nambu + Delta_nambu
        # Hk = Hk0_nambu - Mu_nambu + Chi_nambu + Delta_nambu
        # Hk = Hk0_nambu - Mu_nambu + M_nambu + Delta_nambu
        # Hk = Hk0_nambu - Mu_nambu + M_nambu
        return Hk

    def get_FermiDistri(self, Ek, beta = 100):
        f = 1-1./(exp(-Ek*beta)+1.)
        return f

    def get_UcU(self, Uk_conj, Uk, i, j):
        Ukc_i = Uk_conj[...,i,:]
        Uk_j = Uk[...,j,:]
        return Ukc_i*Uk_j

    def get_obs(self, Ok, fk, phase_k):
        Nx = self.Nx; Ny = self.Ny
        obs_k = (Ok*fk).sum(axis = -1)
        # pdb.set_trace()
        obs = einsum('ij,ij->', obs_k, phase_k)/Nx/Ny
        return obs

    def get_Nup(self, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        nk = self.get_UcU(Uk_conj, Uk, 0, 0)
        nk_Q = self.get_UcU(Uk_conj, Uk, 1, 1)
        N_up1 = einsum('ijk,ijk->', nk, fk)
        N_up2 = einsum('ijk,ijk->', nk_Q, fk)
        N_up = (N_up1 + N_up2)/Nx/Ny/2
        # pdb.set_trace()
        # nk = self.get_UcU(Uk_conj, Uk, 0, 1)
        # N_up = einsum('ijk,ijk->', nk, fk)
        # N_up = N_up/Nx/Ny
        return N_up

    def get_Ndn(self, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        nk = self.get_UcU(Uk_conj, Uk, 2, 2)
        nk_Q = self.get_UcU(Uk_conj, Uk, 3, 3)
        N_dn1 = einsum('ijk,ijk->', nk, fk)
        N_dn2 = einsum('ijk,ijk->', nk_Q, fk)
        N_dn = N_dn1 + N_dn2
        N_dn = 1 - N_dn/Nx/Ny/2
        # pdb.set_trace()
        # nk = self.get_UcU(Uk_conj, Uk, 3, 2)
        # N_dn = einsum('ijk,ijk->', nk, fk)
        # N_dn = 1 - N_dn/Nx/Ny
        return N_dn

    def get_mu(self, n_up, n_dn, doping, mu_old, delta_mu):
        delta_n = n_up + n_dn - (1 - doping)
        # delta_mu = (delta_n**2)*delta_mu
        mu = mu_old - delta_n*delta_mu
        return mu

    def get_moment(self, n_up, n_dn):
        return (n_up - n_dn)/2.

    def get_chi(self, kx, ky, Uk_conj, Uk, fk):
        chi1_k = self.get_UcU(Uk_conj, Uk, 0, 0)
        chi1_k = chi1_k + self.get_UcU(Uk_conj, Uk, 1, 1)
        phase_k = (cos(kx) + cos(ky))/2
        chi1 = self.get_obs(chi1_k, fk, phase_k)
        chi2_k = self.get_UcU(Uk_conj, Uk, 2, 2)
        chi2_k = chi2_k + self.get_UcU(Uk_conj, Uk, 3, 3)
        chi2 = -self.get_obs(chi2_k, fk, phase_k.conj())
        chi = (chi1 + chi2)/2
        return chi

    def get_delta(self, kx, ky, Uk_conj, Uk, fk):
        delta_k = self.get_UcU(Uk_conj, Uk, 2, 0)
        phase_k = (cos(kx) - cos(ky))
        delta1 = self.get_obs(delta_k, fk, phase_k)
        delta_k = self.get_UcU(Uk_conj, Uk, 3, 1)
        phase_k = - phase_k
        delta2 = self.get_obs(delta_k, fk, phase_k)
        delta = delta1 + delta2
        # print delta1, delta2
        delta = delta/2.
        return delta

    def MFT_main(self, doping):
        kmesh = self.get_RBZ()
        kx = kmesh[0]; ky = kmesh[1]
        delta_mu = 0.1
        random.seed(2)
        mu = random.random()
        mA = random.random()
        chi = random.random()
        delta = random.random() + 1j*random.random()
        Hk0_nambu = self.get_MFThk0(kx, ky, doping)
        for i in xrange(100):
            Hk = self.get_MFTHk(kx,ky,Hk0_nambu,mu,mA,chi,delta)
            Ek, Uk = linalg.eigh(Hk)
            # pdb.set_trace()
            Uk_conj = Uk.conj(); fk = self.get_FermiDistri(Ek)
            nA_up = self.get_Nup(Uk_conj, Uk, fk)
            nA_dn = self.get_Ndn(Uk_conj, Uk, fk)
            n = nA_up +  nA_dn
            mu = self.get_mu(nA_up, nA_dn, doping, mu, delta_mu)
            mA = self.get_moment(nA_up, nA_dn)
            chi = self.get_chi(kx, ky, Uk_conj, Uk, fk)
            delta = self.get_delta(kx, ky, Uk_conj, Uk, fk)
            print nA_up, nA_dn
            print chi, delta
            # pdb.set_trace()


    
    def get_Hk(self, k):
        t=array([140.5,-595.1,163.6,-51.9,-111.7,51.])/1000.
        delta0=0.042
        kx=k[0]
        ky=k[1]
        Ek=t[0]+t[1]*(cos(kx)+cos(ky))/2.+t[2]*cos(kx)*cos(ky)+t[3]*(cos(2*kx)+cos(2*ky))/2.\
+t[4]*(cos(2*kx)*cos(ky)+cos(kx)*cos(2*ky))/2.+t[5]*cos(2*kx)*cos(2*ky)
#        
        delta=delta0*(cos(kx)-cos(ky))/2.
        Hk = general_kron(Ek,sigma3) - general_kron(delta,sigma1)

#        delta = delta0
#        Hk = general_kron(Ek,sigma3) + delta0*sigma1
        plt.pcolormesh(kx,ky,Ek[...])
        plt.axis('equal')
        plt.colorbar()
        plt.contour(kx,ky,Ek[...],levels=[-0.,-0.+1e-4])
        plt.show()

        #k = self.get_kmesh()
#        t=array([-155,100,36,10,1.5])/1000.
        #t=array([130.5,-590.8,96.2,-130.6,-50.7,93.9])/1000.


        return Hk

    # def test_G(self,wm,eta = 1e-3):
    #     kmesh = self.get_kmesh()
    #     omega = linspace(-wm,wm,51)
    #     Hk = self.get_Hk(kmesh)
    #     s = Hk.shape[-1]
    #     Gw = zeros([omega.shape[0],s,s],dtype='complex128')
    #     for i,w in enumerate(omega):
    #         Gk0inv = (w+eta*1j)*eye(s) - Hk
    #         Gk0 = linalg.inv(Gk0inv)
    #         Gw[i] = sum(Gk0,axis=(0,1))/prod(Gk0.shape[:2])
    #         print w
    #     pdb.set_trace()

class monolayer():
    def __init__(self, J):
        self.Nx = 200; self.Ny = 200; self.J = J
        self.hopping = [-1, 0.32, -0.16]
    def get_kmesh(self):
        Nx = self.Nx; Ny = self.Ny
        kx = arange(-pi,pi,2*pi/Nx)
        ky = arange(-pi,pi,2*pi/Ny)
        kmesh = meshgrid(kx, ky)
        return kmesh
    def get_Hk0(self, kx, ky):
        t = self.hopping
        Ek1 = t[1]*(cos(kx) + cos(ky))*2
        Ek2 = t[0]*cos(kx/2)*cos(ky/2)*4 + t[2]*cos(kx)*cos(ky)*4
        Hk0 = general_kron(Ek1, sigma0) + general_kron(Ek2,sigma1)
        # pdb.set_trace()
        return Hk0
    def get_Hk0_nambu(self, kx, ky, doping):
        Hk0 = self.get_Hk0(kx, ky)
        Hk0 = Hk0*doping
        Hk0_nambu = kron(sigma3, Hk0)
        return Hk0_nambu

    def get_Hk_M(self, mA):
        J = self.J
        M = J*mA*2
        hk0_M = diag([-M, M, -M, M])
        return hk0_M
    def get_Hk_mu(self, mu):
        hk0_mu = diag([mu, mu, -mu, -mu])
        return hk0_mu
    def get_Hk_chi(self, kx, ky, chi):
        J = self.J
        phase = cos(kx/2)*cos(ky/2)
        # chi_r = chi.real*J*phase*(-3./2.)
        # chi_i = chi.imag*J*phase*(-3./2.)
        # hk0_chi = general_kron(chi_r, sigma1) - general_kron(chi_i, sigma2)
        # hk0_chi = kron(sigma_up, hk0_chi) - kron(sigma_dn, hk0_chi.transpose(0,1,3,2))
        chi_k = chi*J*phase*(-3./2.)
        hk0_chi = general_kron(chi_k, sigma1)
        hk0_chi = kron(sigma3, hk0_chi)
        # pdb.set_trace()
        return hk0_chi

    def get_Hk_delta(self, kx, ky, delta):
        J = self.J
        phase = -sin(kx/2)*sin(ky/2)
        delta_k = delta*J*phase*(-3./2.)
        hk0_delta = general_kron(delta_k, sigma1)
        hk0_delta = kron(sigma1, hk0_delta)
        return hk0_delta

    def get_Hk(self, kx, ky, Hk0, mu, m, chi, delta):
        Hk0_mu = self.get_Hk_mu(mu)
        Hk0_M = self.get_Hk_M(m)
        Hk0_chi = self.get_Hk_chi(kx, ky, chi)
        Hk0_delta = self.get_Hk_delta(kx, ky, delta)
        Hk = Hk0 - Hk0_mu + Hk0_M + Hk0_chi + Hk0_delta
        # Hk = Hk0 - Hk0_mu + Hk0_chi + Hk0_delta
        #Hk = Hk0 - Hk0_mu + Hk0_M + Hk0_chi
        # pdb.set_trace()
        return Hk

    def get_FermiDistri(self, Ek, beta):
        f = ones(Ek.shape)
        f[Ek>0] = 0.
        return f

    def get_UcU(self, Uk_conj, Uk, i, j):
        Ukc_i = Uk_conj[...,i,:]
        Uk_j = Uk[...,j,:]
        return Ukc_i*Uk_j

    def get_obs(self, Ok, fk, phase_k):
        Nx = self.Nx; Ny = self.Ny
        obs_k = (Ok*fk).sum(axis = -1)
        # pdb.set_trace()
        obs = einsum('ij,ij->', obs_k, phase_k)/Nx/Ny
        return obs

    def get_Nup(self, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        nk = self.get_UcU(Uk_conj, Uk, 0, 0)
        N_up = einsum('ijk,ijk->', nk, fk)/Nx/Ny
        return N_up

    def get_Ndn(self, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        nk = self.get_UcU(Uk_conj, Uk, 2, 2)
        N_dn = einsum('ijk,ijk->', nk, fk)/Nx/Ny
        N_dn = 1 - N_dn
        return N_dn

    def get_mu(self, n_up, n_dn, doping, mu_old, delta_mu):
        delta_n = n_up + n_dn - (1 - doping)
        mu = mu_old - delta_n*delta_mu
        return mu

    def get_moment(self, n_up, n_dn):
        return (n_up - n_dn)/2.

    def get_chi(self, kx, ky, Uk_conj, Uk, fk):
        phase = cos(kx/2)*cos(ky/2)
        # phase = exp(-1j*kx/2)*exp(-1j*ky/2)
        chi1_k = self.get_UcU(Uk_conj, Uk, 1, 0)
        chi_up = self.get_obs(chi1_k, fk, phase)
        chi2_k = self.get_UcU(Uk_conj, Uk, 2, 3)
        chi_dn = -self.get_obs(chi2_k, fk, phase)
        chi = chi_up + chi_dn
        # print chi_up, chi_dn
        return chi

    def get_delta(self, kx, ky, Uk_conj, Uk, fk):
        phase = -sin(kx/2)*sin(ky/2)
        # phase = exp(1j*kx/2)*exp(1j*ky/2)
        deltak = self.get_UcU(Uk_conj, Uk, 2, 1)
        delta1 = self.get_obs(deltak, fk, phase)
        deltak = self.get_UcU(Uk_conj, Uk, 3, 0)
        delta2 = self.get_obs(deltak, fk, phase.conj())
        delta = delta1 + delta2
        # print delta1, delta2 
        return delta

    def main(self, doping, beta):
        kmesh = self.get_kmesh()
        kx = kmesh[0]; ky = kmesh[1]
        delta_mu = 0.4
        random.seed(3)
        mu = 0.
        mA = random.random()
        chi = random.random()
        delta = random.random()
        Hk0 = self.get_Hk0_nambu(kx, ky, doping)
        for i in xrange(1000):
            Hk = self.get_Hk(kx,ky,Hk0, mu, mA, chi, delta)
            Ek, Uk = linalg.eigh(Hk)
            Uk_conj = Uk.conj(); fk = self.get_FermiDistri(Ek, beta)
            # pdb.set_trace()
            nA_up = self.get_Nup(Uk_conj, Uk, fk)
            nA_dn = self.get_Ndn(Uk_conj, Uk, fk)
            mu = self.get_mu(nA_up, nA_dn, doping, mu, delta_mu)
            mA = self.get_moment(nA_up, nA_dn)
            chi = self.get_chi(kx, ky, Uk_conj, Uk, fk)
            delta = self.get_delta(kx, ky, Uk_conj, Uk, fk)
            print nA_up, nA_dn
            print chi, delta

class trilayer():
    def __init__(self, J):
        self.Nx = 200; self.Ny = 200; self.J = J
        self.hopping = [-1, 0.32, -0.16]
        self.interhopping = -0.00
        self.deltaE = 0.04
    def get_kmesh(self):
        Nx = self.Nx; Ny = self.Ny
        kx = arange(-pi,pi,2*pi/Nx)
        ky = arange(-pi,pi,2*pi/Ny)
        kmesh=meshgrid(kx,ky)
        return kmesh
    def get_Hk0_intralayer(self, kx, ky,):
        t = self.hopping
        Ek1 = t[1]*(cos(kx) + cos(ky))*2
        Ek2 = t[0]*cos(kx/2)*cos(ky/2)*4 + t[2]*cos(kx)*cos(ky)*4
        h = diag([1,1,1.])
        deltaE = diag([0, self.deltaE, 0])
        H1 = general_kron(Ek1, h) - deltaE
        H2 = general_kron(Ek2, h)
        Hk0 = kron(sigma0, H1) + kron(sigma1, H2)
        return Hk0
    def get_Hk0_interlayer(self, kx, ky):
        t_inter = self.interhopping*sin(kx/2)**2*sin(ky/2)**2
        h = zeros([3,3])
        h[0,1] = 1; h[1,2] = 1
        h = h + h.T
        Hk0 = general_kron(t_inter, h)
        Hk0 = kron(sigma1, Hk0)
        return Hk0
    def get_Hk0_nambu(self, kx, ky, doping):
        Hk0 = self.get_Hk0_intralayer(kx, ky) + self.get_Hk0_interlayer(kx, ky)
        # pdb.set_trace()
        # DL = hstack((doping_layer, doping_layer))
        # SDL = sqrt(DL)
        # Hk0 = einsum('k,ijkl->ijkl', SDL, Hk0)
        # Hk0 = Hk0*SDL
        Hk0 = Hk0*doping
        Hk0 = kron(sigma3,Hk0)
        # pdb.set_trace()
        return Hk0
    
    def get_Hk_mu(self, mu):
        Hk0_mu = diag([mu, mu, mu])
        Hk0_mu = kron(sigma0, Hk0_mu)
        Hk0_mu = kron(sigma3, Hk0_mu)
        return Hk0_mu

    def get_Hk_moment(self, m):
        J = self.J
        M = J*m*2
        Hk0_M = diag(-M)
        Hk0_M = kron(sigma3, Hk0_M)
        Hk0_M = kron(sigma0, Hk0_M)
        return Hk0_M

    def get_Hk_chi(self, kx, ky, chi123):
        J = self.J
        phase = cos(kx/2)*cos(ky/2)
        chi = diag(chi123)
        Hk0_chi = general_kron(phase, -chi*J*3/2.)
        Hk0_chi = kron(sigma1, Hk0_chi)
        Hk0_chi = kron(sigma3, Hk0_chi)
        return Hk0_chi
    
    def get_Hk_delta(self, kx, ky, delta123):
        J = self.J
        phase = -sin(kx/2)*sin(ky/2)
        delta = diag(delta123)
        Hk0_delta = general_kron(phase, -delta*J*3/2.)
        Hk0_delta = kron(sigma1, Hk0_delta)
        Hk0_delta = kron(sigma1, Hk0_delta)
        return Hk0_delta

    def get_Hk(self, kx, ky, Hk0, mu, m, chi, delta):
        Hk0_mu = self.get_Hk_mu(mu)
        Hk0_M = self.get_Hk_moment(m)
        Hk0_chi = self.get_Hk_chi(kx, ky, chi)
        Hk0_delta = self.get_Hk_delta(kx, ky, delta)
        # pdb.set_trace()
        Hk = Hk0 - Hk0_mu + Hk0_M + Hk0_chi + Hk0_delta
        # Hk = Hk0 - Hk0_mu + Hk0_chi + Hk0_delta
        # Hk = Hk0 - Hk0_mu + Hk0_M + Hk0_chi
        # Hk = Hk0 - Hk0_mu + Hk0_M
        # Hk = Hk0 - Hk0_mu + Hk0_delta
        # pdb.set_trace()
        return Hk

    def get_Nup(self, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        Uc = Uk_conj[...,:3,:]
        U = Uk[...,:3,:]
        nk = Uc*U
        Nup = einsum('ijkl,ijl->k', nk, fk)/Nx/Ny
        return Nup

    def get_Ndn(self, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        Uc = Uk_conj[...,6:9,:]
        U = Uk[...,6:9,:]
        nk = Uc*U
        Ndn = einsum('ijkl,ijl->k', nk, fk)/Nx/Ny
        Ndn = 1 - Ndn
        return Ndn

    def get_moment(self, Nup, Ndn):
        m = (Nup - Ndn)/2
        return m

    def get_mu(self, Nup, Ndn, doping, mu_old, delta_mu):
        N = Nup.sum() + Ndn.sum()
        delta_n = N/3. - (1 - doping)
        mu = mu_old - delta_n*delta_mu
        return mu

    def get_chi(self, kx, ky, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        phase = cos(kx/2)*cos(ky/2)
        chi_up = zeros(3)
        chi_dn = zeros(3)
        for i in xrange(3):
            UcU = get_UcU(Uk_conj, Uk, i+3, i)
            chi1 = get_obs(UcU, fk)
            chi_up[i] = get_meanvalue(chi1, phase, Nx, Ny)

            UcU = get_UcU(Uk_conj, Uk, i+6, i+9)
            chi2 = get_obs(UcU, fk)
            chi_dn[i] = -get_meanvalue(chi2, phase, Nx, Ny)
        chi = chi_up + chi_dn
        return chi

    def get_delta(self, kx, ky, Uk_conj, Uk, fk):
        Nx = self.Nx; Ny = self.Ny
        phase = -sin(kx/2)*sin(ky/2)
        delta = zeros(3)
        for i in xrange(3):
            UcU = get_UcU(Uk_conj, Uk, i+9, i)
            delta1 = get_obs(UcU, fk)
            deltaBA = get_meanvalue(delta1, phase, Nx, Ny)

            UcU = get_UcU(Uk_conj, Uk, i+6, i+3)
            delta2 = get_obs(UcU, fk)
            deltaAB = get_meanvalue(delta2, phase, Nx, Ny)
            delta[i] = deltaAB + deltaBA
        return delta

    def get_doping_layer(self, Nup, Ndn):
        doping = abs(1 - Nup - Ndn)
        return doping

    def iteration(self, kx, ky, Hk0, mu, m, chi, delta, doping, beta):
        delta_mu = 0.4
        Hk = self.get_Hk(kx, ky, Hk0, mu, m, chi, delta)
        Ek, Uk = linalg.eigh(Hk)
        Uk_conj = Uk.conj(); fk = get_FermiDistri(Ek, beta)
        Nup = self.get_Nup(Uk_conj, Uk, fk)
        Ndn = self.get_Ndn(Uk_conj, Uk, fk)
        mu = self.get_mu(Nup, Ndn, doping, mu, delta_mu)
        chi = self.get_chi(kx, ky, Uk_conj, Uk, fk)
        delta = self.get_delta(kx, ky, Uk_conj, Uk, fk)
        return mu, Nup, Ndn, chi, delta 

    def main(self, doping, beta):
        kmesh = self.get_kmesh()
        kx = kmesh[0]; ky = kmesh[1]
        random.seed(2)
        mu = random.random()
        # doping_layer = array([doping/3, doping/3, doping/3])
        m = random.random(3)
        chi = random.random(3)
        delta = random.random(3)
        for i in xrange(1000):
            Hk0 = self.get_Hk0_nambu(kx, ky, doping)
            mu, Nup, Ndn, chi, delta = self.iteration(kx, ky, Hk0, mu, m, chi, delta, doping, beta)
            m = self.get_moment(Nup, Ndn)
            # doping_layer = self.get_doping_layer(Nup, Ndn)
            print (Nup)
            print (Ndn)
            print (chi)
            print (delta)
            print ('\n')

if __name__ == '__main__':
    # sc = dwave(0.3)
    # sc.MFT_main(0.13)
    sc = monolayer(0.3)
    sc.main(0.17, 200.)
    # sc = trilayer(0.3)
    # sc.main(0.20,200.)





