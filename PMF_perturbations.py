# =============================================================================
# Imports
# =============================================================================
from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

__all__ = ['pert_sharpcut_at_rec']  # Explicitly export the function

# =============================================================================
# Main Function: pert_sharpcut_at_rec
# =============================================================================

def pert_sharpcut_at_rec(k, S0, kd, Ht, rhodm, rho_tot, photon_mfp_full, alpha, rhob, cb2_full, a_rec):
    """
    Solve dark matter and baryon perturbations with and without primordial magnetic fields (PMF).
    Assumes sharp cutoff of delta_b when theta_b feedbacks on magnetic fields.

    Args:
        k: Wavenumber
        S0: Source term function for PMF
        kd: ODE solution for damping scale
        Ht: Hubble rate function
        rhodm: Dark matter density function
        rho_tot: Total density function
        photon_mfp_full: Photon mean free path function
        alpha: photon Drag coefficient function
        rhob: Baryon density function
        cb2_full: Baryon sound speed squared function
        a_rec: Scale factor at recombination

    Returns:
        List of arrays: [a_dm_only, delta_dm_only, a_pmf, delta_dm_pmf, delta_b_pmf, theta_b_pmf, phi]
    """

    # -------------------------------------------------------------------------
    # Initial normalization and horizon/mfp scales
    # -------------------------------------------------------------------------
    norm = (2.215*10**-9*(k/0.002)**(0.9619-1))**0.5 #normalization of initial density perturbation outside of horizon

    # Find scale factor when mode enters photon mean free path and horizon
    a_mfp = fsolve(lambda a: photon_mfp_full(a)*k-1, a_rec/10*(k*photon_mfp_full(a_rec/10))**-0.5)[0]
    a_hor = fsolve(lambda a: a*Ht(a)-k, (a_rec/10)**2*Ht(a_rec/10)/k)[0]

    # -------------------------------------------------------------------------
    # DM-only perturbation equations (no baryons, no PMF)
    # -------------------------------------------------------------------------
    def ddel_dm(a, delta, k):
        """
        Differential equations for DM-only perturbations.
        """
        phi, dldm, thdm = delta[0], delta[1], delta[2]
        del_phi = -phi/a + (-k**2*phi - 3./2*(a*Ht(a))**2*(rhodm(a)*dldm)/rho_tot(a))/(3*(a*Ht(a))**2)/a
        del_dldm = -(thdm/a**2/Ht(a) - 3*del_phi)
        del_thdm = -thdm/a + k**2/a**2/Ht(a)*phi
        return [del_phi, del_dldm, del_thdm]

    # Initial conditions deep in subhorizon
    delta_dmi = -6.71*norm*np.log(0.5*200*a_hor/a_hor)
    theta_dmi = 6.71*norm*200*a_hor*Ht(200*a_hor)
    deltasolve30 = [0, delta_dmi, theta_dmi]
    deltasolve3 = solve_ivp(
        lambda a, y: ddel_dm(a, y, k),
        [200*a_hor, 1],
        deltasolve30,
        method='BDF',
        dense_output=True,
        atol=1e-6,
        rtol=1e-5
    )
    delta_dm_only = deltasolve3.y[1, :]
    a_dm_only = deltasolve3.t[:]

    # -------------------------------------------------------------------------
    # DM and baryon perturbation equations with PMF
    # -------------------------------------------------------------------------
    def ddel_baryon(a, delta, k):
        """
        Differential equations for DM and baryon perturbations with PMF. Neglect radiation perturbations as they decay inside photon mfp.
        """
        phi, dldm, thdm, dlb, thb = delta[0], delta[1], delta[2], delta[3], delta[4]
        kd_sol_a = float(kd.sol(a)[0])  # Ensure scalar conversion
        S0_kd_sol_a = S0(kd_sol_a)
        del_phi = -phi/a + (-k**2*phi - 3./2*(a*Ht(a))**2*(rhob(a)*dlb + rhodm(a)*dldm)/rho_tot(a))/(3*(a*Ht(a))**2)/a
        del_dldm = -(thdm/a**2/Ht(a) - 3*del_phi)
        del_thdm = -thdm/a + k**2/a**2/Ht(a)*phi
        del_thb = -(1+alpha(a)/Ht(a))*thb/a + k**2/a**2/Ht(a)*phi + cb2_full(a)*k**2/a**2/Ht(a)*dlb + S0_kd_sol_a/a**3/Ht(a)
        del_dlb = -(thb/a**2/Ht(a) - 3*del_phi)
        return np.array([del_phi, del_dldm, del_thdm, del_dlb, del_thb])

    deltasolve0 = [0, 0, 0, 0, 0]
    deltasolve = solve_ivp(
        lambda a, y: ddel_baryon(a, y, k),
        [a_mfp, 1],
        deltasolve0,
        method='BDF',
        dense_output=True,
        atol=1e-6,
        rtol=1e-5
    )

    # -------------------------------------------------------------------------
    # Handling baryon feedback and sharp cutoff
    # -------------------------------------------------------------------------
    if k > kd.sol(10*a_rec):  # For modes smaller than kd, assume turbulence at recombination instantly suppresses delta_b
        i_turb = np.where(deltasolve.t > a_rec)[0][0]
        # Solve DM-only perturbations after baryon feedback, assume delta_b=0
        deltasolve20 = [deltasolve.y[0, i_turb], deltasolve.y[1, i_turb], deltasolve.y[2, i_turb]]
        deltasolve2 = solve_ivp(
            lambda a, y: ddel_dm(a, y, k),
            [deltasolve.t[i_turb], 1],
            deltasolve20,
            method='BDF',
            dense_output=True,
            atol=1e-6,
            rtol=1e-5
        )
        delta_dm_pmf = np.concatenate((deltasolve.y[1, :i_turb], deltasolve2.y[1, :]))
        a_pmf = np.concatenate((deltasolve.t[:i_turb], deltasolve2.t[:]))
        delta_b_pmf = np.concatenate((deltasolve.y[3, :i_turb], np.zeros(deltasolve2.t.shape[0])))
        theta_b_pmf = np.concatenate((deltasolve.y[4, :i_turb], np.zeros(deltasolve2.t.shape[0])))
        phi = np.concatenate((deltasolve.y[0, :i_turb], deltasolve2.y[0, :]))
    else:
        delta_dm_pmf=deltasolve.y[1,:]
        a_pmf=deltasolve.t
        delta_b_pmf=deltasolve.y[3,:]
        theta_b_pmf=deltasolve.y[4,:]
        phi=deltasolve.y[0,:]

    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------
    return [a_dm_only, delta_dm_only, a_pmf, delta_dm_pmf, delta_b_pmf, theta_b_pmf, phi]
