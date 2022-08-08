import numpy as np
import math, random

class MCS:
    def simulate():
        ECM = 200
        M   = 114
        # Number of events to generate
        Neve = 100_000 # (was 10)
        # counter of events generate 
        jj = 0
        # start generating events (i.e. "hit and miss")
        print('generating events...')

        # we will store "generated" costh, cosphi in an array:
        cos_theta = []
        cos_phi = []

        sin_theta = []
        sin_phi = []

        data = np.zeros((Neve, 8))

        delta = 2

        while jj < Neve:
            # random costheta
            costh_ii = -1 + random.random() * delta
            # costh_ii = np.cos(np.arccos(costh_ii)) # + np.random.normal(1, 0.1))
            # calc. phase space point
            phi = random.random() * 2 * math.pi
            sinphi = math.sin(phi)
            cosphi = math.cos(phi)
            sinth = math.sqrt( 1 - costh_ii**2 )
            # if the random number is less than the probability of the PS point
            random_e = random.random() * 200
            random_e = ECM
            # accept
            x1 = np.array([ECM, random_e * sinth * cosphi, random_e * sinth * sinphi, random_e * costh_ii]) * 0.5
            x2 = np.array([ECM, - random_e * sinth * cosphi, - random_e * sinth * sinphi, - random_e * costh_ii]) * 0.5

            pmm = [ 0.5 * ECM, 0.5 * ECM * sinth * cosphi, 0.5 * ECM * sinth * sinphi, 0.5 * ECM * costh_ii ]
            pmp = [ 0.5 * ECM, - 0.5 * ECM * sinth * cosphi, - 0.5 * ECM * sinth * sinphi, - 0.5 * ECM * costh_ii ]

            def dispersion(p):
                return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2

            print(dispersion(x1),dispersion(x2), dispersion(pmm), dispersion(pmp), phi, np.arcsin(sinth))

            if np.allclose([dispersion(x1),dispersion(x2)], [ECM, ECM], rtol=0.1, atol=0.1):
                print("YES!!!!")
                print([dispersion(x1),dispersion(x2)], [ECM, ECM])
                # exit()
                # here we create the four-vectors of the hard process particles
                # generate random phi
                
                phi += (np.random.normal(1, 0.1) * math.pi)
                ECM += (np.random.normal(1, 0.05) * ECM)

                pem = [ 0.5 * ECM, 0., 0., 0.5 * ECM ]
                pep = [ 0.5 * ECM, 0., 0., - 0.5 * ECM ]
                pmm = [ 0.5 * ECM, 0.5 * ECM * sinth * cosphi, 0.5 * ECM * sinth * sinphi, 0.5 * ECM * costh_ii ]
                pmp = [ 0.5 * ECM, - 0.5 * ECM * sinth * cosphi, - 0.5 * ECM * sinth * sinphi, - 0.5 * ECM * costh_ii ]
                data[jj] = np.concatenate((pmm, pmp))

                # here one can either analyze the
                # or store them for later convenience
                cos_theta.append(costh_ii)
                cos_phi.append(cosphi)

                sin_theta.append(sinth)
                sin_phi.append(sinphi)

                jj += 1

        cosths = np.array(cos_theta)
        cosphis = np.array(cos_phi)
        sinths = np.array(sin_theta)
        sinphis = np.array(sin_phi)

        data_plus = data

        # Add raw values of phi and theta to event data
        data_plus = np.column_stack((data_plus, np.arccos(cosths)))
        data_plus = np.column_stack((data_plus, np.arccos(cosphis)))
                
        generate_histo(cosths, f'ee-costheta-{ECM}-{Neve}')
        generate_histo(cosphis, f'ee-cosphi-{ECM}-{Neve}')
        generate_histo(sinths, f'ee-sintheta-{ECM}-{Neve}')
        generate_histo(sinphis, f'ee-sinphi-{ECM}-{Neve}')
        np.savetxt("data/events_naive_mcs.txt", data_plus)


if __name__ in "__main__":
    MCS.simulate()