import numpy as np
import pandas as pd
import math
import Aircraft
import BaryCenter

if __name__ == '__main__':
    rest_oil_mass_np = np.array(Aircraft.OIL_TANK_INIT_OIL_MASS)
    data = pd.read_excel("data.xlsx", sheet_name=0).values

    for time, s1, s2, s3, s4, s5, s6, angle in data:
        time = int(time)
        oil_consume_mass_np = np.array([s1, s2, s3, s4, s5, s6])
        rest_oil_mass_np = rest_oil_mass_np - oil_consume_mass_np
        rest_oil_volume_np = rest_oil_mass_np / Aircraft.OIL_DENSITY_KGpm3
        description = f"time = {time:d}s"
        barycenter, mass = BaryCenter.calc_3d_barycenter_all_tanks(
            Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE,
            rest_oil_volume_np.tolist(), math.radians(angle), description)
        print(f"{time:d}\t{barycenter.x}\t{barycenter.y}\t{barycenter.z}")

    print(rest_oil_mass_np)
