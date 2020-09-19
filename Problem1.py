import numpy as np
import pandas as pd
import math
import Aircraft
import BaryCenter

if __name__ == '__main__':
    rest_oil_np = np.array(Aircraft.OIL_TANK_INIT_OIL)
    data = pd.read_excel("data.xlsx", sheet_name=0).values

    for time, s1, s2, s3, s4, s5, s6, angle in data:
        if time <= 1500:
            continue
        oil_consume_np = np.array([s1, s2, s3, s4, s5, s6])
        rest_oil_np = rest_oil_np - oil_consume_np
        description = f"time = {time}s"
        barycenter, mass = BaryCenter.calc_3d_barycenter_all_tanks(
            Aircraft.OIL_TANK_MIDDLE_POSITION, Aircraft.OIL_TANK_SIZE,
            rest_oil_np.tolist(), math.radians(angle), description)
        print()
