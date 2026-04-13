import numpy as np
import time


class TLE493D:
    def __init__(self, range_mode="full", mode="low_power", f_update=125):
        # -------------------------
        # RANGE CONFIGURATION
        # -------------------------
        self.sensitivity_nominal = {
            "full": 29.5,
            "short": 59,
            "extra_short": 118
        }[range_mode]

        self.range_limit = {
            "full": 160,
            "short": 100,
            "extra_short": 50
        }[range_mode]

        # -------------------------
        # TIMING / MODE
        # -------------------------
        self.mode = mode  # "low_power" or "master"
        self.f_update = f_update
        self.last_sample_time = time.time()

        # -------------------------
        # NOISE / OFFSET
        # -------------------------
        self.noise_std = 0.25  # mT
        self.offset_25C = np.random.normal(0, 0.2)

        # -------------------------
        # TEMPERATURE MODEL
        # -------------------------
        self.temp_ref = 25
        self.temp_sensitivity = 15.2
        self.temp_ref_raw = 4200

        # Drift coefficients (approx)
        self.sens_tempco = 0.0005
        self.offset_tempco = 0.002

        # -------------------------
        # DATASHEET COMPENSATION COEFFICIENTS
        # -------------------------
        self.coeffs = {
            "x": {
                "O0": 52.46501931,
                "O1": -30.828402e-3,
                "O2": 6.06444e-6,
                "O3": -4.20546e-10,
                "L0": -2.109359211,
                "L1": 2.248525e-3,
                "L2": -5.25818e-7,
                "L3": 3.99648e-11,
            },
            "y": {
                "O0": 7.574714985,
                "O1": -4.602293e-3,
                "O2": 8.61016e-7,
                "O3": -7.47545e-11,
                "L0": -2.106808409,
                "L1": 2.234594e-3,
                "L2": -5.22864e-7,
                "L3": 3.97614e-11,
            },
            "z": {
                "O0": 9.233258372,
                "O1": -3.911673e-3,
                "O2": 7.01838e-7,
                "O3": -4.38542e-11,
                "L0": -0.96458813,
                "L1": 1.445091e-3,
                "L2": -3.42739e-7,
                "L3": 2.63e-11,
            }
        }

    # -------------------------
    # HELPERS
    # -------------------------
    def _clip(self, B):
        return np.clip(B, -self.range_limit, self.range_limit)

    def _to_14bit(self, val):
        return int(np.clip(np.round(val), -8192, 8191))

    # -------------------------
    # TEMPERATURE EFFECTS
    # -------------------------
    def _apply_temperature_effects(self, B, temp):
        dT = temp - self.temp_ref

        sensitivity = self.sensitivity_nominal * (1 + self.sens_tempco * dT)
        offset = self.offset_25C + self.offset_tempco * dT

        return B, sensitivity, offset

    # -------------------------
    # RAW MEASUREMENT
    # -------------------------
    def measure_raw(self, Bx, By, Bz, temp):
        Bx = self._clip(Bx)
        By = self._clip(By)
        Bz = self._clip(Bz)

        def axis(B):
            B, sens, offset = self._apply_temperature_effects(B, temp)
            noise = np.random.normal(0, self.noise_std)
            raw = (B + offset + noise) * sens
            return self._to_14bit(raw)

        return {
            "Bx": axis(Bx),
            "By": axis(By),
            "Bz": axis(Bz)
        }

    # -------------------------
    # TEMPERATURE OUTPUT
    # -------------------------
    def measure_temperature(self, temp_c):
        raw = self.temp_ref_raw + (temp_c - 25) * self.temp_sensitivity
        return self._to_14bit(raw)

    # -------------------------
    # COMPENSATION (DATASHEET 4.1.3)
    # -------------------------
    def _compensate_axis(self, raw, T, c):
        O = c["O0"] + c["O1"] * T + c["O2"] * T**2 + c["O3"] * T**3
        L = c["L0"] + c["L1"] * T + c["L2"] * T**2 + c["L3"] * T**3
        return (raw - O) * (1 + L)

    def compensate(self, raw, temp):
        return {
            "Bx": self._compensate_axis(raw["Bx"], temp, self.coeffs["x"]),
            "By": self._compensate_axis(raw["By"], temp, self.coeffs["y"]),
            "Bz": self._compensate_axis(raw["Bz"], temp, self.coeffs["z"]),
        }

    # -------------------------
    # TIMING MODEL
    # -------------------------
    def _ready(self):
        if self.mode == "master":
            return True

        now = time.time()
        if now - self.last_sample_time >= 1 / self.f_update:
            self.last_sample_time = now
            return True
        return False

    # -------------------------
    # FULL SAMPLE PIPELINE
    # -------------------------
    def sample(self, Bx, By, Bz, temp):
        if not self._ready():
            return None

        raw = self.measure_raw(Bx, By, Bz, temp)
        temp_raw = self.measure_temperature(temp)
        compensated = self.compensate(raw, temp)

        return {
            "raw": raw,
            "temp_raw": temp_raw,
            "compensated": compensated
        }


# -------------------------
# EXAMPLE USAGE
# -------------------------
if __name__ == "__main__":
    sensor = TLE493D(mode="low_power", f_update=50)

    for i in range(100):
        data = sensor.sample(Bx=20, By=5, Bz=-10, temp=60)

        if data:
            print("RAW:", data["raw"])
            print("COMP:", data["compensated"])
            print("TEMP RAW:", data["temp_raw"])
            print("---")

        time.sleep(0.01)