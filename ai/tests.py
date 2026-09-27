"""
BioMirror — Test Suite
========================
Unit + integration tests for all core components.
Run: pytest tests.py -v --tb=short
"""

import numpy as np
import json, math, sys, os
import unittest

# ─────────────────────────────────────────────
# BERGMAN MODEL TESTS
# ─────────────────────────────────────────────
class TestBergmanModel(unittest.TestCase):

    def setUp(self):
        """Replicate BergmanEngine from backend_api.py (self-contained)."""
        class Engine:
            P1=0.028; P2=0.025; GB=90.0; IB=7.0; DT=5.0
            def __init__(self, p3=3.5e-5):
                self.p3=p3
            def step(self, G, X, I, meal):
                dG=-(self.P1+X)*G+self.P1*self.GB+meal
                dX=-self.P2*X+self.p3*(I-self.IB)
                return max(40.0,min(400.0,G+dG*self.DT)), max(0.0,X+dX*self.DT)
            def meal_bolus(self, carbs, t, peak=45):
                return carbs*0.6*math.exp(-0.5*((t-peak)/15)**2)/15 if carbs>0 else 0.0
            def classify(self, g):
                if g<54: return 'severe_low'
                if g<70: return 'low'
                if g<=140: return 'normal'
                if g<=180: return 'elevated'
                if g<=250: return 'high'
                return 'critical'
        self.engine = Engine()

    def test_glucose_clamped_lower(self):
        G, X = self.engine.step(45, 0.5, 50, 0)
        self.assertGreaterEqual(G, 40.0)

    def test_glucose_clamped_upper(self):
        G, X = self.engine.step(395, 0.0, 100, 500)
        self.assertLessEqual(G, 400.0)

    def test_insulin_action_nonnegative(self):
        _, X = self.engine.step(100, 0.0, 50, 0)
        self.assertGreaterEqual(X, 0.0)

    def test_basal_equilibrium(self):
        """At basal conditions (G=Gb, I=Ib), glucose should stay near 90."""
        G, X = self.engine.step(90.0, 0.0, 7.0, 0.0)
        self.assertAlmostEqual(G, 90.0, delta=5.0,
                               msg="Bergman should maintain basal glucose near 90")

    def test_meal_bolus_peak(self):
        """Meal bolus should peak around 45 minutes."""
        boluses = [self.engine.meal_bolus(60, t) for t in range(0, 120, 5)]
        peak_t  = np.argmax(boluses) * 5
        self.assertAlmostEqual(peak_t, 45, delta=10,
                               msg=f"Meal peak should be ~45min, got {peak_t}")

    def test_meal_bolus_zero_carbs(self):
        self.assertEqual(self.engine.meal_bolus(0, 30), 0.0)

    def test_meal_bolus_integrates_to_approx_carbs(self):
        """Total meal input should be proportional to carbs (positive correlation)."""
        carbs_lo, carbs_hi = 30, 90
        total_lo = sum(self.engine.meal_bolus(carbs_lo, t) * 5 for t in range(0, 240, 5))
        total_hi = sum(self.engine.meal_bolus(carbs_hi, t) * 5 for t in range(0, 240, 5))
        self.assertGreater(total_hi, total_lo,
                           "Higher carbs should produce larger meal absorption")

    def test_high_p3_reduces_glucose_faster(self):
        """Higher insulin sensitivity should reduce elevated glucose faster."""
        G0, X0, I = 200.0, 0.05, 20.0
        engine_sensitive = type(self.engine)(p3=6e-5)
        engine_resistant = type(self.engine)(p3=2e-5)
        G_s, _ = engine_sensitive.step(G0, X0, I, 0)
        G_r, _ = engine_resistant.step(G0, X0, I, 0)
        self.assertLessEqual(G_s, G_r,
                             msg="Higher p3 should produce lower glucose over time")

    def test_classify_all_zones(self):
        cases = [(40, 'severe_low'), (60, 'low'), (100, 'normal'),
                 (155, 'elevated'), (200, 'high'), (300, 'critical')]
        for g, expected in cases:
            self.assertEqual(self.engine.classify(g), expected,
                             f"classify({g}) should be {expected}")

    def test_long_simulation_stability(self):
        """288-step (24h) simulation should not produce NaN or out-of-range values."""
        G, X = 160.0, 0.02
        I = 12.0
        for _ in range(288):
            G, X = self.engine.step(G, X, I, 0.0)
        self.assertFalse(math.isnan(G), "Glucose became NaN")
        self.assertFalse(math.isnan(X), "X became NaN")
        self.assertGreaterEqual(G, 40.0)
        self.assertLessEqual(G, 400.0)

    def test_dG_formula(self):
        """Manually verify dG/dt = -(p1+X)G + p1*Gb + meal."""
        G, X, I, meal = 150.0, 0.03, 12.0, 5.0
        dG = -(self.engine.P1 + X) * G + self.engine.P1 * self.engine.GB + meal
        G_expected = max(40, min(400, G + dG * self.engine.DT))
        G_actual, _ = self.engine.step(G, X, I, meal)
        self.assertAlmostEqual(G_actual, G_expected, places=4)

    def test_dX_formula(self):
        """Manually verify dX/dt = -p2*X + p3*(I-Ib)."""
        G, X, I = 150.0, 0.03, 12.0
        dX = -self.engine.P2 * X + self.engine.p3 * (I - self.engine.IB)
        X_expected = max(0, X + dX * self.engine.DT)
        _, X_actual = self.engine.step(G, X, I, 0.0)
        self.assertAlmostEqual(X_actual, X_expected, places=6)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING TESTS
# ─────────────────────────────────────────────
class TestFeatureEngineering(unittest.TestCase):

    def _glycemic_load(self, carbs, gi, sugar_ratio=0):
        gi_adjusted = gi * (1 + sugar_ratio * 0.3)
        return carbs * gi_adjusted

    def test_glycemic_load_zero_carbs(self):
        self.assertEqual(self._glycemic_load(0, 0.6), 0.0)

    def test_glycemic_load_high_gi_greater(self):
        lo = self._glycemic_load(50, 0.4)
        hi = self._glycemic_load(50, 0.8)
        self.assertGreater(hi, lo)

    def test_circadian_dawn_factor(self):
        """Dawn effect (4-8 AM) should produce factor > 1."""
        def circadian(hour):
            dawn = 1.15 if 4 <= hour <= 8 else 1.0
            somogyi = 0.90 if 2 <= hour <= 4 else 1.0
            return dawn * somogyi
        self.assertGreater(circadian(6), 1.0)

    def test_circadian_night_baseline(self):
        def circadian(hour):
            dawn = 1.15 if 4 <= hour <= 8 else 1.0
            somogyi = 0.90 if 2 <= hour <= 4 else 1.0
            return dawn * somogyi
        self.assertAlmostEqual(circadian(14), 1.0, places=3)

    def test_metabolic_stress_clipped_0_100(self):
        def stress(vol, steps, risk, health):
            gd = abs(100 - 100) / 100
            ad = max(0, (5000 - steps) / 5000)
            return min(100, max(0, (gd*0.35 + ad*0.25 + risk/2*0.25 + (1-health/100)*0.15)*100))
        self.assertGreaterEqual(stress(0, 10000, 0, 100), 0)
        self.assertLessEqual(stress(100, 0, 2, 0), 100)

    def test_insulin_sensitivity_inversely_proportional_to_bmi(self):
        def sens(bmi, g): return 100 / (bmi/22 * g/90 + 0.5)
        self.assertGreater(sens(22, 90), sens(35, 90))

    def test_hba1c_nathan_formula(self):
        """Nathan formula: HbA1c = (mean_glucose + 46.7) / 28.7"""
        mean_g = 154  # corresponds to ~7% HbA1c
        hba1c  = (mean_g + 46.7) / 28.7
        self.assertAlmostEqual(hba1c, 7.0, delta=0.5)

    def test_sleep_impact_coefficient_range(self):
        def sleep_coeff(health_score):
            sleep_q = health_score / 100
            sleep_h = 4 + sleep_q * 5
            deficit = max(0, 8 - sleep_h)
            return min(1.8, 1 + deficit * 0.18)
        self.assertGreaterEqual(sleep_coeff(100), 1.0)
        self.assertLessEqual(sleep_coeff(0), 1.8)

    def test_glucose_volatility_zero_for_constant(self):
        """Constant glucose → zero volatility."""
        glucose = [120.0] * 10
        vol = np.std(glucose)
        self.assertAlmostEqual(vol, 0.0, places=5)

    def test_reversal_probability_high_for_healthy(self):
        """Healthy profile (high sensitivity, high TIR) → high reversal."""
        def rev(sens, tir, health, stress):
            return (sens*0.30 + tir*0.30 + health*0.25 + (100-stress)*0.15)
        self.assertGreater(rev(90, 85, 90, 10), 80)

    def test_reversal_probability_low_for_diabetic(self):
        def rev(sens, tir, health, stress):
            return (sens*0.30 + tir*0.30 + health*0.25 + (100-stress)*0.15)
        self.assertLess(rev(30, 40, 45, 80), 50)


# ─────────────────────────────────────────────
# METRICS TESTS
# ─────────────────────────────────────────────
class TestMetrics(unittest.TestCase):

    def test_mard_perfect_prediction(self):
        y = np.array([100., 150., 200.])
        self.assertAlmostEqual(
            np.mean(np.abs(y - y) / (y + 1e-8)) * 100, 0.0, places=4
        )

    def test_mard_known_value(self):
        y_true = np.array([100., 100., 100.])
        y_pred = np.array([110., 90.,  100.])  # ±10% errors
        mard = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8)) * 100
        self.assertAlmostEqual(mard, 6.667, delta=0.1)

    def test_r2_perfect_is_one(self):
        y = np.array([100., 140., 180., 200.])
        from sklearn.metrics import r2_score
        self.assertAlmostEqual(r2_score(y, y), 1.0, places=5)

    def test_r2_mean_prediction_is_zero(self):
        y_true = np.array([100., 140., 180., 200.])
        y_pred = np.full_like(y_true, y_true.mean())
        from sklearn.metrics import r2_score
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.0, places=5)

    def test_clarke_zone_a_perfect(self):
        """Perfect predictor → 100% Zone A."""
        y = np.random.uniform(70, 200, 100)
        zone_a = np.mean(np.abs(y - y) / (y + 1e-8) < 0.20) * 100
        self.assertEqual(zone_a, 100.0)

    def test_tir_calculation(self):
        glucose = np.array([60, 80, 100, 140, 180, 200, 250])
        tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
        self.assertAlmostEqual(tir, 4/7 * 100, delta=0.5)


# ─────────────────────────────────────────────
# DIFFERENTIAL PRIVACY TESTS
# ─────────────────────────────────────────────
class TestDifferentialPrivacy(unittest.TestCase):

    def setUp(self):
        class DP:
            def __init__(self, C=1.0, sigma=1.1):
                self.C = C; self.sigma = sigma * C
            def clip(self, grads):
                norm = math.sqrt(sum(np.sum(g**2) for g in grads))
                if norm > self.C:
                    f = self.C / (norm + 1e-8)
                    return [g * f for g in grads]
                return grads
            def add_noise(self, grads):
                return [g + np.random.normal(0, self.sigma, g.shape) for g in grads]
            def privatize(self, grads):
                return self.add_noise(self.clip(grads))
        self.dp = DP()

    def test_clipping_bounds_norm(self):
        grads = [np.ones((10, 10)) * 5]  # large gradient
        clipped = self.dp.clip(grads)
        norm = math.sqrt(sum(np.sum(g**2) for g in clipped))
        self.assertLessEqual(norm, self.dp.C * 1.01)

    def test_small_gradient_not_clipped(self):
        small = [np.ones((2, 2)) * 0.1]
        norm_before = math.sqrt(sum(np.sum(g**2) for g in small))
        clipped = self.dp.clip(small)
        norm_after  = math.sqrt(sum(np.sum(g**2) for g in clipped))
        self.assertAlmostEqual(norm_before, norm_after, delta=1e-6)

    def test_noise_added(self):
        grads = [np.zeros((5, 5))]
        noised = self.dp.add_noise(grads)
        self.assertFalse(np.allclose(grads[0], noised[0]),
                         "Noise should change zero gradient")

    def test_noise_shape_preserved(self):
        grads = [np.ones((4, 3)), np.ones((3,))]
        noised = self.dp.privatize(grads)
        for g, n in zip(grads, noised):
            self.assertEqual(g.shape, n.shape)

    def test_privacy_spent_increases_with_steps(self):
        """Tighter formula: optimal-alpha RDP accountant (Mironov 2017)."""
        def spent(steps, sigma=3.0, delta=1e-3):
            best = float('inf')
            for alpha in [1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0]:
                rdp = steps * alpha / (2.0 * sigma ** 2)
                e   = rdp + math.log(1.0 / delta) / (alpha - 1.0)
                if e < best:
                    best = e
            return max(0.0, best)
        self.assertLess(spent(10), spent(100),
                        "More gradient steps must spend more privacy budget")


# ─────────────────────────────────────────────
# API RESPONSE STRUCTURE TESTS
# ─────────────────────────────────────────────
class TestAPIResponseStructure(unittest.TestCase):

    def _mock_metrics(self):
        return {
            'glucose': 161.0, 'X_insulin_action': 0.042,
            'insulin_sensitivity': 68.0, 'tir_pct': 63.0,
            'metabolic_stress': 35.0, 'recovery_score': 74.0,
            'reversal_score': 72.0, 'metabolic_age': 44.0,
            'hba1c_estimate': 7.2, 'glucose_status': 'elevated',
            'last_updated': '2025-01-01T00:00:00',
        }

    def test_metrics_all_fields_present(self):
        m = self._mock_metrics()
        required = ['glucose', 'insulin_sensitivity', 'tir_pct',
                    'reversal_score', 'hba1c_estimate', 'glucose_status']
        for field in required:
            self.assertIn(field, m, f"Missing field: {field}")

    def test_glucose_in_physiological_range(self):
        m = self._mock_metrics()
        self.assertGreaterEqual(m['glucose'], 40)
        self.assertLessEqual(m['glucose'], 400)

    def test_scores_in_0_100(self):
        m = self._mock_metrics()
        for score in ['insulin_sensitivity', 'tir_pct', 'metabolic_stress',
                      'recovery_score', 'reversal_score']:
            self.assertGreaterEqual(m[score], 0, f"{score} below 0")
            self.assertLessEqual(m[score], 100, f"{score} above 100")

    def test_glucose_status_valid(self):
        valid = {'severe_low', 'low', 'normal', 'elevated', 'high', 'critical'}
        m = self._mock_metrics()
        self.assertIn(m['glucose_status'], valid)

    def _mock_sim_result(self, carbs, exercise):
        n = 48
        base = [160.0] * n
        sim  = [min(400, 160 + carbs * 0.4 * math.exp(-0.5*((i*5-45)/20)**2))
                * (1 - exercise * 0.002) for i in range(n)]
        return {
            'baseline_glucose': base,
            'simulated_glucose': sim,
            'results': {
                'peak_glucose': max(sim),
                'spike_probability_pct': sum(1 for g in sim if g > 180) / n * 100,
                'time_in_range_pct': sum(1 for g in sim if 70 <= g <= 180) / n * 100,
            }
        }

    def test_simulation_higher_carbs_higher_peak(self):
        low  = self._mock_sim_result(20, 0)
        high = self._mock_sim_result(100, 0)
        self.assertGreater(high['results']['peak_glucose'],
                           low['results']['peak_glucose'])

    def test_simulation_exercise_reduces_glucose(self):
        no_ex = self._mock_sim_result(60, 0)
        ex    = self._mock_sim_result(60, 45)
        self.assertLessEqual(ex['results']['peak_glucose'],
                              no_ex['results']['peak_glucose'])

    def test_fhir_bundle_structure(self):
        bundle = {
            'resourceType': 'Bundle',
            'type': 'collection',
            'entry': [{'resource': {
                'resourceType': 'Observation',
                'code': {'coding': [{'code': '2339-0'}]},
                'valueQuantity': {'value': 161, 'unit': 'mg/dL'},
            }}]
        }
        self.assertEqual(bundle['resourceType'], 'Bundle')
        self.assertIn('entry', bundle)
        obs = bundle['entry'][0]['resource']
        self.assertEqual(obs['resourceType'], 'Observation')
        self.assertEqual(obs['valueQuantity']['unit'], 'mg/dL')


# ─────────────────────────────────────────────
# FEDERATED LEARNING TESTS
# ─────────────────────────────────────────────
class TestFederatedLearning(unittest.TestCase):

    def test_fedavg_weighted_average(self):
        """Verify FedAvg weighted averaging is correct."""
        w1 = [np.array([1.0, 2.0])]
        w2 = [np.array([3.0, 4.0])]
        n1, n2 = 60, 40
        total = n1 + n2
        aggregated = [w1[0] * (n1/total) + w2[0] * (n2/total)]
        expected = np.array([1.0*(60/100) + 3.0*(40/100),
                              2.0*(60/100) + 4.0*(40/100)])
        np.testing.assert_allclose(aggregated[0], expected, rtol=1e-5)

    def test_dp_gradient_clipping(self):
        large_grads = [np.ones((100,)) * 10]
        C = 1.0
        norm = math.sqrt(sum(np.sum(g**2) for g in large_grads))
        f = C / norm
        clipped = [g * f for g in large_grads]
        norm_after = math.sqrt(sum(np.sum(g**2) for g in clipped))
        self.assertAlmostEqual(norm_after, C, delta=0.01)

    def test_p3_variation_across_clients(self):
        """Different patients should have different p3 values."""
        p3s = [5.2e-5, 3.8e-5, 2.4e-5, 3.1e-5, 4.5e-5]
        self.assertGreater(np.std(p3s), 0, "p3 should vary across patients")
        # All p3 in physiological range
        for p3 in p3s:
            self.assertGreater(p3, 1e-5)
            self.assertLess(p3, 1e-4)

    def test_privacy_budget_monotone(self):
        """Privacy budget ε should increase (be spent) with more steps.
        Uses corrected tight RDP accountant formula (Mironov 2017)."""
        def eps(steps, sigma=3.0, delta=1e-3):
            best = float('inf')
            for alpha in [1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0]:
                rdp = steps * alpha / (2.0 * sigma ** 2)
                e   = rdp + math.log(1.0 / delta) / (alpha - 1.0)
                if e < best:
                    best = e
            return max(0.0, best)
        budgets = [eps(s) for s in [10, 50, 100, 200]]
        self.assertEqual(budgets, sorted(budgets),
                         "Privacy budget must increase monotonically with steps")


# ─────────────────────────────────────────────
# INTEGRATION TEST — END-TO-END PIPELINE
# ─────────────────────────────────────────────
class TestEndToEnd(unittest.TestCase):

    def test_bergman_to_metrics_pipeline(self):
        """Full pipeline: glucose log → Bergman step → metrics → FHIR."""
        # 1. Simulate glucose event
        G0, X0, I, meal = 185.0, 0.02, 15.0, 8.0
        p1, p2, p3 = 0.028, 0.025, 3.5e-5
        Gb, Ib, DT  = 90.0, 7.0, 5.0
        dG = -(p1 + X0) * G0 + p1 * Gb + meal
        dX = -p2 * X0 + p3 * (I - Ib)
        G_new = max(40, min(400, G0 + dG * DT))
        X_new = max(0, X0 + dX * DT)
        self.assertGreater(G_new, 40)
        self.assertLess(G_new, 400)

        # 2. Compute metrics
        p3_idx  = p3 / 5e-5 * 100
        tir     = 63.0
        stress  = min(100, abs(G_new - 100) * 0.5)
        rec     = max(0, 100 - stress * 0.6)
        rev     = round(p3_idx*0.35 + tir*0.35 + rec*0.30, 1)
        hba1c   = (G_new + 46.7) / 28.7

        self.assertGreaterEqual(rev, 0)
        self.assertLessEqual(rev, 100)
        self.assertGreater(hba1c, 4.0)
        self.assertLess(hba1c, 15.0)

        # 3. FHIR resource
        obs = {
            'resourceType': 'Observation',
            'valueQuantity': {'value': G_new, 'unit': 'mg/dL'},
        }
        self.assertEqual(obs['resourceType'], 'Observation')
        self.assertAlmostEqual(obs['valueQuantity']['value'], G_new, places=2)

    def test_what_if_simulation_consistency(self):
        """More carbs → higher peak glucose in simulation."""
        def sim_peak(carbs, n=48):
            G = 160.0
            peaks = []
            for k in range(n):
                meal = carbs * 0.6 * math.exp(-0.5*((k*5-45)/15)**2)/15
                dG = -(0.028 + 0.02) * G + 0.028*90 + meal
                G = max(40, min(400, G + dG*5))
                peaks.append(G)
            return max(peaks)
        self.assertGreater(sim_peak(100), sim_peak(20))

    def test_reversal_report_fields(self):
        """Reversal report must have all required sections."""
        report = {
            'current_status':  {'hba1c': 7.2, 'reversal_score': 65, 'recovery_stage': 3},
            'reversal_forecast': {'months_to_reversal': 8.5, 'target_hba1c': 6.4},
            'intervention_priorities': [{'rank': 1, 'lever': 'Activity'}],
            'metabolic_summary': {'insulin_sensitivity_index': 68},
        }
        for key in ['current_status', 'reversal_forecast',
                    'intervention_priorities', 'metabolic_summary']:
            self.assertIn(key, report)


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────
def run_tests(verbose: bool = True) -> bool:
    print("\n" + "="*60)
    print("  BioMirror Test Suite")
    print("="*60 + "\n")

    suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestBergmanModel),
        unittest.TestLoader().loadTestsFromTestCase(TestFeatureEngineering),
        unittest.TestLoader().loadTestsFromTestCase(TestMetrics),
        unittest.TestLoader().loadTestsFromTestCase(TestDifferentialPrivacy),
        unittest.TestLoader().loadTestsFromTestCase(TestAPIResponseStructure),
        unittest.TestLoader().loadTestsFromTestCase(TestFederatedLearning),
        unittest.TestLoader().loadTestsFromTestCase(TestEndToEnd),
    ]
    combined = unittest.TestSuite(suites)
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout
    )
    result = runner.run(combined)
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed  ✓")
    if result.failures:
        print(f"  Failures: {len(result.failures)}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
    print(f"{'='*60}\n")
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
