"""CPU unit tests for bench/adaptive_k.py. Run: python -m pytest bench/test_adaptive_k.py -q  (or python bench/test_adaptive_k.py)"""
import json, os, random, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adaptive_k as ak  # noqa: E402


def run(pol, draws):
    for a in draws:
        k = pol.choose()
        pol.observe(min(a, k - 1), k)
    return pol.choose()


class CostCurveTest(unittest.TestCase):
    def test_interp_and_extrap(self):
        c = ak.CostCurve({16: 10.0, 32: 12.0})
        self.assertAlmostEqual(c(16), 10.0)
        self.assertAlmostEqual(c(24), 11.0)
        self.assertAlmostEqual(c(48), 14.0)  # extrapolates with the last slope
        self.assertAlmostEqual(c(8), 9.0)

    def test_measured_curve_monotone(self):
        c = ak.CostCurve(ak.A100_8B_C1_ROUND_MS)
        ks = [8, 12, 16, 24, 32, 48, 64]
        self.assertEqual([c(k) for k in ks], sorted(c(k) for k in ks))

    def test_needs_two_points(self):
        with self.assertRaises(ValueError):
            ak.CostCurve({16: 1.0})


class EmaTest(unittest.TestCase):
    def test_low_acceptance_goes_small(self):
        self.assertEqual(run(ak.EmaThresholdK((8, 16, 32)), [0, 1, 0, 2, 1] * 10), 8)

    def test_full_blocks_climb_to_max(self):
        p = ak.EmaThresholdK((8, 16, 32), init_k=8)
        self.assertEqual(run(p, [100] * 10), 32)

    def test_full_block_never_steps_down(self):
        p = ak.EmaThresholdK((8, 16, 32), ratio=0.1, init_k=16)
        p.choose()
        p.observe(15, 16)  # censored at 16
        self.assertEqual(p.choose(), 32)

    def test_reset_and_validation(self):
        p = ak.EmaThresholdK((8, 16, 32))
        run(p, [0] * 20)
        p.reset()
        self.assertIsNone(p.ema)
        self.assertEqual(p.choose(), 16)
        with self.assertRaises(ValueError):
            p.observe(16, 16)


class HazardTest(unittest.TestCase):
    def test_censoring(self):
        p = ak.HazardK((8, 16), prior=[0.5] * 15, prior_weight=1.0, decay=1.0)
        p.observe(7, 8)  # full block at 8: positions 1..7 accepted, nothing known beyond
        self.assertEqual(p.risk[:8], [1.0] * 7 + [0.0])
        p.observe(3, 16)  # positions 1..3 accepted, 4 rejected
        self.assertEqual((p.risk[3], p.acc[3]), (2.0, 1.0))  # reached twice, accepted once
        self.assertEqual((p.risk[4], p.acc[4]), (1.0, 1.0))  # first round only
        self.assertEqual((p.risk[7], p.acc[7]), (0.0, 0.0))  # position 8: never reached (censored)
        self.assertAlmostEqual(p.hazards()[3], (1 + 0.5) / (2 + 1))

    def test_expected_emitted_geometric(self):
        p = ak.HazardK((8, 16, 32), prior=[0.8] * 31, prior_weight=1.0)
        ee = p.expected_emitted()
        for k in (8, 16, 32):
            self.assertAlmostEqual(ee[k], (1 - 0.8 ** k) / 0.2, places=9)

    def test_steep_cost_prefers_short_flat_cost_prefers_long(self):
        prior = [0.6] * 31
        steep = ak.HazardK((8, 16, 32), cost=ak.CostCurve({8: 10.0, 16: 20.0, 32: 40.0}), prior=prior)
        flat = ak.HazardK((8, 16, 32), cost=ak.CostCurve({8: 10.0, 16: 10.0, 32: 10.0}), prior=prior)
        self.assertEqual(steep.choose(), 8)
        self.assertEqual(flat.choose(), 32)

    def test_learns_easy_request(self):
        cost = ak.CostCurve({8: 10.0, 16: 12.0, 32: 16.0})
        p = ak.HazardK((8, 16, 32), cost=cost, prior=[0.5] * 31, decay=0.95)
        self.assertEqual(p.choose(), 8)
        self.assertEqual(run(p, [40] * 60), 32)


class BatchAndControllerTest(unittest.TestCase):
    def test_allocate_batch_gives_width_to_easy_request(self):
        geo = lambda q: {k: (1 - q ** k) / (1 - q) for k in (8, 16, 32)}
        step = ak.CostCurve({64: 20.0, 128: 23.0, 256: 33.0})
        ks = ak.allocate_batch([geo(0.97)] + [geo(0.5)] * 7, (8, 16, 32), step)
        self.assertEqual(ks[1:], [8] * 7)
        self.assertGreater(ks[0], 8)

    def test_allocate_batch_flat_cost_is_max(self):
        ks = ak.allocate_batch([{8: 3.0, 16: 4.0, 32: 5.0}] * 4, (8, 16, 32), lambda t: 1.0)
        self.assertEqual(ks, [32] * 4)

    def test_controller_and_factory(self):
        c = ak.AdaptiveKController(lambda: ak.make_policy("ema:2.0", (8, 16, 32)))
        k = c.next_k("a")
        c.record("a", 0, k)
        self.assertIn(c.next_k("a"), (8, 16, 32))
        c.finish("a")
        self.assertNotIn("a", c.state)
        self.assertIsInstance(ak.make_policy("fixed:24"), ak.FixedK)
        self.assertIsInstance(ak.make_policy("hazard:0.8"), ak.HazardK)
        with self.assertRaises(ValueError):
            ak.make_policy("nope")
        self.assertEqual(ak.parse_menu("32, 8 16"), [8, 16, 32])

    def test_shadow_logger(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ak.jsonl")
            s = ak.ShadowLogger(path, "ema:3.0", (8, 16, 32))
            s.step(100, 16, 15)
            s.step(116, 16, 99)  # out-of-range accepted is clamped for the policy, logged raw
            s.step(50, 16, 0)  # prefix shrank -> new request -> policy reset
            s.f.close()
            rows = [json.loads(l) for l in open(path)]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["accepted"], 99)
            self.assertTrue(all(r["next_k"] in (8, 16, 32) for r in rows))


class SimulationSanity(unittest.TestCase):
    def test_online_ema_tracks_mixture(self):
        rng = random.Random(0)
        hard = [min(int(rng.expovariate(1 / 1.5)), 40) for _ in range(200)]
        easy = [min(int(rng.expovariate(1 / 25)), 40) for _ in range(200)]
        self.assertEqual(run(ak.EmaThresholdK((8, 16, 32)), hard), 8)
        self.assertEqual(run(ak.EmaThresholdK((8, 16, 32)), easy), 32)


if __name__ == "__main__":
    unittest.main()
