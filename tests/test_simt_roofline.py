"""Check units, architectural differences, and unknown SIMT ceilings."""

import unittest

from llm_optimizer.predefined.gpus import get_gpu_specs, get_precision_tflops


class TestSIMTRoofline(unittest.TestCase):
    def test_h100_fma_rate_is_directly_in_published_flops(self):
        simt = get_gpu_specs("h100")["SIMT"]
        rates = simt["TeraOpsPerSecond"]
        self.assertAlmostEqual(rates["fp32"]["FMA"], 66.90816)
        self.assertEqual(rates["fp32"]["FMA"], 2 * rates["fp32"]["AddOrMultiply"])
        self.assertAlmostEqual(rates["fp16x2"]["FMA"], 133.81632)
        self.assertAlmostEqual(rates["fp32"]["Exp"], 4.18176)
        # One GFLOP of FP32 FMA work at this ceiling takes ~0.015 ms.
        self.assertAlmostEqual(1e9 / (rates["fp32"]["FMA"] * 1e9), 0.01494585974)

    def test_packed_rates_depend_on_architecture_and_format(self):
        ampere = get_gpu_specs("A100")["SIMT"]["OpsPerSMCycle"]
        hopper = get_gpu_specs("H100")["SIMT"]["OpsPerSMCycle"]
        blackwell = get_gpu_specs("B200")["SIMT"]["OpsPerSMCycle"]
        ultra = get_gpu_specs("B300")["SIMT"]["OpsPerSMCycle"]
        self.assertEqual(ampere["fp16x2"]["FMA"], 512)
        self.assertEqual(ampere["bf16x2"]["FMA"], 256)
        self.assertEqual(hopper["fp16x2"]["FMA"], 512)
        self.assertEqual(blackwell["fp16x2"]["FMA"], 256)
        self.assertEqual(ultra["bf16x2"]["FMA"], 256)

    def test_conversion_direction_and_packed_shuffle(self):
        rates = get_gpu_specs("H100")["SIMT"]["OpsPerSMCycle"]
        self.assertEqual(rates["fp16x2"]["ConvToFP32"], 64)
        self.assertEqual(rates["bf16x2"]["ConvToFP32"], 32)
        self.assertEqual(rates["fp16x2"]["ConvFromFP32"], 16)
        self.assertEqual(rates["fp32"]["ConvToFP16"], 16)
        self.assertEqual(rates["fp16x2"]["Shuffle"], 2 * rates["fp32"]["Shuffle"])

    def test_unknowns_are_not_fabricated(self):
        for gpu in ("H100", "B200", "B300"):
            rates = get_gpu_specs(gpu)["SIMT"]["TeraOpsPerSecond"]
            for precision in ("fp16x2", "bf16x2"):
                self.assertIsNone(rates[precision]["Exp"])
                self.assertIsNone(rates[precision]["Rcpf"])
        for gpu in ("B100", "L20"):
            simt = get_gpu_specs(gpu)["SIMT"]
            self.assertIsNone(simt["SM_Clock_MHz"])
            self.assertIsNone(simt["TeraOpsPerSecond"]["fp32"]["FMA"])
            self.assertGreater(simt["OpsPerSMCycle"]["fp32"]["FMA"], 0)

    def test_blackwell_equivalent_clocks_and_tensor_values(self):
        for gpu in ("B200", "B300"):
            simt = get_gpu_specs(gpu)["SIMT"]
            self.assertIn("Inferred", simt["Clock_Basis"])
            self.assertAlmostEqual(simt["TeraOpsPerSecond"]["fp32"]["FMA"], 75)
            self.assertEqual(get_precision_tflops(gpu, "fp16"), 2250)

    def test_returned_nested_specs_are_isolated(self):
        specs = get_gpu_specs("H100")
        specs["SIMT"]["TeraOpsPerSecond"]["fp32"]["FMA"] = 0
        self.assertGreater(
            get_gpu_specs("H100")["SIMT"]["TeraOpsPerSecond"]["fp32"]["FMA"], 0
        )


if __name__ == "__main__":
    unittest.main()
