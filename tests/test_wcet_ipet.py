"""Sanity tests for the IPET WCET solver."""
import pytest

from src.wcet_ipet import _build_layer_cfg, solve_ipet_scipy, solve_ipet_pulp, analyze


def test_scipy_pulp_agree_small_layer():
    layer = _build_layer_cfg("lif_layer3", n_in=32, n_out=2)
    scipy_bound = solve_ipet_scipy(layer)
    pulp_bound = solve_ipet_pulp(layer)
    assert scipy_bound == pytest.approx(pulp_bound, abs=1e-3)


def test_larger_layer_has_larger_bound():
    small = _build_layer_cfg("lif_layer3", n_in=32, n_out=2)
    large = _build_layer_cfg("lif_layer1", n_in=11776, n_out=64)
    assert solve_ipet_scipy(large) > solve_ipet_scipy(small)


def test_extract_cfg_rejects_spurious_leading_loop():
    """Regression test for issue: a `for` loop appearing before the real
    outer/inner pair must not be silently bound as the outer loop -- it
    must raise instead of producing a wrong n_out."""
    from src.wcet_ipet import extract_cfg
    import pytest as _pytest

    c_source = """
    void lif_layer1_step(const float *in, const float *w, float *out) {
        for (int i = 0; i < 999; i++) { }
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 11776; j++) { }
        }
    }
    """
    with _pytest.raises(ValueError):
        extract_cfg(c_source)


def test_extract_cfg_correct_on_well_formed_fallback_codegen_output():
    """The fallback codegen's actual output shape must still extract
    correctly after the stricter validation."""
    from src.wcet_ipet import extract_cfg

    c_source = """
    void lif_layer1_step(const float *in, const float *weight, float *out) {
        for (int i = 0; i < 64; i++) {
            float acc = 0.9f * v_lif_layer1[i];
            for (int j = 0; j < 11776; j++) {
                acc += weight[i * 11776 + j] * in[j];
            }
            int spike = acc >= 1.0f;
            out[i] = (float) spike;
            v_lif_layer1[i] = acc * (1.0f - (float) spike);
        }
    }
    """
    layers = extract_cfg(c_source)
    assert len(layers) == 1
    assert layers[0].n_out == 64
    assert layers[0].n_in == 11776


def test_analyze_uniform_vs_per_layer():
    c_source = """
    void lif_layer1_step(const float *in, const float *w, float *out) {
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 11776; j++) { }
        }
    }
    void lif_layer2_step(const float *in, const float *w, float *out) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 64; j++) { }
        }
    }
    void lif_layer3_step(const float *in, const float *w, float *out) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 32; j++) { }
        }
    }
    """
    uniform = analyze(c_source, uniform_bound=True)
    per_layer = analyze(c_source, uniform_bound=False)
    assert uniform["network_bound_cycles"] >= per_layer["network_bound_cycles"]
    assert len(set(uniform["per_layer_applied"].values())) == 1
