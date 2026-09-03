"""Focused checks for the recovered FUVVIEW3 background algorithms."""

import numpy as np

from icbuilder.fuview_background import (
    _bin_detector_image,
    _fill_array,
    _idl_median_filter,
    _median_arr,
    active_background_p,
    active_background_p2,
    fixed_background,
)


def reference_with_indices():
    dza, sza = np.indices((90, 110))
    return 1000 * dza + sza


def detector_block(size, first_angle=10):
    dza, sza = np.indices((size, size), dtype=float)
    return sza + first_angle, dza + first_angle


def test_fixed_lookup_rounds_before_applying_limits():
    reference = reference_with_indices()
    sza = np.array([0.49, 0.50, 109.49, 109.50, 5.0])
    dza = np.array([0.49, 0.50, 79.49, 79.49, 79.50])

    result = fixed_background(reference, sza, dza)

    np.testing.assert_allclose(
        result["background"][:3],
        [reference[0, 0], reference[1, 1], reference[79, 109]],
    )
    assert result["background"][3] == 0
    assert result["background"][4] == 0
    np.testing.assert_array_equal(
        result["support"], [True, True, True, False, False]
    )


def test_detector_binning_truncates_and_last_pixel_wins():
    image = np.array([2.0, 7.0])
    sza = np.array([10.2, 10.8])
    dza = np.array([20.9, 20.1])
    fit_mask = np.ones(2, dtype=bool)

    binned = _bin_detector_image(image, sza, dza, fit_mask, (90, 110))

    assert binned[20, 10] == 7.0
    assert np.count_nonzero(binned) == 1


def test_p_median_includes_zero_and_leaves_boundary_unchanged():
    values = np.zeros((13, 13))
    values[0, 0] = 7
    values[6, 6] = 9

    filtered = _idl_median_filter(values, width=9)

    assert filtered[0, 0] == 7
    assert filtered[6, 6] == 0


def test_p_recovers_scale_where_a_complete_nine_bin_window_exists():
    sza, dza = detector_block(9)
    image = np.full((9, 9), 2.0)
    mlat = np.full((9, 9), 50.0)
    reference = np.ones((90, 110))

    result = active_background_p(image, reference, sza, dza, mlat)

    assert result["scale"][14] == 2.0
    assert result["background"][4, 4] == 2.0


def test_p_continues_scale_and_background_from_sza100():
    dza, detector_column = np.indices((9, 10), dtype=float)
    dza = dza + 10
    sza = detector_column + 96
    sza[4, 9] = 120

    image = np.full(sza.shape, 2.0)
    mlat = np.full(sza.shape, 50.0)
    reference = np.ones((90, 110))
    reference[:, 101:] = 0

    historical = active_background_p(image, reference, sza, dza, mlat)
    result = active_background_p(
        image, reference, sza, dza, mlat, continue_high_sza=True
    )

    assert historical["background"][4, 9] == 0
    assert not historical["extrapolated"][4, 9]
    assert historical["method"] == "fuview_active_p"

    continuation_scale = result["scale"][100]
    np.testing.assert_allclose(
        result["scale"][100:], continuation_scale
    )
    assert result["background"][4, 9] == continuation_scale
    assert result["extrapolated"][4, 9]
    assert not result["support"][4, 9]
    assert result["background"][4, 8] == continuation_scale
    assert result["extrapolated"][4, 8]
    assert result["support"][4, 8]
    np.testing.assert_allclose(
        result["reference"][:, 100:],
        np.repeat(result["reference"][:, 100, None], 10, axis=1),
    )
    assert result["method"] == "fuview_active_p_edge"
    assert result["continuation_sza"] == 100


def test_p2_median_excludes_zero_and_uses_upper_even_median():
    values = np.zeros((3, 3))
    values[0, 1] = 1
    values[1, 0] = 2
    values[1, 1] = 4
    values[1, 2] = 3

    filtered = _median_arr(values, radius=1)

    assert filtered[1, 1] == 3
    assert filtered[0, 0] == 0


def test_p2_fill_runs_along_dza_then_sza_and_breaks_ties_low():
    values = np.array([
        [np.nan, 0.0, 3.0],
        [1.0, 0.0, np.nan],
        [np.nan, 0.0, 5.0],
    ])

    filled = _fill_array(values)

    np.testing.assert_allclose(
        filled,
        [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [1.0, 1.0, 5.0]],
    )


def test_p2_si_repair_uses_the_si_table_minimum_and_floor():
    sza, dza = detector_block(11)
    image = np.full((11, 11), 8.0)
    mlat = np.full((11, 11), 50.0)

    reference = np.zeros((90, 110))
    reference[:71, 80:] = 4.0
    result = active_background_p2(
        image, reference, sza, dza, mlat, sensor="SI13"
    )

    assert result["p2_si_minimum_repair"]
    assert result["background"][5, 5] == 8.0
    assert np.nanmin(result["reference"]) == 8.0


def test_background_functions_do_not_modify_the_reference():
    sza, dza = detector_block(11)
    image = np.full((11, 11), 2.0)
    mlat = np.full((11, 11), 50.0)
    reference = np.ones((90, 110))
    original = reference.copy()

    active_background_p(image, reference, sza, dza, mlat)
    active_background_p2(image, reference, sza, dza, mlat, sensor="WIC")

    np.testing.assert_array_equal(reference, original)
