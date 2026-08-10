# Proposed Modular Pipeline Redesign

Last reviewed: 2026-08-10
Status: Proposed by the user; architecture under discussion, not authorized for implementation

## Motivation

The current orbit pipeline combines sensor binning, proton correction,
electron energy and flux inference, Robinson conductance conversion, and
serialization in one processing path. This makes a change to one scientific
model look like a reason to regenerate every upstream result.

The user proposes a clean restart that reuses the verified calculations but
separates stable observations from replaceable scientific models. The goal is
not a wholesale rewrite: retain useful binning, response, Zhang--Paxton, and
Robinson calculations while giving each stage a clear input and output.

## Proposed product stages

### 1. Binned FUV observations

Write a model-independent per-orbit product containing the binned WIC, SI12,
and SI13 observations, count uncertainties, masks and support, timestamps,
grid coordinates, viewing geometry, and preprocessing provenance.

This product should contain observations and instrument preprocessing, not a
choice of electron-energy or conductance model. In particular, proton-corrected
WIC and SI13 depend on an assumed proton energy and response model; storing
them in the next stage prevents a changed proton assumption from invalidating
or disguising the reusable binned observations.

The three channels do not need to be forced onto identical frame support in
this product. Later methods can select the channels they require.

### 2. Precipitation inference

Read the binned product, perform the SI12-based proton correction, and store
the corrected WIC and SI13 signals together with electron energy and flux.
Support at least two explicit methods:

- `image_ratio`: infer E0 from corrected WIC/SI13 and then infer Fe;
- `zhang_paxton`: obtain E0 from Kp and MLT and infer Fe from corrected WIC,
  retaining corrected SI13 as an optional diagnostic rather than a required
  input to E0.

The methods should produce separate method-labelled products or directories
rather than silently replacing variables in one ambiguous file. This also
allows their frame support to differ: the ratio method needs WIC, SI12, and
SI13, whereas the Zhang--Paxton method principally needs WIC and SI12.

Each product should preserve method, coefficients or lookup provenance,
assumed proton energy, masks, units, uncertainties, covariance terms, and the
identity of its binned source product. Corrected WIC/SI13 belong here because
they make the energy/flux calculation inspectable without repeating binning.

### 3. Conductance forward model

Read a precipitation product and apply a named forward model to
`(E0, Fe)` and their uncertainty information. Robinson is the initial model,
but it should be replaceable without rerunning sensor binning or precipitation
inference.

The corresponding calculation must also exist as a small public importable
function, independent of orbit files and orchestration. A simple scientific
API should accept arrays and return ordinary arrays or a dictionary containing
Hall, Pedersen, and propagated uncertainty. Avoid factories or a generalized
plugin framework until a second model demonstrates what interface is actually
shared.

Conductance products should identify their precipitation source and forward
model. This yields the dependency chain:

```text
binned FUV
    -> precipitation/image_ratio
    -> precipitation/zhang_paxton
        -> conductance/robinson
        -> conductance/another_model
```

Changing a downstream model then regenerates only its own stage.

## Shared use by icAnalyzer

**Confirmed dependency constraint:** icAnalyzer must not depend on the full
icBuilder package. icBuilder requires pipeline packages such as `fuvpy` that
are not needed in the VAE environment and should not propagate into
icAnalyzer.

The numerical proton correction, camera-response conversion, energy/flux
inference, and conductance forward models should therefore live in a small
shared physics package below both projects, rather than in scripts,
`ConductanceImage` methods, or the icBuilder distribution. `icBuilder` and
`icAnalyzer` can then call exactly the same functions for orbit processing and
each VAE or Monte-Carlo realization.

The intended dependency direction is:

```text
ZhangPaxton2008
        |
        v
shared conductance/precipitation physics
        |                         |
        v                         v
icBuilder                     icAnalyzer
(fuvpy, ApexPy, I/O)          (VAE/analysis stack)
```

Keep `fuvpy`, ApexPy coordinate processing, file reading, NetCDF writing,
orbit iteration, Kp acquisition/matching, and multiprocessing in icBuilder.
The shared functions should have lightweight NumPy/SciPy-level dependencies
and make units, masks, and uncertainty inputs explicit. They may accept Kp as
a numerical input, but should not download or independently assign Kp to
IMAGE frames; icBuilder should serialize the matched value for icAnalyzer.

The shared package should remain small and scientific: ordinary public
functions, no file-product classes, framework, plugin system, or dependence on
either consuming project. ZhangPaxton2008 remains its own focused package;
the shared layer may depend on it rather than copying its equations.

## Probabilistic extension

The longer-term proposal is to train a joint WIC/SI12 VAE and pass sampled FUV
realizations plus sampled Zhang--Paxton E0 through the same precipitation and
conductance functions. The resulting Hall/Pedersen ensemble is the primary
probabilistic product; means, joint covariance blocks, low-rank forms, and
Gaussian approximations are derived representations.

The proposed common log-E0 amplitude perturbation is a transparent rank-one
model, but Zhang--Paxton latitude spread is not measured event-to-event model
error. Until calibrated, describe it as a profile-spread-derived
model-discrepancy or sensitivity prior, not a formal Zhang--Paxton predictive
uncertainty. Preserve the current arithmetic-mean nominal collapse, use
spherical-area weighting, avoid over-weighting wide MLT slices, retain Kp
dependence, and explicitly choose whether the nominal curve is the ensemble
mean or median.

The VAE likelihood must likewise treat current WIC/SI12 uncertainty estimates
as estimates requiring validation, distinguish latent auroral signal from
observation noise, and respect the channels' different effective resolutions.

## Scientific gates retained

- Confirm whether Zhang--Paxton mean energy is compatible with the energy
  definition used by the camera response and conductance model.
- Decide the Zhang--Paxton oval threshold and latitude domain before using its
  profile spread probabilistically.
- Define the assumed proton energy and its uncertainty explicitly.
- Decide how zeros, background, and missing support are represented in
  log-conductance statistics.
- Preserve the Hall--Pedersen cross-covariance rather than calculating only
  separate Hall and Pedersen covariances.
- Do not treat a standard VAE encoder as a calibrated partial-image posterior;
  sparse conditioning needs an explicit masked-observation inference method.

## Next design decision

Decide whether to adopt the three-stage product boundary above. If accepted,
the smallest prototype is one orbit processed as:

1. one reusable binned-FUV file;
2. separate ratio and Zhang--Paxton precipitation files;
3. a Robinson conductance file generated from one precipitation file;
4. direct calls to the same precipitation and Robinson functions from a small
   icAnalyzer-side test.

No full-corpus regeneration or VAE redesign should begin until this boundary
and the variables owned by each stage are agreed.
