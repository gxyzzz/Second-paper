# Visual QA Report

This report records the final plotting-layer quality gate after the visual redesign.

## Inspection method

Each final PNG was read from disk after rendering. Checks combined:

- exact raster dimensions at 400 dpi;
- non-white content bounds / clipping margin audit;
- grayscale raster rendering at reduced manuscript-like size;
- PDF page-size and text-layout extraction;
- semantic review against the question defined in `VISUAL_REDESIGN_PLAN.md`.

The review did not modify scientific data.

## Per-figure review

| Figure | 3-second message | Plot/data fit | Grayscale redundancy | Label/legend audit | Result |
|---|---|---|---|---|---|
| Fig01 progression | two modules move every domain progressively rightward | sequential stages → progression | open circle / filled circle / diamond + labels | legend moved above data; direct gains retained | PASS |
| Fig02 forest | PDBR is positive with CI above zero in all domains | estimate + CI → forest | whisker + diamond; no color dependence | CI text fits page; zero reference retained | PASS |
| Fig03 heatmaps | gains are broad across datasets and four primary metrics | 3×4 matrices → heatmaps | every cell has numeric text | redundant colorbars removed; two panels remain compact | PASS |
| Fig04 lambda | Validation-Test gap varies across four discrete lambda settings | paired discrete values → dumbbell | filled circle vs open square | legend moved above data; formal parameter location subtle | PASS |
| Fig05 margin | 0.25 is strongest paired Validation/Test setting in this sweep | paired discrete values → dumbbell | filled circle vs open square | legend moved above data | PASS |
| Fig06 theta | Validation/Test behavior differs across four angular bounds | paired discrete values → dumbbell | filled circle vs open square | no connecting trend line or interpolation | PASS |
| Fig07 preference gap | TRUE condition stays well above SHUFFLED | paired trajectories + separation → gap/ribbon | solid vs dashed + direct labels | selected epoch annotation uses epoch 23, not later cherry-pick | PASS |
| Fig08 band alignment | TRUE and SHUFFLED are strongly separated in all three bands | paired band summaries → dumbbell | filled circle vs open square + direct % labels | legend above data | PASS |
| Fig09 band refinement | absolute movement decreases Easy → Medium → Hard | ordered scalar → lollipop | position and text labels sufficient | interpretation remains bounded movement, not sample importance | PASS |
| Fig10 efficiency | PDBR reaches best/stop points earlier while max budget remains 60 | best/stop/max → timeline | circle vs square + row labels | best/stop text explicit; max=60 reference explicit | PASS |

## Global gate

- [x] scientific question is legible quickly;
- [x] plot type follows data geometry;
- [x] no decorative gradient, shadow, 3D, or fake smoothing;
- [x] semantic color system is consistent;
- [x] grayscale remains interpretable through redundant encodings;
- [x] labels are concise and legends avoid the data region;
- [x] information density is higher than the replaced bar/line set;
- [x] discrete sensitivity points are not drawn as continuous functions;
- [x] only real bootstrap CI is visualized as uncertainty;
- [x] physical figure sizes are figure-specific and manuscript-oriented;
- [x] vector PDF and 400-dpi PNG are generated;
- [x] no raster clipping detected.
