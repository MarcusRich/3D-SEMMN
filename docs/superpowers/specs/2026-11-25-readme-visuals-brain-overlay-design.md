# README visuals and a brain overlay of the column mapping

Date: 2026-11-25. Status: built the same day (Marcus asked to see the pictures directly; this records the design).

## Intent

3D-SEMMN's purpose is an EEG-to-language system that gives nonspeaking autistic people a voice. The model is meant to be the brain-mappable layer between EEG and an LLM. Today the repository explains this only in text, and its README embeds no images. Marcus wants anyone, not just specialists, to understand the idea and the progress from pictures. The pictures must be pretty, meaningful and grounded in the real research.

**The target, in Marcus's words:** pictures that are "reasonably meaningful and only slightly misrepresenting the reality". The model for this is how math gets taught. The simplified picture is not exact, and someone who knows enough to see the difference also understands why it was simplified. So every figure simplifies on purpose, and it names its simplification for the expert in a one-line caption.

**Audience:**
- **Primary:** curious non-specialists, such as families and the AAC community, potential collaborators, funders and engineers from other fields.
- **Secondary:** EEG and neuroscience readers, who will check the captions.

## Goals

1. The README tells the story top-down in pictures, one idea per figure, in plain language.
2. A brain overlay shows how the 100 model columns map onto a real brain. It also shows how electrodes, word vectors and the model's low-dimensional manifold relate to those columns.
3. Each figure carries one of three badges:
   - **Measured** (from this repository's EEG results).
   - **Planned** (from the roadmap design).
   - **Illustration** (a concept sketch).
   Each figure also gets a plain caption plus one "What's simplified" line.
4. Figures are reproducible. One script regenerates every image from repository data and public atlases, with fixed seeds.
5. Images render on GitHub with no extra hosting.

## Non-goals

- No new science and no new claims. Figures show existing results (P1, P3, and P4/P4b once merged) and the roadmap's planned design.
- The model's columns are not changed to anatomical ones. That is roadmap work (§4.1), and this work only pictures it.
- No source localization is run on real EEG. It is too heavy for this laptop, and the roadmap says 8 to 16 channels can't support it.

## Anti-goals (a failure even if the pictures look good)

- **No implied mind-reading.** No figure may suggest the system already reads thoughts or decodes speech.
- **No implied precision.** No figure may suggest that 8 to 16 electrodes resolve 100 separate brain regions. The overlay is labelled Planned, and network-level colouring carries the message.
- **No product impression.** Nothing should look like a working product to a family reading it. The "user confirms every message" step and the authorship point stay in the story.
- **No live or personal data.** Nothing personal and no raw EEG goes into the repository. Broderick 2018 data stays outside it, as today.

## The figures (in README order)

| # | Title (plain language) | Badge | Content | What's simplified |
|---|---|---|---|---|
| F1 | The big idea | Illustration | A left-to-right strip: a person with an EEG cap, then brain waves, then brain regions, then model columns, then a few-dimensional "meaning map", then the LLM's candidate words, then the person confirms, then a voice. | The real pipeline has filtering, timing and many candidates per step. |
| F2 | Where the model's columns live on the brain | Planned | An inflated brain in lateral and medial views, both hemispheres. The 100 Schaefer-100 regions are coloured by their 7 networks, with one dot per column at the region's centre. A legend ties today's toy areas (vision, hub, audio) to the networks they stand in for. | Today the columns are a flat 10 × 10 grid, not brain regions. This is the roadmap's mapping. |
| F3 | What a headset can hear | Measured | A top-down head. The 128 lab electrodes are shown faint, with the 8- and 16-channel starter headsets highlighted. Colour shows where the measured P3 models predicted the EEG best (envelope, and the gain from word features). | Values are smoothed between electrodes. Region-level means stand in where per-channel values aren't stored. |
| F4 | From electrodes to brain networks to columns | Illustration (built from the planned mapping) | Lines from each starter electrode to the brain networks closest beneath it, with thicker lines for closer regions. Those networks are then linked to the columns they feed. | Every electrode hears the whole brain a little. Real sensitivity is blurry, and this shows only the strongest share. |
| F5 | How words show up in brain waves | Measured | (a) A sentence timeline with each word's surprise (from the LM features) and its word-vector "meaning" as a colour strip. (b) The measured N400 dip: the word-feature response at 300 to 500 ms, mean ± spread over 19 subjects, for the 128-channel cap and each starter headset. | Averaged over people and words; any single word's response is buried in noise. |
| F6 | From 100 columns to a small meaning map | Illustration | Column activity (100 values) folding down into a curved low-dimensional sheet (the manifold), with points coloured by category. | The repository's manifold is from the toy audio-visual digit task, not EEG language data. |
| F7 | Honest scoreboard | Measured | Plain bars. What works so far: word features add prediction, and the N400 is found with 8 channels. Once P4 is on main, also: true words outrank LLM distractors more than shuffled EEG does. What doesn't yet: on the toy task SEMMN scores 0.88 ± 0.13 over seeds against 0.92 for a plain MLP, and it collapses to 0.11 when its connections are cut (`outputs/eval-small/results.md`). | Effect sizes are small, and the captions say how small in plain words. |

## Approaches considered

- **A. Data-generated static figures (chosen for all data figures).**
  - What: a Python script draws the PNG figures from repository results and public atlases with matplotlib and nilearn, then commits them. The README embeds them.
  - Pros: renders everywhere on GitHub, reproducible, and honest because the pictures come from the numbers.
  - Cons: no rotating or hovering.
- **B. A + an interactive rotatable 3D brain (a later slice, not in this design).**
  - What: one plotly page published with GitHub Pages and linked from F2.
  - Pros: the most engaging.
  - Cons: needs a repository settings change (Pages), mesh data on the page, and more build. Worth doing after A lands.
- **C. Hand-drawn diagrams.**
  - Pros: the prettiest and simplest to read.
  - Cons: not tied to the data, which drifts past "slightly misrepresenting".
  - Used only for F1, the concept strip, where there is no data to tie to.

**Pick:** A for F2 to F7, the C style for F1, and B as a follow-up. Marcus handed over the how; this pick is Claude's call.

## Decided

- **Atlas:** Schaefer-100 (7 networks) on fsaverage5 through nilearn, matching roadmap §4.1. The atlas and mesh are downloaded at build time and never committed. Only the rendered PNGs are committed.
- **Electrode positions:** BioSemi-128 and the 10-20 positions already in `semmn/eeg/montages.py`. A template head for scalp drawings.
- **Style:** one consistent palette with network colours fixed across figures, a light background that also reads in GitHub dark mode, large type and short titles.
- **Captions:** each figure has a plain caption plus "What's simplified:" in the README.

## Settled after the builder check (Claude's calls unless marked)

- **Column-to-region assignment (F2):** planned column *i* is Schaefer-100 parcel *i*, at the parcel centroid. This is the roadmap §4.1 rule, one column per parcel (`~/Developer/personal/notes/3d-semmn-architecture-roadmap.md`, not in the repo). The figure says "planned", and it does not place today's 10 × 10 grid columns on the brain.
- **Toy areas to networks (F2 legend):**
  - vision → Visual.
  - audio → Somatomotor. Auditory cortex sits in Schaefer-7's SomMot network; the caption says so.
  - hub → Default + Frontoparietal, per roadmap §4.1 item 6.
- **F3 data:** the committed summaries `outputs/eeg/scalp_forward.md` (region means, top-10 channels) and `summary.md` (montage means). Per-channel values are not recomputed: that would need the full encode run on the laptop. The figure colours scalp regions, not single electrodes. That's the named simplification, and the badge stays Measured.
- **F4 sensitivity:** a geometric proxy. Each starter electrode's three nearest parcels are found on the template head, and lines are weighted by inverse distance. It's labelled Illustration and says it is not a lead field. Network to column follows the F2 assignment.
- **F5 data:**
  - (a) one sentence from Broderick run 1, with word surprisal and embedding taken from the local LM cache (`lm_local.npz`, outside the repository). Only the words and two derived numbers per word go into the figure. The text is the public CC0 story transcript.
  - (b) the committed per-subject N400 kernels (`n400_trf_per_subject_forward.csv`, one value per subject and montage) as dots, over a textbook N400-shaped curve marked as a sketch. No time-course data is committed, so the curve shape is Illustration and the dots are Measured.
- **F1:** a flat schematic icon style (simple shapes, matplotlib or SVG), not illustration-grade drawing.
- **Output format:**
  - PNGs at 2× for sharpness, each under 400 KB and under 3 MB total.
  - Opaque light panels, so they read in GitHub dark mode.
  - Matplotlib on the Agg backend.
  - Generated into `docs/figures/` by `make_figures.py`.
  - Builds run through `~/Developer/rex-inbox/heavy` (the laptop rule).

## Constraints

- **Repository and pushes:** personal public repository `MarcusRich/3D-SEMMN`. Pushes go through the `github-marcusrich` SSH alias.
- **Branches:** named per `BRANCHING.md`, as `feature/<issue>-readme-visuals`, with a GitHub issue opened first. Work lands by PR, and Marcus merges.
- **Laptop heat:** figure builds run through `~/Developer/rex-inbox/heavy`, one at a time, with small meshes (fsaverage5).
- **Dependencies:** nilearn (BSD) is a new dependency for the figure script only, in a separate `requirements-figures.txt`, so training and tests don't need it.
- **Data:** Broderick 2018 (CC0) stays outside the repository. Figures read committed summaries in `outputs/eeg/`. Per-channel values, where needed, come from the cached arrays next to the data. If those values aren't cached, the summary tables are used instead.
- **P4/P4b results:** these live on unmerged branches. F7 uses only results on `main`. Opening PRs for P4/P4b is a separate step.

## Left to the builder

- Exact layout, colours within the palette, and figure sizes.
- Whether F3 uses true per-channel values or region means, depending on what is cached.
- How F4 picks the "closest networks" (for example, the nearest parcels to each electrode projected onto the template cortex).
- The F1 drawing technique (matplotlib shapes or a hand-written SVG).
- The README's section wording, in plain language, keeping the existing technical sections lower down.
