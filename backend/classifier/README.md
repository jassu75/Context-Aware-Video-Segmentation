# Backend Classifier Scorer Guide

The backend classifier is modular. Each non-content category should live in
its own scorer file inside `backend/classifier/`.

Examples:

- `scorer_ad_break.py`
- `scorer_sponsor.py`
- `scorer_self_promo.py`
- `scorer_recap.py`
- `scorer_intro.py`
- `scorer_outro.py`
- `scorer_dead_air.py`

Each scorer file is responsible for defining one segment type. It should
export:

```python
LABEL = "category_name"
```

and a score function:

```python
def score(audio_data, text_data, scene_data, video_data, scores, debug=False):
    ...
```

The classifier passes every scorer the same modality JSON data:

- `audio_data` from `audio_processor.py`
- `text_data` from `text_processor.py`
- `scene_data` from `sceneDetector.py`
- `video_data` from `video_processor.py`

Each scorer assigns raw rule points for every analysis window. The scorer
should also export `MAX_POINTS`, which is the number of points possible if all
rules pass. The classifier normalizes each scorer's raw points onto a shared
0-10 scale, then compares labels. A normalized score greater than or equal to
4 means the window is likely that non-content type. The classifier then picks
the highest scoring label per window and merges adjacent windows into final
timeline segments.

Important: scorer files should not directly output final segments. They should
score windows. The classifier handles combining, tie-breaking, smoothing, and
writing the final timeline JSON.

## Basic Scorer Contract

```python
LABEL = "self_promo"
MAX_POINTS = 10.0


def score(audio_data, text_data, scene_data, video_data, scores, debug=False):
    results = []

    for i, row in enumerate(scores):
        s = 0.0

        # Inspect audio/text/scene/video features here.
        # Add points based on your definition/rules.
        # Add raw rule points. MAX_POINTS should equal all possible points.

        row[LABEL] = round(s, 2)

        results.append({
            "window_index": row["window_index"],
            "label": LABEL,
            "score": round(s, 2),
            "debug": None,
        })

    return results
```

## Adding a New Scorer

When adding a new scorer, make sure:

1. The file is named `backend/classifier/scorer_<label>.py`.
2. `LABEL` matches the segment type string.
3. `MAX_POINTS` equals the total raw points available if all rules pass.
4. The scorer is listed in `classifier.py` under `_EXPECTED_SCORERS`.
5. The label is added to `CATEGORY_PRIORITY` if tie-breaking matters.
6. If evaluating it, add it to the evaluation script or use
   `eval_category.py` with that label.

If a scorer supports debug details, return them under the optional `debug`
field for each window when `debug=True`. If not, accepting the parameter and
returning `None` is fine.

## Definition Guidance

A good scorer should define:

- What this segment type means.
- What signals prove it.
- What signals are only supporting evidence.
- What should not count as this category.

### Self-Promotion / Channel Promotion

Strong evidence:

- Subscribe prompts.
- Like/comment/share calls to action.
- Follow me/us on another platform.
- Join Discord, Patreon, membership, or community.
- Support the channel.
- Merch mentions.
- Link in description when it refers to the creator/channel.

Supporting evidence:

- Near the intro or outro.
- Host speech.
- Sustained host shot.
- End-card or social-handle visuals.

Should not count:

- Phrases like "like this" in normal content.
- Generic intro/outro language without a channel promotion.

### Recap / Repeated Boilerplate

Strong evidence:

- "Previously on..."
- "Last episode..."
- "Quick recap..."
- "To summarize..."
- "Last time we covered..."
- Repeated transcript blocks that appear far apart.

Supporting evidence:

- Near the beginning of the video.
- Narration pace.
- Moderate montage-like shot density.
- Reused boilerplate across videos or sections.

Should not count:

- Casual filler like "anyway."
- Normal content phrases like "last time" unless clearly recap-framed.
- Repeated sounds or single-word fragments without meaningful boilerplate.

## Main Rule

Build each scorer as a window scorer, not as a separate segment generator.
Scorers can use natural raw point totals internally, but they must declare
`MAX_POINTS` so the classifier can normalize scores before comparing labels.
The classifier still uses `CATEGORY_PRIORITY` to break ties if normalized
scores are equal. This keeps every team's work compatible with the shared
classifier and lets the classifier piece all scorer outputs together into one
final timeline JSON for the video player.
