# ScienceCUA project website

This directory contains the complete static paper website, including figures,
result tables, English and Chinese videos, and the interactive scientific
background on the homepage. The title, authors, affiliations, and paper content
retain their original layout.

## Preview locally

From the repository root, run:

```sh
python -m http.server 4177 --bind 127.0.0.1 --directory website
```

Then open <http://127.0.0.1:4177/>. No build step or additional packages are
required. Opening `website/index.html` directly also displays the page.

## Editing

- `index.html`: paper content, figures, tables, and links.
- `styles.css`: page layout and visual styling.
- `hero-background.js`: procedural scientific illustrations and pointer glow.
- Other JavaScript files: dialogs, knowledge examples, result interactions,
  and video controls.
- `assets/`: images, PDFs, data files, and videos.

The background respects the operating system's reduced-motion preference and
pauses when the hero is outside the viewport or the browser tab is hidden.

These files are independent of the research application in `app/`. The existing
[public project page](https://deepautonomy-ai.github.io/ScienceCUA-website/) is
published from a separate repository; adding this directory does not update
that deployment.
