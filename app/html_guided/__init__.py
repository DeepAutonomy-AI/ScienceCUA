"""Opt-in semantic HTML acquisition; the legacy HTML inventory remains intact."""

from .pipeline import HTML_GUIDED_PLAN_PROMPT, build_snapshot, run_guided_html

__all__ = ["HTML_GUIDED_PLAN_PROMPT", "build_snapshot", "run_guided_html"]
