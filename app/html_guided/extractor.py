"""Guided-only preservation of block links and authored math fallbacks.

The shared extractor owns the AST, whitespace and recovery rules. This small
extension records the existing enclosing anchor when its block children are
visited separately, without flattening headings, code, images or card content.
Visible mathematical fractions may use their existing hidden linear spelling;
this never invents operators or reveals unrelated hidden content.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import Comment, NavigableString, Tag

from ..html_sourcepart.extractor import BLOCK, Extractor, all_runs


class GuidedExtractor(Extractor):
    PROFILE = "html-guided-preservation.v2"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._block_anchor_links: dict[int, str] = {}
        self._math_fraction_records: list[dict] = []

    def initialize_visibility(self, soup):
        super().initialize_visibility(soup)
        for fraction in soup.select('.fraction'):
            owners = [fraction, *fraction.parents]
            if any(owner.name in {'pre', 'code', 'textarea', 'listing'} for owner in owners):
                continue
            if not any({'formula', 'math'} & set(owner.get('class', [])) for owner in owners):
                continue
            # Only the linear spelling of an otherwise visible mathematical
            # fraction may be exposed. A hidden equation remains hidden.
            if any(id(owner) in self.hidden_nodes for owner in owners):
                continue
            children = list(fraction.children)
            numerator = [node for node in children if isinstance(node, Tag)
                         and 'numerator' in node.get('class', [])]
            denominator = [node for node in children if isinstance(node, Tag)
                           and 'denominator' in node.get('class', [])]
            if len(numerator) != 1 or len(denominator) != 1:
                continue
            first, second = children.index(numerator[0]), children.index(denominator[0])
            tokens = []

            def punctuation(nodes):
                text = ''
                for node in nodes:
                    if isinstance(node, Comment):
                        continue
                    if isinstance(node, NavigableString):
                        text += str(node)
                    elif isinstance(node, Tag) and node.name == 'span' and not node.find(True):
                        text += node.get_text()
                        tokens.append(node)
                    else:
                        return None
                return re.sub(r'\s+', '', text)

            spelling = (punctuation(children[:first]), punctuation(children[first + 1:second]),
                        punctuation(children[second + 1:])) if first < second else None
            complete = spelling in {('(', ')' + slash + '(', ')') for slash in {'/', '⁄', '∕'}} | {
                ('', slash, '') for slash in {'/', '⁄', '∕'}}
            complete = complete and all(node.get_text(strip=True) for node in [numerator[0], denominator[0]])
            complete = complete and not any(id(node) in self.hidden_nodes
                                            for node in [numerator[0], denominator[0]])
            markup = str(fraction)
            record = {'dom_path': self.path(fraction), 'status': 'authored_fallback_available' if complete else 'unsupported',
                      'raw_markup': markup[:4_000], 'raw_markup_truncated': len(markup) > 4_000,
                      'raw_sha256': hashlib.sha256(markup.encode('utf8', errors='surrogatepass')).hexdigest()}
            if complete:
                restored = [node for node in tokens if id(node) in self.hidden_nodes]
                for node in restored:
                    self.hidden_nodes.pop(id(node))
                record.update(status='authored_fallback_restored' if restored else 'authored_fallback_visible',
                              restored_tokens=[{'dom_path': self.path(node), 'text': node.get_text()}
                                               for node in restored])
            else:
                record['reason'] = 'missing_or_ambiguous_authored_linear_fraction_fallback'
            self._math_fraction_records.append(record)

    def _extract_structured(self, html_text: str, max_chars: int = 10_000) -> dict:
        document = super()._extract_structured(html_text, max_chars)
        if self._math_fraction_records:
            document['guided_math_fractions'] = list(self._math_fraction_records)
        return document

    def extract(self, html_text: str, max_chars: int = 10_000) -> dict:
        document = super().extract(html_text, max_chars)
        unsupported = [record for record in document.get('guided_math_fractions', [])
                       if record['status'] == 'unsupported']
        if unsupported:
            reason = {'code': 'guided_math_fraction_unsupported', 'count': len(unsupported),
                      'dom_paths': [record['dom_path'] for record in unsupported]}
            document['warnings'].append(reason)
            document['extraction']['degraded'] = True
            document['extraction']['reasons'].append(reason)
            for part in document['parts']:
                part['metadata']['degraded'] = True
        return document

    @staticmethod
    def _block_anchor(node: Any) -> Tag | None:
        if isinstance(node, Tag) and node.name == "a" and node.has_attr("href"):
            anchor = node
        else:
            parent = getattr(node, "find_parent", None)
            anchor = parent("a", href=True) if callable(parent) else None
        if isinstance(anchor, Tag) and anchor.find(list(BLOCK)) is not None:
            return anchor
        return None

    def block(self, kind: str, node: Any, **kwargs: Any) -> dict:
        block = super().block(kind, node, **kwargs)
        anchor = self._block_anchor(node)
        if anchor is None:
            return block
        link_id = self._block_anchor_links.get(id(anchor))
        if link_id is None:
            link_id = self.link(anchor)
            if link_id is None:
                return block
            self._block_anchor_links[id(anchor)] = link_id
        for run in all_runs([block]):
            if run.get("type") in {"text", "image"} and not run.get("link_id"):
                run["link_id"] = link_id
        block["enclosing_link_id"] = link_id
        return block

    def sequence(self, nodes) -> list[dict]:
        # A card footer is authored link content, whereas the shared policy
        # intentionally omits page footers. Visit its original children without
        # mutating the DOM, its attributes, CSS ancestry or source locators.
        expanded = []
        for node in nodes:
            if (isinstance(node, Tag) and node.name == "footer"
                    and self._block_anchor(node) is not None
                    and not any(id(owner) in self.hidden_nodes for owner in [node, *node.parents])):
                expanded.extend(node.contents)
            else:
                expanded.append(node)
        return super().sequence(expanded)
