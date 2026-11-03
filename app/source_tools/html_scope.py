"""Chapter-limited HTML acquisition; the entry page is outside the child budget."""
from __future__ import annotations

from collections import Counter, deque
import posixpath
from urllib.parse import unquote, urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from ..html_navigation import classify_html_link, html_base_url, is_authored_contents
from ..html_sourcepart.scope import flatten_sections, page_url, selected_sections


def selected_html_inventory(executor, *, max_pages=None):
    workspace, settings, store = executor.workspace, executor.settings, executor.store
    manifest = workspace.manifest
    scope = manifest.source_metadata['extraction_scope']
    selected = selected_sections(scope)
    child_budget = max(0, int(settings.max_html_pages))
    if manifest.source_origin == 'websearch':
        child_budget = min(child_budget, max(0, int(settings.websearch_html_max_pages)))
    if max_pages is not None:
        child_budget = min(child_budget, max(0, int(max_pages)))
    root_url = manifest.final_url or manifest.origin_url or f'artifact://{manifest.source_id}/root'
    root_url = page_url(root_url)
    raw_id = executor._require_raw_artifact()
    selected_ids = {node['section_id'] for node in selected}
    chapter_by_id = {node['section_id']:node for node in selected}
    root_directory = posixpath.dirname(urlparse(root_url).path).rstrip('/') + '/'
    known_owners, targets, queues, unselected_targets = {}, {}, {}, {}
    for chapter in scope['sections']:
        for node in flatten_sections([chapter]):
            if node.get('url'):
                known_owners.setdefault(page_url(node['url']), set()).add(chapter['section_id'])
                if chapter['section_id'] not in selected_ids:
                    unselected_targets.setdefault(page_url(node['url']), []).append(node)
    for chapter in selected:
        queue, queued = deque(), set()
        for node in flatten_sections([chapter]):
            if not node.get('url'):
                continue
            url = page_url(node['url'])
            targets.setdefault(url, []).append(dict(node))
            if url != root_url and url not in queued:
                queue.append({'url':url, 'depth':1, 'parent_url':root_url})
                queued.add(url)
        queues[chapter['section_id']] = queue
    preview_urls = set(targets)

    def refresh_exclusions(url):
        page_targets = targets.get(url, [])
        whole_page = any(not target.get('fragment') and not target.get('heading_dom_path') for target in page_targets)
        excluded = [target for target in unselected_targets.get(url, [])
                    if whole_page or target.get('fragment') or target.get('heading_dom_path')]
        for target in page_targets:
            target['excluded_targets'] = excluded

    for url in targets:
        refresh_exclusions(url)

    # A scope artifact exposes the selected hierarchy, never the unselected body.
    groups = [{'group_id':chapter['section_id'], 'label':chapter['title'],
               'labels':[chapter['title']], 'link_count':chapter.get('known_page_count', 0)} for chapter in selected]
    root_structure = {
        'schema_version':'root-structure.v2', 'source_id':manifest.source_id,
        'root_url':root_url, 'document_title':manifest.display_name,
        'primary_region_tag':None, 'headings':[], 'disclosures':[],
        'python_visual_group_candidates':groups,
        'nested_link_trees':[{'tag':'ul', 'context_label':None, 'heading_node_id':None,
                             'authored_contents':True, 'items':[_structure_item(chapter) for chapter in selected]}],
        'instructions':'Only the user-selected chapter hierarchy is included. Preserve its authored order.',
        'extraction_scope':{'mode':'selected', 'section_ids':list(scope['section_ids'])},
    }
    structure_artifact = store.put_json_artifact(root_structure, kind='html_root_structure',
                                                source_id=manifest.source_id, preserve_text_integrity=True)
    workspace.root_structure_artifact_id = structure_artifact.artifact_id
    workspace.allowed_artifacts.add(structure_artifact.artifact_id)
    root_soup, root_part, root_page, _sections = executor._materialize_html_page(
        page_url=root_url, data=store.artifact_path(raw_id).read_bytes(), artifact_id=raw_id,
        ordinal=1, navigation_depth=0, capability_branch=None, parent_url=None,
        scope_targets=targets.get(root_url, []))
    navigation_scope = executor._infer_tutorial_scope(root_soup, root_url)
    root_part.metadata.update(root_structure_artifact_id=structure_artifact.artifact_id,
                              navigation_groups=groups)
    root_page.update(root_structure_artifact_id=structure_artifact.artifact_id, navigation_groups=groups)
    branches = [{'branch_id':chapter['section_id'], 'label':chapter['title'],
                 'links':[item['url'] for item in queues[chapter['section_id']]],
                 'link_count':len(queues[chapter['section_id']])} for chapter in selected]
    root_part.metadata['navigation_branches'] = root_page['navigation_branches'] = branches
    root_links = list(dict.fromkeys(item['url'] for queue in queues.values() for item in queue))
    root_part.metadata['controlled_navigation'] = root_page['controlled_navigation'] = root_links
    pages, gaps = [root_page], []
    visited, discovered = {root_url}, {root_url, *root_links}
    stats = {chapter['section_id']:{'branch_id':chapter['section_id'], 'label':chapter['title'],
                                  'discovered_by_depth':Counter({1:len(queues[chapter['section_id']])}),
                                  'fetched_by_depth':Counter(), 'failed_by_depth':Counter()}
             for chapter in selected}
    # An admitted child URL has a single capture even when several chapters use
    # different anchors in that page. All selected intervals are already known.
    captured_cache = getattr(workspace, 'html_selected_page_cache', {})
    workspace.html_selected_page_cache = captured_cache
    captures, captures_by_url, dirty_urls = [], {}, set()

    def candidate_links(soup, current_url, branch_id, page_targets):
        base = html_base_url(soup, current_url)
        result = []
        admitted = None
        if (any(target.get('excluded_targets') for target in page_targets)
                or not any(not target.get('fragment') and not target.get('heading_dom_path') for target in page_targets)):
            from ..html_sourcepart.scope import selected_anchor_urls
            admitted = set(selected_anchor_urls(soup, current_url, page_targets))
        for anchor in soup.find_all('a', href=True):
            role, _reason = classify_html_link(anchor)
            if role != 'directory_entry':
                continue
            # A global navigation/sidebar or cross-chapter contents widget does
            # not establish a new child of the selected body chapter.
            if anchor.find_parent(['nav', 'aside', 'header', 'footer']):
                continue
            absolute = urljoin(base, str(anchor['href']))
            if admitted is not None and absolute not in admitted:
                continue
            url, fragment = urldefrag(absolute)
            if not workspace.acquisition.is_within_tutorial_scope(navigation_scope, url):
                continue
            owners = known_owners.get(url, set())
            if owners and branch_id not in owners:
                continue
            if not owners:
                # A chapter in its own directory owns that subtree. A sibling
                # chapter's unlisted page must not be admitted merely because
                # a directory-looking cross-reference appears in this body.
                chapter_url = chapter_by_id[branch_id].get('url') or current_url
                chapter_path = urlparse(chapter_url).path
                chapter_directory = (chapter_path if chapter_path.endswith('/') else posixpath.dirname(chapter_path) + '/')
                current_path = urlparse(current_url).path
                current_directory = (current_path if current_path.endswith('/') else posixpath.dirname(current_path) + '/')
                boundary = chapter_directory if chapter_directory != root_directory else current_directory
                if boundary != root_directory and not urlparse(url).path.startswith(boundary):
                    continue
                if boundary == root_directory:
                    candidate_path = urlparse(url).path
                    candidate_directory = (candidate_path if candidate_path.endswith('/')
                                           else posixpath.dirname(candidate_path) + '/')
                    # Flat manuals can discover sibling topic files from their
                    # body directories; entering an unlisted directory requires
                    # an explicit ownership edge from the preview subtree.
                    if candidate_directory != current_directory:
                        continue
                other_directories = {
                    (urlparse(node['url']).path if urlparse(node['url']).path.endswith('/')
                     else posixpath.dirname(urlparse(node['url']).path) + '/')
                    for other in scope['sections'] if other['section_id'] not in selected_ids
                    for node in [other] if node.get('url')
                }
                if any(directory != root_directory and urlparse(url).path.startswith(directory)
                       for directory in other_directories):
                    continue
                # Ignore complete global TOCs embedded into every content page.
                owner_list = anchor.find_parent(['ul', 'ol', 'dl'])
                if isinstance(owner_list, Tag) and is_authored_contents(owner_list) and any(
                        known_owners.get(page_url(urljoin(base, str(other['href']))), set()) - selected_ids
                        for other in owner_list.find_all('a', href=True)):
                    continue
            if url == current_url or url == root_url:
                continue
            if urlparse(url).scheme not in {'http', 'https', 'file'}:
                continue
            # A fragment-only child reference grants its interval, not the rest
            # of a shared page. Existing preview targets are never broadened.
            if url not in preview_urls:
                new_target = {'url':url, 'fragment':unquote(fragment) or None}
                existing = targets.setdefault(url, [])
                if not any(target.get('fragment') == new_target['fragment'] for target in existing):
                    existing.append(new_target)
                    refresh_exclusions(url)
                    if url in captures_by_url:
                        dirty_urls.add(url)
            result.append(url)
        return list(dict.fromkeys(result))

    # Root content can itself be a selected same-page chapter with local child
    # links. Inspect only anchors inside its selected DOM intervals.
    if targets.get(root_url):
        for chapter in selected:
            branch_id = chapter['section_id']
            chapter_targets = [node for node in flatten_sections([chapter]) if page_url(node.get('url')) == root_url]
            for url in candidate_links(root_soup, root_url, branch_id, chapter_targets):
                owners = known_owners.get(url, set())
                if url != root_url and (not owners or branch_id in owners) and url not in discovered:
                    queues[branch_id].append({'url':url, 'depth':1, 'parent_url':root_url})
                    discovered.add(url)
                    stats[branch_id]['discovered_by_depth'][1] += 1

    child_count = 0
    while dirty_urls or child_count < child_budget and any(queues.values()):
        # Discovering a second selected anchor does not fetch its page again.
        # Revisit only its cached navigation; materialization happens once after
        # all selected targets have been accumulated.
        while dirty_urls:
            dirty = dirty_urls.pop()
            capture = captures_by_url[dirty]
            branch_id = capture['branch_id']
            refreshed_links = candidate_links(capture['soup'], dirty, branch_id, targets.get(dirty, []))
            capture['controlled_navigation'] = list(dict.fromkeys([*capture.get('controlled_navigation', []), *refreshed_links]))
            for link in refreshed_links:
                if link not in discovered and link not in visited:
                    discovered.add(link)
                    queues[branch_id].append({'url':link, 'depth':capture['depth']+1, 'parent_url':dirty})
                    stats[branch_id]['discovered_by_depth'][capture['depth']+1] += 1
        if child_count >= child_budget:
            break
        progressed = False
        for chapter in selected:
            branch_id = chapter['section_id']
            queue = queues[branch_id]
            while queue and queue[0]['url'] in visited:
                queue.popleft()
            if not queue:
                continue
            workspace.acquisition._raise_if_cancelled()
            item = queue.popleft()
            url, depth = item['url'], item['depth']
            visited.add(url)
            progressed = True
            fetched = captured_cache.get(url)
            if fetched is None:
                fetched = workspace.acquisition.fetch_html_child(manifest.source_id, navigation_scope, url)
                if fetched.get('status') == 'succeeded':
                    captured_cache[url] = fetched
            if fetched.get('status') != 'succeeded':
                gaps.append({'url':url, 'reason':fetched.get('reason', 'fetch_failed'),
                             'error_type':fetched.get('error_type'),
                             'message':str(fetched.get('message') or '')[:1_600],
                             'navigation_depth':depth, 'capability_branch':chapter['title']})
                stats[branch_id]['failed_by_depth'][depth] += 1
                continue
            final_url = page_url(fetched.get('final_url') or url)
            # Redirects must still obey the source boundary and cannot enter an
            # unselected chapter just because its selected URL redirected there.
            owners = known_owners.get(final_url, set())
            if (not workspace.acquisition.is_within_tutorial_scope(navigation_scope, final_url)
                    or owners and not owners.intersection(selected_ids)):
                gaps.append({'url':url, 'reason':'redirect_outside_selected_scope'})
                continue
            if final_url != url and final_url in visited:
                continue
            visited.add(final_url)
            page_targets = targets.get(url, [])
            if final_url != url:
                targets.setdefault(final_url, []).extend(page_targets)
                refresh_exclusions(final_url)
                page_targets = targets[final_url]
            artifact_id = str(fetched['artifact_id'])
            workspace.allowed_artifacts.add(artifact_id)
            html = executor._decode_html(fetched['data'])
            from ..html_sourcepart.fallback import Limits, scan
            _probe, reasons = scan(html, final_url, Limits())
            soup = BeautifulSoup('' if any('budget' in reason['code'] for reason in reasons) else html, 'html.parser')
            capture = {'url':final_url, 'fetched':fetched, 'soup':soup, 'branch_id':branch_id,
                       'depth':depth, 'parent_url':item['parent_url']}
            captures.append(capture)
            captures_by_url[final_url] = capture
            child_count += 1
            # One shared capture covers every selected chapter that explicitly
            # declared this page, while consuming just one child-page slot.
            covered_branches = (known_owners.get(url, set()) | known_owners.get(final_url, set()) | {branch_id}) & selected_ids
            for covered_branch in covered_branches:
                covered_depth = next((queued['depth'] for queued in queues[covered_branch]
                                      if queued['url'] in {url, final_url}), depth)
                stats[covered_branch]['fetched_by_depth'][covered_depth] += 1
            links = candidate_links(soup, final_url, branch_id, page_targets)
            capture['controlled_navigation'] = links
            for link in links:
                if link in discovered or link in visited:
                    continue
                discovered.add(link)
                queue.append({'url':link, 'depth':depth+1, 'parent_url':final_url})
                stats[branch_id]['discovered_by_depth'][depth+1] += 1
            if child_count >= child_budget:
                break
        if not progressed:
            break

    for capture in captures:
        url, branch_id, fetched = capture['url'], capture['branch_id'], capture['fetched']
        groups = [chapter['title'] for chapter in selected if chapter['section_id'] in known_owners.get(url, {branch_id})]
        _soup, part, page, _sections = executor._materialize_html_page(
            page_url=url, data=fetched['data'], artifact_id=str(fetched['artifact_id']),
            ordinal=len(pages)+1, navigation_depth=capture['depth'],
            capability_branch=chapter_by_id[branch_id]['title'], parent_url=capture['parent_url'],
            authored_navigation_groups=groups, scope_targets=targets.get(url, []))
        links = capture.get('controlled_navigation', [])
        part.metadata['controlled_navigation'] = page['controlled_navigation'] = links
        pages.append(page)

    pending = list(dict.fromkeys(item['url'] for queue in queues.values() for item in queue if item['url'] not in visited))
    if pending:
        workspace.gaps.append(f'The {child_budget}-child-page HTML budget left {len(pending)} selected-chapter pages unfetched.')
    if gaps:
        workspace.gaps.append(f'{len(gaps)} selected HTML pages could not be acquired or admitted; see the inventory.')
        if child_count == 0:
            workspace.gaps.append('No selected HTML child-page body content was captured; only available selected entry-page content and source links are retained as fallback.')
    coverage = []
    for chapter in selected:
        record = stats[chapter['section_id']]
        totals = {key:sum(record[key].values()) for key in ('discovered_by_depth', 'fetched_by_depth', 'failed_by_depth')}
        coverage.append({'branch_id':record['branch_id'], 'label':record['label'],
                         'discovered_pages':totals['discovered_by_depth'], 'fetched_pages':totals['fetched_by_depth'],
                         'failed_pages':totals['failed_by_depth'],
                         'unfetched_pages':max(0, totals['discovered_by_depth']-totals['fetched_by_depth']-totals['failed_by_depth']),
                         **{key:{str(depth):count for depth,count in record[key].items()}
                            for key in ('discovered_by_depth','fetched_by_depth','failed_by_depth')}})
    workspace.inventory_completed = True
    workspace.update_document_version('\n'.join([manifest.display_name, *(page['title'] for page in pages[:3])]))
    payload = {'root_structure_artifact_id':structure_artifact.artifact_id, 'page_count':len(pages),
               'root_page_count':1, 'child_page_count':child_count, 'part_count':len(workspace.parts),
               'pages':pages, 'navigation_gaps':gaps, 'truncated':bool(pending),
               'acquisition_complete':not pending and not gaps,
               'unfetched_discovered_urls':pending, 'page_budget_scope':'child_pages_only',
               'max_pages':child_budget, 'max_child_pages':child_budget,
               'navigation_sampling':{'strategy':'selected_chapters_round_robin_breadth_first.v1'},
               'capability_branch_count':len(selected), 'branch_coverage':coverage, 'depth_coverage':[],
               'extraction_scope':{'mode':'selected', 'section_ids':list(scope['section_ids'])}}
    artifact = store.put_json_artifact(payload, kind='html_inventory', source_id=manifest.source_id,
                                       preserve_text_integrity=True)
    workspace.allowed_artifacts.add(artifact.artifact_id)
    workspace.persist()
    return {'inventory_artifact_id':artifact.artifact_id, **payload}


def _structure_item(node):
    url = node.get('url')
    if url and node.get('fragment'):
        url += '#' + node['fragment']
    return {'label':node['title'], 'href':url, 'fragment':node.get('fragment'),
            'link_role':'directory_entry', 'children':[_structure_item(child) for child in node.get('children', [])]}
