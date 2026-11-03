"""Navigation can nominate chapters without replacing the current page's body."""
from bs4 import BeautifulSoup
import pytest

from app.html_guided.boundary import boundary_selection_error, container_boundary_facts


def roots(html):
    soup = BeautifulSoup(html, "html.parser")
    return {"C1": soup.body, "C2": soup.find(id="catalog"), "C3": soup.find(id="text")}


@pytest.mark.parametrize("body_tag", ['main', 'article', 'div role="main"', 'div role="article"'])
def test_navigation_only_selection_cannot_drop_a_disjoint_explicit_authored_body(body_tag):
    name = body_tag.split()[0]
    nodes = roots('<body><div id="catalog"><nav><ul><li><a href="chapter.html">Chapter</a></li></ul></nav>'
                  '<footer>Manual navigation</footer></div>'
                  f'<{body_tag} id="text"><h1>Manual introduction</h1><p>How to use this document.</p></{name}></body>')
    assert container_boundary_facts(nodes['C2'])["navigation_only"] is True
    assert container_boundary_facts(nodes['C3'])["authored_body"] is True
    assert "separate explicit authored main/article body" in boundary_selection_error("C2", nodes)
    assert boundary_selection_error("C1", nodes) is None
    assert boundary_selection_error("C3", nodes) is None


def test_semantic_navigation_role_is_recognized_without_publisher_names():
    soup = BeautifulSoup('<div role="navigation"><h2>Catalog</h2><p><a href="x">Lesson</a></p></div>', 'html.parser')
    facts = container_boundary_facts(soup.div)
    assert facts == {'role': 'navigation', 'navigation_only': True,
                     'body_semantics': None, 'authored_body': False}


def test_brand_classes_do_not_establish_navigation_only_status():
    soup = BeautifulSoup('<div class="side-bar"><h2>Article</h2><p>Actual authored material.</p></div>', 'html.parser')
    facts = container_boundary_facts(soup.div)
    assert facts['navigation_only'] is False and facts['authored_body'] is True


def test_generated_div_prose_does_not_disappear_when_navigation_has_more_links():
    nodes = roots('<body><div id="catalog"><nav><a href="x">Chapter</a></nav></div>'
                  '<div role="main" id="text"><h1>Manual title</h1><div><span>Document authors and acknowledgments.</span></div></div></body>')
    assert container_boundary_facts(nodes['C3'])['authored_body'] is True
    assert boundary_selection_error('C2', nodes)


def test_plain_heading_and_link_only_directory_is_not_mislabeled_as_prose():
    nodes = roots('<body><nav id="catalog">Catalog</nav><main id="text"><h1>Index</h1>'
                  '<div><a href="lesson">Tutorial</a></div></main></body>')
    assert container_boundary_facts(nodes['C3'])['authored_body'] is False


@pytest.mark.parametrize("main_contents", ['<h1>Welcome</h1>', '<p>One paragraph.</p>', '<nav><h1>Index</h1><p>Links.</p></nav>'])
def test_navigation_portal_without_disjoint_substantive_body_is_not_rejected(main_contents):
    nodes = roots('<body><aside id="catalog"><a href="x">Catalog</a></aside>'
                  '<main id="text">' + main_contents + '</main></body>')
    assert boundary_selection_error('C2', nodes) is None


@pytest.mark.parametrize("hidden", ['hidden', 'aria-hidden="true"'])
def test_hidden_main_and_comments_do_not_create_a_competing_authored_body(hidden):
    nodes = roots('<body><nav id="catalog">Catalog<!-- main article --><script>fake prose</script></nav>'
                  f'<main id="text" {hidden}><h1>Hidden heading</h1><p>Hidden prose.</p></main></body>')
    assert boundary_selection_error('C2', nodes) is None
    assert container_boundary_facts(nodes['C3'])['authored_body'] is False


def test_code_is_substantive_but_empty_containers_are_not_navigation():
    nodes = roots('<body><nav id="catalog">Catalog</nav><main id="text"><h1>Run</h1><pre>run --example</pre></main></body>')
    assert boundary_selection_error('C2', nodes)
    assert container_boundary_facts(BeautifulSoup('<div hidden>Empty</div>', 'html.parser').div)['navigation_only'] is False


def test_nested_semantic_body_inside_selected_navigation_does_not_trigger_disjoint_guard():
    nodes = roots('<body><aside id="catalog">Catalog<main id="text"><h1>Heading</h1><p>Text.</p></main></aside></body>')
    assert boundary_selection_error('C2', nodes) is None
