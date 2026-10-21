"""The viewer, served as static files, shows the rows and owners the trace holds.

Runs a real browser (Playwright, headless Chromium with software WebGL) against the viewer
directory served over HTTP, the way any static host serves it.
"""

import copy
import functools
import http.server
import json
import math
import threading
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
TRACE = json.loads((VIEWER / "traces/alexander-the-great.json").read_text())


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(VIEWER))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture()
def page(server):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.errors = []
        page.on("pageerror", lambda error: page.errors.append(str(error)))
        page.on("console", lambda message: message.type == "error" and page.errors.append(message.text))
        page.base = server
        yield page
        browser.close()


def owner(row, num_rows, world):
    return row // math.ceil(num_rows / world)


def table_rows(page):
    page.wait_for_selector("table.rows tbody tr")
    return [
        [cell.inner_text().strip() for cell in row.query_selector_all("td")]
        for row in page.query_selector_all("table.rows tbody tr")
    ]


def test_rows_owners_and_reuse_match_the_trace(page):
    t, world = 11, 8
    page.goto(f"{page.base}/index.html?trace=traces/alexander-the-great.json&t={t}&world={world}")
    rows = table_rows(page)
    layer = TRACE["layers"][0]
    assert [int(r[2].replace(",", "")) for r in rows] == TRACE["hash_rows"][t][0]
    assert [r[3] for r in rows] == [f"rank {owner(x, layer['num_rows'], world)}" for x in TRACE["hash_rows"][t][0]]
    # the second "Alexander the Great": its 2-gram and 3-gram rows were first read at position 3
    first_read = [r[4] for r in rows]
    assert first_read[:16] == ["#3"] * 16
    assert page.errors == []


def test_world_size_and_layer_controls(page):
    t = 3
    page.goto(f"{page.base}/index.html?trace=traces/alexander-the-great.json&t={t}")
    table_rows(page)
    layer = TRACE["layers"][0]
    rows = TRACE["hash_rows"][t][0]
    # the check is only meaningful if some owner changes between world sizes 8 and 16
    assert any(owner(x, layer["num_rows"], 8) != owner(x, layer["num_rows"], 16) for x in rows)
    page.fill("#world", "16")
    page.press("#world", "Enter")
    expected = [f"rank {owner(x, layer['num_rows'], 16)}" for x in rows]
    page.wait_for_function(
        "expected => [...document.querySelectorAll('table.rows tbody tr td:nth-child(4)')].map(c => c.innerText.trim()).join() === expected.join()",
        arg=expected)
    assert "world=16" in page.url
    page.select_option("#layer", "1")
    expect(page.locator(".wall-title")).to_contain_text("Layer 14")
    assert [int(r[2].replace(",", "")) for r in table_rows(page)] == TRACE["hash_rows"][t][1]
    assert page.errors == []


def test_play_advances(page):
    page.goto(f"{page.base}/index.html?trace=traces/alexander-the-great.json&t=0")
    table_rows(page)  # the keys act once a trace is loaded
    page.keyboard.press("ArrowRight")
    expect(page.locator("#where")).to_have_text(f"1 / {len(TRACE['tokens']) - 1}")
    page.click("#play")
    page.wait_for_function("Number(document.getElementById('scrub').value) >= 3", timeout=15000)
    assert page.errors == []


def test_keys_still_step_after_a_button_click(page):
    page.goto(f"{page.base}/index.html?trace=traces/alexander-the-great.json&t=0")
    table_rows(page)
    page.click("#next")
    expect(page.locator("#where")).to_have_text(f"1 / {len(TRACE['tokens']) - 1}")
    page.keyboard.press("ArrowRight")
    expect(page.locator("#where")).to_have_text(f"2 / {len(TRACE['tokens']) - 1}")
    page.fill("#world", "")
    page.press("#world", "Enter")
    expect(page.locator("#world")).to_have_value("8")
    assert page.errors == []


def test_a_copied_url_restores_the_view(page):
    page.goto(f"{page.base}/index.html?trace=traces/alexander-the-great.json")
    table_rows(page)
    page.fill("#world", "16")
    page.press("#world", "Enter")
    page.select_option("#layer", "1")
    page.fill("#scrub", "21")
    copied = page.url
    fresh = page.context.browser.new_page()
    fresh.goto(copied)
    fresh.wait_for_selector("table.rows tbody tr")
    expect(fresh.locator("#where")).to_have_text(f"21 / {len(TRACE['tokens']) - 1}")
    expect(fresh.locator("#world")).to_have_value("16")
    expect(fresh.locator("#layer")).to_have_value("1")
    shown = [int(c.inner_text().replace(",", "")) for c in fresh.query_selector_all("table.rows tbody tr td:nth-child(3)")]
    assert shown == TRACE["hash_rows"][21][1]


def test_defaults_without_url_values(page):
    page.goto(f"{page.base}/index.html")
    table_rows(page)
    expect(page.locator("#world")).to_have_value("8")
    expect(page.locator("#where")).to_have_text(f"0 / {len(TRACE['tokens']) - 1}")
    expect(page.locator("#layer")).to_have_value("0")
    assert page.errors == []


def test_bad_url_values_fall_back_to_a_valid_view(page):
    page.goto(f"{page.base}/index.html?trace=traces/alexander-the-great.json&t=abc&layer=-1&world=384006168")
    table_rows(page)
    expect(page.locator("#where")).to_have_text(f"0 / {len(TRACE['tokens']) - 1}")
    expect(page.locator("#world")).to_have_value("4096")
    expect(page.locator("#layer")).to_have_value("0")
    assert page.errors == []


def test_gates_are_listed_with_their_values(page):
    gated = copy.deepcopy(TRACE)
    hc = gated["model"]["hc_mult"]
    # a synthetic fixture for the viewer's handling of the optional field, not model output;
    # every value differs so a position, layer or copy mix-up shows
    gated["gates"] = [[[round((t * 7 + l * 3 + k) % 97 / 97, 3) for k in range(hc)] for l in range(len(gated["layers"]))]
                      for t in range(len(gated["tokens"]))]
    gated["provenance"]["gates"] = "synthetic test fixture"
    page.route("**/traces/gated.json", lambda route: route.fulfill(json=gated))
    page.goto(f"{page.base}/index.html?trace=traces/gated.json&t=5&layer=1")
    expect(page.locator(".badge.on", has_text="gates: present")).to_be_visible()
    expect(page.locator(".provenance-notes")).to_contain_text("synthetic test fixture")
    values = page.locator(".gate .num").all_inner_texts()
    assert values == [f"{g:.3f}" for g in gated["gates"][5][1]]
    assert page.errors == []


@pytest.mark.parametrize("break_it, message", [
    (lambda tr: tr["hash_rows"][2][0].__setitem__(5, 0), "hash_rows[2][0][5] = 0 is outside column 5"),
    (lambda tr: tr["layers"][1]["columns"].pop(), "layer 14 has 23 columns, expected 24"),
    (lambda tr: tr["layers"][0]["columns"][0].__setitem__("head", 3), "column 0 is (n=2, head=3), expected (n=2, head=0)"),
    (lambda tr: tr.__setitem__("gates", [[[0.5]] * 2] * len(tr["tokens"])), "gates[0][0] must hold 4 values in [0, 1]"),
    (lambda tr: tr["layers"][0]["columns"].__setitem__(3, None), "layer 1 column 3 is not an object"),
    (lambda tr: tr["tokens"][0].__setitem__("display", 123), "tokens[0].display must be a string"),
    (lambda tr: tr["model"].__setitem__("n_heads", 1_000_000_000), "has 24 columns, expected 3000000000"),
])
def test_a_broken_trace_says_what_is_wrong(page, break_it, message):
    broken = copy.deepcopy(TRACE)
    break_it(broken)
    page.route("**/traces/broken.json", lambda route: route.fulfill(json=broken))
    page.goto(f"{page.base}/index.html?trace=traces/broken.json")
    expect(page.locator("#error")).to_contain_text(message)


def test_opening_a_local_file_is_not_shared_as_the_old_trace(page):
    page.goto(f"{page.base}/index.html?trace=traces/alexander-the-great.json")
    table_rows(page)
    page.set_input_files("#file", str(VIEWER / "traces/tokyo-japanese.json"))
    expect(page.locator("#header h2")).to_have_text("Tokyo Metropolitan Government (Japanese)")
    for key in ("trace=", "t=", "layer="):
        assert key not in page.url
    expect(page.locator("#catalog")).to_contain_text("tokyo-japanese.json (local file)")
    assert page.errors == []


LADDER = json.loads((VIEWER / "traces/nanogpt-hastings.json").read_text())


def settled_block(blocks):
    final = blocks[-1]["top"][0][0]
    k = len(blocks) - 1
    while k > 0 and blocks[k - 1]["top"][0][0] == final:
        k -= 1
    return k


def test_the_ladder_trace_matches_the_schema():
    import jsonschema

    jsonschema.validate(LADDER, json.loads((VIEWER / "schema/engram-trace.schema.json").read_text()))


def test_the_ladder_summary_is_the_traces_own_count(page):
    page.goto(f"{page.base}/ladder.html?trace=traces/nanogpt-hastings.json&a=engram&b=baseline")
    expect(page.locator("#summary .stat").first).to_be_visible()
    lens = LADDER["lens"]
    n = len(LADDER["tokens"]) - 1
    sa = [settled_block(b) for b in lens["values"]["engram"][:n]]
    sb = [settled_block(b) for b in lens["values"]["baseline"][:n]]
    final = lambda name, t: lens["values"][name][t][-1]["top"][0][0]  # noqa: E731
    nxt = [token["id"] for token in LADDER["tokens"][1:]]
    right = [t for t in range(n) if final("engram", t) == nxt[t] == final("baseline", t)]

    def line(title, positions):
        sooner = sum(sa[t] < sb[t] for t in positions)
        later = sum(sa[t] > sb[t] for t in positions)
        above = lambda s: sum(lens["n_layer"] - 1 - s[t] for t in positions) / len(positions)  # noqa: E731
        return (f"{title} ({len(positions)}): with Engram settles sooner at {sooner}, later at {later}, same at "
                f"{len(positions) - sooner - later}; blocks above the settled one, on average {above(sa):.2f} against {above(sb):.2f}")

    stats = page.locator("#summary .stat").all_inner_texts()
    assert stats == [line("all positions", range(n)), line("both end on the actual next token", right)]
    assert page.locator(".chart .col").count() == n
    assert page.errors == []


def test_the_ladder_shows_every_block_of_one_position(page):
    t = next(i for i, token in enumerate(LADDER["tokens"][:-1]) if token["text"] == " Sax")
    page.goto(f"{page.base}/ladder.html?trace=traces/nanogpt-hastings.json&a=engram&b=baseline&t={t}")
    expect(page.locator(".focus-text")).to_have_text("·Sax → on")
    lens = LADDER["lens"]
    for column, name in enumerate(["engram", "baseline"]):
        blocks = lens["values"][name][t]
        col = page.locator(".ladder-col").nth(column)
        words = col.locator(".rung-word").all_inner_texts()
        texts = [lens["vocab"][str(blocks[k]["top"][0][0])] for k in range(lens["n_layer"] - 1, -1, -1)]
        expected = ["·" + text[1:] if text.startswith(" ") else text for text in texts]
        assert words == expected
        pct = lambda p: f"{100 * p:.{0 if p >= 0.1 else 1}f} %"  # noqa: E731
        assert col.locator(".rung-p").all_inner_texts() == [pct(blocks[k]["top"][0][1]) for k in range(lens["n_layer"] - 1, -1, -1)]
        expect(col.locator(".rung.settles .rung-block")).to_have_text(str(settled_block(blocks)))
    gate = LADDER["gates"][t][0][0]
    expect(page.locator(".ladder-col").nth(0).locator(".rung-engram")).to_have_text(f"Engram added, gate {gate:.2f}")
    expect(page.locator(".ladder-col").nth(1).locator(".rung-engram")).to_have_text("no Engram")
    assert page.errors == []


def test_the_ladder_steps_and_keeps_the_view_in_the_url(page):
    page.goto(f"{page.base}/ladder.html?trace=traces/nanogpt-hastings.json&a=engram_off&b=engram&t=3")
    expect(page.locator("#where")).to_have_text(f"3 / {len(LADDER['tokens']) - 2}")
    page.click("#next")
    expect(page.locator("#where")).to_have_text(f"4 / {len(LADDER['tokens']) - 2}")
    expect(page.locator("#a")).to_have_value("engram_off")
    assert "t=4" in page.url and "a=engram_off" in page.url and "b=engram" in page.url
    page.select_option("#b", "baseline")
    assert "b=baseline" in page.url
    assert page.errors == []


@pytest.mark.parametrize("tops, expected", [
    ([1, 1, 1, 1], 0),
    ([2, 1, 1, 1], 1),
    ([1, 2, 1, 1], 2),  # matches the final answer at block 0, changes, then comes back
    ([1, 1, 1, 2], 3),
])
def test_a_prediction_settles_where_its_top_1_stops_changing(page, tops, expected):
    page.goto(f"{page.base}/ladder.html?trace=traces/nanogpt-hastings.json")
    blocks = [{"top": [[i, 0.5]]} for i in tops]
    assert page.evaluate("async (b) => (await import('./js/ladder.js')).settledBlock(b)", blocks) == expected
    assert settled_block(blocks) == expected


@pytest.mark.parametrize("break_it, message", [
    (lambda tr: tr["lens"]["values"]["engram"].pop(), "lens.values.engram has 83 positions, tokens has 84"),
    (lambda tr: tr["lens"]["values"]["baseline"][4].pop(), "lens.values.baseline[4] has 11 blocks, expected 12"),
    (lambda tr: tr["lens"]["values"]["engram"][2][3].__setitem__("top", []), "lens.values.engram[2][3] has no top-1 token"),
    (lambda tr: tr["lens"]["variants"].append(dict(tr["lens"]["variants"][0])), "must name each variant once"),
    (lambda tr: tr["lens"]["values"]["engram"][5][2].__setitem__("p_final", None), "lens.values.engram[5][2] needs a top-1 probability"),
    (lambda tr: tr["lens"].__setitem__("n_layer", 0), "lens.n_layer must be a positive integer"),
])
def test_a_broken_lens_says_what_is_wrong(page, break_it, message):
    broken = copy.deepcopy(LADDER)
    break_it(broken)
    page.route("**/traces/broken.json", lambda route: route.fulfill(json=broken))
    page.goto(f"{page.base}/ladder.html?trace=traces/broken.json")
    expect(page.locator("#error")).to_contain_text(message)
    page.click("#next")
    assert page.errors == []


def test_the_ladder_says_a_trace_without_lens_is_for_the_table_view(page):
    page.goto(f"{page.base}/ladder.html?trace=traces/alexander-the-great.json")
    expect(page.locator("#error")).to_contain_text('has no "lens" field')


def test_the_table_view_opens_the_ladder_trace(page):
    page.goto(f"{page.base}/index.html?trace=traces/nanogpt-hastings.json")
    rows = table_rows(page)
    assert len(rows) == sum(len(layer["columns"]) for layer in LADDER["layers"][:1])
    assert page.errors == []
