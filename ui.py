"""Pure HTML card renderer for match predictions.

No streamlit import — functions are unit-testable.
"""
import html
from model.predict import MatchPrediction

# ── colour constants ──────────────────────────────────────────────────────────
_CARD_BG        = "#fff"
_CARD_BORDER    = "#ddd"
_POOL1_BG       = "#0f4c2a"
_POOL2_BG       = "#1a6b3c"
_BAR_HOME       = "#2e7d32"
_BAR_DRAW       = "#9e9e9e"
_BAR_AWAY       = "#c62828"
_CAPTION_COLOR  = "#888"
_NOTE_COLOR     = "#c62828"


def render_card(p: MatchPrediction, flag_home: str, flag_away: str) -> str:
    """Return an HTML string representing one match prediction card.

    Parameters
    ----------
    p:          MatchPrediction dataclass instance.
    flag_home:  Flag emoji for the home team (may be '').
    flag_away:  Flag emoji for the away team (may be '').
    """
    f = p.fixture
    home = html.escape(f["home"])
    away = html.escape(f["away"])
    flag_home = html.escape(flag_home)
    flag_away = html.escape(flag_away)

    # ── title ─────────────────────────────────────────────────────────────────
    title = (
        f'<div style="font-weight:700;font-size:17px;margin-bottom:4px;">'
        f'{flag_home} {home} vs {away} {flag_away}'
        f'</div>'
    )

    # ── caption ───────────────────────────────────────────────────────────────
    stage  = html.escape(f["stage"])
    kickoff = f["kickoff_et"].strftime("%a %b %d, %I:%M %p ET")

    members = getattr(p, "members", [])
    expert_picks = getattr(p, "expert_picks", None)

    if p.source == "none":
        caption_body = f'{stage} · {kickoff}'
    elif p.source == "elo":
        caption_body = (
            f'{stage} · {kickoff}'
            ' · \U0001f7e1 model-only (no market odds)'
        )
        if p.elo_disagrees:
            caption_body += ' · ⚠️ market and Elo disagree'
    else:
        # Build members label: books count only when 'market' is a member.
        # Fall back to checking source string when members list is empty
        # (supports MatchPrediction instances created without explicit members).
        market_present = "market" in members or (not members and "market" in p.source)
        if market_present:
            caption_body = f'{stage} · {kickoff} · {p.books_count} bookmakers'
        else:
            caption_body = f'{stage} · {kickoff}'
        if p.elo_disagrees:
            caption_body += ' · ⚠️ market and Elo disagree'

    caption = (
        f'<div style="color:{_CAPTION_COLOR};font-size:12px;margin-bottom:10px;">'
        f'{caption_body}'
        f'</div>'
    )

    # ── source==none: just title + caption + red note ─────────────────────────
    if p.source == "none":
        note = (
            f'<div style="color:{_NOTE_COLOR};font-size:13px;">'
            f'{html.escape(p.note)}'
            f'</div>'
        )
        body = title + caption + note
        return _wrap_card(body)

    # ── badges row ────────────────────────────────────────────────────────────
    pool1_score = html.escape(str(p.pool1["score"]))
    pool2_score = html.escape(str(p.pool2["score"]))

    badge_style = (
        "flex:1;color:#fff;border-radius:12px;"
        "padding:10px;text-align:center;"
    )
    label_style = "font-size:11px;opacity:.8;"
    score_style = "font-size:26px;font-weight:800;"

    badges = (
        f'<div style="display:flex;gap:8px;margin-bottom:10px;">'
        f'  <div style="{badge_style}background:{_POOL1_BG};">'
        f'    <div style="{label_style}">POOL 1</div>'
        f'    <div style="{score_style}">{pool1_score}</div>'
        f'  </div>'
        f'  <div style="{badge_style}background:{_POOL2_BG};">'
        f'    <div style="{label_style}">POOL 2</div>'
        f'    <div style="{score_style}">{pool2_score}</div>'
        f'  </div>'
        f'</div>'
    )

    # ── expert picks chip row ─────────────────────────────────────────────────
    expert_chip_row = ""
    if expert_picks:
        davo_esc = html.escape(expert_picks["davo"])
        maldini_esc = html.escape(expert_picks["maldini"])
        chip_style = (
            "display:inline-block;font-size:13px;padding:2px 8px;"
            "border:1px solid #444;border-radius:10px;margin-right:6px;"
            "color:#333;background:none;"
        )
        expert_chip_row = (
            f'<div style="margin-bottom:8px;">'
            f'<span style="{chip_style}">DAVO {davo_esc}</span>'
            f'<span style="{chip_style}">MALDINI {maldini_esc}</span>'
            f'</div>'
        )

    # ── probability bar ───────────────────────────────────────────────────────
    probs = p.probs
    ph = probs["home"]
    pd = probs["draw"]
    pa = probs["away"]

    def _seg_label(prob, base_label, prepend_flag, append_flag):
        if prob < 0.12:
            return ""
        text = f"{prob:.0%}"
        if prob >= 0.18 and prepend_flag:
            text = f"{prepend_flag} {text}"
        if prob >= 0.18 and append_flag:
            text = f"{text} {append_flag}"
        return text

    home_label = _seg_label(ph, f"{ph:.0%}", flag_home, None)
    draw_label = _seg_label(pd, f"{pd:.0%}", None, None)
    away_label = _seg_label(pa, f"{pa:.0%}", None, flag_away)

    seg_style = (
        "display:inline-block;color:#fff;font-size:11px;"
        "text-align:center;line-height:22px;overflow:hidden;"
        "white-space:nowrap;"
    )

    bar = (
        f'<div style="display:flex;height:22px;border-radius:6px;overflow:hidden;">'
        f'<div style="{seg_style}width:{ph:.0%};background:{_BAR_HOME};">{home_label}</div>'
        f'<div style="{seg_style}width:{pd:.0%};background:{_BAR_DRAW};">{draw_label}</div>'
        f'<div style="{seg_style}width:{pa:.0%};background:{_BAR_AWAY};">{away_label}</div>'
        f'</div>'
    )

    body = title + caption + badges + expert_chip_row + bar
    return _wrap_card(body)


def _wrap_card(body: str) -> str:
    style = (
        "border:1px solid #ddd;border-radius:18px;"
        "padding:14px 16px;margin-bottom:14px;background:#fff;"
    )
    return f'<div style="{style}">{body}</div>'
