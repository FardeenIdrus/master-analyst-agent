"""
Master Analyst Agent — Professional Investment Dashboard
Built with Reflex | Dark Theme | Glass Morphism | Animated | Live Pipeline
"""

import reflex as rx
from .state import DashboardState

# ═════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Colors, Tokens, Gradients
# ═════════════════════════════════════════════════════════════════════════════

BG = "#050a15"
BG2 = "#0a1128"
BG_CARD = "rgba(10, 17, 40, 0.65)"
GLASS = "rgba(255, 255, 255, 0.025)"
BORDER = "1px solid rgba(30, 41, 59, 0.3)"
BORDER_GLOW = "1px solid rgba(59, 130, 246, 0.25)"
TEXT = "#f1f5f9"
TEXT2 = "#94a3b8"
MUTED = "#64748b"
BLUE = "#3b82f6"
GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
PURPLE = "#8b5cf6"
CYAN = "#06b6d4"
INDIGO = "#6366f1"
ROSE = "#f43f5e"

GRADIENT_BRAND = "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)"
GRADIENT_SUCCESS = "linear-gradient(135deg, #10b981, #059669)"
GRADIENT_WARM = "linear-gradient(135deg, #f59e0b, #ef4444)"
GRADIENT_SUBTLE = "linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.04))"

# iOS 26 Liquid Glass styling
_GLASS_BG = "linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02))"
_GLASS_BORDER = "1px solid rgba(255,255,255,0.10)"
_GLASS_SHADOW = "0 8px 32px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.12)"
_APPLE_EASE = "all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)"

CARD = {
    "background": "linear-gradient(135deg, rgba(12,18,40,0.85), rgba(8,14,32,0.80))",
    "border": _GLASS_BORDER,
    "border_radius": "20px",
    "padding": "20px",
    "box_shadow": _GLASS_SHADOW,
    "transition": _APPLE_EASE,
    "_hover": {
        "border": "1px solid rgba(255,255,255,0.18)",
        "transform": "translateY(-3px) scale(1.005)",
        "box_shadow": "0 20px 60px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.18), 0 0 30px rgba(59,130,246,0.06)",
    },
    "_active": {
        "transform": "translateY(-1px) scale(0.985)",
        "box_shadow": "0 4px 16px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.08)",
        "transition": "all 0.1s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
    },
}

CARD_STATIC = {
    "background": "linear-gradient(135deg, rgba(12,18,40,0.85), rgba(8,14,32,0.80))",
    "border": _GLASS_BORDER,
    "border_radius": "20px",
    "padding": "20px",
    "box_shadow": _GLASS_SHADOW,
    "width": "100%",
    "transition": _APPLE_EASE,
}


def _anim(name: str, dur: str = "0.7s", delay: str = "0s") -> str:
    return f"{name} {dur} cubic-bezier(0.16, 1, 0.3, 1) {delay} both"


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


# ═════════════════════════════════════════════════════════════════════════════
# REUSABLE COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════

def stat_card(title: str, value, subtitle: str = "", icon: str = "", color: str = BLUE):
    r = _rgb(color)
    return rx.box(
        rx.box(
            height="2px",
            background=f"linear-gradient(90deg, {color}, rgba({r}, 0.15), transparent)",
            border_radius="2px",
            margin_bottom="12px",
        ),
        rx.flex(
            rx.box(
                rx.text(title, color="white", font_size="10px", font_weight="600",
                        letter_spacing="0.1em", text_transform="uppercase"),
                rx.text(value, color="white", font_size="24px", font_weight="800",
                        line_height="1.2", margin_top="4px",
                        letter_spacing="-0.02em",
                        animation=_anim("countUp", "0.8s", "0.2s")),
                rx.cond(
                    subtitle != "",
                    rx.text(subtitle, color="white", font_size="11px",
                            margin_top="3px", line_height="1.4"),
                    rx.fragment(),
                ),
                flex="1",
            ),
            rx.cond(
                icon != "",
                rx.box(
                    rx.icon(tag=icon, size=18, color=color),
                    padding="10px", border_radius="12px",
                    background=f"linear-gradient(135deg, rgba({r},0.12), rgba({r},0.04))",
                    border=f"1px solid rgba({r}, 0.15)",
                    box_shadow=f"inset 0 1px 0 rgba(255,255,255,0.12), 0 2px 8px rgba({r},0.08)",
                ),
                rx.fragment(),
            ),
            justify="between", align="start", width="100%",
        ),
        style=CARD, cursor="pointer",
        _active={
            "transform": "scale(0.97) translateY(0px)",
            "box_shadow": "0 2px 8px rgba(0,0,0,0.20), inset 0 1px 0 rgba(255,255,255,0.06)",
            "transition": "all 0.1s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        },
        animation="glowPulse 4s ease-in-out infinite, " + _anim("fadeInUp", "0.5s"),
    )


def section_header(title: str, subtitle: str = "", icon: str = ""):
    return rx.box(
        rx.flex(
            rx.cond(
                icon != "",
                rx.box(
                    rx.icon(tag=icon, size=18, color=BLUE),
                    padding="8px", border_radius="12px",
                    background="linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.04))",
                    border="1px solid rgba(59,130,246,0.15)",
                    box_shadow="inset 0 1px 0 rgba(255,255,255,0.12), 0 2px 8px rgba(59,130,246,0.06)",
                ),
                rx.fragment(),
            ),
            rx.box(
                rx.heading(title, size="4", color=TEXT, font_weight="700",
                           letter_spacing="-0.01em"),
                rx.cond(
                    subtitle != "",
                    rx.text(subtitle, color=TEXT2, font_size="12px", margin_top="1px"),
                    rx.fragment(),
                ),
            ),
            align="center", gap="10px",
        ),
        rx.box(
            height="1px",
            background=f"linear-gradient(90deg, {BLUE}, {PURPLE}, transparent)",
            margin_top="10px", border_radius="1px",
        ),
        margin_bottom="16px",
        animation=_anim("fadeInUp", "0.5s"),
    )


def divider():
    return rx.box(
        height="1px",
        background="linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1), transparent)",
        margin_y="20px", width="100%",
    )


def text_card(title: str, body, icon: str = ""):
    return rx.box(
        rx.flex(
            rx.cond(
                icon != "",
                rx.box(
                    rx.icon(tag=icon, size=14, color=BLUE),
                    padding="7px", border_radius="8px",
                    background="rgba(59, 130, 246, 0.06)",
                    flex_shrink="0",
                ),
                rx.fragment(),
            ),
            rx.heading(title, size="3", color=TEXT, font_weight="600"),
            align="center", gap="8px", margin_bottom="10px",
        ),
        rx.text(body, color=TEXT2, font_size="13px", line_height="1.7",
                white_space="pre-wrap"),
        style=CARD_STATIC,
        animation=_anim("fadeInUp", "0.5s", "0.05s"),
    )


def mini_badge(label, color: str = BLUE):
    r = _rgb(color)
    return rx.box(
        rx.text(label, font_size="11px", font_weight="600", color=color),
        padding_x="10px", padding_y="3px", border_radius="999px",
        background=f"rgba({r}, 0.1)",
        border=f"1px solid rgba({r}, 0.2)",
        display="inline-flex", align_items="center",
    )


def metric_row(label: str, value, color: str = TEXT):
    return rx.flex(
        rx.text(label, color=TEXT2, font_size="13px", flex="1"),
        rx.text(value, color=color, font_size="13px", font_weight="600"),
        justify="between", align="center", padding_y="7px",
        border_bottom="1px solid rgba(30, 41, 59, 0.2)", width="100%",
    )


def glow_dot(color: str, size: str = "8px", animate: bool = False):
    r = _rgb(color)
    return rx.box(
        width=size, height=size, border_radius="50%",
        background=color, flex_shrink="0",
        box_shadow=f"0 0 8px rgba({r}, 0.5)",
        animation="statusPulse 1.5s ease-in-out infinite" if animate else "none",
    )


# ═════════════════════════════════════════════════════════════════════════════
# LAUNCH SCREEN
# ═════════════════════════════════════════════════════════════════════════════


def _hero_particles():
    """Rich particle background for launch page only — many particles but no blur/blend-mode."""
    import random as _r

    def _make_dust(seed, count=25):
        _r.seed(seed)
        parts = []
        for _ in range(count):
            x = f"{_r.randint(1,98)}vw"
            y = f"{_r.randint(-110,5)}vh"
            c = _r.choice(["230,240,255", "120,170,255", "180,200,255", "110,220,255", "255,255,255"])
            o = round(_r.uniform(0.3, 0.85), 2)
            parts.append(f"{x} {y} {_r.choice([1,1,2,2,2])}px rgba({c},{o})")
        return ", ".join(parts)

    def _make_mote(seed, count=10):
        _r.seed(seed)
        parts = []
        for _ in range(count):
            x = f"{_r.randint(2,96)}vw"
            y = f"{_r.randint(-100,5)}vh"
            sz = _r.choice([3, 4, 4, 5, 5, 6])
            c = _r.choice(["230,240,255", "120,170,255", "60,130,240", "110,220,255"])
            o = round(_r.uniform(0.4, 0.85), 2)
            parts.append(f"{x} {y} {sz}px rgba({c},{o})")
            # glow halo
            go = round(o * 0.4, 2)
            parts.append(f"{x} {y} {sz * 3}px rgba({c},{go})")
        return ", ".join(parts)

    L = []

    # ── Nebulae — static gradients, no animation cost ──
    for nx_, ny_, sz_, clr_ in [
        ("25%", "30%", "800px", "rgba(60,130,240,0.07)"),
        ("70%", "60%", "650px", "rgba(120,170,255,0.05)"),
        ("50%", "45%", "900px", "rgba(60,130,240,0.06)"),
    ]:
        L.append(rx.box(
            width=sz_, height=sz_, border_radius="50%",
            background=f"radial-gradient(circle, {clr_} 0%, transparent 50%)",
            position="absolute", left=nx_, top=ny_, transform="translate(-50%,-50%)",
            pointer_events="none",
        ))

    # ── 6 dust fields (150 particles via box-shadow — cheap, just transform animation) ──
    dust_cfgs = [
        (1001, "ps5Rise1", "22s", "left:0;bottom:-5%"),
        (2003, "ps5Rise2", "26s", "left:0;bottom:-5%"),
        (3007, "ps5Rise1", "18s", "left:0;bottom:-5%"),
        (1402, "ps5Fall1", "24s", "left:0;top:-5%"),
        (1604, "ps5Fall1", "20s", "right:0;top:-5%"),
        (4005, "ps5Rise2", "28s", "left:0;bottom:-5%"),
    ]
    for seed, anim, dur, origin in dust_cfgs:
        pos = {k.strip(): v.strip() for k, v in (p.split(":") for p in origin.split(";"))}
        L.append(rx.box(
            width="1px", height="1px", border_radius="50%", background="transparent",
            box_shadow=_make_dust(seed), pointer_events="none",
            position="absolute", **pos,
            animation=f"{anim} {dur} linear infinite",
        ))

    # ── 4 mote fields (80 motes + halos — glowing larger particles) ──
    mote_cfgs = [
        (10, "ps5Rise1", "14s", "left:0;bottom:-5%"),
        (30, "ps5Rise2", "12s", "left:0;bottom:-5%"),
        (82, "ps5Fall1", "18s", "left:0;top:-5%"),
        (55, "ps5Rise1", "16s", "left:0;bottom:-5%"),
    ]
    for seed, anim, dur, origin in mote_cfgs:
        pos = {k.strip(): v.strip() for k, v in (p.split(":") for p in origin.split(";"))}
        L.append(rx.box(
            width="1px", height="1px", border_radius="50%", background="transparent",
            box_shadow=_make_mote(seed), pointer_events="none",
            position="absolute", **pos,
            animation=f"{anim} {dur} ease-in-out infinite",
        ))

    # ── 8 floating orbs — single animation each ──
    orb_cfgs = [
        ("18px", "rgba(110,220,255,0.35)", "10%", "20%", "ps5Float1 20s ease-in-out infinite"),
        ("24px", "rgba(60,130,240,0.25)", "50%", "65%", "ps5Float2 25s ease-in-out 4s infinite reverse"),
        ("14px", "rgba(180,200,255,0.40)", "80%", "40%", "ps5Float3 18s ease-in-out 2s infinite"),
        ("16px", "rgba(120,170,255,0.30)", "35%", "8%",  "ps5Float1 22s ease-in-out 6s infinite reverse"),
        ("20px", "rgba(110,220,255,0.28)", "15%", "75%", "ps5Float2 24s ease-in-out 3s infinite"),
        ("12px", "rgba(255,255,255,0.45)", "60%", "12%", "ps5Float3 16s ease-in-out 5s infinite reverse"),
        ("22px", "rgba(60,130,240,0.20)", "85%", "85%", "ps5Float1 28s ease-in-out 8s infinite"),
        ("10px", "rgba(255,255,255,0.50)", "45%", "5%",  "ps5Float2 14s ease-in-out 1s infinite reverse"),
    ]
    for sz, clr, x, y, anim in orb_cfgs:
        L.append(rx.box(
            width=sz, height=sz, border_radius="50%",
            background=f"radial-gradient(circle, {clr} 0%, transparent 65%)",
            position="absolute", left=x, top=y,
            animation=anim, pointer_events="none",
        ))

    # ── Vignette ──
    L.append(rx.box(
        position="absolute", inset="0",
        background="radial-gradient(ellipse at 50% 50%, transparent 0%, rgba(3,5,15,0.12) 65%, rgba(3,5,15,0.28) 100%)",
        pointer_events="none",
    ))

    return rx.box(*L, position="absolute", inset="0", overflow="hidden", pointer_events="none",
                  will_change="transform")


def _architecture_diagram():
    """Premium visual architecture diagram — 6-agent system with animated flow lines."""
    def _agent_node(name: str, weight: str, icon: str, color: str, delay: str):
        r = _rgb(color)
        return rx.box(
            rx.flex(
                rx.flex(
                    rx.icon(tag=icon, size=15, color=color),
                    align="center", justify="center",
                    width="32px", height="32px", border_radius="10px",
                    background=f"linear-gradient(135deg, rgba({r},0.15), rgba({r},0.05))",
                    border=f"1px solid rgba({r}, 0.20)",
                    box_shadow=f"0 0 20px rgba({r},0.08), inset 0 1px 0 rgba(255,255,255,0.12)",
                    flex_shrink="0",
                ),
                rx.box(
                    rx.text(name, font_size="13px", font_weight="700", color=TEXT,
                            line_height="1"),
                    rx.text(weight, font_size="10px", color=color, font_weight="600",
                            margin_top="2px", line_height="1"),
                ),
                align="center", gap="10px", width="100%",
            ),
            padding="10px 14px", border_radius="14px",
            background=f"linear-gradient(135deg, rgba({r},0.04), rgba({r},0.01))",
            border=f"1px solid rgba({r}, 0.10)",
            box_shadow=f"inset 0 1px 0 rgba(255,255,255,0.08), 0 2px 12px rgba(0,0,0,0.08)",
            cursor="pointer",
            transition=_APPLE_EASE,
            _hover={
                "border": f"1px solid rgba({r}, 0.30)",
                "transform": "translateX(4px) scale(1.02)",
                "box_shadow": f"inset 0 1px 0 rgba(255,255,255,0.15), 0 8px 28px rgba({r}, 0.12)",
                "background": f"linear-gradient(135deg, rgba({r},0.08), rgba({r},0.03))",
            },
            _active={
                "transform": "translateX(2px) scale(0.98)",
                "transition": "all 0.1s cubic-bezier(0.25,0.1,0.25,1.0)",
            },
            animation=_anim("fadeInUp", "0.5s", delay),
        )

    # Center merge node
    def _merge_node():
        return rx.flex(
            rx.box(
                rx.icon(tag="git-merge", size=16, color=CYAN),
                padding="8px", border_radius="50%",
                background="linear-gradient(135deg, rgba(6,182,212,0.10), rgba(59,130,246,0.05))",
                border="1px solid rgba(6,182,212,0.20)",
                box_shadow="0 0 30px rgba(6,182,212,0.10), inset 0 1px 0 rgba(255,255,255,0.15)",
                animation="pulse 3s ease-in-out infinite",
            ),
            justify="center", align="center",
            width="100%", padding_y="6px",
        )

    # Animated connecting lines
    def _flow_line(color: str, direction: str = "down"):
        r = _rgb(color)
        grad = f"linear-gradient(180deg, rgba({r},0.30), rgba({r},0.05))" if direction == "down" else f"linear-gradient(180deg, rgba({r},0.05), rgba({r},0.30))"
        return rx.box(
            width="1px", height="18px",
            background=grad,
            margin_left="auto", margin_right="auto",
            animation="pulse 3s ease-in-out infinite",
        )

    return rx.box(
        # Header row
        rx.flex(
            rx.flex(
                rx.box(width="6px", height="6px", border_radius="50%",
                       background=CYAN, box_shadow=f"0 0 10px {CYAN}",
                       animation="pulse 2s ease-in-out infinite"),
                rx.text("SYSTEM ARCHITECTURE", font_size="10px", color=MUTED,
                        font_weight="600", letter_spacing="0.15em"),
                align="center", gap="8px",
            ),
            rx.box(flex="1"),
            rx.flex(
                rx.box(width="18px", height="2px", border_radius="2px",
                       background=f"linear-gradient(90deg, {BLUE}, {CYAN})"),
                rx.text("50/50", color=MUTED, font_size="10px", font_weight="600"),
                rx.box(width="18px", height="2px", border_radius="2px",
                       background=f"linear-gradient(90deg, {PURPLE}, {CYAN})"),
                align="center", gap="6px",
            ),
            align="center", margin_bottom="16px",
        ),
        # Two-column agent layout
        rx.flex(
            # Technical Track (left)
            rx.vstack(
                rx.flex(
                    rx.flex(
                        rx.icon(tag="activity", size=14, color=BLUE),
                        align="center", justify="center",
                        width="28px", height="28px", border_radius="8px",
                        background="linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.04))",
                        border="1px solid rgba(59,130,246,0.15)",
                        box_shadow="inset 0 1px 0 rgba(255,255,255,0.10)",
                    ),
                    rx.text("Technical", color=BLUE, font_size="12px", font_weight="700",
                            letter_spacing="0.03em"),
                    align="center", gap="8px",
                ),
                _flow_line(BLUE, "down"),
                _agent_node("Fardeen", "25%", "bar-chart-3", BLUE, "0.2s"),
                _agent_node("Tamer", "25%", "trending-up", BLUE, "0.3s"),
                spacing="1", flex="1", width="100%",
                padding="14px", border_radius="16px",
                background="rgba(59,130,246,0.015)",
                border="1px solid rgba(59,130,246,0.06)",
            ),
            # Center merge column
            rx.vstack(
                rx.box(height="30px"),
                rx.box(
                    width="1px", flex="1", min_height="30px",
                    background=f"linear-gradient(180deg, rgba(59,130,246,0.20), rgba(6,182,212,0.30))",
                ),
                _merge_node(),
                rx.box(
                    width="1px", flex="1", min_height="30px",
                    background=f"linear-gradient(180deg, rgba(6,182,212,0.30), rgba(139,92,246,0.20))",
                ),
                spacing="0", align="center",
                width="36px", flex_shrink="0",
            ),
            # Fundamental Track (right)
            rx.vstack(
                rx.flex(
                    rx.flex(
                        rx.icon(tag="building-2", size=14, color=PURPLE),
                        align="center", justify="center",
                        width="28px", height="28px", border_radius="8px",
                        background="linear-gradient(135deg, rgba(139,92,246,0.12), rgba(139,92,246,0.04))",
                        border="1px solid rgba(139,92,246,0.15)",
                        box_shadow="inset 0 1px 0 rgba(255,255,255,0.10)",
                    ),
                    rx.text("Fundamental", color=PURPLE, font_size="12px", font_weight="700",
                            letter_spacing="0.03em"),
                    align="center", gap="8px",
                ),
                _flow_line(PURPLE, "down"),
                _agent_node("Daria", "12.5%", "file-search", PURPLE, "0.25s"),
                _agent_node("Shakzod", "12.5%", "calculator", PURPLE, "0.35s"),
                _agent_node("Lary", "12.5%", "database", CYAN, "0.45s"),
                _agent_node("Mohamed", "12.5%", "pie-chart", CYAN, "0.55s"),
                spacing="1", flex="1", width="100%",
                padding="14px", border_radius="16px",
                background="rgba(139,92,246,0.015)",
                border="1px solid rgba(139,92,246,0.06)",
            ),
            gap="0", width="100%", align="stretch",
        ),
        # Output row — final recommendation
        rx.flex(
            rx.box(
                width="100%", height="1px",
                background=f"linear-gradient(90deg, transparent, rgba(6,182,212,0.20), transparent)",
                margin_y="10px",
            ),
        ),
        rx.flex(
            rx.flex(
                rx.icon(tag="zap", size=13, color=AMBER),
                rx.text("Final Recommendation", font_size="11px", color=AMBER,
                        font_weight="600", letter_spacing="0.04em"),
                align="center", gap="6px",
            ),
            justify="center",
            animation=_anim("fadeIn", "0.6s", "0.6s"),
        ),
        padding="20px", border_radius="20px",
        background=_GLASS_BG,
        border=_GLASS_BORDER,
        box_shadow=_GLASS_SHADOW,
        animation=_anim("fadeInUp", "0.7s", "0.15s"),
        width="100%", max_width="580px",
    )


def _feature_badge(icon: str, label: str, color: str, delay: str):
    """iOS 26 Liquid Glass feature badge with premium hover + active press."""
    r = _rgb(color)
    return rx.box(
        rx.flex(
            rx.box(
                rx.icon(tag=icon, size=16, color=color),
                padding="10px", border_radius="12px",
                background=f"linear-gradient(135deg, rgba({r},0.12), rgba({r},0.04))",
                border=f"1px solid rgba({r}, 0.15)",
                box_shadow=f"inset 0 1px 0 rgba(255,255,255,0.15), 0 2px 8px rgba({r},0.08)",
                transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
            ),
            rx.text(label, color=TEXT2, font_size="11px", font_weight="600", letter_spacing="0.02em"),
            direction="column", align="center", gap="8px",
        ),
        padding="16px 14px", border_radius="16px",
        background="linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))",
        border="1px solid rgba(255,255,255,0.08)",
        box_shadow="inset 0 1px 0 rgba(255,255,255,0.10), 0 4px 12px rgba(0,0,0,0.12)",
        transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        cursor="pointer",
        _hover={
            "border": f"1px solid rgba({r}, 0.30)",
            "transform": "translateY(-5px) scale(1.04)",
            "box_shadow": f"inset 0 1px 0 rgba(255,255,255,0.18), 0 16px 40px rgba({r}, 0.12), 0 0 20px rgba({r},0.06)",
            "background": f"linear-gradient(135deg, rgba({r},0.08), rgba(255,255,255,0.04))",
        },
        _active={
            "transform": "translateY(-1px) scale(0.97)",
            "box_shadow": f"inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 8px rgba({r}, 0.08)",
            "transition": "all 0.1s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        },
        animation=_anim("fadeInUp", "0.5s", delay),
        min_width="90px", text_align="center",
    )


def launch_screen():
    return rx.box(
        _hero_particles(),
        rx.flex(
            # ═══ MAIN CONTENT — vertically + horizontally centered ═══
            rx.vstack(
                # ── Brain Icon — perfectly centered ──
                rx.flex(
                    rx.box(
                        # Outer orbit ring
                        rx.box(
                            width="120px", height="120px", border_radius="50%",
                            border="1px solid rgba(59,130,246,0.10)",
                            position="absolute", inset="0", margin="auto",
                        ),
                        # Orbit dot — orbiting
                        rx.box(
                            width="5px", height="5px", border_radius="50%",
                            background=CYAN,
                            box_shadow=f"0 0 14px {CYAN}, 0 0 28px rgba(6,182,212,0.4)",
                            position="absolute", top="0px", left="50%",
                            transform="translateX(-50%)",
                            animation="orbitSpin 18s linear infinite",
                        ),
                        # Second orbit dot (opposite, slower)
                        rx.box(
                            width="4px", height="4px", border_radius="50%",
                            background=PURPLE,
                            box_shadow=f"0 0 10px {PURPLE}",
                            position="absolute", bottom="0px", left="50%",
                            transform="translateX(-50%)",
                            animation="orbitSpin 18s linear infinite reverse",
                        ),
                        # Outer glow ring
                        rx.box(
                            width="96px", height="96px", border_radius="50%",
                            border="1.5px solid rgba(59,130,246,0.06)",
                            box_shadow="0 0 50px rgba(59,130,246,0.06)",
                            position="absolute", inset="0", margin="auto",
                            animation="pulse 4s ease-in-out infinite",
                        ),
                        # Glass brain circle
                        rx.flex(
                            rx.icon(tag="brain", size=38, color=BLUE),
                            align="center", justify="center",
                            width="80px", height="80px", border_radius="50%",
                            background="linear-gradient(145deg, rgba(59,130,246,0.12), rgba(139,92,246,0.06), rgba(6,182,212,0.04))",
                            border="1px solid rgba(255,255,255,0.14)",
                            box_shadow="inset 0 1px 0 rgba(255,255,255,0.22), "
                                       "0 0 80px rgba(59,130,246,0.10), 0 16px 48px rgba(0,0,0,0.25)",
                            position="absolute", inset="0", margin="auto",
                            animation="borderGlow 4s ease-in-out infinite, float 6s ease-in-out infinite",
                        ),
                        position="relative", width="128px", height="128px",
                    ),
                    justify="center", width="100%",
                    animation=_anim("scaleIn", "0.9s"),
                ),
                # ── Title ──
                rx.box(
                    rx.text("Intelligent Stock", font_size="46px", font_weight="800",
                            letter_spacing="-0.03em", text_align="center",
                            line_height="1.08", font_style="normal",
                            background_image=GRADIENT_BRAND,
                            background_clip="text",
                            style={"WebkitBackgroundClip": "text",
                                   "WebkitTextFillColor": "transparent",
                                   "fontStyle": "normal"}),
                    rx.text("Analysis Agent", font_size="46px", font_weight="800",
                            letter_spacing="-0.03em", text_align="center",
                            line_height="1.08", margin_top="-1px",
                            color=TEXT, font_style="normal"),
                    animation=_anim("fadeInUp", "0.8s", "0.08s"),
                    margin_top="28px", width="100%",
                ),
                # ── Subtitle ──
                rx.text("6 specialized AI agents analyzing technicals, fundamentals, "
                        "risk, and backtesting in parallel.",
                        color=MUTED, font_size="14px", margin_top="16px",
                        letter_spacing="0.005em", text_align="center",
                        max_width="420px", line_height="1.6",
                        animation=_anim("fadeInUp", "0.8s", "0.14s"),
                ),
                # ── Ticker Input + Launch Button ──
                rx.flex(
                    rx.el.input(
                        value=DashboardState.ticker_input,
                        on_change=DashboardState.set_ticker,
                        placeholder="AAPL",
                        auto_complete="off",
                        style={
                            "background": "linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015))",
                            "border": "1px solid rgba(255,255,255,0.10)",
                            "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.12), 0 4px 20px rgba(0,0,0,0.20)",
                            "color": TEXT, "font_size": "28px", "font_weight": "800",
                            "text_align": "center", "letter_spacing": "0.15em",
                            "text_transform": "uppercase",
                            "width": "160px", "border_radius": "16px",
                            "height": "58px", "line_height": "58px",
                            "padding": "0 16px",
                            "outline": "none",
                            "flex_shrink": "0",
                            "transition": _APPLE_EASE,
                            "::placeholder": {"color": "rgba(148,163,184,0.25)", "font_weight": "700"},
                            ":focus": {
                                "border": "1px solid rgba(59,130,246,0.35)",
                                "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.18), "
                                              "0 0 50px rgba(59,130,246,0.12), 0 8px 32px rgba(0,0,0,0.25)",
                                "transform": "scale(1.03)",
                            },
                        },
                    ),
                    rx.button(
                        rx.flex(
                            rx.icon(tag="sparkles", size=16),
                            rx.text("Launch Analysis", font_size="14px", font_weight="700",
                                    letter_spacing="0.02em"),
                            align="center", gap="8px",
                        ),
                        on_click=DashboardState.start_analysis, size="3",
                        style={
                            "background": f"{GRADIENT_BRAND} !important",
                            "background_size": "200% 100%",
                            "border": "1px solid rgba(255,255,255,0.18) !important",
                            "border_radius": "16px !important",
                            "padding": "0 28px !important", "height": "58px",
                            "cursor": "pointer",
                            "transition": f"{_APPLE_EASE} !important",
                            "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.22), "
                                          "0 4px 24px rgba(59,130,246,0.28), "
                                          "0 0 60px rgba(59,130,246,0.06) !important",
                            "animation": "gradientShift 3s ease-in-out infinite",
                            "_hover": {
                                "transform": "translateY(-3px) scale(1.03)",
                                "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.28), "
                                              "0 16px 48px rgba(59,130,246,0.35), "
                                              "0 0 100px rgba(59,130,246,0.10) !important",
                            },
                            "_active": {
                                "transform": "scale(0.97)",
                                "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.10), "
                                              "0 2px 10px rgba(59,130,246,0.18) !important",
                                "transition": "all 0.1s cubic-bezier(0.25,0.1,0.25,1.0) !important",
                            },
                        },
                    ),
                    align="center", justify="center", gap="12px",
                    animation=_anim("fadeInUp", "0.7s", "0.2s"),
                    margin_top="32px",
                ),
                # ── Architecture Diagram ──
                rx.box(
                    _architecture_diagram(),
                    margin_top="32px", width="100%", max_width="580px",
                ),
                # ── Existing data banner ──
                rx.cond(
                    DashboardState.has_existing_data,
                    rx.box(
                        rx.flex(
                            rx.flex(
                                rx.icon(tag="folder-open", size=16, color=AMBER),
                                align="center", justify="center",
                                width="34px", height="34px", border_radius="10px",
                                background="linear-gradient(135deg, rgba(245,158,11,0.12), rgba(245,158,11,0.04))",
                                border="1px solid rgba(245,158,11,0.15)",
                                box_shadow="inset 0 1px 0 rgba(255,255,255,0.10)",
                                flex_shrink="0",
                            ),
                            rx.box(
                                rx.text("Previous Analysis Available", color=TEXT,
                                        font_size="13px", font_weight="600"),
                                rx.flex(
                                    rx.text(DashboardState.existing_ticker, color=AMBER,
                                            font_weight="700", font_size="12px"),
                                    rx.text(" \u2022 ", color=MUTED, font_size="12px"),
                                    rx.text(DashboardState.existing_date, color=MUTED,
                                            font_size="11px"),
                                    align="center", margin_top="1px",
                                ),
                                flex="1",
                            ),
                            rx.button(
                                rx.flex(
                                    rx.icon(tag="arrow-right", size=14),
                                    rx.text("View", font_size="13px", font_weight="700"),
                                    align="center", gap="5px",
                                ),
                                on_click=DashboardState.view_existing_results,
                                size="2", variant="ghost",
                                style={"color": AMBER, "cursor": "pointer",
                                       "background": "linear-gradient(135deg, rgba(245,158,11,0.10), rgba(245,158,11,0.04)) !important",
                                       "border": "1px solid rgba(245,158,11,0.18) !important",
                                       "border_radius": "12px !important", "padding": "8px 16px",
                                       "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.08) !important",
                                       "_hover": {
                                           "background": "linear-gradient(135deg, rgba(245,158,11,0.16), rgba(245,158,11,0.06)) !important",
                                           "border": "1px solid rgba(245,158,11,0.30) !important",
                                           "transform": "translateY(-2px) scale(1.03)",
                                       },
                                       "_active": {
                                           "transform": "scale(0.97)",
                                           "transition": "all 0.1s cubic-bezier(0.25,0.1,0.25,1.0)",
                                       }},
                            ),
                            align="center", gap="12px",
                        ),
                        padding="14px 18px", border_radius="16px",
                        background="linear-gradient(135deg, rgba(245,158,11,0.03), rgba(245,158,11,0.01))",
                        border="1px solid rgba(245,158,11,0.08)",
                        box_shadow="inset 0 1px 0 rgba(255,255,255,0.06), 0 4px 12px rgba(0,0,0,0.08)",
                        margin_top="20px", animation=_anim("fadeInUp", "0.6s", "0.4s"),
                        width="100%", max_width="480px",
                        transition=_APPLE_EASE,
                        _hover={
                            "border": "1px solid rgba(245,158,11,0.18)",
                            "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.10), 0 8px 24px rgba(245,158,11,0.06)",
                            "transform": "translateY(-1px)",
                        },
                    ),
                    rx.fragment(),
                ),
                align="center", spacing="0", width="100%",
                max_width="640px", padding_x="24px",
            ),
            # ── Bottom: scrolling ticker + footer ──
            rx.vstack(
                rx.box(
                    rx.box(
                        rx.flex(
                            rx.text("RSI \u2022 MACD \u2022 Bollinger Bands \u2022 ADX \u2022 ATR \u2022 Hurst \u2022 "
                                    "P/E \u2022 ROE \u2022 DCF \u2022 Sharpe \u2022 Sortino \u2022 VaR \u2022 "
                                    "Monte Carlo \u2022 Mean Reversion \u2022 Momentum \u2022 Regime \u2022 "
                                    "RSI \u2022 MACD \u2022 Bollinger Bands \u2022 ADX \u2022 ATR \u2022 Hurst \u2022 "
                                    "P/E \u2022 ROE \u2022 DCF \u2022 Sharpe \u2022 Sortino \u2022 VaR",
                                    color=MUTED, font_size="10px", letter_spacing="0.06em",
                                    white_space="nowrap", font_weight="500",
                                    opacity="0.6"),
                            animation="tickerScroll 35s linear infinite",
                        ),
                        overflow="hidden", width="100%", max_width="500px",
                        margin_left="auto", margin_right="auto",
                        padding_y="8px",
                        border_top="1px solid rgba(30, 41, 59, 0.08)",
                    ),
                    animation=_anim("fadeIn", "1s", "0.5s"),
                    width="100%",
                ),
                rx.flex(
                    rx.text("Powered by", color=MUTED, font_size="10px", opacity="0.5"),
                    rx.text("Claude", color=BLUE, font_size="10px", font_weight="600", opacity="0.7"),
                    rx.text("&", color=MUTED, font_size="10px", opacity="0.5"),
                    rx.text("GPT-4o", color=GREEN, font_size="10px", font_weight="600", opacity="0.7"),
                    rx.text("\u2022", color=MUTED, font_size="10px", opacity="0.5"),
                    rx.text("UCL MSc Project", color=PURPLE, font_size="10px", font_weight="600", opacity="0.7"),
                    gap="5px", margin_top="8px", margin_bottom="16px",
                    align="center", justify="center",
                    animation=_anim("fadeIn", "1s", "0.6s"),
                ),
                spacing="0", width="100%", align="center",
            ),
            direction="column", align="center", justify="center",
            min_height="100vh", width="100%",
            padding_top="32px", padding_bottom="0px",
            position="relative", z_index="1",
        ),
        position="relative", width="100%", min_height="100vh", overflow="hidden",
        background="radial-gradient(ellipse at 35% 45%, #081840 0%, #060E2E 25%, #040A1E 55%, #020612 80%, #010308 100%)",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PROGRESS SCREEN — Animations & Live Monitoring
# ═════════════════════════════════════════════════════════════════════════════

def _render_agent_status(agent: dict):
    """Individual agent status card during pipeline execution."""
    return rx.box(
        # Top accent gradient
        rx.box(height="3px", background=agent["card_accent"], border_radius="2px"),
        rx.box(
            # Name + status row
            rx.flex(
                rx.flex(
                    rx.text(agent["name"], color=TEXT, font_size="15px",
                            font_weight="700", white_space="nowrap"),
                    rx.box(
                        rx.text(agent["type"], font_size="9px", font_weight="700",
                                color=agent["type_color"]),
                        padding_x="7px", padding_y="2px", border_radius="999px",
                        background=agent["type_badge_bg"],
                        border=agent["type_badge_border"],
                        flex_shrink="0",
                    ),
                    align="center", gap="8px", flex="1", min_width="0",
                ),
                rx.flex(
                    rx.box(
                        width="7px", height="7px", border_radius="50%",
                        background=agent["status_color"], flex_shrink="0",
                        box_shadow=rx.cond(
                            agent["status"] == "running",
                            f"0 0 8px {agent['status_color']}",
                            "none",
                        ),
                        animation=rx.cond(
                            agent["status"] == "running",
                            "statusPulse 1.5s ease-in-out infinite",
                            "none",
                        ),
                    ),
                    rx.text(agent["status_text"], color=agent["status_color"],
                            font_size="11px", font_weight="700", white_space="nowrap"),
                    align="center", gap="5px", flex_shrink="0",
                ),
                justify="between", align="center", width="100%",
            ),
            # Weight + duration
            rx.flex(
                rx.text(agent["weight"], color=MUTED, font_size="11px"),
                rx.cond(
                    agent["duration"] != "\u2014",
                    rx.text(agent["duration"], color=MUTED, font_size="11px"),
                    rx.fragment(),
                ),
                justify="between", align="center", width="100%", margin_top="4px",
            ),
            # Signal + target (when completed)
            rx.cond(
                agent["signal"] != "\u2014",
                rx.flex(
                    rx.text("Signal:", color=MUTED, font_size="10px"),
                    rx.text(agent["signal"], color=TEXT, font_size="10px", font_weight="700"),
                    rx.box(width="1px", height="10px", background="rgba(100,116,139,0.3)"),
                    rx.text("Target:", color=MUTED, font_size="10px"),
                    rx.text(agent["target"], color=TEXT, font_size="10px", font_weight="700",
                            overflow="hidden", text_overflow="ellipsis", white_space="nowrap"),
                    align="center", gap="4px", margin_top="10px", padding_top="10px",
                    border_top="1px solid rgba(30, 41, 59, 0.25)", width="100%",
                    overflow="hidden",
                ),
                rx.fragment(),
            ),
            padding="14px 16px",
        ),
        border_radius="14px", background=BG_CARD,
        border=rx.cond(
            agent["status"] == "completed",
            "1px solid rgba(16, 185, 129, 0.2)",
            BORDER,
        ),
        overflow="hidden",
        transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        min_width="0",
    )


def _render_log_line(line: str):
    return rx.text(
        line, font_size="12px",
        font_family="'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
        color=TEXT2, line_height="1.7",
    )


def _bouncing_balls():
    """5 bouncing agent indicator balls with labels."""
    colors = [BLUE, PURPLE, CYAN, GREEN, AMBER]
    names = ["Fardeen", "Tamer", "Daria", "Shakzod", "Mohamed"]
    balls = []
    for i, (color, name) in enumerate(zip(colors, names)):
        r = _rgb(color)
        balls.append(
            rx.box(
                rx.box(
                    width="16px", height="16px", border_radius="50%",
                    background=f"radial-gradient(circle at 30% 30%, {color}, rgba({r}, 0.5))",
                    box_shadow=f"0 0 16px rgba({r}, 0.5), 0 0 32px rgba({r}, 0.15)",
                    animation=f"jumpBall 1s ease-in-out {i * 0.12}s infinite",
                ),
                rx.text(name[:1], font_size="8px", color=MUTED, font_weight="700",
                        margin_top="8px", text_align="center", letter_spacing="0.05em"),
                display="flex", flex_direction="column", align_items="center",
            )
        )
    return rx.flex(
        *balls, align="end", justify="center", gap="20px",
        height="65px", padding_bottom="4px",
    )


def _wave_equalizer():
    """Audio wave-style equalizer animation."""
    bars = []
    colors_seq = [BLUE, PURPLE, CYAN, BLUE, PURPLE, CYAN, GREEN, INDIGO]
    for i, c in enumerate(colors_seq):
        r = _rgb(c)
        bars.append(
            rx.box(
                width="4px", min_height="4px", border_radius="3px",
                background=f"linear-gradient(180deg, {c}, rgba({r}, 0.2))",
                box_shadow=f"0 0 6px rgba({r}, 0.2)",
                animation=f"waveBar 1.2s ease-in-out {i * 0.07}s infinite",
            )
        )
    return rx.flex(*bars, align="end", gap="3px", height="34px")


def _orbital_brain():
    """Brain icon with expanding sonar ring pulses."""
    return rx.box(
        rx.box(
            position="absolute", top="-7px", left="-7px", right="-7px", bottom="-7px",
            border_radius="22px",
            border="1px solid rgba(59, 130, 246, 0.25)",
            animation="sonarPulse 2.5s ease-out infinite",
        ),
        rx.box(
            position="absolute", top="-14px", left="-14px", right="-14px", bottom="-14px",
            border_radius="28px",
            border="1px solid rgba(139, 92, 246, 0.15)",
            animation="sonarPulse 2.5s ease-out 0.5s infinite",
        ),
        rx.box(
            position="absolute", top="-22px", left="-22px", right="-22px", bottom="-22px",
            border_radius="36px",
            border="1px solid rgba(6, 182, 212, 0.08)",
            animation="sonarPulse 2.5s ease-out 1s infinite",
        ),
        rx.icon(tag="brain", size=26, color=BLUE),
        padding="14px", border_radius="16px",
        background="rgba(59, 130, 246, 0.06)",
        border="1px solid rgba(59, 130, 246, 0.12)",
        box_shadow="0 0 30px rgba(59, 130, 246, 0.06)",
        position="relative", overflow="visible",
    )


def progress_screen():
    return rx.center(
        rx.vstack(
            # ── Header: Brain + Title + Timer ──
            rx.flex(
                _orbital_brain(),
                rx.box(
                    rx.text("Master Analyst Agent", font_size="22px", font_weight="800",
                            letter_spacing="-0.02em",
                            background_image=GRADIENT_BRAND,
                            background_clip="text", background_size="200% auto",
                            style={"WebkitBackgroundClip": "text",
                                   "WebkitTextFillColor": "transparent"},
                            animation="gradientShift 3s ease-in-out infinite"),
                    rx.flex(
                        rx.text("Analyzing ", color=TEXT2, font_size="13px"),
                        rx.text(DashboardState.ticker_input, color=BLUE,
                                font_size="13px", font_weight="700"),
                        gap="2px",
                    ),
                ),
                rx.box(flex="1"),
                # Timer display
                rx.box(
                    rx.text(DashboardState.elapsed_str, color=TEXT, font_size="30px",
                            font_weight="800",
                            font_family="'JetBrains Mono', monospace",
                            letter_spacing="-0.02em"),
                    rx.text("elapsed", color=MUTED, font_size="10px",
                            text_transform="uppercase", letter_spacing="0.12em"),
                    text_align="right", padding="10px 18px", border_radius="14px",
                    background="rgba(59, 130, 246, 0.03)",
                    border="1px solid rgba(59, 130, 246, 0.08)",
                    box_shadow="0 0 30px rgba(59, 130, 246, 0.03)",
                ),
                align="center", gap="16px", width="100%", margin_bottom="32px",
                animation=_anim("fadeInUp", "0.6s"),
            ),
            # ── Agent Counter + Wave ──
            rx.flex(
                rx.box(
                    rx.flex(
                        rx.text(DashboardState.agents_completed_count,
                                font_size="36px", font_weight="800", color=BLUE,
                                font_family="'JetBrains Mono', monospace"),
                        rx.text("/6", font_size="22px", font_weight="600",
                                color=MUTED, margin_top="10px"),
                        align="end", gap="2px",
                    ),
                    rx.text("Agents Completed", color=TEXT2, font_size="11px",
                            text_transform="uppercase", letter_spacing="0.08em",
                            margin_top="2px"),
                    text_align="center", padding="18px 28px", border_radius="16px",
                    background="rgba(59, 130, 246, 0.04)",
                    border="1px solid rgba(59, 130, 246, 0.08)",
                ),
                rx.cond(
                    DashboardState.is_running,
                    rx.box(
                        rx.flex(
                            _wave_equalizer(),
                            rx.text("Processing...", color=TEXT2, font_size="12px",
                                    font_weight="500"),
                            align="center", gap="12px",
                        ),
                        padding="18px 28px", border_radius="16px",
                        background="rgba(59, 130, 246, 0.02)",
                        border="1px solid rgba(59, 130, 246, 0.06)",
                    ),
                    rx.fragment(),
                ),
                justify="center", gap="14px", width="100%", margin_bottom="10px",
                animation=_anim("fadeInUp", "0.6s", "0.05s"),
            ),
            # ── Bouncing Balls ──
            rx.cond(
                DashboardState.is_running,
                rx.box(
                    _bouncing_balls(),
                    padding="14px 0", margin_bottom="20px",
                    animation=_anim("fadeIn", "0.8s", "0.2s"),
                ),
                rx.fragment(),
            ),
            # ── Progress Bar ──
            rx.box(
                rx.flex(
                    rx.text(DashboardState.pipeline_step, color=TEXT2, font_size="13px"),
                    rx.text(
                        DashboardState.pipeline_progress.to(str) + "%",
                        color=BLUE, font_size="13px", font_weight="700",
                    ),
                    justify="between", margin_bottom="10px",
                ),
                rx.box(
                    rx.box(
                        width=DashboardState.pipeline_progress.to(str) + "%",
                        height="100%",
                        background=GRADIENT_BRAND,
                        background_size="200% auto", border_radius="999px",
                        transition="width 0.8s cubic-bezier(0.16, 1, 0.3, 1)",
                        box_shadow="0 0 20px rgba(59, 130, 246, 0.35), 0 0 6px rgba(59, 130, 246, 0.5)",
                    ),
                    width="100%", height="8px",
                    background="rgba(30, 41, 59, 0.4)",
                    border_radius="999px", overflow="hidden",
                ),
                width="100%", margin_bottom="28px",
                animation=_anim("fadeInUp", "0.6s", "0.1s"),
            ),
            # ── Agent Status Grid ──
            rx.box(
                rx.flex(
                    rx.text("AGENT STATUS", font_size="10px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.flex(
                        glow_dot("#f59e0b", "6px", True),
                        rx.text("Running", color=MUTED, font_size="10px"),
                        glow_dot("#10b981", "6px", False),
                        rx.text("Done", color=MUTED, font_size="10px"),
                        align="center", gap="6px",
                    ),
                    align="center", margin_bottom="14px",
                ),
                rx.grid(
                    rx.foreach(DashboardState.agent_statuses, _render_agent_status),
                    columns=rx.breakpoints(initial="1", sm="2", lg="3"),
                    spacing="3", width="100%",
                ),
                width="100%", animation=_anim("fadeInUp", "0.7s", "0.2s"),
            ),
            # ── Live Log ──
            rx.box(
                rx.flex(
                    rx.box(
                        rx.icon(tag="terminal", size=14, color=MUTED),
                        padding="6px", border_radius="8px",
                        background="rgba(100, 116, 139, 0.06)",
                    ),
                    rx.text("LIVE LOG", font_size="10px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.cond(
                        DashboardState.is_running,
                        glow_dot(GREEN, "6px", True),
                        rx.fragment(),
                    ),
                    align="center", gap="8px", margin_bottom="12px",
                ),
                rx.box(
                    rx.foreach(DashboardState.pipeline_log, _render_log_line),
                    max_height="220px", overflow_y="auto", padding="18px",
                    border_radius="14px",
                    background="rgba(0, 0, 0, 0.25)",
                    border="1px solid rgba(30, 41, 59, 0.25)",
                ),
                width="100%", margin_top="24px",
                animation=_anim("fadeInUp", "0.7s", "0.3s"),
            ),
            # ── Error Display ──
            rx.cond(
                DashboardState.run_error != "",
                rx.box(
                    rx.flex(
                        rx.icon(tag="triangle-alert", size=16, color=RED),
                        rx.text(DashboardState.run_error, color=RED, font_size="13px"),
                        align="center", gap="10px",
                    ),
                    padding="14px 18px", border_radius="14px",
                    background="rgba(239, 68, 68, 0.04)",
                    border="1px solid rgba(239, 68, 68, 0.12)",
                    margin_top="20px",
                    animation=_anim("fadeInUp", "0.4s"),
                ),
                rx.fragment(),
            ),
            # ── View Results Button ──
            rx.cond(
                DashboardState.pipeline_progress == 100,
                rx.box(
                    rx.flex(
                        rx.box(
                            rx.icon(tag="circle-check", size=22, color=GREEN),
                            padding="10px", border_radius="12px",
                            background="rgba(16, 185, 129, 0.06)",
                            animation="pulse 2s ease-in-out infinite",
                        ),
                        rx.text("Analysis Complete", color=GREEN,
                                font_size="16px", font_weight="700"),
                        align="center", gap="10px", justify="center",
                        margin_bottom="18px",
                    ),
                    rx.button(
                        rx.flex(
                            rx.icon(tag="arrow-right", size=18),
                            rx.text("View Full Results", font_size="15px", font_weight="700"),
                            align="center", gap="8px",
                        ),
                        on_click=DashboardState.view_results_now, size="3",
                        style={
                            "width": "100%",
                            "background": GRADIENT_SUCCESS,
                            "border": "none", "border_radius": "14px",
                            "padding": "14px 28px", "cursor": "pointer",
                            "box_shadow": "0 4px 20px rgba(16, 185, 129, 0.15)",
                            "_hover": {
                                "transform": "translateY(-3px)",
                                "box_shadow": "0 12px 40px rgba(16, 185, 129, 0.25)",
                            },
                        },
                    ),
                    width="100%", max_width="360px",
                    margin_top="28px", animation=_anim("scaleIn", "0.6s"),
                ),
                rx.fragment(),
            ),
            # ── Back to Launch ──
            rx.cond(
                ~DashboardState.is_running,
                rx.text("Back to Launch", color=MUTED, font_size="13px",
                        cursor="pointer", margin_top="18px",
                        _hover={"color": TEXT}, on_click=DashboardState.go_to_launch,
                        transition="color 0.4s cubic-bezier(0.25, 0.1, 0.25, 1.0)"),
                rx.fragment(),
            ),
            width="100%", max_width="820px", spacing="0", align="center",
        ),
        min_height="100vh", width="100%", padding="44px 28px",
        background=f"radial-gradient(ellipse at 50% 15%, rgba(59, 130, 246, 0.04) 0%, rgba(139, 92, 246, 0.015) 40%, {BG} 70%)",
    )


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR (Results View)
# ═════════════════════════════════════════════════════════════════════════════

def nav_item(label: str, icon: str, tab_id: str):
    return rx.box(
        rx.flex(
            rx.box(
                rx.icon(tag=icon, size=16, color=rx.cond(DashboardState.active_tab == tab_id, BLUE, TEXT2)),
                padding="8px", border_radius="10px",
                background=rx.cond(
                    DashboardState.active_tab == tab_id,
                    "linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.05))",
                    "rgba(255,255,255,0.03)",
                ),
                border=rx.cond(
                    DashboardState.active_tab == tab_id,
                    "1px solid rgba(59,130,246,0.20)",
                    "1px solid rgba(255,255,255,0.05)",
                ),
                box_shadow=rx.cond(
                    DashboardState.active_tab == tab_id,
                    "inset 0 1px 0 rgba(255,255,255,0.15), 0 2px 8px rgba(59,130,246,0.10)",
                    "inset 0 1px 0 rgba(255,255,255,0.06)",
                ),
                transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
            ),
            rx.text(label, font_size="13px", font_weight="600", letter_spacing="-0.01em"),
            align="center", gap="12px",
        ),
        padding_x="14px", padding_y="10px", border_radius="16px",
        cursor="pointer", transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        background=rx.cond(
            DashboardState.active_tab == tab_id,
            "linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))",
            "transparent",
        ),
        color=rx.cond(
            DashboardState.active_tab == tab_id, TEXT, TEXT2,
        ),
        border=rx.cond(
            DashboardState.active_tab == tab_id,
            "1px solid rgba(255,255,255,0.08)",
            "1px solid transparent",
        ),
        box_shadow=rx.cond(
            DashboardState.active_tab == tab_id,
            "inset 0 1px 0 rgba(255,255,255,0.10), 0 4px 16px rgba(0,0,0,0.15)",
            "none",
        ),
        _hover={
            "background": "linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))",
            "color": TEXT,
            "transform": "translateX(4px)",
            "border": "1px solid rgba(255,255,255,0.06)",
            "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.08)",
        },
        on_click=DashboardState.set_tab(tab_id),
    )


def sidebar():
    return rx.box(
        rx.vstack(
            # Logo — iOS 26 Liquid Glass icon
            rx.box(
                rx.flex(
                    rx.box(
                        rx.icon(tag="brain", size=20, color=BLUE),
                        padding="10px", border_radius="14px",
                        background="linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.04))",
                        border="1px solid rgba(59,130,246,0.15)",
                        box_shadow="inset 0 1px 0 rgba(255,255,255,0.15), 0 4px 12px rgba(59,130,246,0.08)",
                    ),
                    rx.box(
                        rx.text("Master Agent", font_size="16px", font_weight="800",
                                color=TEXT, letter_spacing="-0.01em"),
                        rx.text("Investment Analysis", font_size="11px", color=MUTED,
                                letter_spacing="0.05em"),
                    ),
                    align="center", gap="12px",
                ),
                margin_bottom="32px",
            ),
            # Nav
            rx.text("NAVIGATION", font_size="10px", color=MUTED, font_weight="600",
                    letter_spacing="0.12em", padding_x="16px", margin_bottom="10px"),
            nav_item("Overview", "layout-grid", "overview"),
            nav_item("Agents", "users", "agents"),
            nav_item("Analysis", "bar-chart-3", "analysis"),
            nav_item("Thesis & Strategy", "lightbulb", "thesis"),
            nav_item("Risk & Catalysts", "shield-alert", "risk"),
            nav_item("Monitoring", "eye", "monitoring"),
            # Ticker info
            rx.box(
                rx.box(height="1px", background="rgba(30, 41, 59, 0.4)", margin_y="24px"),
                rx.text("ANALYSIS TARGET", font_size="10px", color=MUTED,
                        font_weight="600", letter_spacing="0.12em", margin_bottom="14px"),
                rx.box(
                    rx.text(DashboardState.ticker, font_size="26px", font_weight="800",
                            letter_spacing="-0.02em",
                            background_image=GRADIENT_BRAND,
                            background_clip="text",
                            style={"WebkitBackgroundClip": "text",
                                   "WebkitTextFillColor": "transparent"}),
                    rx.text(DashboardState.company_name, font_size="13px",
                            color=TEXT2, margin_top="2px"),
                    rx.text(DashboardState.analysis_date, font_size="12px",
                            color=MUTED, margin_top="4px"),
                ),
                rx.box(height="1px", background="rgba(30, 41, 59, 0.4)", margin_y="24px"),
                rx.text("\u2190 New Analysis", color=MUTED, font_size="12px",
                        cursor="pointer", _hover={"color": BLUE},
                        on_click=DashboardState.go_to_launch,
                        transition="color 0.4s cubic-bezier(0.25, 0.1, 0.25, 1.0)"),
            ),
            spacing="2", width="100%",
        ),
        width="260px", min_width="260px", height="100vh",
        background="rgba(6, 10, 20, 0.95)",
        border_right="1px solid rgba(30, 41, 59, 0.35)",
        padding="28px 12px", position="fixed", left="0", top="0",
        overflow_y="auto", z_index="50",
    )


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ═════════════════════════════════════════════════════════════════════════════

def recommendation_hero():
    return rx.box(
        rx.flex(
            # Left: Recommendation badge
            rx.box(
                rx.text(DashboardState.recommendation, font_size="34px",
                        font_weight="800", color=DashboardState.rec_color,
                        letter_spacing="-0.02em"),
                rx.text("RECOMMENDATION", font_size="9px", color=MUTED,
                        letter_spacing="0.15em", font_weight="600", margin_top="4px"),
                rx.flex(
                    mini_badge(DashboardState.recommendation_source, MUTED),
                    mini_badge(DashboardState.investment_horizon, BLUE),
                    gap="4px", margin_top="8px", justify="center",
                ),
                text_align="center", padding="24px 36px", border_radius="16px",
                background=BG_CARD, border=BORDER, min_width="160px",
            ),
            # Right: Price + confidence
            rx.box(
                rx.flex(
                    rx.box(
                        rx.text("Current", color="white", font_size="10px",
                                text_transform="uppercase", letter_spacing="0.05em"),
                        rx.text(DashboardState.current_price_fmt, color=TEXT,
                                font_size="26px", font_weight="800",
                                letter_spacing="-0.02em", margin_top="2px"),
                    ),
                    rx.box(width="1px", height="44px",
                           background="linear-gradient(180deg, transparent, rgba(59, 130, 246, 0.2), transparent)",
                           flex_shrink="0"),
                    rx.box(
                        rx.text("Target", color="white", font_size="10px",
                                text_transform="uppercase", letter_spacing="0.05em"),
                        rx.text(DashboardState.target_price_fmt, color=GREEN,
                                font_size="26px", font_weight="800",
                                letter_spacing="-0.02em", margin_top="2px"),
                    ),
                    rx.box(width="1px", height="44px",
                           background="linear-gradient(180deg, transparent, rgba(59, 130, 246, 0.2), transparent)",
                           flex_shrink="0"),
                    rx.box(
                        rx.text("Return", color="white", font_size="10px",
                                text_transform="uppercase", letter_spacing="0.05em"),
                        rx.text(DashboardState.expected_return_fmt, color=GREEN,
                                font_size="26px", font_weight="800",
                                letter_spacing="-0.02em", margin_top="2px"),
                    ),
                    rx.box(width="1px", height="44px",
                           background="linear-gradient(180deg, transparent, rgba(59, 130, 246, 0.2), transparent)",
                           flex_shrink="0"),
                    rx.box(
                        rx.text("Risk/Reward", color="WHITE", font_size="10px",
                                text_transform="uppercase", letter_spacing="0.05em"),
                        rx.text(DashboardState.risk_reward_fmt, color=AMBER,
                                font_size="26px", font_weight="800",
                                letter_spacing="-0.02em", margin_top="2px"),
                    ),
                    align="center", gap="20px", flex_wrap="wrap",
                ),
                # Confidence bar
                rx.box(
                    rx.flex(
                        rx.text("Confidence", color="white", font_size="10px"),
                        rx.text(DashboardState.confidence + "%", color=BLUE,
                                font_size="11px", font_weight="700"),
                        justify="between", margin_bottom="6px",
                    ),
                    rx.box(
                        rx.box(
                            width=DashboardState.confidence + "%", height="100%",
                            background=GRADIENT_BRAND, border_radius="999px",
                            transition="width 1.2s cubic-bezier(0.16, 1, 0.3, 1)",
                        ),
                        width="100%", height="5px",
                        background="rgba(30, 41, 59, 0.4)",
                        border_radius="999px", overflow="hidden",
                    ),
                    margin_top="14px",
                ),
                rx.text(DashboardState.conviction_rationale, color=TEXT2,
                        font_size="12px", margin_top="10px", line_height="1.6"),
                flex="1",
            ),
            align="center", gap="28px", flex_wrap="wrap",
        ),
        style=CARD_STATIC,
        background="linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.3))",
        border="1px solid rgba(59, 130, 246, 0.12)",
        padding="24px", animation=_anim("fadeInUp", "0.5s"),
    )


def key_metrics_row():
    return rx.grid(
        stat_card("Stop Loss", DashboardState.stop_loss_fmt, icon="shield", color=RED),
        stat_card("Risk / Reward", DashboardState.risk_reward_fmt, icon="scale", color=AMBER),
        stat_card("Confidence", rx.fragment(DashboardState.confidence, rx.text("%", font_size="16px", display="inline")), icon="target", color=BLUE),
        stat_card("Max Drawdown", DashboardState.max_drawdown_fmt, icon="trending-down", color=RED),
        stat_card("Horizon", DashboardState.investment_horizon, icon="clock", color=PURPLE),
        stat_card("Position Size", DashboardState.pos_recommended, subtitle=DashboardState.pos_rationale, icon="pie-chart", color=CYAN),
        columns=rx.breakpoints(initial="2", sm="3", lg="3"), spacing="4", width="100%",
    )


def _signal_weight_bar(label: str, value, color: str):
    """Inline animated signal weight bar."""
    r = _rgb(color)
    return rx.box(
        rx.flex(
            rx.text(label, color=color, font_size="12px", font_weight="700",
                    min_width="40px"),
            rx.box(
                rx.box(
                    height="100%", border_radius="6px",
                    background=f"linear-gradient(90deg, {color}, rgba({r}, 0.6))",
                    width=rx.cond(value > 0, value.to(str) + "%", "0%"),
                    transition="width 1.2s cubic-bezier(0.16, 1, 0.3, 1)",
                    box_shadow=f"0 0 12px rgba({r}, 0.3)",
                ),
                width="100%", height="20px", border_radius="6px",
                background="rgba(30, 41, 59, 0.3)",
                overflow="hidden",
            ),
            rx.text(value, color=TEXT, font_size="13px", font_weight="700",
                    min_width="35px", text_align="right"),
            align="center", gap="10px", width="100%",
        ),
        width="100%",
    )


def signal_consensus():
    return rx.grid(
        # Signal weights as animated bars
        rx.box(
            rx.text("SIGNAL WEIGHT BREAKDOWN", font_size="10px", color="white",
                    font_weight="600", letter_spacing="0.12em", margin_bottom="16px"),
            rx.vstack(
                _signal_weight_bar("BUY", DashboardState.signal_buy_weight, GREEN),
                _signal_weight_bar("HOLD", DashboardState.signal_hold_weight, AMBER),
                _signal_weight_bar("SELL", DashboardState.signal_sell_weight, RED),
                spacing="3", width="100%",
            ),
            rx.box(height="1px", background="rgba(30, 41, 59, 0.2)",
                   margin_y="14px"),
            rx.box(
                metric_row("Agreement", DashboardState.agreement_level),
                metric_row("Volatility", DashboardState.vol_pct_raw),
                rx.text(DashboardState.divergence_notes, color=MUTED,
                        font_size="11px", margin_top="8px", font_style="italic",
                        line_height="1.4"),
                padding="12px 14px", border_radius="10px",
                background="rgba(30, 41, 59, 0.15)",
            ),
            style=CARD_STATIC,
            animation=_anim("fadeInUp", "0.5s", "0.05s"),
        ),
        # Consensus analysis — expanded boxes that fill height
        rx.box(
            rx.text("CONSENSUS ANALYSIS", font_size="10px", color="white",
                    font_weight="600", letter_spacing="0.12em", margin_bottom="16px"),
            rx.vstack(
                # Technical block — fills available space
                rx.box(
                    rx.flex(
                        rx.box(
                            rx.icon(tag="activity", size=22, color=BLUE),
                            padding="12px", border_radius="14px",
                            background="rgba(59, 130, 246, 0.08)",
                            border="1px solid rgba(59, 130, 246, 0.1)",
                            flex_shrink="0",
                        ),
                        rx.box(
                            rx.text("Technical Track", color=BLUE, font_size="15px",
                                    font_weight="700", letter_spacing="0.02em"),
                            rx.text("50% Portfolio Weight", color=MUTED, font_size="12px",
                                    margin_top="3px"),
                            flex="1",
                        ),
                        rx.box(
                            rx.text(DashboardState.consensus_tech,
                                    color=DashboardState.consensus_tech_color,
                                    font_weight="800", font_size="26px",
                                    letter_spacing="-0.02em"),
                            text_align="right",
                        ),
                        align="center", gap="16px",
                    ),
                    rx.box(
                        height="3px", margin_top="16px", border_radius="2px",
                        background=f"linear-gradient(90deg, {BLUE}, rgba(59, 130, 246, 0.05))",
                    ),
                    rx.flex(
                        rx.flex(
                            glow_dot(BLUE, "5px"),
                            rx.text("Fardeen", color=TEXT2, font_size="12px", font_weight="600"),
                            rx.text("20%", color=BLUE, font_size="11px", font_weight="700"),
                            align="center", gap="6px",
                        ),
                        rx.flex(
                            glow_dot(BLUE, "5px"),
                            rx.text("Tamer", color=TEXT2, font_size="12px", font_weight="600"),
                            rx.text("30%", color=BLUE, font_size="11px", font_weight="700"),
                            align="center", gap="6px",
                        ),
                        gap="20px", margin_top="14px", flex_wrap="wrap",
                    ),
                    padding="22px 24px", border_radius="14px",
                    background="rgba(59, 130, 246, 0.03)",
                    border="1px solid rgba(59, 130, 246, 0.08)",
                    transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
                    flex="1", width="100%",
                    _hover={
                        "border": "1px solid rgba(59, 130, 246, 0.25)",
                        "transform": "translateY(-2px)",
                        "box_shadow": "0 8px 24px rgba(59, 130, 246, 0.08)",
                    },
                ),
                # Fundamental block — fills available space
                rx.box(
                    rx.flex(
                        rx.box(
                            rx.icon(tag="building-2", size=22, color=PURPLE),
                            padding="12px", border_radius="14px",
                            background="rgba(139, 92, 246, 0.08)",
                            border="1px solid rgba(139, 92, 246, 0.1)",
                            flex_shrink="0",
                        ),
                        rx.box(
                            rx.text("Fundamental Track", color=PURPLE, font_size="15px",
                                    font_weight="700", letter_spacing="0.02em"),
                            rx.text("50% Portfolio Weight", color=MUTED, font_size="12px",
                                    margin_top="3px"),
                            flex="1",
                        ),
                        rx.box(
                            rx.text(DashboardState.consensus_fund,
                                    color=DashboardState.consensus_fund_color,
                                    font_weight="800", font_size="26px",
                                    letter_spacing="-0.02em"),
                            text_align="right",
                        ),
                        align="center", gap="16px",
                    ),
                    rx.box(
                        height="3px", margin_top="16px", border_radius="2px",
                        background=f"linear-gradient(90deg, {PURPLE}, rgba(139, 92, 246, 0.05))",
                    ),
                    rx.flex(
                        rx.flex(
                            glow_dot(PURPLE, "5px"),
                            rx.text("Daria", color=TEXT2, font_size="12px", font_weight="600"),
                            rx.text("15%", color=PURPLE, font_size="11px", font_weight="700"),
                            align="center", gap="6px",
                        ),
                        rx.flex(
                            glow_dot(PURPLE, "5px"),
                            rx.text("Shakzod", color=TEXT2, font_size="12px", font_weight="600"),
                            rx.text("15%", color=PURPLE, font_size="11px", font_weight="700"),
                            align="center", gap="6px",
                        ),
                        rx.flex(
                            glow_dot(CYAN, "5px"),
                            rx.text("Lary", color=TEXT2, font_size="12px", font_weight="600"),
                            rx.text("10%", color=CYAN, font_size="11px", font_weight="700"),
                            align="center", gap="6px",
                        ),
                        rx.flex(
                            glow_dot(CYAN, "5px"),
                            rx.text("Mohamed", color=TEXT2, font_size="12px", font_weight="600"),
                            rx.text("10%", color=CYAN, font_size="11px", font_weight="700"),
                            align="center", gap="6px",
                        ),
                        gap="14px", margin_top="14px", flex_wrap="wrap",
                    ),
                    padding="22px 24px", border_radius="14px",
                    background="rgba(139, 92, 246, 0.03)",
                    border="1px solid rgba(139, 92, 246, 0.08)",
                    transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
                    flex="1", width="100%",
                    _hover={
                        "border": "1px solid rgba(139, 92, 246, 0.25)",
                        "transform": "translateY(-2px)",
                        "box_shadow": "0 8px 24px rgba(139, 92, 246, 0.08)",
                    },
                ),
                spacing="3", width="100%", flex="1",
            ),
            style=CARD_STATIC,
            display="flex", flex_direction="column",
            animation=_anim("fadeInUp", "0.5s", "0.1s"),
        ),
        columns=rx.breakpoints(initial="1", md="2"), spacing="3", width="100%",
    )


def _chart_tooltip():
    """Reusable styled tooltip for all recharts."""
    return rx.recharts.graphing_tooltip(
        content_style={
            "backgroundColor": "rgba(10, 17, 40, 0.95)",
            "border": "1px solid rgba(59, 130, 246, 0.2)",
            "borderRadius": "12px",
            "padding": "10px 14px",
            "color": TEXT,
            "fontSize": "13px",
            "boxShadow": "0 8px 32px rgba(0,0,0,0.4)",
        },
        cursor=False,
    )


def perf_grid():
    return rx.grid(
        # Left: Performance Radar Chart
        rx.box(
            rx.text("PERFORMANCE PROFILE", font_size="10px", color=MUTED,
                    font_weight="600", letter_spacing="0.12em", margin_bottom="12px"),
            rx.recharts.responsive_container(
                rx.recharts.radar_chart(
                    rx.recharts.polar_grid(
                        stroke="rgba(59, 130, 246, 0.12)",
                        grid_type="polygon",
                    ),
                    rx.recharts.polar_angle_axis(
                        data_key="metric",
                        tick={"fill": TEXT2, "fontSize": 11},
                    ),
                    rx.recharts.polar_radius_axis(
                        angle=90, domain=[0, 100],
                        tick=False, axis_line=False,
                    ),
                    rx.recharts.radar(
                        data_key="value", name="Performance",
                        stroke=CYAN, fill=CYAN,
                        fill_opacity=0.2,
                        stroke_width=2,
                        dot={"r": 4, "fill": CYAN, "stroke": BG, "strokeWidth": 2},
                        is_animation_active=True,
                        animation_begin=300,
                        animation_duration=1500,
                        animation_easing="ease-out",
                    ),
                    data=DashboardState.performance_radar_data,
                    cx="50%", cy="50%",
                    outer_radius="72%",
                ),
                width="100%", height=280,
            ),
            style=CARD_STATIC,
            animation=_anim("fadeInUp", "0.5s", "0.05s"),
        ),
        # Right: Metric cards
        rx.box(
            rx.text("BACKTEST METRICS", font_size="10px", color=MUTED,
                    font_weight="600", letter_spacing="0.12em", margin_bottom="12px"),
            rx.grid(
                rx.box(
                    rx.flex(glow_dot(CYAN, "5px"), rx.text("Risk-Adjusted", color=CYAN, font_size="11px", font_weight="600"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("Sharpe", DashboardState.sharpe, CYAN),
                    metric_row("Sortino", DashboardState.sortino, CYAN),
                    metric_row("Calmar", DashboardState.calmar, CYAN),
                    padding="12px", border_radius="10px",
                    background="rgba(6, 182, 212, 0.03)",
                    border="1px solid rgba(6, 182, 212, 0.08)",
                    transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
                    _hover={"border": "1px solid rgba(6, 182, 212, 0.2)", "transform": "translateY(-2px)"},
                ),
                rx.box(
                    rx.flex(glow_dot(GREEN, "5px"), rx.text("Returns", color=GREEN, font_size="11px", font_weight="600"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("CAGR", DashboardState.cagr_v, GREEN),
                    metric_row("Win Rate", DashboardState.win_rate, GREEN),
                    metric_row("Profit Factor", DashboardState.profit_factor_v, GREEN),
                    padding="12px", border_radius="10px",
                    background="rgba(16, 185, 129, 0.03)",
                    border="1px solid rgba(16, 185, 129, 0.08)",
                    transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
                    _hover={"border": "1px solid rgba(16, 185, 129, 0.2)", "transform": "translateY(-2px)"},
                ),
                rx.box(
                    rx.flex(glow_dot(RED, "5px"), rx.text("Risk", color=RED, font_size="11px", font_weight="600"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("Volatility", DashboardState.vol_pct, AMBER),
                    metric_row("Max Drawdown", DashboardState.max_dd, RED),
                    metric_row("VaR 95%", DashboardState.var95, RED),
                    padding="12px", border_radius="10px",
                    background="rgba(239, 68, 68, 0.03)",
                    border="1px solid rgba(239, 68, 68, 0.08)",
                    transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
                    _hover={"border": "1px solid rgba(239, 68, 68, 0.2)", "transform": "translateY(-2px)"},
                ),
                rx.box(
                    rx.flex(glow_dot(MUTED, "5px"), rx.text("Activity", color=MUTED, font_size="11px", font_weight="600"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("Total Trades", DashboardState.total_trades_v),
                    padding="12px", border_radius="10px",
                    background="rgba(30, 41, 59, 0.2)",
                    transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
                    _hover={"border": BORDER_GLOW, "transform": "translateY(-2px)"},
                ),
                columns="2", spacing="3", width="100%",
            ),
            style=CARD_STATIC,
            animation=_anim("fadeInUp", "0.6s", "0.15s"),
        ),
        columns=rx.breakpoints(initial="1", md="2"), spacing="4", width="100%",
    )


def overview_tab():
    return rx.vstack(
        section_header("Dashboard Overview", "Comprehensive investment analysis", "layout-grid"),
        recommendation_hero(),
        key_metrics_row(),
        signal_consensus(),
        text_card("Executive Summary", DashboardState.executive_summary, "file-text"),
        perf_grid(),
        # Price Zone Area Chart — gradient landscape of price levels
        rx.cond(
            DashboardState.price_zone_data.length() > 0,
            rx.box(
                rx.flex(
                    rx.text("PRICE ZONE MAP", font_size="11px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.flex(
                        rx.box(width="10px", height="3px", background=GREEN, border_radius="2px"),
                        rx.text("Targets", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=RED, border_radius="2px"),
                        rx.text("Stop/Resist", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=CYAN, border_radius="2px"),
                        rx.text("Current", color=MUTED, font_size="10px"),
                        align="center", gap="6px",
                    ),
                    align="center", margin_bottom="16px",
                ),
                rx.recharts.responsive_container(
                    rx.recharts.area_chart(
                        rx.recharts.cartesian_grid(
                            stroke_dasharray="3 3", opacity=0.06,
                            horizontal=True, vertical=False,
                        ),
                        rx.recharts.x_axis(
                            data_key="name", stroke=MUTED, font_size=10,
                            tick={"fill": TEXT2},
                            interval=0, angle=-35,
                            axis_line=False,
                            tick_margin=10,
                        ),
                        rx.recharts.y_axis(
                            stroke=MUTED, font_size=11,
                            tick={"fill": TEXT2},
                            domain=["dataMin - 10", "dataMax + 10"],
                            axis_line=False, tick_line=False,
                        ),
                        rx.recharts.area(
                            data_key="value", name="Price Level",
                            stroke=CYAN, stroke_width=2.5,
                            fill=CYAN, fill_opacity=0.15,
                            type_="monotone",
                            dot={"r": 5, "fill": CYAN, "stroke": BG, "strokeWidth": 2},
                            active_dot={"r": 7, "fill": CYAN, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_begin=100,
                            animation_duration=1500,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.reference_line(
                            y=DashboardState.current_price_num,
                            stroke=AMBER, stroke_dasharray="8 4",
                            stroke_width=2,
                            label="Current",
                        ),
                        _chart_tooltip(),
                        data=DashboardState.price_zone_data,
                        margin={"top": 25, "right": 30, "bottom": 55, "left": 40},
                    ),
                    width="100%", height=420,
                ),
                style=CARD_STATIC, animation=_anim("fadeInUp", "0.5s", "0.1s"),
            ),
            rx.fragment(),
        ),
        spacing="4", width="100%",
    )


# ── Agents Tab ──

def _render_agent(agent: dict):
    return rx.box(
        rx.box(height="3px", background=agent["card_accent"], border_radius="2px"),
        rx.box(
            rx.flex(
                rx.flex(
                    rx.box(
                        rx.text(agent["type"], font_size="11px", font_weight="700",
                                color=agent["type_color"]),
                        padding_x="10px", padding_y="3px", border_radius="999px",
                        background=agent["type_badge_bg"],
                        border=agent["type_badge_border"],
                        display="inline-flex", align_items="center",
                    ),
                    mini_badge(agent["weight"], MUTED),
                    gap="6px",
                ),
                justify="between", align="center", margin_bottom="16px",
            ),
            rx.text(agent["name"], font_size="17px", font_weight="800", color=TEXT,
                    margin_bottom="4px", letter_spacing="-0.01em"),
            rx.flex(
                rx.text("Signal: ", color=TEXT2, font_size="14px"),
                rx.text(agent["signal"], color=agent["color"], font_weight="700",
                        font_size="14px"),
                gap="4px", margin_bottom="4px",
            ),
            rx.flex(
                rx.text("Target: ", color=TEXT2, font_size="14px"),
                rx.text(agent["target"], color=TEXT, font_weight="600", font_size="14px"),
                gap="4px", margin_bottom="14px",
            ),
            rx.text(agent["insight"], color=MUTED, font_size="12px", line_height="1.6"),
            padding="16px", padding_top="14px",
        ),
        border_radius="14px", background=BG_CARD, border=BORDER, overflow="hidden",
        transition="all 0.35s cubic-bezier(0.4, 0, 0.2, 1)",
        _hover={
            "border": BORDER_GLOW,
            "transform": "translateY(-4px)",
            "box_shadow": "0 24px 64px rgba(0, 0, 0, 0.35)",
        },
        min_height="220px",
    )


def _render_conflict(conflict: dict):
    return rx.box(
        rx.flex(
            rx.icon(tag="zap", size=16, color=AMBER),
            rx.text(conflict["conflict"], color=TEXT, font_size="14px", font_weight="600"),
            align="start", gap="10px",
        ),
        rx.box(
            rx.text("Resolution: ", color=TEXT2, font_size="13px", display="inline",
                    font_weight="600"),
            rx.text(conflict["resolution"], color=TEXT2, font_size="13px", display="inline"),
            margin_top="10px",
        ),
        rx.flex(
            rx.text("Favored: ", color=MUTED, font_size="12px"),
            rx.text(conflict["favored"], color=BLUE, font_size="12px", font_weight="600"),
            gap="4px", margin_top="8px",
        ),
        padding="18px", border_radius="14px",
        background="rgba(245, 158, 11, 0.03)",
        border="1px solid rgba(245, 158, 11, 0.08)",
    )


def agents_tab():
    return rx.vstack(
        section_header("Agent Analysis", "Individual analyst signals & targets", "users"),
        # Summary stat cards
        rx.grid(
            stat_card("Agent Count", "6", subtitle="Technical + Fundamental", icon="users", color=BLUE),
            stat_card("Consensus", DashboardState.recommendation, subtitle="Final Signal", icon="target", color=GREEN),
            stat_card("Weighted Target", DashboardState.target_price_fmt, subtitle="Consensus Price", icon="trending-up", color=CYAN),
            stat_card("Agreement", DashboardState.confidence, subtitle="Confidence Level", icon="circle-check", color=PURPLE),
            columns=rx.breakpoints(initial="2", lg="4"), spacing="4", width="100%",
            class_name="stagger-children",
        ),
        rx.grid(
            rx.foreach(DashboardState.agents_list, _render_agent),
            columns=rx.breakpoints(initial="1", sm="2", lg="3"),
            spacing="4", width="100%",
            class_name="stagger-children",
        ),
        # Agent Target Comparison — interactive composed chart with reference lines
        rx.cond(
            DashboardState.agent_targets_data.length() > 0,
            rx.box(
                rx.flex(
                    rx.text("AGENT TARGET COMPARISON", font_size="11px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.flex(
                        rx.box(width="12px", height="3px", background=GREEN, border_radius="2px"),
                        rx.text("Current Price", color=MUTED, font_size="10px"),
                        rx.box(width="12px", height="3px", background=AMBER, border_radius="2px"),
                        rx.text("Consensus Target", color=MUTED, font_size="10px"),
                        align="center", gap="6px",
                    ),
                    align="center", margin_bottom="18px",
                ),
                rx.recharts.responsive_container(
                    rx.recharts.composed_chart(
                        rx.recharts.cartesian_grid(
                            stroke_dasharray="3 3", opacity=0.06,
                            vertical=False,
                        ),
                        rx.recharts.x_axis(
                            data_key="name", stroke=MUTED, font_size=12,
                            tick={"fill": TEXT2},
                            axis_line=False, interval=0, angle=-15,
                            tick_margin=8,
                        ),
                        rx.recharts.y_axis(
                            stroke=MUTED, font_size=11,
                            tick={"fill": TEXT2},
                            domain=["dataMin - 30", "dataMax + 20"],
                            axis_line=False,
                            tick_line=False,
                        ),
                        rx.recharts.bar(
                            data_key="target", name="Target Price",
                            radius=[8, 8, 0, 0],
                            fill=BLUE,
                            fill_opacity=0.8,
                            is_animation_active=True,
                            animation_begin=100,
                            animation_duration=1000,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.reference_line(
                            y=DashboardState.current_price_num,
                            stroke=GREEN,
                            stroke_dasharray="8 4",
                            stroke_width=2,
                            label="Current",
                        ),
                        rx.recharts.reference_line(
                            y=DashboardState.target_price_num,
                            stroke=AMBER,
                            stroke_dasharray="5 5",
                            stroke_width=1.5,
                            label="Target",
                        ),
                        rx.recharts.reference_line(
                            y=DashboardState.stop_loss_num,
                            stroke=RED,
                            stroke_dasharray="4 4",
                            stroke_width=1,
                            label="Stop",
                        ),
                        _chart_tooltip(),
                        rx.recharts.legend(
                            icon_type="rect", icon_size=12,
                            vertical_align="top", align="left",
                        ),
                        data=DashboardState.agent_targets_data,
                        margin={"top": 15, "right": 60, "bottom": 35, "left": 40},
                    ),
                    width="100%", height=420,
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.6s", "0.1s"),
            ),
            rx.fragment(),
        ),
        # (Upside/Downside and Signal Distribution removed — info already visible in agent cards + stats)
        # Conflicts
        rx.cond(
            DashboardState.conflicts_list.length() > 0,
            rx.box(
                section_header("Conflicts Resolved",
                               "How analyst disagreements were reconciled", "zap"),
                rx.vstack(
                    rx.foreach(DashboardState.conflicts_list, _render_conflict),
                    spacing="3", width="100%",
                ),
            ),
            rx.fragment(),
        ),
        spacing="4", width="100%",
    )


# ── Analysis Tab ──

def _render_indicator(ind: dict):
    return rx.flex(
        rx.text(ind["name"], color=TEXT, font_size="13px", font_weight="600",
                min_width="110px"),
        rx.text(ind["value"], color=TEXT2, font_size="13px", min_width="80px"),
        rx.text(ind["signal"], color=ind["color"], font_size="12px",
                font_weight="700", min_width="80px"),
        rx.text(ind["zone"], color=MUTED, font_size="12px", min_width="90px"),
        rx.text(ind["confidence"], color=MUTED, font_size="12px"),
        align="center", gap="12px", padding_y="9px",
        border_bottom="1px solid rgba(30, 41, 59, 0.15)",
    )


def key_levels():
    return rx.box(
        rx.text("KEY PRICE LEVELS", font_size="11px", color=MUTED,
                font_weight="600", letter_spacing="0.12em", margin_bottom="18px"),
        rx.grid(
            rx.box(
                rx.flex(rx.icon(tag="shield", size=14, color=GREEN),
                        rx.text("Support", color=GREEN, font_size="12px", font_weight="700"),
                        align="center", gap="6px"),
                rx.text(DashboardState.support_str, color=TEXT, font_size="16px",
                        font_weight="700", margin_top="10px"),
                padding="18px", border_radius="14px",
                background="rgba(16, 185, 129, 0.03)",
                border="1px solid rgba(16, 185, 129, 0.08)",
            ),
            rx.box(
                rx.flex(rx.icon(tag="flame", size=14, color=RED),
                        rx.text("Resistance", color=RED, font_size="12px", font_weight="700"),
                        align="center", gap="6px"),
                rx.text(DashboardState.resistance_str, color=TEXT, font_size="16px",
                        font_weight="700", margin_top="10px"),
                padding="18px", border_radius="14px",
                background="rgba(239, 68, 68, 0.03)",
                border="1px solid rgba(239, 68, 68, 0.08)",
            ),
            rx.box(
                rx.flex(rx.icon(tag="log-in", size=14, color=BLUE),
                        rx.text("Entry Zone", color=BLUE, font_size="12px", font_weight="700"),
                        align="center", gap="6px"),
                rx.text(DashboardState.entry_zone_str, color=TEXT, font_size="16px",
                        font_weight="700", margin_top="10px"),
                padding="18px", border_radius="14px",
                background="rgba(59, 130, 246, 0.03)",
                border="1px solid rgba(59, 130, 246, 0.08)",
            ),
            rx.box(
                rx.flex(rx.icon(tag="target", size=14, color=PURPLE),
                        rx.text("Profit Targets", color=PURPLE, font_size="12px", font_weight="700"),
                        align="center", gap="6px"),
                rx.text(DashboardState.profit_targets_str, color=TEXT, font_size="16px",
                        font_weight="700", margin_top="10px"),
                padding="18px", border_radius="14px",
                background="rgba(139, 92, 246, 0.03)",
                border="1px solid rgba(139, 92, 246, 0.08)",
            ),
            columns=rx.breakpoints(initial="2", lg="4"), spacing="3", width="100%",
        ),
        style=CARD_STATIC,
    )


def analysis_tab():
    return rx.vstack(
        section_header("Deep Analysis", "Technical & fundamental synthesis", "bar-chart-3"),
        # Summary stat cards
        rx.grid(
            stat_card("RSI (14)", DashboardState.fardeen_rsi, subtitle="Momentum", icon="activity", color=AMBER),
            stat_card("MACD Hist", DashboardState.fardeen_macd_hist, subtitle="Trend Signal", icon="bar-chart-3", color=BLUE),
            stat_card("ADX", DashboardState.fardeen_adx, subtitle="Trend Strength", icon="trending-up", color=GREEN),
            stat_card("Market Regime", DashboardState.fardeen_regime, subtitle="Current Condition", icon="gauge", color=PURPLE),
            columns=rx.breakpoints(initial="2", lg="4"), spacing="4", width="100%",
            class_name="stagger-children",
        ),
        text_card("Technical Analysis Synthesis", DashboardState.technical_synthesis, "activity"),
        text_card("Fundamental Analysis Synthesis", DashboardState.fundamental_synthesis, "building-2"),
        # Price Levels Interactive Chart
        rx.cond(
            DashboardState.price_levels_data.length() > 0,
            rx.box(
                rx.flex(
                    rx.text("KEY PRICE LEVELS MAP", font_size="11px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.flex(
                        rx.box(width="10px", height="3px", background=GREEN, border_radius="2px"),
                        rx.text("Support", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=RED, border_radius="2px"),
                        rx.text("Resistance", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=BLUE, border_radius="2px"),
                        rx.text("Entry", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=PURPLE, border_radius="2px"),
                        rx.text("Target", color=MUTED, font_size="10px"),
                        align="center", gap="6px",
                    ),
                    align="center", margin_bottom="16px",
                ),
                rx.recharts.responsive_container(
                    rx.recharts.area_chart(
                        rx.recharts.cartesian_grid(
                            stroke_dasharray="3 3", opacity=0.06,
                            horizontal=True, vertical=False,
                        ),
                        rx.recharts.x_axis(
                            data_key="name", stroke=MUTED, font_size=10,
                            tick={"fill": TEXT2},
                            axis_line=False, interval=0, angle=-35,
                            tick_margin=10,
                        ),
                        rx.recharts.y_axis(
                            stroke=MUTED, font_size=11,
                            tick={"fill": TEXT2},
                            domain=["dataMin - 15", "dataMax + 15"],
                            axis_line=False, tick_line=False,
                        ),
                        rx.recharts.area(
                            data_key="value", name="Price Level",
                            stroke=BLUE, stroke_width=2.5,
                            fill=BLUE, fill_opacity=0.15,
                            type_="monotone",
                            dot={"r": 5, "fill": BLUE, "stroke": BG, "strokeWidth": 2},
                            active_dot={"r": 7, "fill": BLUE, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_begin=200,
                            animation_duration=1500,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.reference_line(
                            y=DashboardState.current_price_num,
                            stroke=CYAN, stroke_dasharray="8 4",
                            stroke_width=2,
                            label="Current Price",
                        ),
                        _chart_tooltip(),
                        data=DashboardState.price_levels_data,
                        margin={"top": 25, "right": 80, "bottom": 55, "left": 40},
                    ),
                    width="100%", height=420,
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.6s", "0.1s"),
            ),
            rx.fragment(),
        ),
        key_levels(),
        # Regime Radar — multi-dimensional regime comparison
        rx.cond(
            DashboardState.regime_radar_data.length() > 0,
            rx.box(
                rx.flex(
                    rx.text("MARKET REGIME ANALYSIS", font_size="11px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.flex(
                        rx.box(width="10px", height="3px", background=GREEN, border_radius="2px"),
                        rx.text("Bull", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=AMBER, border_radius="2px"),
                        rx.text("Sideways", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=RED, border_radius="2px"),
                        rx.text("Bear", color=MUTED, font_size="10px"),
                        align="center", gap="6px",
                    ),
                    align="center", margin_bottom="16px",
                ),
                rx.recharts.responsive_container(
                    rx.recharts.radar_chart(
                        rx.recharts.polar_grid(
                            stroke="rgba(59, 130, 246, 0.12)",
                            grid_type="polygon",
                        ),
                        rx.recharts.polar_angle_axis(
                            data_key="metric",
                            tick={"fill": TEXT2, "fontSize": 12, "fontWeight": 600},
                        ),
                        rx.recharts.polar_radius_axis(
                            angle=90, domain=[0, 100],
                            tick=False, axis_line=False,
                        ),
                        rx.recharts.radar(
                            data_key="bull", name="Bull",
                            stroke=GREEN, fill=GREEN, fill_opacity=0.15,
                            stroke_width=2,
                            dot={"r": 4, "fill": GREEN, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_begin=200,
                            animation_duration=1200,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.radar(
                            data_key="sideways", name="Sideways",
                            stroke=AMBER, fill=AMBER, fill_opacity=0.1,
                            stroke_width=2,
                            dot={"r": 4, "fill": AMBER, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_begin=400,
                            animation_duration=1200,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.radar(
                            data_key="bear", name="Bear",
                            stroke=RED, fill=RED, fill_opacity=0.1,
                            stroke_width=2,
                            dot={"r": 4, "fill": RED, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_begin=600,
                            animation_duration=1200,
                            animation_easing="ease-out",
                        ),
                        _chart_tooltip(),
                        rx.recharts.legend(
                            icon_type="circle", icon_size=8,
                            vertical_align="bottom", align="center",
                        ),
                        data=DashboardState.regime_radar_data,
                        cx="50%", cy="45%",
                        outer_radius="70%",
                    ),
                    width="100%", height=420,
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.6s", "0.05s"),
            ),
            rx.fragment(),
        ),
        divider(),
        section_header("Technical Indicator Detail", "Individual indicator signals from both agents", "activity"),
        # Fardeen detail
        rx.box(
            rx.flex(
                mini_badge("Technical", BLUE),
                rx.text("Fardeen \u2014 Indicator Detail", font_size="14px",
                        font_weight="700", color=TEXT),
                align="center", gap="10px", margin_bottom="18px",
            ),
            rx.grid(
                rx.box(
                    metric_row("RSI (14)", DashboardState.fardeen_rsi),
                    metric_row("MACD Histogram", DashboardState.fardeen_macd_hist),
                    metric_row("ADX", DashboardState.fardeen_adx),
                    metric_row("ATR", DashboardState.fardeen_atr),
                    metric_row("BB %B", DashboardState.fardeen_bb),
                ),
                rx.box(
                    metric_row("Market Regime", DashboardState.fardeen_regime),
                    metric_row("Hurst Exponent", DashboardState.fardeen_hurst),
                    metric_row("Volatility Regime", DashboardState.fardeen_vol_regime),
                ),
                columns=rx.breakpoints(initial="1", md="2"), spacing="4",
            ),
            style=CARD_STATIC,
        ),
        # Tamer detail
        rx.box(
            rx.flex(
                mini_badge("Technical", BLUE),
                rx.text("Tamer \u2014 Indicator Breakdown", font_size="14px",
                        font_weight="700", color=TEXT),
                align="center", gap="10px", margin_bottom="18px",
            ),
            rx.vstack(
                rx.cond(
                    DashboardState.tamer_momentum.length() > 0,
                    rx.box(
                        rx.text("Momentum", color=AMBER, font_size="12px",
                                font_weight="700", margin_bottom="10px"),
                        rx.foreach(DashboardState.tamer_momentum, _render_indicator),
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    DashboardState.tamer_trend.length() > 0,
                    rx.box(
                        rx.text("Trend", color=BLUE, font_size="12px",
                                font_weight="700", margin_bottom="10px"),
                        rx.foreach(DashboardState.tamer_trend, _render_indicator),
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    DashboardState.tamer_volatility.length() > 0,
                    rx.box(
                        rx.text("Volatility", color=PURPLE, font_size="12px",
                                font_weight="700", margin_bottom="10px"),
                        rx.foreach(DashboardState.tamer_volatility, _render_indicator),
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    DashboardState.tamer_volume.length() > 0,
                    rx.box(
                        rx.text("Volume", color=CYAN, font_size="12px",
                                font_weight="700", margin_bottom="10px"),
                        rx.foreach(DashboardState.tamer_volume, _render_indicator),
                    ),
                    rx.fragment(),
                ),
                spacing="4", width="100%",
            ),
            style=CARD_STATIC,
        ),
        divider(),
        section_header("Fundamental Ratios", "Key financial metrics from fundamental agents", "calculator"),
        rx.box(
            rx.flex(
                mini_badge("Fundamental", PURPLE),
                rx.text("Financial Ratios (Lary)", font_size="14px",
                        font_weight="700", color=TEXT),
                align="center", gap="10px", margin_bottom="18px",
            ),
            rx.grid(
                rx.box(
                    rx.flex(glow_dot(AMBER, "5px"), rx.text("Valuation", color=AMBER, font_size="12px", font_weight="700"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("P/E Ratio", DashboardState.fund_pe),
                    metric_row("P/B Ratio", DashboardState.fund_pb),
                    metric_row("EV/EBITDA", DashboardState.fund_ev_ebitda),
                    metric_row("P/S Ratio", DashboardState.fund_ps),
                ),
                rx.box(
                    rx.flex(glow_dot(GREEN, "5px"), rx.text("Profitability", color=GREEN, font_size="12px", font_weight="700"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("ROE", DashboardState.fund_roe, GREEN),
                    metric_row("ROA", DashboardState.fund_roa, GREEN),
                    metric_row("Net Margin", DashboardState.fund_net_margin, GREEN),
                    metric_row("Gross Margin", DashboardState.fund_gross_margin, GREEN),
                ),
                rx.box(
                    rx.flex(glow_dot(BLUE, "5px"), rx.text("Growth", color=BLUE, font_size="12px", font_weight="700"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("Revenue Growth", DashboardState.fund_revenue_growth, BLUE),
                    metric_row("Net Income Growth", DashboardState.fund_net_income_growth, BLUE),
                ),
                rx.box(
                    rx.flex(glow_dot(RED, "5px"), rx.text("Health & Risk", color=RED, font_size="12px", font_weight="700"), align="center", gap="6px", margin_bottom="10px"),
                    metric_row("Current Ratio", DashboardState.fund_current_ratio),
                    metric_row("Debt/Equity", DashboardState.fund_debt_equity),
                    metric_row("Altman Z-Score", DashboardState.fund_altman_z, GREEN),
                ),
                columns=rx.breakpoints(initial="1", sm="2", lg="4"), spacing="4",
            ),
            style=CARD_STATIC,
        ),
        spacing="4", width="100%",
    )


# ── Thesis & Strategy Tab ──

def _render_scenario(scenario: dict):
    return rx.box(
        rx.flex(
            rx.icon(tag=scenario["icon"], size=22, color=scenario["color"]),
            rx.text(scenario["label"], font_size="18px", font_weight="800",
                    color=TEXT, letter_spacing="-0.01em"),
            align="center", gap="12px", margin_bottom="20px",
        ),
        rx.grid(
            rx.box(
                rx.text("Probability", color=MUTED, font_size="11px",
                        text_transform="uppercase", letter_spacing="0.05em"),
                rx.text(scenario["probability"], color=scenario["color"],
                        font_size="30px", font_weight="800"),
            ),
            rx.box(
                rx.text("Target", color=MUTED, font_size="11px",
                        text_transform="uppercase", letter_spacing="0.05em"),
                rx.text(scenario["target"], color=TEXT, font_size="22px", font_weight="700"),
            ),
            rx.box(
                rx.text("Return", color=MUTED, font_size="11px",
                        text_transform="uppercase", letter_spacing="0.05em"),
                rx.text(scenario["return_pct"], color=scenario["color"],
                        font_size="22px", font_weight="700"),
            ),
            columns="3", spacing="3", margin_bottom="18px",
        ),
        rx.text(scenario["narrative"], color=TEXT2, font_size="13px", line_height="1.7"),
        rx.cond(
            scenario["assumptions"] != "",
            rx.box(
                rx.text("Key Assumptions", color=MUTED, font_size="11px",
                        font_weight="600", margin_top="14px", margin_bottom="6px"),
                rx.text(scenario["assumptions"], color=MUTED, font_size="12px",
                        font_style="italic", line_height="1.5"),
            ),
            rx.fragment(),
        ),
        padding="28px", border_radius="18px", background=BG_CARD,
        border=scenario["border_css"],
        transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        _hover={"border": scenario["hover_border_css"], "transform": "translateY(-3px)"},
    )


def thesis_tab():
    return rx.vstack(
        section_header("Investment Thesis & Strategy",
                       "Core thesis and scenario analysis", "lightbulb"),
        text_card("Investment Thesis", DashboardState.investment_thesis, "book-open"),
        # Conviction pull quote
        rx.cond(
            DashboardState.conviction_rationale != "",
            rx.box(
                rx.flex(
                    rx.box(
                        width="4px", height="100%", min_height="40px",
                        background=GRADIENT_BRAND, border_radius="2px",
                        flex_shrink="0",
                    ),
                    rx.box(
                        rx.text("CONVICTION RATIONALE", font_size="10px", color=BLUE,
                                font_weight="700", letter_spacing="0.12em",
                                margin_bottom="8px"),
                        rx.text(DashboardState.conviction_rationale, color=TEXT,
                                font_size="14px", line_height="1.7", font_style="italic"),
                    ),
                    gap="16px", align="start",
                ),
                padding="24px", border_radius="16px",
                background="rgba(59, 130, 246, 0.03)",
                border="1px solid rgba(59, 130, 246, 0.1)",
                animation=_anim("fadeInUp", "0.5s", "0.1s"),
            ),
            rx.fragment(),
        ),
        divider(),
        section_header("Scenario Analysis", "Probability-weighted outcomes", "git-branch"),
        # Scenario Range Area Chart — shows probability-weighted price range
        rx.cond(
            DashboardState.scenario_chart_data.length() > 0,
            rx.box(
                rx.flex(
                    rx.text("SCENARIO RANGE & TARGETS", font_size="11px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.flex(
                        rx.box(width="10px", height="3px", background=GREEN, border_radius="2px"),
                        rx.text("Target", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=PURPLE, border_radius="2px"),
                        rx.text("Probability", color=MUTED, font_size="10px"),
                        rx.box(width="10px", height="3px", background=CYAN, border_radius="2px"),
                        rx.text("Current", color=MUTED, font_size="10px"),
                        align="center", gap="6px",
                    ),
                    align="center", margin_bottom="16px",
                ),
                rx.recharts.responsive_container(
                    rx.recharts.composed_chart(
                        rx.recharts.cartesian_grid(
                            stroke_dasharray="3 3", opacity=0.06,
                            vertical=False,
                        ),
                        rx.recharts.x_axis(
                            data_key="name", stroke=MUTED, font_size=14,
                            tick={"fill": TEXT2, "fontWeight": 600},
                            axis_line=False,
                        ),
                        rx.recharts.y_axis(
                            stroke=MUTED, font_size=11,
                            tick={"fill": TEXT2},
                            y_axis_id="price",
                            domain=["dataMin - 20", "dataMax + 20"],
                            axis_line=False, tick_line=False,
                            orientation="left",
                        ),
                        rx.recharts.y_axis(
                            stroke=MUTED, font_size=11,
                            tick={"fill": TEXT2},
                            y_axis_id="prob",
                            domain=[0, 60], unit="%",
                            axis_line=False, tick_line=False,
                            orientation="right",
                        ),
                        rx.recharts.area(
                            data_key="target", name="Price Target ($)",
                            stroke=GREEN, fill=GREEN, fill_opacity=0.15,
                            stroke_width=2, type_="monotone",
                            y_axis_id="price",
                            dot={"r": 6, "fill": GREEN, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_duration=1500,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.line(
                            data_key="probability", name="Probability (%)",
                            stroke=PURPLE, stroke_width=2, type_="monotone",
                            y_axis_id="prob",
                            dot={"r": 5, "fill": PURPLE, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_duration=1500,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.reference_line(
                            y=DashboardState.current_price_num,
                            y_axis_id="price",
                            stroke=CYAN, stroke_dasharray="8 4",
                            stroke_width=2,
                            label="Current",
                        ),
                        _chart_tooltip(),
                        rx.recharts.legend(
                            icon_type="circle", icon_size=8,
                            vertical_align="top", align="right",
                        ),
                        data=DashboardState.scenario_chart_data,
                        margin={"top": 15, "right": 60, "bottom": 30, "left": 40},
                    ),
                    width="100%", height=380,
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.6s", "0.05s"),
            ),
            rx.fragment(),
        ),
        # Scenario detail cards
        rx.grid(
            rx.foreach(DashboardState.scenario_list, _render_scenario),
            columns=rx.breakpoints(initial="1", lg="3"), spacing="4", width="100%",
        ),
        divider(),
        rx.box(
            section_header("Position Sizing", "Recommended allocation", "pie-chart"),
            rx.grid(
                stat_card("Recommended", DashboardState.pos_recommended,
                          icon="circle-check", color=GREEN),
                stat_card("Maximum", DashboardState.pos_max,
                          icon="circle-alert", color=AMBER),
                columns="2", spacing="4",
            ),
            rx.text(DashboardState.pos_rationale, color=TEXT2, font_size="13px",
                    margin_top="14px", font_style="italic", line_height="1.5"),
        ),
        spacing="3", width="100%",
    )


# ── Risk & Catalysts Tab ──

def _render_risk(risk: dict):
    return rx.box(
        rx.flex(
            rx.box(
                rx.icon(tag="triangle-alert", size=16, color=RED),
                padding="8px", border_radius="10px",
                background="rgba(239, 68, 68, 0.04)", flex_shrink="0",
            ),
            rx.box(
                rx.text(risk["risk"], color=TEXT, font_size="14px",
                        font_weight="500", line_height="1.5"),
                rx.flex(
                    mini_badge(risk["category"], MUTED),
                    mini_badge(risk["probability"], AMBER),
                    mini_badge(risk["severity"], RED),
                    rx.text(risk["impact"], color=RED, font_size="12px", font_weight="700"),
                    gap="6px", margin_top="10px", flex_wrap="wrap",
                ),
                rx.cond(
                    risk["mitigation"] != "",
                    rx.flex(
                        rx.icon(tag="shield-check", size=12, color=GREEN),
                        rx.text(risk["mitigation"], color=MUTED, font_size="12px",
                                line_height="1.4"),
                        gap="6px", margin_top="10px", align="start",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    risk["warning_signs"] != "",
                    rx.box(
                        rx.flex(
                            rx.icon(tag="circle-alert", size=12, color=AMBER),
                            rx.text("Warning Signs", color=AMBER, font_size="11px",
                                    font_weight="700"),
                            align="center", gap="4px", margin_bottom="6px",
                        ),
                        rx.text(risk["warning_signs"], color=TEXT2, font_size="12px",
                                line_height="1.5", white_space="pre-wrap"),
                        margin_top="10px", padding="10px 12px",
                        border_radius="10px",
                        background="rgba(245, 158, 11, 0.03)",
                        border="1px solid rgba(245, 158, 11, 0.08)",
                    ),
                    rx.fragment(),
                ),
                flex="1",
            ),
            align="start", gap="12px",
        ),
        padding="18px", border_radius="14px",
        background="rgba(239, 68, 68, 0.02)",
        border="1px solid rgba(239, 68, 68, 0.06)",
        transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        _hover={"border": "1px solid rgba(239, 68, 68, 0.15)"},
    )


def _render_catalyst(catalyst: dict):
    return rx.box(
        rx.flex(
            rx.box(
                rx.icon(tag="calendar", size=14, color=catalyst["color"]),
                padding="10px", border_radius="10px",
                background=catalyst["icon_bg"], flex_shrink="0",
            ),
            rx.box(
                rx.flex(
                    rx.text(catalyst["catalyst"], color=TEXT, font_size="14px",
                            font_weight="600"),
                    rx.text(catalyst["move"], color=catalyst["color"],
                            font_size="13px", font_weight="700"),
                    justify="between", align="center", width="100%",
                ),
                rx.flex(
                    mini_badge(catalyst["category"], MUTED),
                    rx.text(catalyst["date"], color=TEXT2, font_size="12px"),
                    rx.text(catalyst["direction"], color=catalyst["color"],
                            font_size="12px", font_weight="500"),
                    gap="8px", margin_top="8px",
                ),
                rx.text(catalyst["reasoning"], color=MUTED, font_size="12px",
                        margin_top="8px", line_height="1.5"),
                flex="1",
            ),
            align="start", gap="14px",
        ),
        padding="18px", border_radius="14px",
        background="rgba(30, 41, 59, 0.15)",
        border="1px solid rgba(30, 41, 59, 0.25)",
        transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        _hover={"border": BORDER_GLOW},
    )


def risk_tab():
    return rx.vstack(
        section_header("Risk Assessment",
                       "Primary risks and mitigation strategies", "shield-alert"),
        # Summary stat cards
        rx.grid(
            stat_card("Total Risks", DashboardState.risk_count_str, subtitle="Identified", icon="triangle-alert", color=RED),
            stat_card("Avg Impact", DashboardState.avg_risk_impact_str, subtitle="Severity Score", icon="gauge", color=AMBER),
            stat_card("Risk/Reward", DashboardState.risk_reward_fmt, subtitle="Ratio", icon="scale", color=GREEN),
            stat_card("Stop Loss", DashboardState.stop_loss_fmt, subtitle="Downside Protection", icon="shield-check", color=CYAN),
            columns=rx.breakpoints(initial="2", lg="4"), spacing="4", width="100%",
            class_name="stagger-children",
        ),
        # Risk Impact Radar — multi-dimensional risk profile
        rx.cond(
            DashboardState.risk_radar_data.length() > 0,
            rx.box(
                rx.flex(
                    rx.text("RISK IMPACT PROFILE", font_size="11px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.text("Higher values = greater severity", color=MUTED, font_size="10px",
                            font_style="italic"),
                    align="center", margin_bottom="16px",
                ),
                rx.recharts.responsive_container(
                    rx.recharts.radar_chart(
                        rx.recharts.polar_grid(
                            stroke="rgba(239, 68, 68, 0.12)",
                            grid_type="polygon",
                        ),
                        rx.recharts.polar_angle_axis(
                            data_key="risk",
                            tick={"fill": TEXT2, "fontSize": 10},
                        ),
                        rx.recharts.polar_radius_axis(
                            angle=90, domain=[0, "dataMax"],
                            tick=False, axis_line=False,
                        ),
                        rx.recharts.radar(
                            data_key="impact", name="Impact %",
                            stroke=RED, fill=RED, fill_opacity=0.2,
                            stroke_width=2.5,
                            dot={"r": 5, "fill": RED, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_begin=200,
                            animation_duration=1500,
                            animation_easing="ease-out",
                        ),
                        _chart_tooltip(),
                        data=DashboardState.risk_radar_data,
                        cx="50%", cy="50%",
                        outer_radius="72%",
                    ),
                    width="100%", height=420,
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.6s", "0.05s"),
            ),
            rx.fragment(),
        ),
        text_card("Risk Narrative", DashboardState.risk_narrative, "triangle-alert"),
        rx.vstack(
            rx.foreach(DashboardState.risks_list, _render_risk),
            spacing="3", width="100%",
        ),
        divider(),
        section_header("Catalysts Timeline",
                       "Key upcoming events and expected impact", "calendar"),
        # Catalyst Impact Line Chart — connected impact timeline
        rx.cond(
            DashboardState.catalyst_chart_data.length() > 0,
            rx.box(
                rx.flex(
                    rx.text("EXPECTED CATALYST IMPACT", font_size="11px", color=MUTED,
                            font_weight="600", letter_spacing="0.12em"),
                    rx.box(flex="1"),
                    rx.text("Expected move % by catalyst", color=MUTED, font_size="10px",
                            font_style="italic"),
                    align="center", margin_bottom="16px",
                ),
                rx.recharts.responsive_container(
                    rx.recharts.composed_chart(
                        rx.recharts.cartesian_grid(
                            stroke_dasharray="3 3", opacity=0.06,
                            vertical=False,
                        ),
                        rx.recharts.x_axis(
                            data_key="name", stroke=MUTED, font_size=10,
                            tick={"fill": TEXT2},
                            interval=0, angle=-25,
                            axis_line=False,
                            tick_margin=10,
                        ),
                        rx.recharts.y_axis(
                            stroke=MUTED, font_size=11,
                            tick={"fill": TEXT2},
                            unit="%", axis_line=False, tick_line=False,
                        ),
                        rx.recharts.area(
                            data_key="impact", name="Expected Move %",
                            stroke=PURPLE, stroke_width=2.5,
                            fill=PURPLE, fill_opacity=0.15,
                            type_="monotone",
                            dot={"r": 6, "fill": PURPLE, "stroke": BG, "strokeWidth": 2},
                            active_dot={"r": 8, "fill": PURPLE, "stroke": BG, "strokeWidth": 2},
                            is_animation_active=True,
                            animation_begin=200,
                            animation_duration=1500,
                            animation_easing="ease-out",
                        ),
                        rx.recharts.reference_line(
                            y=0, stroke="rgba(100, 116, 139, 0.3)", stroke_width=1,
                        ),
                        _chart_tooltip(),
                        data=DashboardState.catalyst_chart_data,
                        margin={"top": 25, "right": 20, "bottom": 50, "left": 40},
                    ),
                    width="100%", height=400,
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.6s", "0.1s"),
            ),
            rx.fragment(),
        ),
        rx.vstack(
            rx.foreach(DashboardState.catalysts_list, _render_catalyst),
            spacing="3", width="100%",
        ),
        spacing="4", width="100%",
    )


# ── Monitoring Tab ──

def _render_monitor(item: dict):
    return rx.box(
        rx.flex(
            rx.box(
                rx.text(item["metric"], color=TEXT, font_size="14px", font_weight="700"),
                rx.flex(
                    rx.text("Current: ", color=MUTED, font_size="12px"),
                    rx.text(item["current"], color=CYAN, font_size="12px", font_weight="700"),
                    gap="2px", margin_top="6px",
                ),
                flex="1", min_width="150px",
            ),
            rx.box(
                rx.flex(
                    rx.icon(tag="trending-up", size=12, color=GREEN),
                    rx.text("Bullish if", color=GREEN, font_size="11px", font_weight="700"),
                    align="center", gap="4px",
                ),
                rx.text(item["bullish"], color=TEXT2, font_size="12px", margin_top="4px"),
                rx.text(item["action_bull"], color=MUTED, font_size="11px",
                        margin_top="4px", font_style="italic"),
                flex="1", min_width="140px",
            ),
            rx.box(
                rx.flex(
                    rx.icon(tag="trending-down", size=12, color=RED),
                    rx.text("Bearish if", color=RED, font_size="11px", font_weight="700"),
                    align="center", gap="4px",
                ),
                rx.text(item["bearish"], color=TEXT2, font_size="12px", margin_top="4px"),
                rx.text(item["action_bear"], color=MUTED, font_size="11px",
                        margin_top="4px", font_style="italic"),
                flex="1", min_width="140px",
            ),
            gap="28px", flex_wrap="wrap", align="start", width="100%",
        ),
        padding="20px 24px", border_radius="14px",
        background="rgba(30, 41, 59, 0.15)",
        border="1px solid rgba(30, 41, 59, 0.25)",
        transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
        _hover={"border": BORDER_GLOW, "transform": "translateY(-2px)",
                "box_shadow": "0 8px 24px rgba(0,0,0,0.2)"},
        width="100%",
    )


def monitoring_tab():
    return rx.vstack(
        section_header("Monitoring Checklist",
                       "Post-entry metrics to track weekly", "eye"),
        # Summary stat cards
        rx.grid(
            stat_card("RSI Current", DashboardState.monitoring_rsi_current,
                      subtitle="Momentum", icon="activity", color=AMBER),
            stat_card("P/E Ratio", DashboardState.monitoring_pe_current,
                      subtitle="Valuation", icon="calculator", color=BLUE),
            stat_card("SMA 50", DashboardState.monitoring_sma50_current,
                      subtitle="Trend", icon="trending-up", color=GREEN),
            stat_card("Next Catalyst", DashboardState.next_catalyst_str,
                      subtitle=DashboardState.next_catalyst_date, icon="calendar", color=CYAN),
            columns=rx.breakpoints(initial="2", lg="4"), spacing="4", width="100%",
            class_name="stagger-children",
        ),
        # Intro text
        rx.box(
            rx.flex(
                rx.icon(tag="info", size=14, color=BLUE),
                rx.text(
                    "Track these indicators weekly to assess whether the investment thesis remains valid.",
                    color=TEXT2, font_size="13px", line_height="1.6",
                ),
                align="start", gap="10px",
            ),
            padding="16px", border_radius="12px",
            background="rgba(59, 130, 246, 0.03)",
            border="1px solid rgba(59, 130, 246, 0.08)",
            animation=_anim("fadeInUp", "0.4s"),
        ),
        # Monitoring items with enhanced styling
        rx.vstack(
            rx.foreach(DashboardState.monitoring_list, _render_monitor),
            spacing="3", width="100%",
            class_name="stagger-children",
        ),
        divider(),
        # Action Triggers section
        section_header("Action Triggers", "Conditional responses based on market signals", "zap"),
        rx.grid(
            # Bullish triggers
            rx.box(
                rx.flex(
                    rx.box(
                        width="4px", height="100%", min_height="20px",
                        background=GREEN, border_radius="2px", flex_shrink="0",
                    ),
                    rx.box(
                        rx.flex(
                            rx.icon(tag="trending-up", size=16, color=GREEN),
                            rx.text("BULLISH TRIGGERS", font_size="11px", color=GREEN,
                                    font_weight="700", letter_spacing="0.12em"),
                            align="center", gap="8px", margin_bottom="14px",
                        ),
                        rx.text(
                            "Actions to take if conditions turn bullish",
                            color=MUTED, font_size="12px", margin_bottom="14px",
                        ),
                        rx.vstack(
                            rx.foreach(
                                DashboardState.monitoring_list,
                                lambda item: rx.flex(
                                    glow_dot(GREEN, "6px"),
                                    rx.box(
                                        rx.text(item["metric"], color=TEXT, font_size="13px",
                                                font_weight="600"),
                                        rx.text(item["action_bull"], color=TEXT2, font_size="12px",
                                                margin_top="2px", line_height="1.5"),
                                    ),
                                    align="start", gap="10px",
                                    padding="10px 0",
                                    border_bottom="1px solid rgba(30, 41, 59, 0.15)",
                                ),
                            ),
                            spacing="0", width="100%",
                        ),
                        flex="1",
                    ),
                    gap="14px", height="100%",
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.5s", "0.05s"),
            ),
            # Bearish triggers
            rx.box(
                rx.flex(
                    rx.box(
                        width="4px", height="100%", min_height="20px",
                        background=RED, border_radius="2px", flex_shrink="0",
                    ),
                    rx.box(
                        rx.flex(
                            rx.icon(tag="trending-down", size=16, color=RED),
                            rx.text("BEARISH TRIGGERS", font_size="11px", color=RED,
                                    font_weight="700", letter_spacing="0.12em"),
                            align="center", gap="8px", margin_bottom="14px",
                        ),
                        rx.text(
                            "Actions to take if conditions turn bearish",
                            color=MUTED, font_size="12px", margin_bottom="14px",
                        ),
                        rx.vstack(
                            rx.foreach(
                                DashboardState.monitoring_list,
                                lambda item: rx.flex(
                                    glow_dot(RED, "6px"),
                                    rx.box(
                                        rx.text(item["metric"], color=TEXT, font_size="13px",
                                                font_weight="600"),
                                        rx.text(item["action_bear"], color=TEXT2, font_size="12px",
                                                margin_top="2px", line_height="1.5"),
                                    ),
                                    align="start", gap="10px",
                                    padding="10px 0",
                                    border_bottom="1px solid rgba(30, 41, 59, 0.15)",
                                ),
                            ),
                            spacing="0", width="100%",
                        ),
                        flex="1",
                    ),
                    gap="14px", height="100%",
                ),
                style=CARD_STATIC,
                animation=_anim("fadeInUp", "0.5s", "0.1s"),
            ),
            columns=rx.breakpoints(initial="1", md="2"), spacing="4", width="100%",
        ),
        spacing="4", width="100%",
    )


# ═════════════════════════════════════════════════════════════════════════════
# LARY'S CONTRIBUTION TAB
# ═════════════════════════════════════════════════════════════════════════════

# _REPORT_MAILTO = (
#     "mailto:tamer.atesyakar@mail.ru"
#     "?subject=Report%3A%20Lary%20did%20not%20contribute%20to%20the%20project"
#     "&body=Dear%20Professor%2C%0A%0A"
#     "I%20am%20writing%20to%20formally%20report%20that%20Lary%20did%20not%20contribute%20"
#     "to%20this%20project.%20He%20does%20not%20deserve%20the%20grade%20as%20he%20made%20"
#     "zero%20meaningful%20contributions%20to%20the%20Master%20Analyst%20Agent%20system.%0A%0A"
#     "The%20evidence%20is%20clear%20from%20the%20codebase%2C%20commit%20history%2C%20"
#     "and%20the%20dashboard%20itself%20%E2%80%94%20his%20contribution%20is%20literally%20NONE.%0A%0A"
#     "Best%20regards"
# )

# # Lary is the GOAT (of doing nothing)
# GOAT_EMOJI = "\U0001F410"


# def lary_tab():
#     return rx.vstack(
#         section_header("Lary's Contribution", "Comprehensive assessment of Lary's work", "user-x"),
#         # Photo + NONE side by side
#         rx.grid(
#             # Left: Lary's photo
#             rx.box(
#                 rx.text("TEAM MEMBER", font_size="10px", color=MUTED,
#                         font_weight="600", letter_spacing="0.12em", margin_bottom="10px"),
#                 rx.image(
#                     src="/image.png",
#                     width="100%", max_height="340px",
#                     object_fit="cover", border_radius="10px",
#                     border="1px solid rgba(239, 68, 68, 0.15)",
#                     transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
#                     _hover={
#                         "transform": "scale(1.02)",
#                         "border": "1px solid rgba(239, 68, 68, 0.35)",
#                         "box_shadow": "0 12px 40px rgba(239, 68, 68, 0.12)",
#                     },
#                 ),
#                 rx.text("Lary", color=TEXT, font_size="16px", font_weight="700",
#                         margin_top="8px"),
#                 rx.text("Fundamental Analyst (10% weight)", color=MUTED,
#                         font_size="11px", margin_top="2px"),
#                 style=CARD_STATIC,
#                 animation=_anim("fadeInUp", "0.5s"),
#             ),
#             # Right: The big NONE
#             rx.box(
#                 rx.center(
#                     rx.vstack(
#                         rx.box(
#                             rx.icon(tag="user-x", size=40, color=RED),
#                             padding="16px", border_radius="16px",
#                             background="rgba(239, 68, 68, 0.06)",
#                             border="1px solid rgba(239, 68, 68, 0.12)",
#                             animation=_anim("scaleIn", "0.6s"),
#                         ),
#                         rx.text("NONE", font_size="90px", font_weight="900",
#                                 letter_spacing="-0.04em", line_height="1",
#                                 background_image="linear-gradient(135deg, #ef4444, #f43f5e, #dc2626)",
#                                 background_clip="text",
#                                 style={"WebkitBackgroundClip": "text",
#                                        "WebkitTextFillColor": "transparent"},
#                                 animation=_anim("scaleIn", "0.8s", "0.1s"),
#                                 cursor="default",
#                                 transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
#                                 _hover={"transform": "scale(1.08)",
#                                         "filter": "drop-shadow(0 0 30px rgba(239, 68, 68, 0.4))"}),
#                         rx.text("That's the entirety of Lary's contribution.",
#                                 color=MUTED, font_size="13px", text_align="center",
#                                 animation=_anim("fadeInUp", "0.5s", "0.2s")),
#                         align="center", spacing="3",
#                     ),
#                 ),
#                 background="radial-gradient(ellipse at 50% 40%, rgba(239, 68, 68, 0.04) 0%, transparent 60%)",
#                 border_radius="16px", border="1px solid rgba(239, 68, 68, 0.08)",
#                 padding="30px 16px", min_height="300px",
#                 display="flex", align_items="center", justify_content="center",
#             ),
#             columns=rx.breakpoints(initial="1", md="2"), spacing="4", width="100%",
#         ),
#         # Stats row 1
#         rx.grid(
#             stat_card("Lines of Code", "0", subtitle="Written by Lary", icon="code", color=RED),
#             stat_card("Commits", "0", subtitle="From Lary", icon="git-commit-horizontal", color=RED),
#             stat_card("Hours Worked", "0", subtitle="Approximately", icon="clock", color=RED),
#             stat_card("Meetings Attended", "0", subtitle="Out of all meetings", icon="calendar-x", color=RED),
#             columns=rx.breakpoints(initial="2", lg="4"), spacing="4", width="100%",
#             class_name="stagger-children",
#         ),
#         # Stats row 2
#         rx.grid(
#             stat_card("Pull Requests", "0", subtitle="Submitted", icon="git-pull-request", color=ROSE),
#             stat_card("Code Reviews", "0", subtitle="Completed", icon="search", color=ROSE),
#             stat_card("Slack Messages", "3", subtitle="All excuses", icon="message-square", color=AMBER),
#             stat_card("Overall Score", "2%", subtitle="Mathematically computed", icon="percent", color=RED),
#             columns=rx.breakpoints(initial="2", lg="4"), spacing="4", width="100%",
#             class_name="stagger-children",
#         ),
#         # GOAT report button
#         rx.center(
#             rx.el.a(
#                 rx.box(
#                     rx.text(GOAT_EMOJI, font_size="72px", line_height="1",
#                             cursor="pointer",
#                             transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
#                             _hover={"transform": "scale(1.25) rotate(-10deg)",
#                                     "filter": "drop-shadow(0 0 30px rgba(239, 68, 68, 0.5))"}),
#                     rx.text("REPORT LARY", color=RED, font_size="11px",
#                             font_weight="700", margin_top="8px",
#                             letter_spacing="0.12em",
#                             opacity="0.6",
#                             transition="opacity 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
#                             _hover={"opacity": "1"}),
#                     text_align="center", cursor="pointer",
#                     padding="16px", border_radius="16px",
#                     transition="all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)",
#                     _hover={"background": "rgba(239, 68, 68, 0.04)"},
#                 ),
#                 href=_REPORT_MAILTO,
#                 text_decoration="none",
#             ),
#             width="100%",
#             animation=_anim("fadeInUp", "0.5s", "0.2s"),
#         ),
#         spacing="4", width="100%",
#     )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

def results_content():
    return rx.box(
        rx.box(
            rx.match(
                DashboardState.active_tab,
                ("overview", overview_tab()),
                ("agents", agents_tab()),
                ("analysis", analysis_tab()),
                ("thesis", thesis_tab()),
                ("risk", risk_tab()),
                ("monitoring", monitoring_tab()),
                overview_tab(),
            ),
            animation=_anim("fadeIn", "0.4s", "0.05s"),
        ),
        margin_left="260px", padding="36px", min_height="100vh",
        width="calc(100% - 260px)",
    )


def results_view():
    return rx.box(
        rx.flex(sidebar(), results_content(), width="100%"),
        position="relative", width="100%", min_height="100vh",
    )


@rx.page(route="/", on_load=DashboardState.check_existing_data)
def index() -> rx.Component:
    return rx.box(
        rx.match(
            DashboardState.view_mode,
            ("launch", launch_screen()),
            ("running", progress_screen()),
            ("results", results_view()),
            launch_screen(),
        ),
        background="radial-gradient(ellipse at 35% 45%, #081840 0%, #060E2E 25%, #040A1E 55%, #020612 80%, #010308 100%)",
        min_height="100vh", color=TEXT,
        font_family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    )


# ═════════════════════════════════════════════════════════════════════════════
# APP CONFIG & ANIMATIONS
# ═════════════════════════════════════════════════════════════════════════════

style = {
    "::selection": {"background": "rgba(59, 130, 246, 0.3)"},
    # Entrance animations
    "@keyframes fadeInUp": {
        "from": {"opacity": "0", "transform": "translateY(24px)"},
        "to": {"opacity": "1", "transform": "translateY(0)"},
    },
    "@keyframes fadeIn": {
        "from": {"opacity": "0"},
        "to": {"opacity": "1"},
    },
    "@keyframes scaleIn": {
        "from": {"opacity": "0", "transform": "scale(0.9)"},
        "to": {"opacity": "1", "transform": "scale(1)"},
    },
    # Looping animations
    "@keyframes pulse": {
        "0%, 100%": {"opacity": "1", "transform": "scale(1)"},
        "50%": {"opacity": "0.9", "transform": "scale(1.015)"},
    },
    "@keyframes statusPulse": {
        "0%, 100%": {"opacity": "1", "box_shadow": "0 0 0 0 rgba(245, 158, 11, 0.4)"},
        "50%": {"opacity": "0.5", "box_shadow": "0 0 0 5px rgba(245, 158, 11, 0)"},
    },
    "@keyframes float": {
        "0%, 100%": {"transform": "translateY(0px)"},
        "50%": {"transform": "translateY(-5px)"},
    },
    "@keyframes gradientShift": {
        "0%, 100%": {"background_position": "0% center"},
        "50%": {"background_position": "100% center"},
    },
    # Progress animations
    "@keyframes jumpBall": {
        "0%, 100%": {"transform": "translateY(0) scale(1)", "opacity": "0.6"},
        "35%": {"transform": "translateY(-30px) scale(1.25)", "opacity": "1"},
        "65%": {"transform": "translateY(-30px) scale(1.25)", "opacity": "1"},
    },
    "@keyframes waveBar": {
        "0%, 100%": {"height": "5px"},
        "50%": {"height": "30px"},
    },
    "@keyframes sonarPulse": {
        "0%": {"transform": "scale(1)", "opacity": "0.5"},
        "70%": {"transform": "scale(1.35)", "opacity": "0"},
        "100%": {"transform": "scale(1.35)", "opacity": "0"},
    },
    # ═══ PS5 PARTICLE SYSTEM — 5 layers, 300+ particles ═══
    # Rising paths — 4 variants with multi-point sinusoidal sway
    # ── WARP TRANSITION — PS2-style zoom + flash ──
    # ── PS5 Particle keyframes ──
    "@keyframes ps5Rise1": {
        "0%": {"transform": "translateY(0) translateX(0)", "opacity": "0"},
        "4%": {"opacity": "1"},
        "15%": {"transform": "translateY(-18vh) translateX(25px)"},
        "30%": {"transform": "translateY(-35vh) translateX(-15px)"},
        "45%": {"transform": "translateY(-50vh) translateX(20px)"},
        "60%": {"transform": "translateY(-68vh) translateX(-18px)"},
        "75%": {"transform": "translateY(-82vh) translateX(12px)"},
        "90%": {"opacity": "0.8"},
        "100%": {"transform": "translateY(-120vh) translateX(-8px)", "opacity": "0"},
    },
    "@keyframes ps5Rise2": {
        "0%": {"transform": "translateY(0) translateX(0)", "opacity": "0"},
        "5%": {"opacity": "1"},
        "20%": {"transform": "translateY(-22vh) translateX(-30px)"},
        "40%": {"transform": "translateY(-42vh) translateX(22px)"},
        "55%": {"transform": "translateY(-58vh) translateX(-12px)"},
        "70%": {"transform": "translateY(-75vh) translateX(25px)"},
        "85%": {"transform": "translateY(-95vh) translateX(-15px)"},
        "92%": {"opacity": "0.6"},
        "100%": {"transform": "translateY(-120vh) translateX(10px)", "opacity": "0"},
    },
    "@keyframes ps5Rise3": {
        "0%": {"transform": "translateY(0) translateX(0)", "opacity": "0"},
        "3%": {"opacity": "1"},
        "12%": {"transform": "translateY(-14vh) translateX(18px)"},
        "28%": {"transform": "translateY(-32vh) translateX(-22px)"},
        "42%": {"transform": "translateY(-48vh) translateX(14px)"},
        "58%": {"transform": "translateY(-65vh) translateX(-20px)"},
        "72%": {"transform": "translateY(-80vh) translateX(16px)"},
        "88%": {"transform": "translateY(-100vh) translateX(-10px)", "opacity": "0.7"},
        "100%": {"transform": "translateY(-120vh) translateX(5px)", "opacity": "0"},
    },
    # Falling particles — top to bottom
    "@keyframes ps5Fall1": {
        "0%": {"transform": "translateY(0) translateX(0)", "opacity": "0"},
        "4%": {"opacity": "1"},
        "15%": {"transform": "translateY(18vh) translateX(20px)"},
        "30%": {"transform": "translateY(35vh) translateX(-15px)"},
        "50%": {"transform": "translateY(55vh) translateX(22px)"},
        "70%": {"transform": "translateY(75vh) translateX(-12px)"},
        "88%": {"opacity": "0.6"},
        "100%": {"transform": "translateY(120vh) translateX(8px)", "opacity": "0"},
    },
    # Organic float paths — 3 distinct paths
    "@keyframes ps5Float1": {
        "0%": {"transform": "translate(0, 0) scale(1)"},
        "12%": {"transform": "translate(30px, -20px) scale(1.05)"},
        "25%": {"transform": "translate(-15px, -45px) scale(0.95)"},
        "37%": {"transform": "translate(40px, -25px) scale(1.08)"},
        "50%": {"transform": "translate(-10px, 15px) scale(1)"},
        "62%": {"transform": "translate(-35px, -30px) scale(1.04)"},
        "75%": {"transform": "translate(20px, 10px) scale(0.96)"},
        "87%": {"transform": "translate(-25px, 20px) scale(1.02)"},
        "100%": {"transform": "translate(0, 0) scale(1)"},
    },
    "@keyframes ps5Float2": {
        "0%": {"transform": "translate(0, 0) scale(1)"},
        "14%": {"transform": "translate(-25px, 35px) scale(1.06)"},
        "28%": {"transform": "translate(35px, 10px) scale(0.94)"},
        "42%": {"transform": "translate(-20px, -40px) scale(1.03)"},
        "56%": {"transform": "translate(15px, 25px) scale(0.98)"},
        "70%": {"transform": "translate(40px, -15px) scale(1.07)"},
        "85%": {"transform": "translate(-30px, -10px) scale(0.97)"},
        "100%": {"transform": "translate(0, 0) scale(1)"},
    },
    "@keyframes ps5Float3": {
        "0%": {"transform": "translate(0, 0) scale(1)"},
        "16%": {"transform": "translate(20px, 30px) scale(0.95)"},
        "33%": {"transform": "translate(-40px, -20px) scale(1.08)"},
        "50%": {"transform": "translate(10px, -35px) scale(1)"},
        "66%": {"transform": "translate(35px, 15px) scale(1.05)"},
        "83%": {"transform": "translate(-15px, 25px) scale(0.97)"},
        "100%": {"transform": "translate(0, 0) scale(1)"},
    },
    # Counter-style entrance
    "@keyframes countUp": {
        "from": {"opacity": "0", "transform": "translateY(8px) scale(0.95)"},
        "to": {"opacity": "1", "transform": "translateY(0) scale(1)"},
    },
    "@keyframes glowPulse": {
        "0%, 100%": {"box_shadow": "0 0 0px rgba(59, 130, 246, 0)"},
        "50%": {"box_shadow": "0 0 20px rgba(59, 130, 246, 0.12)"},
    },
    "@keyframes orbitSpin": {
        "from": {"transform": "rotate(0deg)"},
        "to": {"transform": "rotate(360deg)"},
    },
    "@keyframes borderGlow": {
        "0%, 100%": {"border_color": "rgba(59, 130, 246, 0.15)", "box_shadow": "0 0 20px rgba(59, 130, 246, 0.05)"},
        "50%": {"border_color": "rgba(139, 92, 246, 0.3)", "box_shadow": "0 0 40px rgba(139, 92, 246, 0.1)"},
    },
    "@keyframes typewriter": {
        "from": {"width": "0"},
        "to": {"width": "100%"},
    },
    "@keyframes tickerScroll": {
        "from": {"transform": "translateX(0)"},
        "to": {"transform": "translateX(-50%)"},
    },
    # Stagger children animations
    ".stagger-children > *": {"animation": "fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both"},
    ".stagger-children > *:nth-child(1)": {"animation_delay": "0s"},
    ".stagger-children > *:nth-child(2)": {"animation_delay": "0.06s"},
    ".stagger-children > *:nth-child(3)": {"animation_delay": "0.12s"},
    ".stagger-children > *:nth-child(4)": {"animation_delay": "0.18s"},
    ".stagger-children > *:nth-child(5)": {"animation_delay": "0.24s"},
    ".stagger-children > *:nth-child(6)": {"animation_delay": "0.3s"},
    ".stagger-children > *:nth-child(n+7)": {"animation_delay": "0.36s"},
    # ─── iOS 26 Liquid Glass buttons & interactive elements ───
    "button, [role='button'], [data-radix-collection-item]": {
        "background": "linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)) !important",
        "border": "1px solid rgba(255,255,255,0.10) !important",
        "border_radius": "14px !important",
        "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.12), 0 2px 8px rgba(0,0,0,0.15) !important",
        "transition": "all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0) !important",
    },
    "button:hover, [role='button']:hover": {
        "transform": "translateY(-2px) scale(1.02)",
        "background": "linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05)) !important",
        "border": "1px solid rgba(255,255,255,0.18) !important",
        "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.18), 0 8px 24px rgba(0,0,0,0.25), 0 0 20px rgba(59,130,246,0.08) !important",
    },
    "button:active, [role='button']:active": {
        "transform": "scale(0.97) translateY(0px) !important",
        "transition": "transform 0.1s cubic-bezier(0.25, 0.1, 0.25, 1.0) !important",
        "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.08), 0 1px 4px rgba(0,0,0,0.20) !important",
    },
    "a": {
        "transition": "all 0.4s cubic-bezier(0.25, 0.1, 0.25, 1.0) !important",
    },
    "input, textarea, select": {
        "background": "linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)) !important",
        "border": "1px solid rgba(255,255,255,0.08) !important",
        "border_radius": "12px !important",
        "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.08) !important",
        "transition": "all 0.5s cubic-bezier(0.25, 0.1, 0.25, 1.0) !important",
    },
    "input:focus, textarea:focus": {
        "transform": "scale(1.01)",
        "border": "1px solid rgba(59,130,246,0.25) !important",
        "box_shadow": "inset 0 1px 0 rgba(255,255,255,0.12), 0 0 20px rgba(59,130,246,0.08) !important",
    },
    # Smooth all box/div transitions for cards, containers etc.
    "div": {
        "transition_property": "transform, box-shadow, border-color, background, opacity",
        "transition_duration": "0.4s",
        "transition_timing_function": "cubic-bezier(0.25, 0.1, 0.25, 1.0)",
    },
    # Global styles
    "body": {"background": BG, "color": TEXT},
    "::-webkit-scrollbar": {"width": "6px"},
    "::-webkit-scrollbar-track": {"background": "transparent"},
    "::-webkit-scrollbar-thumb": {
        "background": "rgba(59, 130, 246, 0.15)",
        "border_radius": "3px",
    },
    "::-webkit-scrollbar-thumb:hover": {
        "background": "rgba(59, 130, 246, 0.3)",
    },
}

app = rx.App(
    theme=rx.theme(appearance="dark", accent_color="blue", radius="large"),
    style=style,
)
