"""Declarative specifications of the demo / study interfaces.

Each :class:`UISpec` is rendered to a PNG by :mod:`gazegrounded.data.ui_render`, and its
elements marked ``is_aoi`` become the Areas of Interest (AOIs). Keeping layout, AOIs, tasks and
the *designed* friction manipulation in one place guarantees that screenshots, AOI tables and
study stimuli never drift apart.

Coordinates are normalised to [0, 1] (x_min, y_min, x_max, y_max).

``salience`` is a hand-set visual prominence (size x contrast x emphasis) used ONLY by the
synthetic gaze simulator. ``design_friction`` is the intended severity of a task's friction
manipulation, again only used for simulation. Neither is visible to the VLM providers.
"""

from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Element:
    id: str
    kind: str  # header|heading|text|dense_text|card|button|button_outline|link|input|banner|icon|image_card|footer
    label: str
    box: Box
    role: str = "text"
    description: str = ""
    salience: float = 0.3
    is_aoi: bool = True
    fill: str | None = None
    fg: str | None = None
    font_size: int = 18
    glyph: str | None = None  # for kind == "icon": lock|pin|more


@dataclass(frozen=True)
class TaskSpec:
    task_key: str  # shared across clean/friction variants of the same task
    description: str
    target_aoi_ids: tuple[str, ...]
    friction_type: str  # one of FEEDBACK_CATEGORIES (designed manipulation)
    design_friction: float  # intended severity in [0, 1]; simulation only


@dataclass(frozen=True)
class UISpec:
    ui_id: str
    title: str
    domain: str
    condition: str  # clean | friction | single
    pair_id: str | None
    elements: tuple[Element, ...]
    tasks: tuple[TaskSpec, ...]
    background: str = "#F6F7F9"
    width: int = 1440
    height: int = 900
    notes: str = ""

    def aoi_elements(self) -> list[Element]:
        return [e for e in self.elements if e.is_aoi]


# ---------------------------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------------------------


def _header(brand: str, color: str, nav: str) -> list[Element]:
    return [
        Element(
            "header",
            "header",
            brand,
            (0.0, 0.0, 1.0, 0.075),
            role="navigation",
            description=f"Top navigation bar with the {brand} logo",
            salience=0.35,
            fill=color,
            fg="#FFFFFF",
            font_size=26,
        ),
        Element(
            "header_nav",
            "text",
            nav,
            (0.55, 0.022, 0.97, 0.06),
            is_aoi=False,
            fg="#E8ECF4",
            font_size=17,
        ),
    ]


def _footer(text: str, box: Box = (0.25, 0.935, 0.75, 0.97)) -> list[Element]:
    return [
        Element("footer_bg", "footer", "", (0.0, 0.92, 1.0, 1.0), is_aoi=False, fill="#E9ECF1"),
        Element(
            "footer",
            "text",
            text,
            box,
            role="navigation",
            description="Footer links to help, returns and legal pages",
            salience=0.08,
            fg="#7A828E",
            font_size=14,
        ),
    ]


# ---------------------------------------------------------------------------------------------
# E-commerce checkout (paired clean / friction)
# ---------------------------------------------------------------------------------------------

_SHOP_TASKS_CLEAN = (
    TaskSpec(
        "change_address",
        "Change the delivery address for this order.",
        ("edit_address",),
        "no_friction",
        0.12,
    ),
    TaskSpec(
        "apply_discount",
        "Apply the discount code SAVE10 to your order.",
        ("discount_input", "discount_apply"),
        "no_friction",
        0.15,
    ),
)
_SHOP_TASKS_FRICTION = (
    TaskSpec(
        "change_address",
        "Change the delivery address for this order.",
        ("edit_address_link",),
        "findability",
        0.82,
    ),
    TaskSpec(
        "apply_discount",
        "Apply the discount code SAVE10 to your order.",
        ("discount_link",),
        "ambiguous_wording",
        0.72,
    ),
)

_ADDRESS_TEXT = "Alex Example\nMusterstrasse 12\n1010 Vienna, Austria"
_SUMMARY_TEXT = "Wool sweater x1          EUR 79.00\nCanvas sneakers x1     EUR 64.00\n\nSubtotal                     EUR 143.00\nDelivery                     Free"

SHOP_CLEAN = UISpec(
    ui_id="shop_checkout_clean",
    title="Checkout (clean)",
    domain="e-commerce",
    condition="clean",
    pair_id="shop_checkout",
    elements=(
        *_header("ShopNow", "#1F2A44", "Women    Men    Home    Sale    Cart (2)"),
        Element(
            "progress",
            "text",
            "Cart   >   Delivery   >   Payment   >   Review",
            (0.06, 0.1, 0.5, 0.135),
            role="navigation",
            description="Checkout progress steps",
            salience=0.25,
            fg="#5F6B7A",
            font_size=16,
        ),
        Element(
            "title",
            "heading",
            "Checkout",
            (0.06, 0.145, 0.4, 0.2),
            role="heading",
            description="Page title",
            salience=0.5,
            font_size=34,
        ),
        Element(
            "address_card",
            "card",
            "Delivery address",
            (0.06, 0.22, 0.6, 0.44),
            role="region",
            description="Card showing the current delivery address",
            salience=0.5,
        ),
        Element(
            "address_text",
            "text",
            _ADDRESS_TEXT,
            (0.08, 0.28, 0.4, 0.42),
            is_aoi=False,
            fg="#2B2F36",
        ),
        Element(
            "edit_address",
            "button",
            "Edit address",
            (0.43, 0.355, 0.58, 0.41),
            role="button",
            description="Button to edit or change the delivery address",
            salience=0.7,
            fill="#2557D6",
            fg="#FFFFFF",
        ),
        Element(
            "delivery_options",
            "card",
            "Delivery options",
            (0.06, 0.47, 0.6, 0.64),
            role="region",
            description="Choice between standard and express shipping speed",
            salience=0.4,
        ),
        Element(
            "delivery_text",
            "text",
            "(o) Standard, 3-5 days      Free\n( ) Express, 1-2 days        EUR 9.90",
            (0.08, 0.53, 0.5, 0.62),
            is_aoi=False,
            fg="#2B2F36",
        ),
        Element(
            "payment",
            "card",
            "Payment method",
            (0.06, 0.67, 0.6, 0.86),
            role="region",
            description="Selected payment card",
            salience=0.4,
        ),
        Element(
            "payment_text",
            "text",
            "Visa **** 4242      Expires 08/28",
            (0.08, 0.74, 0.5, 0.8),
            is_aoi=False,
            fg="#2B2F36",
        ),
        Element(
            "order_summary",
            "card",
            "Order summary",
            (0.65, 0.22, 0.94, 0.64),
            role="region",
            description="Items, subtotal and total of the order",
            salience=0.5,
        ),
        Element(
            "summary_text",
            "text",
            _SUMMARY_TEXT,
            (0.67, 0.28, 0.93, 0.5),
            is_aoi=False,
            fg="#2B2F36",
            font_size=16,
        ),
        Element(
            "discount_input",
            "input",
            "Discount code",
            (0.67, 0.53, 0.83, 0.585),
            role="input",
            description="Text field to enter a discount or promo code",
            salience=0.45,
        ),
        Element(
            "discount_apply",
            "button",
            "Apply",
            (0.84, 0.53, 0.92, 0.585),
            role="button",
            description="Button to apply the entered discount code",
            salience=0.5,
            fill="#2557D6",
            fg="#FFFFFF",
        ),
        Element(
            "place_order",
            "button",
            "Place order  -  EUR 143.00",
            (0.65, 0.68, 0.94, 0.76),
            role="button",
            description="Primary button to pay for and place the order",
            salience=0.85,
            fill="#1E8E3E",
            fg="#FFFFFF",
            font_size=22,
        ),
        *_footer("Help    Returns    Privacy    Terms"),
    ),
    tasks=_SHOP_TASKS_CLEAN,
)

SHOP_FRICTION = UISpec(
    ui_id="shop_checkout_friction",
    title="Checkout (friction)",
    domain="e-commerce",
    condition="friction",
    pair_id="shop_checkout",
    notes="Edit-address control moved to a low-contrast footer link; discount field hidden "
    "behind an ambiguous 'Have a promo?' link; competing promotional banner.",
    elements=(
        *_header("ShopNow", "#1F2A44", "Women    Men    Home    Sale    Cart (2)"),
        Element(
            "progress",
            "text",
            "Cart   >   Delivery   >   Payment   >   Review",
            (0.06, 0.1, 0.5, 0.135),
            role="navigation",
            description="Checkout progress steps",
            salience=0.25,
            fg="#5F6B7A",
            font_size=16,
        ),
        Element(
            "title",
            "heading",
            "Checkout",
            (0.06, 0.145, 0.4, 0.2),
            role="heading",
            description="Page title",
            salience=0.5,
            font_size=34,
        ),
        Element(
            "promo_banner",
            "banner",
            "Members get FREE express delivery - join ShopNow+ today!",
            (0.06, 0.215, 0.94, 0.325),
            role="banner",
            description="Large promotional banner advertising a paid membership",
            salience=0.95,
            fill="#FFD966",
            fg="#3A2E00",
            font_size=28,
        ),
        Element(
            "address_card",
            "card",
            "Delivery",
            (0.06, 0.35, 0.6, 0.55),
            role="region",
            description="Card showing the current delivery address (read-only)",
            salience=0.45,
        ),
        Element(
            "address_text",
            "text",
            _ADDRESS_TEXT,
            (0.08, 0.41, 0.4, 0.54),
            is_aoi=False,
            fg="#2B2F36",
        ),
        Element(
            "delivery_options",
            "card",
            "Delivery options",
            (0.06, 0.58, 0.6, 0.73),
            role="region",
            description="Choice between standard and express shipping speed",
            salience=0.4,
        ),
        Element(
            "delivery_text",
            "text",
            "(o) Standard, 3-5 days      Free\n( ) Express, 1-2 days        EUR 9.90",
            (0.08, 0.635, 0.5, 0.72),
            is_aoi=False,
            fg="#2B2F36",
        ),
        Element(
            "payment",
            "card",
            "Payment method",
            (0.06, 0.76, 0.6, 0.9),
            role="region",
            description="Selected payment card",
            salience=0.4,
        ),
        Element(
            "payment_text",
            "text",
            "Visa **** 4242      Expires 08/28",
            (0.08, 0.82, 0.5, 0.87),
            is_aoi=False,
            fg="#2B2F36",
        ),
        Element(
            "order_summary",
            "card",
            "Order summary",
            (0.65, 0.35, 0.94, 0.73),
            role="region",
            description="Items, subtotal and total of the order",
            salience=0.5,
        ),
        Element(
            "summary_text",
            "text",
            _SUMMARY_TEXT,
            (0.67, 0.41, 0.93, 0.63),
            is_aoi=False,
            fg="#2B2F36",
            font_size=16,
        ),
        Element(
            "discount_link",
            "link",
            "Have a promo?",
            (0.67, 0.66, 0.77, 0.69),
            role="link",
            description="Small grey link that reveals the field for entering a discount "
            "or promo code",
            salience=0.12,
            fg="#A3A9B2",
            font_size=14,
        ),
        Element(
            "place_order",
            "button",
            "Place order  -  EUR 143.00",
            (0.65, 0.76, 0.94, 0.84),
            role="button",
            description="Primary button to pay for and place the order",
            salience=0.85,
            fill="#1E8E3E",
            fg="#FFFFFF",
            font_size=22,
        ),
        Element("footer_bg", "footer", "", (0.0, 0.92, 1.0, 1.0), is_aoi=False, fill="#E9ECF1"),
        Element(
            "edit_address_link",
            "link",
            "Shipping preferences",
            (0.06, 0.937, 0.19, 0.967),
            role="link",
            description="Footer link to shipping preferences, where the "
            "delivery address can be changed",
            salience=0.08,
            fg="#9AA0A6",
            font_size=14,
        ),
        Element(
            "footer",
            "text",
            "Help    Returns    Privacy    Terms",
            (0.3, 0.937, 0.75, 0.967),
            role="navigation",
            description="Footer links to help, returns and legal pages",
            salience=0.08,
            fg="#7A828E",
            font_size=14,
        ),
    ),
    tasks=_SHOP_TASKS_FRICTION,
)

# ---------------------------------------------------------------------------------------------
# Banking card management (paired clean / friction)
# ---------------------------------------------------------------------------------------------

_SIDEBAR = Element(
    "sidebar_nav",
    "text",
    "Overview\n\nAccounts\n\nCards\n\nPayments\n\nSettings",
    (0.02, 0.12, 0.15, 0.6),
    role="navigation",
    description="Sidebar navigation: overview, accounts, cards, payments, settings",
    salience=0.35,
    fg="#24413E",
    font_size=19,
)
_SIDEBAR_BG = Element(
    "sidebar_bg", "footer", "", (0.0, 0.075, 0.17, 1.0), is_aoi=False, fill="#E6EEEC"
)
_TX_TEXT = "12 Oct   Grocery store        -EUR 54.20\n11 Oct   Coffee bar             -EUR 3.80\n10 Oct   Train ticket            -EUR 29.00\n09 Oct   Salary                  +EUR 2,950.00\n08 Oct   Pharmacy                -EUR 12.45"

_CONDITIONS_TEXT = (
    "Card type: Debit Mastercard. Issued 03/2024, valid until 03/2029. Contactless payments are\n"
    "enabled up to EUR 50 per transaction. Foreign transaction fee 1.75%. ATM withdrawals:\n"
    "4 free per month, then EUR 2.00 each. Monthly card limit EUR 1,500 (adjustable in branch\n"
    "or via phone banking). Daily ATM limit EUR 400. Online purchases protected by 3-D Secure.\n"
    "Replacement card fee EUR 10. Statement cycle ends on the last business day of the month.\n"
    "Chargeback requests must be filed within 120 days. Interest does not apply to debit cards.\n"
    "Card insurance: purchase protection up to EUR 1,000 per item for 90 days after purchase.\n"
    "Travel notification not required within the EEA. Emergency cash available abroad 24/7."
)

BANK_CLEAN = UISpec(
    ui_id="bank_cards_clean",
    title="Card management (clean)",
    domain="banking",
    condition="clean",
    pair_id="bank_cards",
    background="#FFFFFF",
    elements=(
        *_header("NordBank", "#0B3D3A", "Help    Messages    Log out"),
        _SIDEBAR_BG,
        _SIDEBAR,
        Element(
            "title",
            "heading",
            "Your cards",
            (0.21, 0.105, 0.5, 0.165),
            role="heading",
            description="Page title",
            salience=0.5,
            font_size=34,
        ),
        Element(
            "card_image",
            "image_card",
            "NordBank  Debit   **** 4821",
            (0.21, 0.2, 0.47, 0.5),
            role="image",
            description="Picture of the debit card ending in 4821",
            salience=0.7,
            fill="#145C57",
            fg="#FFFFFF",
        ),
        Element(
            "card_status",
            "text",
            "Debit card **** 4821\nStatus: Active",
            (0.5, 0.2, 0.75, 0.29),
            role="text",
            description="Card number and current status",
            salience=0.45,
            fg="#1E2B2A",
            font_size=20,
        ),
        Element(
            "freeze_card",
            "button",
            "Freeze card",
            (0.5, 0.33, 0.66, 0.39),
            role="button",
            description="Button to temporarily freeze or block the card",
            salience=0.7,
            fill="#0B6E4F",
            fg="#FFFFFF",
        ),
        Element(
            "report_lost",
            "button_outline",
            "Report lost or stolen",
            (0.68, 0.33, 0.88, 0.39),
            role="button",
            description="Button to report the card as lost or stolen and " "order a replacement",
            salience=0.55,
            fg="#0B6E4F",
        ),
        Element(
            "spending_limit",
            "card",
            "Spending limit",
            (0.21, 0.55, 0.55, 0.74),
            role="region",
            description="Panel showing the card's monthly spending limit",
            salience=0.5,
        ),
        Element(
            "limit_text",
            "text",
            "Monthly limit:   EUR 1,500\nUsed this month:  EUR 412",
            (0.23, 0.61, 0.53, 0.71),
            is_aoi=False,
            fg="#1E2B2A",
            font_size=20,
        ),
        Element(
            "transactions",
            "card",
            "Recent transactions",
            (0.58, 0.47, 0.94, 0.88),
            role="region",
            description="List of recent card transactions",
            salience=0.45,
        ),
        Element(
            "tx_text",
            "text",
            _TX_TEXT,
            (0.6, 0.53, 0.93, 0.85),
            is_aoi=False,
            fg="#36403F",
            font_size=16,
        ),
    ),
    tasks=(
        TaskSpec(
            "freeze_card",
            "You have lost your debit card. Freeze it.",
            ("freeze_card",),
            "no_friction",
            0.12,
        ),
        TaskSpec(
            "find_limit",
            "Find the card's monthly spending limit.",
            ("spending_limit",),
            "no_friction",
            0.14,
        ),
    ),
)

BANK_FRICTION = UISpec(
    ui_id="bank_cards_friction",
    title="Card management (friction)",
    domain="banking",
    condition="friction",
    pair_id="bank_cards",
    background="#FFFFFF",
    notes="Freeze action is an unlabeled padlock icon; spending limit is buried in dense small "
    "print; a large upsell card competes for attention.",
    elements=(
        *_header("NordBank", "#0B3D3A", "Help    Messages    Log out"),
        _SIDEBAR_BG,
        _SIDEBAR,
        Element(
            "title",
            "heading",
            "Your cards",
            (0.21, 0.105, 0.5, 0.165),
            role="heading",
            description="Page title",
            salience=0.5,
            font_size=34,
        ),
        Element(
            "card_image",
            "image_card",
            "NordBank  Debit   **** 4821",
            (0.21, 0.2, 0.47, 0.5),
            role="image",
            description="Picture of the debit card ending in 4821",
            salience=0.7,
            fill="#145C57",
            fg="#FFFFFF",
        ),
        Element(
            "card_status",
            "text",
            "Debit card **** 4821\nStatus: Active",
            (0.5, 0.2, 0.75, 0.29),
            role="text",
            description="Card number and current status",
            salience=0.45,
            fg="#1E2B2A",
            font_size=20,
        ),
        Element(
            "icon_freeze",
            "icon",
            "",
            (0.5, 0.325, 0.54, 0.39),
            role="button",
            description="Icon-only button with a padlock symbol that freezes or blocks the "
            "card (no text label)",
            salience=0.22,
            glyph="lock",
            fg="#5B6B69",
        ),
        Element(
            "icon_pin",
            "icon",
            "",
            (0.555, 0.325, 0.595, 0.39),
            role="button",
            description="Icon-only button that shows the card PIN",
            salience=0.2,
            glyph="pin",
            fg="#5B6B69",
        ),
        Element(
            "icon_more",
            "icon",
            "",
            (0.61, 0.325, 0.65, 0.39),
            role="button",
            description="Icon-only overflow menu with more card options",
            salience=0.2,
            glyph="more",
            fg="#5B6B69",
        ),
        Element(
            "upsell",
            "banner",
            "Upgrade to Platinum\n2% cashback on everything",
            (0.21, 0.54, 0.55, 0.7),
            role="banner",
            description="Large advertisement for a premium card upgrade",
            salience=0.85,
            fill="#C9A227",
            fg="#2A2100",
            font_size=26,
        ),
        Element(
            "card_conditions",
            "dense_text",
            _CONDITIONS_TEXT,
            (0.58, 0.44, 0.97, 0.7),
            role="text",
            description="Dense block of card terms and details, including the "
            "monthly spending limit",
            salience=0.2,
            fg="#7C8584",
            font_size=12,
        ),
        Element(
            "transactions",
            "card",
            "Recent transactions",
            (0.21, 0.73, 0.94, 0.95),
            role="region",
            description="List of recent card transactions",
            salience=0.4,
        ),
        Element(
            "tx_text",
            "text",
            _TX_TEXT.split("\n09")[0],
            (0.23, 0.79, 0.6, 0.94),
            is_aoi=False,
            fg="#36403F",
            font_size=16,
        ),
    ),
    tasks=(
        TaskSpec(
            "freeze_card",
            "You have lost your debit card. Freeze it.",
            ("icon_freeze",),
            "ambiguous_wording",
            0.74,
        ),
        TaskSpec(
            "find_limit",
            "Find the card's monthly spending limit.",
            ("card_conditions",),
            "information_overload",
            0.8,
        ),
    ),
)

# ---------------------------------------------------------------------------------------------
# Travel booking (single version, intermediate friction)
# ---------------------------------------------------------------------------------------------

_FARE_TEXT = (
    "Fare: Economy Light. Changes permitted for a fee of EUR 70 plus any fare difference.\n"
    "Cancellation: non-refundable; taxes refundable on request. Cabin bag: 1 x 8 kg.\n"
    "Checked baggage: 1 x 23 kg included on this booking. Seat selection from EUR 9.\n"
    "Name changes are not permitted. Check-in closes 45 minutes before departure."
)

TRAVEL = UISpec(
    ui_id="travel_booking",
    title="Manage booking",
    domain="travel",
    condition="single",
    pair_id=None,
    notes="Date change hidden in a 'Manage booking' dropdown; baggage allowance in small print.",
    elements=(
        *_header("SkyWays", "#3B1F5C", "Book    My trips    Check-in    Help"),
        Element(
            "title",
            "heading",
            "Booking ABC123  -  Vienna to Lisbon",
            (0.06, 0.105, 0.62, 0.165),
            role="heading",
            description="Booking reference and route",
            salience=0.5,
            font_size=32,
        ),
        Element(
            "flight_details",
            "card",
            "Outbound flight",
            (0.06, 0.2, 0.62, 0.45),
            role="region",
            description="Flight date, times, airports and passenger count",
            salience=0.6,
        ),
        Element(
            "flight_text",
            "text",
            "Fri 14 Nov 2026      07:35  ->  10:10\nVIE Terminal 3      ->  LIS Terminal 1\n1 adult   -   Economy Light",
            (0.08, 0.26, 0.6, 0.43),
            is_aoi=False,
            fg="#2A2236",
            font_size=21,
        ),
        Element(
            "manage_booking",
            "button_outline",
            "Manage booking  v",
            (0.66, 0.2, 0.86, 0.26),
            role="button",
            description="Dropdown menu with options to change the date, "
            "change seats or cancel the booking",
            salience=0.5,
            fg="#3B1F5C",
        ),
        Element(
            "check_in",
            "button",
            "Check in online",
            (0.66, 0.29, 0.94, 0.37),
            role="button",
            description="Primary button to start online check-in",
            salience=0.85,
            fill="#E4572E",
            fg="#FFFFFF",
            font_size=22,
        ),
        Element(
            "extras",
            "card",
            "Add extras",
            (0.66, 0.41, 0.94, 0.62),
            role="region",
            description="Offers to add seats, meals and insurance",
            salience=0.5,
        ),
        Element(
            "extras_text",
            "text",
            "Seats\nMeals\nTravel insurance",
            (0.68, 0.47, 0.9, 0.6),
            is_aoi=False,
            fg="#2A2236",
        ),
        Element(
            "passengers",
            "card",
            "Passengers",
            (0.06, 0.48, 0.62, 0.62),
            role="region",
            description="Passenger names on the booking",
            salience=0.4,
        ),
        Element(
            "passenger_text",
            "text",
            "Alex Example  (adult)",
            (0.08, 0.54, 0.5, 0.6),
            is_aoi=False,
            fg="#2A2236",
        ),
        Element(
            "fare_conditions",
            "dense_text",
            _FARE_TEXT,
            (0.06, 0.66, 0.62, 0.8),
            role="text",
            description="Small-print fare conditions including change fees and the checked "
            "baggage allowance",
            salience=0.2,
            fg="#8A8496",
            font_size=12,
        ),
        *_footer("Help    Contact    Privacy    Terms"),
    ),
    tasks=(
        TaskSpec(
            "change_date", "Change the date of your flight.", ("manage_booking",), "navigation", 0.5
        ),
        TaskSpec(
            "find_baggage",
            "Find out how much checked baggage is included.",
            ("fare_conditions",),
            "visual_hierarchy",
            0.45,
        ),
    ),
)

# ---------------------------------------------------------------------------------------------
# Additional study stimuli (healthcare, subscription). Not used by the synthetic demo.
# ---------------------------------------------------------------------------------------------

HEALTH_CLEAN = UISpec(
    ui_id="health_portal_clean",
    title="Patient portal (clean)",
    domain="healthcare",
    condition="clean",
    pair_id="health_portal",
    background="#FFFFFF",
    elements=(
        *_header("CareLink", "#0F4C81", "Messages    Prescriptions    Profile"),
        Element(
            "title",
            "heading",
            "Good morning, Alex",
            (0.06, 0.105, 0.5, 0.165),
            role="heading",
            description="Greeting",
            salience=0.45,
            font_size=32,
        ),
        Element(
            "next_appointment",
            "card",
            "Your next appointment",
            (0.06, 0.2, 0.55, 0.42),
            role="region",
            description="Upcoming appointment with date, time and doctor",
            salience=0.75,
        ),
        Element(
            "appt_text",
            "text",
            "Tue 18 Nov 2026, 09:30\nDr. Weber  -  General practice\nRoom 2.14",
            (0.08, 0.26, 0.5, 0.4),
            is_aoi=False,
            fg="#1C2A39",
            font_size=21,
        ),
        Element(
            "reschedule",
            "button",
            "Reschedule",
            (0.06, 0.45, 0.2, 0.51),
            role="button",
            description="Button to move the appointment to another time",
            salience=0.6,
            fill="#0F4C81",
            fg="#FFFFFF",
        ),
        Element(
            "results",
            "card",
            "Test results",
            (0.6, 0.2, 0.94, 0.42),
            role="region",
            description="Latest lab test results",
            salience=0.5,
        ),
        Element(
            "results_text",
            "text",
            "Blood panel  -  ready\nVitamin D  -  pending",
            (0.62, 0.26, 0.92, 0.38),
            is_aoi=False,
            fg="#1C2A39",
        ),
        Element(
            "save_contact",
            "button_outline",
            "Save emergency contact",
            (0.6, 0.55, 0.85, 0.61),
            role="button",
            description="Button that saves the emergency contact details",
            salience=0.5,
            fg="#0F4C81",
        ),
        Element(
            "save_confirmation",
            "text",
            "Saved. Your emergency contact was updated.",
            (0.6, 0.63, 0.94, 0.67),
            role="text",
            description="Confirmation message that the contact was saved",
            salience=0.55,
            fg="#1E7B3A",
            font_size=17,
        ),
        *_footer("Help    Accessibility    Privacy"),
    ),
    tasks=(
        TaskSpec(
            "find_appointment",
            "Find the time of your next appointment.",
            ("next_appointment",),
            "no_friction",
            0.12,
        ),
        TaskSpec(
            "confirm_saved",
            "Check whether your emergency contact was saved.",
            ("save_confirmation",),
            "no_friction",
            0.15,
        ),
    ),
)

HEALTH_FRICTION = UISpec(
    ui_id="health_portal_friction",
    title="Patient portal (friction)",
    domain="healthcare",
    condition="friction",
    pair_id="health_portal",
    background="#FFFFFF",
    notes="Appointment hidden in a generic 'Activity' feed; save confirmation shown far from the "
    "button in faint text.",
    elements=(
        *_header("CareLink", "#0F4C81", "Messages    Prescriptions    Profile"),
        Element(
            "title",
            "heading",
            "Good morning, Alex",
            (0.06, 0.105, 0.5, 0.165),
            role="heading",
            description="Greeting",
            salience=0.45,
            font_size=32,
        ),
        Element(
            "news",
            "banner",
            "Flu season: book your vaccination now",
            (0.06, 0.2, 0.94, 0.3),
            role="banner",
            description="Large health campaign banner",
            salience=0.9,
            fill="#9AD1F5",
            fg="#0B2C47",
            font_size=26,
        ),
        Element(
            "activity",
            "dense_text",
            "Activity\n03 Nov  Prescription renewed (Ibuprofen 400 mg)\n29 Oct  Message from Dr. Weber\n"
            "27 Oct  Appointment scheduled: Tue 18 Nov 2026, 09:30, Dr. Weber, Room 2.14\n"
            "20 Oct  Lab results available: blood panel\n12 Oct  Invoice paid  EUR 25.00",
            (0.06, 0.34, 0.55, 0.6),
            role="text",
            description="Chronological activity feed that also contains the next appointment",
            salience=0.3,
            fg="#56606B",
            font_size=14,
        ),
        Element(
            "results",
            "card",
            "Test results",
            (0.6, 0.34, 0.94, 0.55),
            role="region",
            description="Latest lab test results",
            salience=0.5,
        ),
        Element(
            "save_contact",
            "button_outline",
            "Save emergency contact",
            (0.6, 0.6, 0.85, 0.66),
            role="button",
            description="Button that saves the emergency contact details",
            salience=0.5,
            fg="#0F4C81",
        ),
        Element(
            "save_confirmation",
            "text",
            "changes stored",
            (0.06, 0.88, 0.2, 0.91),
            role="text",
            description="Faint status text that the contact was saved",
            salience=0.06,
            fg="#C3C8CE",
            font_size=13,
        ),
        *_footer("Help    Accessibility    Privacy"),
    ),
    tasks=(
        TaskSpec(
            "find_appointment",
            "Find the time of your next appointment.",
            ("activity",),
            "information_overload",
            0.7,
        ),
        TaskSpec(
            "confirm_saved",
            "Check whether your emergency contact was saved.",
            ("save_confirmation",),
            "feedback_uncertainty",
            0.75,
        ),
    ),
)

SUBSCRIPTION_CLEAN = UISpec(
    ui_id="subscription_clean",
    title="Subscription settings (clean)",
    domain="subscription",
    condition="clean",
    pair_id="subscription",
    elements=(
        *_header("StreamBox", "#141414", "Home    Series    Films    Account"),
        Element(
            "title",
            "heading",
            "Your plan",
            (0.06, 0.105, 0.5, 0.165),
            role="heading",
            description="Page title",
            salience=0.5,
            font_size=34,
        ),
        Element(
            "plan_card",
            "card",
            "Premium  -  EUR 17.99 / month",
            (0.06, 0.2, 0.6, 0.42),
            role="region",
            description="Current subscription plan and price",
            salience=0.6,
        ),
        Element(
            "change_plan",
            "button",
            "Change plan",
            (0.08, 0.33, 0.22, 0.39),
            role="button",
            description="Button to switch to a different subscription plan",
            salience=0.65,
            fill="#E50914",
            fg="#FFFFFF",
        ),
        Element(
            "cancel_plan",
            "button_outline",
            "Cancel subscription",
            (0.24, 0.33, 0.42, 0.39),
            role="button",
            description="Button to cancel the subscription",
            salience=0.55,
            fg="#141414",
        ),
        Element(
            "billing",
            "card",
            "Billing",
            (0.06, 0.46, 0.6, 0.64),
            role="region",
            description="Next payment date and payment method",
            salience=0.4,
        ),
        *_footer("Help    Privacy    Terms"),
    ),
    tasks=(
        TaskSpec("cancel_plan", "Cancel your subscription.", ("cancel_plan",), "no_friction", 0.15),
    ),
)

SUBSCRIPTION_FRICTION = UISpec(
    ui_id="subscription_friction",
    title="Subscription settings (friction)",
    domain="subscription",
    condition="friction",
    pair_id="subscription",
    notes="Cancellation hidden behind 'Manage membership' among similar-looking links.",
    elements=(
        *_header("StreamBox", "#141414", "Home    Series    Films    Account"),
        Element(
            "title",
            "heading",
            "Your plan",
            (0.06, 0.105, 0.5, 0.165),
            role="heading",
            description="Page title",
            salience=0.5,
            font_size=34,
        ),
        Element(
            "plan_card",
            "card",
            "Premium  -  EUR 17.99 / month",
            (0.06, 0.2, 0.6, 0.42),
            role="region",
            description="Current subscription plan and price",
            salience=0.6,
        ),
        Element(
            "upgrade",
            "button",
            "Upgrade to Ultra 4K",
            (0.08, 0.33, 0.28, 0.39),
            role="button",
            description="Button to upgrade to a more expensive plan",
            salience=0.85,
            fill="#E50914",
            fg="#FFFFFF",
        ),
        Element(
            "account_links",
            "text",
            "Manage profiles\nManage devices\nManage membership\nManage downloads",
            (0.66, 0.2, 0.9, 0.42),
            role="navigation",
            description="List of similar 'Manage ...' links; 'Manage membership' leads to "
            "cancellation",
            salience=0.3,
            fg="#5E6670",
            font_size=17,
        ),
        Element(
            "billing",
            "card",
            "Billing",
            (0.06, 0.46, 0.6, 0.64),
            role="region",
            description="Next payment date and payment method",
            salience=0.4,
        ),
        *_footer("Help    Privacy    Terms"),
    ),
    tasks=(
        TaskSpec(
            "cancel_plan", "Cancel your subscription.", ("account_links",), "navigation", 0.78
        ),
    ),
)

#: The five interfaces used by the synthetic demo dataset.
SYNTHETIC_UIS: tuple[UISpec, ...] = (SHOP_CLEAN, SHOP_FRICTION, BANK_CLEAN, BANK_FRICTION, TRAVEL)

#: Interfaces available as study stimuli (the demo five plus healthcare and subscription).
STUDY_UIS: tuple[UISpec, ...] = (
    *SYNTHETIC_UIS,
    HEALTH_CLEAN,
    HEALTH_FRICTION,
    SUBSCRIPTION_CLEAN,
    SUBSCRIPTION_FRICTION,
)


def get_spec(ui_id: str) -> UISpec:
    for spec in STUDY_UIS:
        if spec.ui_id == ui_id:
            return spec
    raise KeyError(ui_id)


def all_specs() -> dict[str, UISpec]:
    return {s.ui_id: s for s in STUDY_UIS}
