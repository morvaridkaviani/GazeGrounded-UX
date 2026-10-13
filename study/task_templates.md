# Tasks and interfaces

Generated from `src/gazegrounded/data/ui_specs.py`, which also defines the AOIs used for
scoring. Stimuli: `study/webgazer/stimuli/`. Edit the specs, then run
`python -m gazegrounded.cli build-study`. Do not edit `tasks.json` by hand.

| Domain | Interface (clean / friction) | Task | Friction manipulation | Designed category |
|---|---|---|---|---|
| E-commerce | `shop_checkout_clean` / `_friction` | Change the delivery address for this order. | “Edit address” button → low-contrast footer link “Shipping preferences” | findability |
| E-commerce | same | Apply the discount code SAVE10 to your order. | visible field → small grey link “Have a promo?” | ambiguous wording |
| Banking | `bank_cards_clean` / `_friction` | You have lost your debit card. Freeze it. | labelled button → unlabelled padlock icon among icons | ambiguous wording |
| Banking | same | Find the card's monthly spending limit. | dedicated panel → buried in dense small print; upsell banner | information overload |
| Healthcare | `health_portal_clean` / `_friction` | Find the time of your next appointment. | appointment card → line in an activity feed | information overload |
| Healthcare | same | Check whether your emergency contact was saved. | clear confirmation next to the button → faint text far away | feedback uncertainty |
| Subscription | `subscription_clean` / `_friction` | Cancel your subscription. | “Cancel subscription” button → “Manage membership” among similar links | navigation |
| Travel | `travel_booking` (single) | Change the date of your flight. | inside “Manage booking ▾” dropdown | navigation |
| Travel | same | Find out how much checked baggage is included. | small-print fare conditions | visual hierarchy |

**Counterbalancing.** Group A sees clean `bank`, `shop` and friction `health`, `subscription`;
group B the opposite. Both groups see `travel_booking`. 9 tasks per participant.

**Ethics of manipulations.** None of the manipulations is deceptive towards the participant:
all are common design patterns, the interfaces are fictitious, and no real account, money or
data is involved. Avoid manipulations that pressure or shame (e.g. confirm-shaming copy).

**Simplification.** Real tasks often need several steps. Here the participant clicks where they
would *start* the task on a static screenshot; success = clicking inside a target AOI.
