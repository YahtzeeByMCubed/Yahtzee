"""PyQt6 dashboard — design doc §2.3 and §6.

Layout (top-level QMainWindow):
    Left panel:
        - Live YOLOv8 feed with bounding-box overlay (real or sim image).
        - DiceReadout: "Detected dice: [3, 4, 4, 6, 1]" with kept dice
          rendered distinctly (per the engine's physical keep mask).
        - "Confirm Human Move" button (advances state machine from human turn).
        - Software E-Stop (cancels current robot action goal).
    Center panel:
        - Dual scorecard (human | AI), 13 rows + upper bonus + total.
    Right panel:
        - Q-value bar chart for the AI's last decision (45 bars).
        - Current roll indicator (1/2/3).

Refresh rate target: 60 Hz for state sync (§2.3 telemetry-accuracy req).

Namespaces (read this before touching the file):
    The GUI lives entirely in the *physical* namespace — the dice array as
    the camera sees it, left-to-right. It never sees the agent's 24-D state
    vector, the canonical (sorted) dice ordering used by the action-space
    bridge in YahtzeeStateMachine, or face counts. The engine exposes a
    physical keep mask (already translated from the canonical action mask),
    and the GUI just renders it.

    The Q-chart is the one place where *abstract* action indices appear
    (0..31 hold, 32..44 score). Their slot semantics are canonical, so
    the chart labels them opaquely ("Action 14") rather than as physical
    slots. The DiceReadout's kept/reroll overlay is the concrete view of
    what those decisions mean for the dice currently on the table.

State propagation:
    Logic Engine -> emits RenderState         -> Dashboard.refresh_ui(state)
    DQNAgent     -> emits q_values + chosen   -> Dashboard.display_q_chart(...)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RenderState:
    """Everything the GUI needs to redraw, in physical / human-facing terms.

    Built by the outer demo loop from YahtzeeStateMachine each tick. Keeps
    refresh_ui's signature stable as we add panels.
    """

    physical_dice: np.ndarray              # shape (5,), face values 1..6 in camera order
    roll_number: int                       # 1, 2, or 3
    keep_mask_physical: np.ndarray | None  # shape (5,) bool; None outside a hold step
    human_scorecard: object                # ScorecardManager (human side)
    ai_scorecard: object                   # ScorecardManager (AI side)


class Dashboard:
    """Top-level QMainWindow. Owns the three panels described above.

    Signals (Qt):
        human_move_confirmed   — emitted by the "Confirm Human Move" button.
        e_stop_requested       — emitted by the software E-Stop button.
                                 Wired to robot_client.cancel_goal() at init.
    The outer demo loop subscribes to human_move_confirmed; the GUI itself
    never advances the state machine.
    """

    def __init__(self, robot_client=None) -> None:
        # TODO:
        #   - Build QMainWindow with a 3-column layout (left/center/right).
        #   - Instantiate DiceReadout, ScorecardView, QChartView.
        #   - Declare pyqtSignals: human_move_confirmed, e_stop_requested.
        #   - Wire E-Stop QPushButton.clicked -> robot_client.cancel_goal()
        #     AND e_stop_requested.emit() (so observers can log/UI-react).
        #   - Wire "Confirm Human Move" QPushButton.clicked ->
        #     human_move_confirmed.emit().
        raise NotImplementedError

    def refresh_ui(self, state: RenderState) -> None:
        # TODO: 60 Hz refresh — fan `state` out to child widgets:
        #   - DiceReadout.update(state.physical_dice, state.keep_mask_physical)
        #   - ScorecardView.update_human(state.human_scorecard)
        #   - ScorecardView.update_ai(state.ai_scorecard)
        #   - roll-indicator label = f"Roll {state.roll_number}/3"
        #   - (when vision is wired) blit the latest camera frame.
        raise NotImplementedError

    def display_q_chart(
        self,
        q_values: np.ndarray,
        legal_mask: np.ndarray | None = None,
        chosen_idx: int | None = None,
    ) -> None:
        # TODO: forward to QChartView.update(q_values, legal_mask, chosen_idx).
        # q_values is 45-D in *abstract* action-index order (0..31 hold,
        # 32..44 score). chosen_idx highlights the AI's pick. legal_mask
        # uses 0.0 / -inf per get_legal_mask().
        raise NotImplementedError

    def show_error(self, message: str) -> None:
        # TODO: surface vision sanity-check failures (§4.2), robot faults,
        # and protective-stop events in a non-blocking status bar.
        raise NotImplementedError


class DiceReadout:
    """Left-panel widget: the 5 dice as the camera sees them, plus a
    kept/reroll overlay driven by the engine's physical keep mask.

    Operates entirely in the physical namespace — slot i in the readout
    corresponds to slot i in `physical_dice` and slot i in
    `keep_mask_physical`. No re-ordering happens here.
    """

    def __init__(self) -> None:
        # TODO: QWidget with a horizontal row of 5 die-face labels (or
        # QLabel-with-QPixmap if we render pip art). A border / fill color
        # toggles per slot to show kept vs. reroll.
        raise NotImplementedError

    def update(
        self,
        physical_dice: np.ndarray,
        keep_mask_physical: np.ndarray | None,
    ) -> None:
        # TODO: render the 5 face values; for each slot i, style "kept" if
        # keep_mask_physical is not None and keep_mask_physical[i], else
        # "reroll". When keep_mask_physical is None (e.g., just after a
        # score action or before any action this turn), render all dice
        # neutrally.
        raise NotImplementedError


class ScorecardView:
    """Dual scorecard widget — human on the left, AI on the right.

    Each row is one of the 13 categories, plus the upper-bonus and grand-
    total rows. Open cells render blank; locked cells show the committed
    score.
    """

    def __init__(self) -> None:
        # TODO: QTableWidget with 15 rows x 2 columns.
        raise NotImplementedError

    def update_human(self, scorecard_manager) -> None:
        # TODO: read scorecard_manager and write into the human column.
        raise NotImplementedError

    def update_ai(self, scorecard_manager) -> None:
        # TODO: read scorecard_manager and write into the AI column.
        raise NotImplementedError


class QChartView:
    """Bar chart of all 45 Q-values for the AI's current decision.

    Color convention:
        blue   indices 0..31  (hold combinations)
        green  indices 32..44 (score categories)
        grey   illegal indices (mask == -inf)
    The chosen bar (argmax over legal actions) is rendered solid; others
    are translucent.

    Action indices are *abstract*: hold-action slot semantics are canonical,
    not physical. The DiceReadout shows the physical consequence of the
    chosen action; this chart shows the decision itself.
    """

    def __init__(self) -> None:
        # TODO: subclass a PyQt6 chart widget — either QtCharts.QChartView
        # with QBarSeries, or pyqtgraph.PlotWidget if we add pyqtgraph as
        # a dep. PyQt6's built-in QtCharts is fine.
        raise NotImplementedError

    def update(
        self,
        q_values: np.ndarray,
        legal_mask: np.ndarray | None = None,
        chosen_idx: int | None = None,
    ) -> None:
        # TODO: plot q_values; grey out bars where legal_mask == -inf;
        # render bar `chosen_idx` solid, others translucent.
        raise NotImplementedError
