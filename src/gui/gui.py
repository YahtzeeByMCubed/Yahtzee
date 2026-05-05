"""PyQt6 dashboard — design doc §2.3 and §6.

Layout (top-level QMainWindow):
    Left panel:
        - Live YOLOv8 feed with bounding-box overlay (real or sim image).
        - "Detected dice: [3, 4, 4, 6, 1]" readout.
        - "Confirm Human Move" button (advances state machine from human turn).
        - Software E-Stop (cancels current robot action goal).
    Center panel:
        - Dual scorecard (human | AI), 13 rows + upper bonus + total.
    Right panel:
        - Q-value bar chart for the AI's last decision (45 bars).
        - Current roll indicator (1/2/3).

Refresh rate target: 60 Hz for state sync (§2.3 telemetry-accuracy req).

The GUI is a passive observer of the Logic Engine — it never mutates the
game state. State propagation:
    Logic Engine -> emits state_vector  -> Dashboard.refresh_ui(state)
    DQNAgent     -> emits q_values      -> Dashboard.display_q_chart(qs)
"""

from __future__ import annotations


class Dashboard:
    """Top-level QMainWindow. Owns the three panels described above."""

    def __init__(self, robot_client=None) -> None:
        # TODO:
        #   - Build QMainWindow with a 3-column layout (left/center/right).
        #   - Instantiate ScorecardView and QChartView (below).
        #   - Wire E-Stop QPushButton.clicked -> robot_client.cancel_goal().
        #   - Wire "Confirm Human Move" QPushButton.clicked -> emit a Qt
        #     signal that the outer demo loop subscribes to (so the GUI
        #     stays decoupled from the state machine).
        raise NotImplementedError

    def refresh_ui(self, state_vector) -> None:
        # TODO: 60 Hz refresh — update scorecard view, dice-face readout,
        # roll indicator, and (if vision is wired) the camera feed.
        raise NotImplementedError

    def display_q_chart(self, q_values, legal_mask=None) -> None:
        # TODO: bar chart of all 45 Q-values. Highlight the argmax bar.
        # If legal_mask is provided, render illegal-action bars greyed-out
        # so the user can visually confirm the agent is correctly ignoring
        # illegal moves.
        raise NotImplementedError

    def show_error(self, message: str) -> None:
        # TODO: surface vision sanity-check failures (§4.2), robot faults,
        # and protective-stop events in a non-blocking status bar.
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
    The argmax bar is rendered solid; others are translucent.
    """

    def __init__(self) -> None:
        # TODO: subclass a PyQt6 chart widget — either QtCharts.QChartView
        # with QBarSeries, or pyqtgraph.PlotWidget if we add pyqtgraph as
        # a dep. PyQt6's built-in QtCharts is fine.
        raise NotImplementedError

    def update(self, q_values, legal_mask) -> None:
        # TODO: plot q_values; grey out bars where legal_mask == -inf.
        raise NotImplementedError
