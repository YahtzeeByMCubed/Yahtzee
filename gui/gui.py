"""PySide6 dashboard — port of the Pygame concept.

Layout (top-level QMainWindow):
    - Left panel: Yahtzee card background and scores.
    - Right panel: Dice state and controls.

To test standalone:
    python -m src.gui.gui
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from environment.constants import (
    SCORE_OFFSET,
    MAX_ROLLS_PER_TURN,
    CATEGORY_NAMES,
)
from environment.yahtzee_env import IllegalActionError, GameOverError

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel
)
from PyQt6.QtGui import QPainter, QPixmap, QFontDatabase, QFont, QColor
from PyQt6.QtCore import Qt, QRectF

# Ensure project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.misc import generate_random_yahtzee_game
from gui.dice import Dice
from environment.yahtzee_env import YahtzeeEnv

def get_centers(start, end, count):
    step = (end - start) / count
    return [start + i * step + step / 2 for i in range(count)]


class ScoreQChart(QWidget):
    """Custom widget to draw Q-value bars left of the scorecard."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(400)
        self.q_values = None
        self.legal_mask = None
        self.is_human_turn = False
        self.is_viewing_human = False
        
        self.y_centers = (
            get_centers(140, 405, 6) +
            get_centers(410, 518, 3) +
            get_centers(552, 811, 7) +
            get_centers(840, 960, 4)
        )
        self.score_row_indices = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15]
        self.custom_font = QFont("Arial", 10)

        self.on_score_clicked = None

        # Scorecard row index -> Yahtzee category index
        self.score_row_to_category = {
            0: 0,    # Ones
            1: 1,    # Twos
            2: 2,    # Threes
            3: 3,    # Fours
            4: 4,    # Fives
            5: 5,    # Sixes
            9: 6,    # Three of a Kind
            10: 7,   # Four of a Kind
            11: 8,   # Full House
            12: 9,   # Small Straight
            13: 10,  # Large Straight
            14: 11,  # Yahtzee
            15: 12,  # Chance
        }

    def update_data(self, q_values, legal_mask, is_human_turn):
        self.q_values = q_values
        self.legal_mask = legal_mask
        self.is_human_turn = is_human_turn
        self.update()

    def paintEvent(self, event):
        if self.is_human_turn or self.is_viewing_human or self.q_values is None or self.legal_mask is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.custom_font)
        
        scoring_qs = self.q_values[32:45]
        valid_qs = [v for v in scoring_qs if v != float('-inf') and v != float('inf') and not np.isinf(v)]
        max_q = max(valid_qs + [0.1])
        
        legal_scoring_qs = [scoring_qs[j] for j in range(13) if self.legal_mask[32 + j] == 0.0]
        max_legal_q = max(legal_scoring_qs) if legal_scoring_qs else float('-inf')
        
        for i, row_idx in enumerate(self.score_row_indices):
            q_val = scoring_qs[i]
            is_legal = self.legal_mask[32 + i] == 0.0
            
            if is_legal:
                if q_val == max_legal_q and max_legal_q != float('-inf'):
                    painter.setBrush(QColor("#228B22")) # Dark green
                else:
                    painter.setBrush(QColor("#bbf7a3")) # Green
            else:
                painter.setBrush(QColor("#d3d3d3")) # Gray
                
            painter.setPen(QColor("black"))
            
            y = self.y_centers[row_idx]
            bar_height = 20
            
            display_val = max(0, q_val) if not np.isinf(q_val) else 0
            bar_width = (display_val / max_q) * 300 if max_q > 0 else 0
            
            x = 390 - bar_width
            
            rect = QRectF(x, y - bar_height/2, bar_width, bar_height)
            painter.drawRect(rect)
            
            painter.setPen(QColor("black"))
            text_val = f"{q_val:.2f}" if not np.isinf(q_val) else "-inf"
            painter.drawText(QRectF(x - 55, y - bar_height/2, 50, bar_height), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text_val)


class YahtzeeBoard(QWidget):
    """Custom widget to draw the Yahtzee scorecard and game state."""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        # Load assets
        base_path = Path(__file__).resolve().parent
        image_path = base_path / "assets" / "yahtzee_card.png"
        font_path = base_path / "assets" / "KOMIKAHB.ttf"
        
        self.background = QPixmap(str(image_path))
        self.setFixedSize(self.background.size())
        
        # Load custom font
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            # Size 18 roughly matches Pygame's 24pt depending on DPI
            self.custom_font = QFont(font_family, 18)
        else:
            self.custom_font = QFont("Arial", 18)
            
        self.games = []
        self.on_mouse_move = None
        self.setMouseTracking(True)
        
        x_offset = 0
        self.y_centers = (
            get_centers(140, 405, 6) +
            get_centers(410, 518, 3) +
            get_centers(552, 811, 7) +
            get_centers(840, 960, 4)
        )
        self.x_centers = get_centers(x_offset + 270, x_offset + 578, 5)
        self.is_viewing_human = False

        self.on_score_clicked = None

        # Scorecard row index -> Yahtzee category index
        self.score_row_to_category = {
            0: 0,    # Ones
            1: 1,    # Twos
            2: 2,    # Threes
            3: 3,    # Fours
            4: 4,    # Fives
            5: 5,    # Sixes
            9: 6,    # Three of a Kind
            10: 7,   # Four of a Kind
            11: 8,   # Full House
            12: 9,   # Small Straight
            13: 10,  # Large Straight
            14: 11,  # Yahtzee
            15: 12,  # Chance
        }

    def set_games(self, games) -> None:
        self.games = games
        self.update()

    def mouseMoveEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        if self.on_mouse_move:
            self.on_mouse_move(x, y)

    def _category_from_click(self, x, y):
        """
        Convert a scorecard click position into a Yahtzee category index.
        Returns category index 0-12, or None.
        """

        # Only allow clicking in the first visible score column.
        active_score_col_x = self.x_centers[0]

        if abs(x - active_score_col_x) > 45:
            return None

        closest_row = None
        closest_distance = float("inf")

        for row_idx in self.score_row_to_category:
            distance = abs(y - self.y_centers[row_idx])

            if distance < closest_distance:
                closest_distance = distance
                closest_row = row_idx

        if closest_distance > 22:
            return None

        return self.score_row_to_category[closest_row]


    def mousePressEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()

        category_idx = self._category_from_click(x, y)

        print(f"Clicked board at x={int(x)}, y={int(y)}, category={category_idx}")

        if category_idx is not None and self.on_score_clicked is not None:
            self.on_score_clicked(category_idx)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.drawPixmap(0, 0, self.background)
        
        # Draw text
        painter.setFont(self.custom_font)
        painter.setPen(QColor("black"))
        
        # Draw AI/Human indicator
        player_text = "AI" if not self.is_viewing_human else "Human"
        painter.drawText(QRectF(303, 22, 100, 50), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, player_text)
        
        # Draw games
        for col_idx, game in enumerate(self.games):
            if col_idx >= len(self.x_centers):
                break
            cell_center_x = self.x_centers[col_idx]
            
            attributes = [
                game.ones, game.twos, game.threes, game.fours, game.fives, game.sixes,
                game.upper_section_total, game.upper_section_bonus, game.upper_section_total_with_bonus,
                game.three_of_a_kind, game.four_of_a_kind, game.full_house,
                game.small_straight, game.large_straight, game.yahtzee, game.chance,
                game.yahtzee_bonus, game.lower_section_total, game.upper_section_total_with_bonus, game.grand_total
            ]
            
            for row_idx, val in enumerate(attributes):
                if val is not None:
                    text = str(val)
                    rect = QRectF(cell_center_x - 50, self.y_centers[row_idx] - 25, 100, 50)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class DiceButton(QPushButton):
    """A button representing a single die in the GUI."""
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFixedSize(80, 80)
        self.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.is_kept = False
        self.update_style()

    def update_style(self):
        bg_color = "#bbf7a3" if self.is_kept else "white"
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: black;
                border: 2px solid black;
                border-radius: 5px;
            }}
        """)


class DicePanel(QWidget):
    """Right-side panel for displaying and interacting with dice."""
    def __init__(self, dice_manager, parent=None, on_reroll_clicked=None):
        super().__init__(parent)
        self.dice_manager = dice_manager
        self.setFixedWidth(300)
        self.on_reroll_clicked = on_reroll_clicked
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Roll Count Label
        self.roll_label = QLabel("Dice roll: 1")
        self.roll_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(self.roll_label)
        
        layout.addSpacing(10)
        
        # Main dice grid (3 on top, 2 below)
        self.dice_buttons = []
        for i in range(5):
            btn = DiceButton(i)
            btn.clicked.connect(lambda checked, idx=i: self.toggle_die(idx))
            self.dice_buttons.append(btn)
        
        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for i in range(3):
            top_row.addWidget(self.dice_buttons[i])
        
        bottom_row = QHBoxLayout()
        bottom_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_row.addWidget(self.dice_buttons[3])
        bottom_row.addWidget(self.dice_buttons[4])
        
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)
        
        layout.addSpacing(30)
        
        # Kept Dice Section
        self.kept_label = QLabel("Kept dice:")
        self.kept_label.setFont(QFont("Arial", 14))
        layout.addWidget(self.kept_label)
        
        self.hint_label = QLabel("Click dice to keep between rolls")
        self.hint_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.hint_label)
        
        layout.addSpacing(10)
        
        self.kept_row = QHBoxLayout()
        self.kept_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.kept_widgets = []
        
        for i in range(5):
            lbl = QLabel("")
            lbl.setFixedSize(50, 50)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            lbl.setStyleSheet("background-color: white; color: black; border: 1px solid black;")
            lbl.hide()
            self.kept_row.addWidget(lbl)
            self.kept_widgets.append(lbl)
            
        layout.addLayout(self.kept_row)
        
        # Add spacing instead of a full stretch to keep the button closer
        layout.addSpacing(20)
        
        self.action_button = QPushButton("Reroll")
        self.action_button.setFixedHeight(50)
        self.action_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.action_button.clicked.connect(self.on_action_clicked)
        layout.addWidget(self.action_button)
        
        # Add the instructional text below the button
        self.sub_action_label = QLabel("or select a score to fill in")
        self.sub_action_label.setFont(QFont("Arial", 10))
        self.sub_action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sub_action_label)
        
        layout.addSpacing(20)
        
        self.toggle_view_button = QPushButton("View Human Scorecard")
        self.toggle_view_button.setFixedHeight(40)
        self.toggle_view_button.setFont(QFont("Arial", 12))
        layout.addWidget(self.toggle_view_button)
        
        # Push everything up to the top
        layout.addStretch()
        
        # Add mouse coordinates label at the bottom right
        self.mouse_label = QLabel("X: 0, Y: 0")
        self.mouse_label.setFont(QFont("Arial", 10))
        self.mouse_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.mouse_label)
        
        self.sync_ui()

    def update_mouse_coords(self, x, y):
        self.mouse_label.setText(f"X: {int(x)}, Y: {int(y)}")

    def toggle_die(self, index):
        if self.dice_manager.roll_count < MAX_ROLLS_PER_TURN:
            self.dice_manager.toggle_keep(index)
            self.sync_ui()

    def on_action_clicked(self):
        if self.dice_manager.roll_count < MAX_ROLLS_PER_TURN:
            if self.on_reroll_clicked is not None:
                self.on_reroll_clicked()
            else:
                self.dice_manager.reroll()
                self.sync_ui()
        else:
            print("Action: Select a cell on the sheet to place a score")

    def sync_ui(self):
        self.roll_label.setText(f"Dice roll: {self.dice_manager.roll_count}")
        
        for i, val in enumerate(self.dice_manager.get_dice()):
            self.dice_buttons[i].setText(str(val))
            self.dice_buttons[i].is_kept = self.dice_manager.keep_mask[i]
            self.dice_buttons[i].update_style()
            
        kept_vals = self.dice_manager.get_kept_dice()
        for i in range(5):
            if i < len(kept_vals):
                self.kept_widgets[i].setText(str(kept_vals[i]))
                self.kept_widgets[i].show()
            else:
                self.kept_widgets[i].hide()
                        
        if self.dice_manager.roll_count >= MAX_ROLLS_PER_TURN:
            self.action_button.setText("No rerolls left")
            self.sub_action_label.setText("select an open score category")
            self.sub_action_label.show()
        else:
            self.action_button.setText("Reroll")
            self.sub_action_label.setText("or select an open score category now")
            self.sub_action_label.show()

class Dashboard(QMainWindow):
    """Top-level QMainWindow that holds our YahtzeeBoard and DicePanel."""

    def __init__(self, env=None, dice=None, robot_client=None) -> None:
        super().__init__()
        self.setWindowTitle("Yahtzee AI Dashboard")
        
        self.is_human_turn = False
        self.game_counter = 0

        self.env = env
        self.last_q_values = None
        self.last_legal_mask = None
        
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot for Q values replaces addStretch()
        self.q_chart = ScoreQChart(self)
        layout.addWidget(self.q_chart)
        
        self.board = YahtzeeBoard(self)
        layout.addWidget(self.board)
        
        if self.env is not None:
            dice = self.env.dice_manager
        elif dice is None:
            dice = Dice()
            dice.roll()
            
        self.dice_panel = DicePanel(
            dice_manager=dice,
            parent=self,
            on_reroll_clicked=self.handle_reroll_clicked if self.env is not None else None,
        )
        layout.addWidget(self.dice_panel)
        if self.env is not None:
            self.board.set_games([self.env.scorecard])
        
        self.dice_panel.toggle_view_button.clicked.connect(self.toggle_scorecard_view)
        
        # Wire up mouse tracking from the board to the panel
        self.board.on_mouse_move = self.dice_panel.update_mouse_coords
        self.board.on_score_clicked = self.handle_score_clicked

        if self.env is not None:
            self.board.set_games([self.env.scorecard])
        
        self.setCentralWidget(main_widget)
        
        # Lock window size to board width + panel width + extra space for new feature
        self.setFixedSize(self.board.width() + self.dice_panel.width() + 420, self.board.height())

    def toggle_scorecard_view(self):
        self.board.is_viewing_human = not self.board.is_viewing_human
        if self.board.is_viewing_human:
            self.dice_panel.toggle_view_button.setText("View AI Scorecard")
        else:
            self.dice_panel.toggle_view_button.setText("View Human Scorecard")
        self.q_chart.is_viewing_human = self.board.is_viewing_human
        self.q_chart.update()
        self.board.update()

    def refresh_ui(self, state_vector=None, q_values=None, legal_mask=None) -> None:
        """
        Refresh the GUI using the real environment state.

        Updates:
        - Q-value chart
        - scorecard display
        - dice panel
        """

        if q_values is not None:
            self.last_q_values = np.asarray(q_values, dtype=np.float32)

        if legal_mask is not None:
            self.last_legal_mask = np.asarray(legal_mask, dtype=np.float32)

        if self.last_q_values is not None and self.last_legal_mask is not None:
            self.q_chart.update_data(
                self.last_q_values,
                self.last_legal_mask,
                self.is_human_turn,
            )

        if self.env is not None:
            self.board.set_games([self.env.scorecard])
            self.dice_panel.dice_manager = self.env.dice_manager
            self.dice_panel.sync_ui()

        self.board.update()

    def update_q(self, q_values, legal_mask=None) -> None:
        self.q_chart.update_data(q_values, legal_mask, self.is_human_turn)

    def show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        print(f"GUI Error: {message}")

    def _current_hold_action_from_gui_mask(self) -> int:
        """
        Convert the GUI keep_mask into a DQN hold action index.

        Example:
            [True, False, True, False, False] -> 5
        """

        action_idx = 0

        for bit_position, keep in enumerate(self.env.dice_manager.keep_mask):
            if keep:
                action_idx += 2 ** bit_position

        return action_idx


    def handle_reroll_clicked(self):
        """
        Called when the GUI reroll button is clicked.

        Instead of directly mutating dice, this sends a hold action through
        the real YahtzeeEnv.
        """

        if self.env is None:
            return

        try:
            action_idx = self._current_hold_action_from_gui_mask()

            state, reward, done = self.env.step(action_idx)

            self.refresh_ui(
                state_vector=state,
                legal_mask=self.env.get_legal_mask(),
            )

            self.statusBar().showMessage(
                f"Rerolled using hold action {action_idx}.",
                3000,
            )

        except (IllegalActionError, GameOverError, Exception) as exc:
            self.show_error(str(exc))


    def handle_score_clicked(self, category_idx: int):
        """
        Called when a scorecard row is clicked.

        Converts the clicked category into a DQN score action:
            action = 32 + category_idx

        GUI rules now match YahtzeeEnv:
            - Cannot score after game is complete
            - Cannot score while viewing human scorecard
            - Can score at roll 1, 2, or 3
            - Can score 0
            - Can only score open categories
            - Final legality is decided by env.get_legal_mask()
        """

        if self.env is None:
            self.show_error("No environment connected to dashboard.")
            return

        if self.env.done:
            self.show_error("Game is already complete. Start/reset a new game.")
            return

        if self.board.is_viewing_human:
            self.show_error("Switch to AI Scorecard before filling AI scores.")
            return

        category_idx = int(category_idx)
        category_name = CATEGORY_NAMES[category_idx]

        action_idx = SCORE_OFFSET + category_idx
        legal_mask = self.env.get_legal_mask()

        # This is the important part:
        # Let the environment's legal mask define the rule.
        # Do NOT block scoring before roll 3.
        if not np.isfinite(legal_mask[action_idx]) or legal_mask[action_idx] != 0.0:
            self.show_error(
                f"{category_name} is not a legal score choice right now."
            )
            return

        current_dice = self.env.dice_manager.get_dice().astype(int).tolist()
        score_value = self.env.scorecard.calculate_score(category_idx, current_dice)

        try:
            state, reward, done = self.env.step(action_idx)

            if done:
                self.statusBar().showMessage(
                    f"Scored {category_name} = {score_value} using dice {current_dice}. "
                    f"Game complete. Final score: {self.env.scorecard.total()} | Reward: {reward:.2f}",
                    8000,
                )
            else:
                self.statusBar().showMessage(
                    f"Scored {category_name} = {score_value} using dice {current_dice}. "
                    f"Now Turn {self.env.turn}, Roll {self.env.current_roll}.",
                    5000,
                )

            self.refresh_ui(
                state_vector=state,
                legal_mask=self.env.get_legal_mask(),
            )

        except (IllegalActionError, GameOverError, Exception) as exc:
            self.show_error(str(exc))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QWidget { color: black; }
        QLabel { color: black; }
        QPushButton { color: black; }
    """)

    from environment.yahtzee_env import YahtzeeEnv

    env = YahtzeeEnv(seed=42)
    state = env.reset()

    window = Dashboard(env=env)

    mock_q = np.random.rand(45).astype(np.float32) * 3.5
    legal_mask = env.get_legal_mask()

    window.refresh_ui(
        state_vector=state,
        q_values=mock_q,
        legal_mask=legal_mask,
    )

    window.show()
    sys.exit(app.exec())

     