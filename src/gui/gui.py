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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel
)
from PyQt6.QtGui import QPainter, QPixmap, QFontDatabase, QFont, QColor
from PyQt6.QtCore import Qt, QRectF

# Ensure both the project root and old_pygame are in the path for standalone execution
project_root = Path(__file__).resolve().parent.parent.parent
old_pygame_dir = Path(__file__).resolve().parent / "old_pygame"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(old_pygame_dir))

from misc import generate_random_yahtzee_game
from src.gui.dice import Dice

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

    def set_games(self, games) -> None:
        self.games = games
        self.update()

    def mouseMoveEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        if self.on_mouse_move:
            self.on_mouse_move(x, y)

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
    def __init__(self, dice_manager, parent=None):
        super().__init__(parent)
        self.dice_manager = dice_manager
        self.setFixedWidth(300)
        
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
        if self.dice_manager.roll_count < 3:
            self.dice_manager.toggle_keep(index)
            self.sync_ui()

    def on_action_clicked(self):
        if self.dice_manager.roll_count < 3:
            self.dice_manager.reroll()
            self.sync_ui()
        else:
            print("Action: Select a cell on the sheet to place a score")

    def sync_ui(self):
        self.roll_label.setText(f"Dice roll: {self.dice_manager.roll_count}")
        
        for i, val in enumerate(self.dice_manager.values):
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
                
        if self.dice_manager.roll_count >= 3:
            self.action_button.setText("Select a cell on the sheet\nto place a score")
            self.sub_action_label.hide()
        else:
            self.action_button.setText("Reroll")
            self.sub_action_label.show()


class Dashboard(QMainWindow):
    """Top-level QMainWindow that holds our YahtzeeBoard and DicePanel."""

    def __init__(self, dice=None, robot_client=None) -> None:
        super().__init__()
        self.setWindowTitle("Yahtzee AI Dashboard")
        
        self.is_human_turn = False
        self.game_counter = 0
        
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot for Q values replaces addStretch()
        self.q_chart = ScoreQChart(self)
        layout.addWidget(self.q_chart)
        
        self.board = YahtzeeBoard(self)
        layout.addWidget(self.board)
        
        if dice is None:
            dice = Dice()
            dice.roll()
            
        self.dice_panel = DicePanel(dice_manager=dice, parent=self)
        layout.addWidget(self.dice_panel)
        
        self.dice_panel.toggle_view_button.clicked.connect(self.toggle_scorecard_view)
        
        # Wire up mouse tracking from the board to the panel
        self.board.on_mouse_move = self.dice_panel.update_mouse_coords
        
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

    def refresh_ui(self, state_vector, q_values, legal_mask) -> None:
        self.q_chart.update_data(q_values, legal_mask, self.is_human_turn)
        # TODO: Implement later for real engine state
        pass

    def update_q(self, q_values, legal_mask=None) -> None:
        self.q_chart.update_data(q_values, legal_mask, self.is_human_turn)

    def show_error(self, message: str) -> None:
        # TODO: Implement later
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Enforce a global light theme to prevent OS dark mode from overriding text colors
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QWidget { color: black; }
        QLabel { color: black; }
        QPushButton { color: black; }
    """)
    
    # Initialize the shared Dice state
    dice = Dice()
    dice.roll()
    
    # --- MOCK AI ACTION ---
    # Simulate the AI choosing to keep the 1st, 2nd, and 4th dice
    dice.keep_mask = [True, True, False, True, False]
    
    window = Dashboard(dice=dice)
    
    # Load 5 mock games for the standalone test
    mock_games = [generate_random_yahtzee_game() for _ in range(1)]
    window.board.set_games(mock_games)
    
    # Mock Q-values and legal mask for visual testing
    mock_q = np.random.rand(45) * 3.5
    mock_mask = np.zeros(45, dtype=np.float32)
    # Set some to illegal
    mock_mask[32] = -np.inf
    mock_mask[35] = -np.inf
    mock_mask[44] = -np.inf
    
    window.refresh_ui(None, mock_q, mock_mask)
    
    window.show()
    sys.exit(app.exec())