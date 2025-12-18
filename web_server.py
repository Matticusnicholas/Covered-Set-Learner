"""
Web-Based Visualization Server
==============================
Beautiful browser-based interface for streaming the neural network training.
Uses Flask + SocketIO for real-time updates.
"""

import os
import json
import threading
import time
from queue import Queue, Empty
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from typing import Dict, List, Tuple, Optional

# Create Flask app
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = 'lottery-neural-network-secret'

# Use threading mode but with a message queue for updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Message queue for thread-safe updates
update_queue = Queue()

# Global state for the training
training_state = {
    'running': False,
    'generation': 0,
    'coverage': 0.0,
    'best_coverage': 0.0,
    'num_tickets': 0,
    'best_num_tickets': 0,
    'tickets': [],
    'heat_map': {},
    'current_ticket': [],
    'ghost_tickets': None,
    'ghost_coverage': None,
    'is_new_record': False,
    'pool_size': 36,
    'draw_size': 5,
    'match_required': 3,
    'wheel_name': '3 of 5 from 36',
    'theoretical_min': 0,
    'elapsed_time': 0,
    'temperature': 1.0
}


class WebVisualizer:
    """Web-based visualizer that sends updates via WebSocket."""

    def __init__(self, pool_size: int = 36, draw_size: int = 5, match_required: int = 3):
        self.pool_size = pool_size
        self.draw_size = draw_size
        self.match_required = match_required
        self.running = True

        # Update global state
        training_state['pool_size'] = pool_size
        training_state['draw_size'] = draw_size
        training_state['match_required'] = match_required
        training_state['wheel_name'] = f"{match_required} of {draw_size} from {pool_size}"
        training_state['running'] = True

    def set_ghost(self, ghost_tickets: int, ghost_coverage: float):
        """Set the ghost record to beat."""
        training_state['ghost_tickets'] = ghost_tickets
        training_state['ghost_coverage'] = ghost_coverage
        self._emit_update()

    def set_theoretical_min(self, min_tickets: int):
        """Set the theoretical minimum tickets."""
        training_state['theoretical_min'] = min_tickets
        self._emit_update()

    def update(self, dt: float, stats: Dict = None, current_ticket: Tuple[int, ...] = None,
               heat_map: Dict[int, float] = None):
        """Update visualization with new data."""
        if stats:
            training_state['generation'] = stats.get('generation', 0)
            training_state['coverage'] = stats.get('coverage', 0)
            training_state['best_coverage'] = stats.get('best_coverage', 0)
            training_state['num_tickets'] = stats.get('num_tickets', 0)
            training_state['efficiency'] = stats.get('efficiency', 0)
            training_state['temperature'] = stats.get('temperature', 1.0)
            training_state['elapsed_time'] = stats.get('elapsed_time', 0)

            # Track best number of tickets (fewest to reach 100%)
            if 'best_num_tickets' in stats:
                training_state['best_num_tickets'] = stats.get('best_num_tickets', 0)

            # Check for new record - at 100% coverage with fewer tickets than ghost
            current_coverage = stats.get('coverage', 0)
            current_tickets = stats.get('num_tickets', 0)
            if current_coverage >= 99.999:
                # Track session record (fewest tickets to 100%)
                if training_state['best_num_tickets'] == 0 or current_tickets < training_state['best_num_tickets']:
                    training_state['best_num_tickets'] = current_tickets
                    print(f"🏆 NEW SESSION RECORD: {current_tickets} tickets for 100% coverage!")

            # Check if we beat the ghost
            if training_state['ghost_tickets'] and current_coverage >= 99.999:
                training_state['is_new_record'] = (current_tickets <= training_state['ghost_tickets'])

        if current_ticket:
            training_state['current_ticket'] = list(current_ticket)

        if heat_map:
            training_state['heat_map'] = heat_map

        self._emit_update()

    def add_ticket(self, ticket: Tuple[int, ...]):
        """Add a new ticket."""
        training_state['tickets'].append(list(ticket))
        # Keep only last 50 tickets for display
        if len(training_state['tickets']) > 50:
            training_state['tickets'] = training_state['tickets'][-50:]
        self._emit_update()

    def new_generation(self):
        """Start a new generation - clears ticket list."""
        training_state['tickets'] = []
        training_state['current_ticket'] = []
        training_state['coverage'] = 0
        training_state['num_tickets'] = 0
        self._emit_update()
        print(f"🔄 Generation reset - tickets cleared")

    def set_best_tickets(self, tickets: List[Tuple[int, ...]]):
        """Set the best ticket set found."""
        training_state['best_tickets'] = [list(t) for t in tickets]
        training_state['best_num_tickets'] = len(tickets)
        self._emit_update()

    def _emit_update(self):
        """Emit update to all connected clients."""
        # Make a copy of the state to avoid threading issues
        state_copy = dict(training_state)
        state_copy['tickets'] = list(training_state['tickets'])
        state_copy['current_ticket'] = list(training_state['current_ticket'])
        state_copy['heat_map'] = dict(training_state['heat_map'])

        # Emit directly - Flask-SocketIO handles cross-thread emit
        try:
            socketio.emit('training_update', state_copy, namespace='/')
        except Exception as e:
            print(f"  ⚠️ Emit error: {e}")

    def run_frame(self) -> float:
        """Compatibility method - returns a small dt."""
        socketio.sleep(0.05)  # 20 FPS update rate
        return 0.05

    def cleanup(self):
        """Mark training as complete."""
        training_state['running'] = False
        self._emit_update()


# Flask routes
@app.route('/')
def index():
    """Serve the main visualization page."""
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current training state."""
    return jsonify(training_state)


@app.route('/api/records')
def get_records():
    """Get hall of fame records."""
    try:
        from records import RecordsManager
        manager = RecordsManager()
        return jsonify(manager.get_all_records())
    except:
        return jsonify([])


# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"🌐 Client connected")
    emit('training_update', training_state)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"🌐 Client disconnected")


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the web server."""
    print(f"\n🌐 Starting web visualization server...")
    print(f"   Open http://localhost:{port} in your browser")
    print(f"   Press Ctrl+C to stop\n")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


def run_server_background(host='0.0.0.0', port=5000):
    """Run the web server in a background thread."""
    server_thread = threading.Thread(
        target=lambda: socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True),
        daemon=True
    )
    server_thread.start()
    print(f"\n🌐 Web visualization running at http://localhost:{port}")
    print(f"   Open this URL in your browser to watch training!\n")
    time.sleep(1)  # Give server time to start
    return server_thread


if __name__ == '__main__':
    run_server(debug=True)
