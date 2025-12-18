"""
Records & Persistence System
=============================
Saves and loads best covered sets, tracking records like a time trial ghost.
"""

import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class RecordsManager:
    """
    Manages saving/loading of best covered sets and records.
    Acts like a 'ghost time trial' - always trying to beat the best.
    """

    def __init__(self, save_dir: str = "records"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.records_file = self.save_dir / "hall_of_fame.json"
        self.records = self._load_records()

    def _load_records(self) -> Dict:
        """Load all records from file."""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_records(self):
        """Save all records to file."""
        with open(self.records_file, 'w') as f:
            json.dump(self.records, f, indent=2)

    def get_wheel_key(self, pool: int, draw: int, match: int) -> str:
        """Generate unique key for a wheel configuration."""
        return f"{match}of{draw}from{pool}"

    def get_record(self, pool: int, draw: int, match: int) -> Optional[Dict]:
        """Get the current record for a wheel configuration."""
        key = self.get_wheel_key(pool, draw, match)
        return self.records.get(key)

    def check_new_record(self, pool: int, draw: int, match: int,
                         coverage: float, num_tickets: int) -> bool:
        """
        Check if this is a new record.
        A new record is set if:
        - No previous record exists
        - Same coverage with fewer tickets
        - Higher coverage
        """
        current = self.get_record(pool, draw, match)

        if current is None:
            return True

        # Better coverage always wins
        if coverage > current['coverage']:
            return True

        # Same coverage but fewer tickets
        if coverage >= current['coverage'] and num_tickets < current['num_tickets']:
            return True

        return False

    def save_record(self, pool: int, draw: int, match: int,
                   coverage: float, tickets: List[Tuple[int, ...]],
                   generations: int, time_seconds: float) -> Dict:
        """
        Save a new record.

        Returns the record data.
        """
        key = self.get_wheel_key(pool, draw, match)

        # Get previous record for comparison
        previous = self.records.get(key)

        record = {
            'wheel': f"{match} of {draw} from {pool}",
            'pool': pool,
            'draw': draw,
            'match': match,
            'coverage': coverage,
            'num_tickets': len(tickets),
            'tickets': [list(t) for t in tickets],  # Convert tuples to lists for JSON
            'generations': generations,
            'time_seconds': time_seconds,
            'timestamp': datetime.now().isoformat(),
            'previous_record': previous['num_tickets'] if previous else None
        }

        self.records[key] = record
        self._save_records()

        # Also save individual wheel file with full ticket list
        wheel_file = self.save_dir / f"wheel_{key}.json"
        with open(wheel_file, 'w') as f:
            json.dump(record, f, indent=2)

        # Save human-readable ticket list
        ticket_file = self.save_dir / f"tickets_{key}.txt"
        with open(ticket_file, 'w') as f:
            f.write(f"# Abbreviated Wheel: {match} of {draw} from {pool}\n")
            f.write(f"# Coverage: {coverage:.2f}%\n")
            f.write(f"# Tickets: {len(tickets)}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#" + "="*50 + "\n\n")
            for i, ticket in enumerate(tickets, 1):
                # Numbers are 0-indexed internally, display as 1-indexed
                numbers = [str(n + 1) for n in ticket]
                f.write(f"Ticket {i:3d}: {' '.join(f'{n:>2}' for n in numbers)}\n")

        return record

    def get_ghost_to_beat(self, pool: int, draw: int, match: int) -> Optional[Dict]:
        """
        Get the 'ghost' record to beat for this configuration.
        Returns None if no record exists.
        """
        return self.get_record(pool, draw, match)

    def get_all_records(self) -> List[Dict]:
        """Get all records sorted by wheel configuration."""
        return sorted(self.records.values(), key=lambda x: (x['pool'], x['draw'], x['match']))

    def print_hall_of_fame(self):
        """Print the hall of fame to console."""
        records = self.get_all_records()

        if not records:
            print("\n🏆 HALL OF FAME - No records yet! Be the first!")
            return

        print("\n" + "="*70)
        print("🏆 HALL OF FAME - BEST COVERED SETS")
        print("="*70)
        print(f"{'Wheel':<18} {'Coverage':>10} {'Tickets':>10} {'Date':>20}")
        print("-"*70)

        for r in records:
            date = datetime.fromisoformat(r['timestamp']).strftime('%Y-%m-%d %H:%M')
            print(f"{r['wheel']:<18} {r['coverage']:>9.2f}% {r['num_tickets']:>10} {date:>20}")

        print("="*70 + "\n")

    def export_tickets_formatted(self, pool: int, draw: int, match: int,
                                 format_type: str = 'simple') -> str:
        """
        Export tickets in various formats.

        format_type options:
        - 'simple': Just the numbers
        - 'csv': Comma-separated
        - 'lottery': Ready to play format
        """
        record = self.get_record(pool, draw, match)
        if not record:
            return "No record found for this configuration."

        tickets = [tuple(t) for t in record['tickets']]
        lines = []

        if format_type == 'simple':
            for i, ticket in enumerate(tickets, 1):
                numbers = [str(n + 1) for n in ticket]
                lines.append(f"{i:3d}. {' '.join(numbers)}")

        elif format_type == 'csv':
            lines.append(','.join([f'N{i+1}' for i in range(record['draw'])]))
            for ticket in tickets:
                lines.append(','.join(str(n + 1) for n in ticket))

        elif format_type == 'lottery':
            lines.append(f"=== {record['wheel']} Wheel ===")
            lines.append(f"Coverage: {record['coverage']:.1f}%")
            lines.append(f"Total Tickets: {len(tickets)}")
            lines.append("")
            for i, ticket in enumerate(tickets, 1):
                numbers = [str(n + 1).zfill(2) for n in ticket]
                lines.append(f"Play {i:3d}: [ {' - '.join(numbers)} ]")

        return '\n'.join(lines)


class GhostTracker:
    """
    Tracks progress against the 'ghost' (previous best record).
    Provides real-time comparison during training.
    """

    def __init__(self, records_manager: RecordsManager,
                 pool: int, draw: int, match: int):
        self.records = records_manager
        self.pool = pool
        self.draw = draw
        self.match = match

        self.ghost = self.records.get_ghost_to_beat(pool, draw, match)
        self.ghost_tickets = self.ghost['num_tickets'] if self.ghost else None
        self.ghost_coverage = self.ghost['coverage'] if self.ghost else None

    @property
    def has_ghost(self) -> bool:
        """Check if there's a ghost to beat."""
        return self.ghost is not None

    def get_ghost_status(self, current_coverage: float, current_tickets: int) -> Dict:
        """
        Compare current progress to ghost.

        Returns status dict with:
        - ahead: bool - are we ahead of the ghost?
        - tickets_diff: int - difference in ticket count (negative = better)
        - coverage_diff: float - difference in coverage
        - message: str - status message
        """
        if not self.has_ghost:
            return {
                'ahead': True,
                'tickets_diff': 0,
                'coverage_diff': 0,
                'message': "🆕 First attempt - setting the record!",
                'ghost_tickets': None,
                'ghost_coverage': None
            }

        tickets_diff = current_tickets - self.ghost_tickets
        coverage_diff = current_coverage - self.ghost_coverage

        # Determine if we're ahead
        if current_coverage >= self.ghost_coverage:
            if current_tickets <= self.ghost_tickets:
                ahead = True
                if current_tickets < self.ghost_tickets:
                    message = f"🏆 NEW RECORD! {-tickets_diff} fewer tickets!"
                else:
                    message = f"✅ Matched record with {current_tickets} tickets"
            else:
                ahead = False
                message = f"⚠️ {tickets_diff} more tickets than record"
        else:
            ahead = False
            message = f"📈 {coverage_diff:+.1f}% vs record ({self.ghost_coverage:.1f}%)"

        return {
            'ahead': ahead,
            'tickets_diff': tickets_diff,
            'coverage_diff': coverage_diff,
            'message': message,
            'ghost_tickets': self.ghost_tickets,
            'ghost_coverage': self.ghost_coverage
        }

    def is_new_record(self, coverage: float, num_tickets: int) -> bool:
        """Check if current result beats the ghost."""
        return self.records.check_new_record(
            self.pool, self.draw, self.match, coverage, num_tickets
        )


if __name__ == "__main__":
    # Test the records system
    manager = RecordsManager()

    # Show hall of fame
    manager.print_hall_of_fame()

    # Test saving a record
    test_tickets = [
        (0, 5, 10, 15, 20),
        (1, 6, 11, 16, 21),
        (2, 7, 12, 17, 22),
    ]

    print("Testing record save...")
    if manager.check_new_record(36, 5, 3, 45.5, len(test_tickets)):
        record = manager.save_record(36, 5, 3, 45.5, test_tickets, 100, 30.5)
        print(f"Saved record: {record['wheel']} - {record['num_tickets']} tickets")

    manager.print_hall_of_fame()

    # Test ghost tracker
    ghost = GhostTracker(manager, 36, 5, 3)
    print(f"\nGhost to beat: {ghost.ghost_tickets} tickets at {ghost.ghost_coverage}%")
    print(ghost.get_ghost_status(50.0, 4))
