"""
Covered Set Logic with CUDA Acceleration
=========================================
Calculates coverage for lottery combinations using GPU-accelerated operations.

A covered set guarantees that at least one ticket matches at least M numbers
with ANY possible drawn combination.
"""

import torch
import numpy as np
from itertools import combinations
from typing import List, Tuple, Set
import math


class CoveredSetCalculator:
    """GPU-accelerated covered set calculator."""

    def __init__(self, pool_size: int = 36, draw_size: int = 5,
                 match_required: int = 3, device: str = None):
        """
        Initialize the covered set calculator.

        Args:
            pool_size: Total numbers in the lottery pool (e.g., 36)
            draw_size: Numbers drawn per ticket (e.g., 5)
            match_required: Minimum matches needed for coverage (e.g., 3)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.pool_size = pool_size
        self.draw_size = draw_size
        self.match_required = match_required

        # Auto-detect CUDA
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🎰 Covered Set Calculator initialized on {self.device}")
        print(f"   Pool: {pool_size} numbers | Draw: {draw_size} | Match: {match_required}+")

        # Pre-compute all possible draws for coverage calculation
        self._precompute_all_draws()

    def _precompute_all_draws(self):
        """Pre-compute all possible lottery draws as GPU tensors."""
        # Generate all C(pool_size, draw_size) combinations
        all_draws = list(combinations(range(self.pool_size), self.draw_size))
        self.total_draws = len(all_draws)

        # Convert to GPU tensor for fast computation
        self.all_draws_tensor = torch.tensor(all_draws, device=self.device, dtype=torch.int32)

        # Create binary representation for faster matching
        self.all_draws_binary = torch.zeros(
            (self.total_draws, self.pool_size),
            device=self.device,
            dtype=torch.bool
        )

        for i, draw in enumerate(all_draws):
            for num in draw:
                self.all_draws_binary[i, num] = True

        print(f"   Pre-computed {self.total_draws:,} possible draws")

    def tickets_to_binary(self, tickets: List[Tuple[int, ...]]) -> torch.Tensor:
        """Convert ticket list to binary GPU tensor."""
        num_tickets = len(tickets)
        binary = torch.zeros(
            (num_tickets, self.pool_size),
            device=self.device,
            dtype=torch.bool
        )

        for i, ticket in enumerate(tickets):
            for num in ticket:
                binary[i, num] = True

        return binary

    def calculate_coverage(self, tickets: List[Tuple[int, ...]],
                          return_details: bool = False) -> dict:
        """
        Calculate coverage percentage for a set of tickets.

        Args:
            tickets: List of ticket tuples (each tuple contains draw_size numbers)
            return_details: Whether to return detailed coverage info

        Returns:
            Dictionary with coverage statistics
        """
        if not tickets:
            return {'coverage': 0.0, 'covered_draws': 0, 'total_draws': self.total_draws}

        # Convert tickets to binary format
        tickets_binary = self.tickets_to_binary(tickets)

        # Calculate matches between all tickets and all possible draws
        # This is the key GPU-accelerated operation
        # Shape: (num_tickets, total_draws)
        matches = torch.zeros(
            (len(tickets), self.total_draws),
            device=self.device,
            dtype=torch.int32
        )

        # Batch compute intersections using matrix operations
        for i, ticket_bin in enumerate(tickets_binary):
            # Count matching numbers between this ticket and all draws
            intersection = ticket_bin.unsqueeze(0) & self.all_draws_binary
            matches[i] = intersection.sum(dim=1)

        # A draw is covered if ANY ticket matches >= match_required numbers
        max_matches_per_draw = matches.max(dim=0).values
        covered_mask = max_matches_per_draw >= self.match_required
        covered_count = covered_mask.sum().item()

        coverage_pct = (covered_count / self.total_draws) * 100

        result = {
            'coverage': coverage_pct,
            'covered_draws': covered_count,
            'total_draws': self.total_draws,
            'num_tickets': len(tickets),
            'efficiency': coverage_pct / len(tickets) if tickets else 0
        }

        if return_details:
            result['match_distribution'] = self._get_match_distribution(max_matches_per_draw)
            result['uncovered_indices'] = (~covered_mask).nonzero().squeeze(-1).cpu().tolist()

        return result

    def _get_match_distribution(self, max_matches: torch.Tensor) -> dict:
        """Get distribution of match counts."""
        distribution = {}
        for i in range(self.draw_size + 1):
            count = (max_matches == i).sum().item()
            if count > 0:
                distribution[i] = count
        return distribution

    def calculate_incremental_coverage(self, existing_tickets: List[Tuple[int, ...]],
                                       new_ticket: Tuple[int, ...]) -> float:
        """Calculate how much coverage a new ticket would add."""
        current = self.calculate_coverage(existing_tickets)
        combined = self.calculate_coverage(existing_tickets + [new_ticket])
        return combined['coverage'] - current['coverage']

    def get_uncovered_draws(self, tickets: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """Get list of draws not yet covered by the tickets."""
        if not tickets:
            return [tuple(self.all_draws_tensor[i].cpu().tolist())
                    for i in range(self.total_draws)]

        result = self.calculate_coverage(tickets, return_details=True)
        uncovered_indices = result['uncovered_indices']

        if isinstance(uncovered_indices, int):
            uncovered_indices = [uncovered_indices]

        return [tuple(self.all_draws_tensor[i].cpu().tolist())
                for i in uncovered_indices]

    def theoretical_minimum_tickets(self) -> int:
        """Estimate theoretical minimum tickets needed for full coverage."""
        # This is a rough estimate based on covering design theory
        # Actual minimum is NP-hard to compute
        n = self.pool_size
        k = self.draw_size
        m = self.match_required

        # Use covering design bound
        numerator = math.comb(n, m)
        denominator = math.comb(k, m)

        return math.ceil(numerator / denominator)

    def generate_random_ticket(self) -> Tuple[int, ...]:
        """Generate a random valid ticket."""
        numbers = torch.randperm(self.pool_size, device=self.device)[:self.draw_size]
        return tuple(sorted(numbers.cpu().tolist()))

    def generate_random_tickets(self, count: int) -> List[Tuple[int, ...]]:
        """Generate multiple random tickets."""
        return [self.generate_random_ticket() for _ in range(count)]


class CoverageTracker:
    """Track coverage progress over training."""

    def __init__(self, calculator: CoveredSetCalculator):
        self.calculator = calculator
        self.history = []
        self.best_coverage = 0.0
        self.best_tickets = []

    def update(self, tickets: List[Tuple[int, ...]], generation: int = 0):
        """Update tracker with new ticket set."""
        result = self.calculator.calculate_coverage(tickets)
        result['generation'] = generation
        self.history.append(result)

        if result['coverage'] > self.best_coverage:
            self.best_coverage = result['coverage']
            self.best_tickets = tickets.copy()

        return result

    def get_recent_improvement(self, window: int = 10) -> float:
        """Calculate improvement over recent generations."""
        if len(self.history) < 2:
            return 0.0

        recent = self.history[-min(window, len(self.history)):]
        return recent[-1]['coverage'] - recent[0]['coverage']


if __name__ == "__main__":
    # Test the calculator
    calc = CoveredSetCalculator(pool_size=36, draw_size=5, match_required=3)

    # Generate some random tickets
    tickets = calc.generate_random_tickets(10)
    print(f"\nGenerated {len(tickets)} random tickets:")
    for i, t in enumerate(tickets[:5]):
        print(f"  Ticket {i+1}: {t}")

    # Calculate coverage
    result = calc.calculate_coverage(tickets, return_details=True)
    print(f"\n📊 Coverage Results:")
    print(f"   Coverage: {result['coverage']:.2f}%")
    print(f"   Covered: {result['covered_draws']:,} / {result['total_draws']:,} draws")
    print(f"   Efficiency: {result['efficiency']:.2f}% per ticket")
    print(f"\n   Theoretical minimum: ~{calc.theoretical_minimum_tickets()} tickets")
