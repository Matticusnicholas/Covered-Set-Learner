#!/usr/bin/env python3
"""
🎰 Neural Network Lottery Covered Set Learner
=============================================

A deep learning AI that learns to find optimal covered sets for lottery combinations.
Uses reinforcement learning with CUDA acceleration and a beautiful streaming interface.

Usage:
    python main.py                    # Run with default settings
    python main.py --headless         # Run without visualization
    python main.py --pool 36 --draw 5 --match 3  # Custom lottery config

Author: AI-Generated
License: MIT
"""

import argparse
import time
import sys
import os
import random
from typing import List, Tuple, Dict, Optional
from collections import deque

# Check for CUDA availability early
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  CUDA not available, using CPU (slower)")

from covered_set import CoveredSetCalculator, CoverageTracker
from neural_network import CoveredSetPolicy, ReinforcementTrainer, EvolutionaryOptimizer


class LotteryLearner:
    """
    Main class that orchestrates the neural network learning process
    for finding optimal lottery covered sets.
    """

    def __init__(self, pool_size: int = 36, draw_size: int = 5, match_required: int = 3,
                 headless: bool = False, target_coverage: float = 100.0):
        """
        Initialize the lottery learner.

        Args:
            pool_size: Total numbers in the lottery pool
            draw_size: Numbers drawn per ticket
            match_required: Minimum matches needed for coverage
            headless: If True, run without visualization
            target_coverage: Target coverage percentage (0-100)
        """
        self.pool_size = pool_size
        self.draw_size = draw_size
        self.match_required = match_required
        self.headless = headless
        self.target_coverage = target_coverage

        # Abbreviated wheel notation
        self.wheel_name = f"{match_required} of {draw_size} from {pool_size}"

        print("\n" + "="*60)
        print("🎰 NEURAL NETWORK ABBREVIATED WHEEL LEARNER")
        print("="*60)
        print(f"   Wheel: {self.wheel_name}")
        print(f"   ├─ Pool Size: {pool_size} numbers")
        print(f"   ├─ Draw Size: {draw_size} numbers per ticket")
        print(f"   ├─ Match Required: {match_required}+ numbers")
        print(f"   ├─ Target Coverage: {target_coverage}%")
        print(f"   └─ Mode: {'Headless' if headless else 'Visual (Streaming)'}")
        print("="*60 + "\n")

        # Initialize components
        print("Initializing components...")

        self.calculator = CoveredSetCalculator(pool_size, draw_size, match_required)
        self.policy = CoveredSetPolicy(pool_size, draw_size, match_required,
                                       embed_dim=128, num_heads=4, num_layers=3)
        self.trainer = ReinforcementTrainer(self.policy, learning_rate=3e-4)
        self.tracker = CoverageTracker(self.calculator)

        # Visualization (if not headless)
        self.viz = None
        if not headless:
            try:
                from visualization import StreamingVisualizer
                self.viz = StreamingVisualizer(
                    width=1280, height=720,
                    pool_size=pool_size,
                    draw_size=draw_size,
                    match_required=match_required
                )
                print("✅ Visualization initialized")
            except ImportError as e:
                print(f"⚠️  Could not initialize visualization: {e}")
                print("   Running in headless mode")

        # Training state
        self.generation = 0
        self.best_coverage = 0.0
        self.best_tickets = []
        self.running = True

        # Statistics
        self.stats_history = deque(maxlen=1000)

        print("\n🎯 Ready to start training!")
        print(f"   Theoretical minimum: ~{self.calculator.theoretical_minimum_tickets()} tickets")
        print()

    def compute_heat_map(self, uncovered_draws: List[Tuple[int, ...]]) -> Dict[int, float]:
        """Compute heat map showing which numbers appear most in uncovered draws."""
        heat = {i: 0.0 for i in range(self.pool_size)}

        if not uncovered_draws:
            return heat

        for draw in uncovered_draws:
            for num in draw:
                heat[num] += 1

        # Normalize
        max_heat = max(heat.values()) if heat.values() else 1
        for k in heat:
            heat[k] /= max_heat

        return heat

    def train_generation(self, max_tickets: int = 100,
                        temperature: float = 1.0) -> Dict:
        """
        Train one generation of ticket generation.

        Returns:
            Dictionary with generation statistics
        """
        self.generation += 1
        tickets = []

        # Initial state
        uncovered = self.calculator.get_uncovered_draws([])
        heat_map = self.compute_heat_map(uncovered)

        start_time = time.time()

        for step in range(max_tickets):
            # Compute current state
            coverage_state = self.policy.compute_coverage_state(tickets, uncovered)

            # Generate new ticket
            ticket, log_prob = self.policy.generate_ticket(
                coverage_state,
                temperature=temperature,
                greedy=(temperature < 0.3)  # More greedy at low temps
            )

            # Avoid duplicate tickets
            attempts = 0
            while ticket in tickets and attempts < 10:
                ticket, log_prob = self.policy.generate_ticket(
                    coverage_state,
                    temperature=min(2.0, temperature + 0.5),
                    greedy=False
                )
                attempts += 1

            tickets.append(ticket)

            # Calculate new coverage
            result = self.calculator.calculate_coverage(tickets)
            coverage = result['coverage']

            # Update visualization
            if self.viz and self.viz.running:
                dt = self.viz.run_frame()

                # Update heat map periodically
                if step % 5 == 0:
                    uncovered = self.calculator.get_uncovered_draws(tickets)
                    heat_map = self.compute_heat_map(uncovered)

                stats = {
                    'generation': self.generation,
                    'coverage': coverage,
                    'num_tickets': len(tickets),
                    'best_coverage': self.best_coverage,
                    'efficiency': coverage / len(tickets) if tickets else 0,
                    'step': step
                }

                self.viz.add_ticket(ticket)
                self.viz.update(dt, stats, ticket, heat_map)
                self.viz.draw()

                if not self.viz.running:
                    self.running = False
                    break

            # Check if we've reached target
            if coverage >= self.target_coverage:
                break

            # Update uncovered draws for next iteration
            uncovered = self.calculator.get_uncovered_draws(tickets)

        elapsed = time.time() - start_time

        # Final coverage
        final_result = self.calculator.calculate_coverage(tickets)

        # Update best
        if final_result['coverage'] > self.best_coverage:
            self.best_coverage = final_result['coverage']
            self.best_tickets = tickets.copy()
            print(f"\n🎉 NEW BEST! Generation {self.generation}: {self.best_coverage:.2f}% with {len(tickets)} tickets")

        # Prepare statistics
        stats = {
            'generation': self.generation,
            'coverage': final_result['coverage'],
            'num_tickets': len(tickets),
            'best_coverage': self.best_coverage,
            'best_num_tickets': len(self.best_tickets),
            'efficiency': final_result['efficiency'],
            'time': elapsed,
            'tickets': tickets
        }

        self.stats_history.append(stats)
        self.tracker.update(tickets, self.generation)

        return stats

    def train_episode_rl(self) -> Dict:
        """
        Train using reinforcement learning (policy gradients).
        """
        result = self.trainer.train_episode(
            self.calculator,
            max_tickets=100,
            target_coverage=self.target_coverage
        )

        # Update visualization
        if self.viz and self.viz.running:
            self.viz.new_generation()
            for ticket in result['tickets']:
                self.viz.add_ticket(ticket)

        if result['final_coverage'] > self.best_coverage:
            self.best_coverage = result['final_coverage']
            self.best_tickets = result['tickets']
            print(f"\n🎉 NEW BEST (RL)! {self.best_coverage:.2f}% with {len(self.best_tickets)} tickets")

        return result

    def run_training(self, num_generations: int = 1000,
                    training_mode: str = 'hybrid') -> Dict:
        """
        Run the full training loop.

        Args:
            num_generations: Number of generations to train
            training_mode: 'greedy', 'rl', or 'hybrid'

        Returns:
            Final training statistics
        """
        print(f"\n🚀 Starting training for {num_generations} generations...")
        print(f"   Training mode: {training_mode}")
        print("   Press ESC to stop (in visual mode)\n")

        start_time = time.time()

        # Temperature schedule for exploration
        initial_temp = 2.0
        final_temp = 0.1

        try:
            for gen in range(num_generations):
                if not self.running:
                    print("\n⏹️  Training stopped by user")
                    break

                # Compute temperature (annealing)
                progress = gen / max(1, num_generations - 1)
                temperature = initial_temp - (initial_temp - final_temp) * progress

                if training_mode == 'greedy':
                    stats = self.train_generation(
                        max_tickets=100,
                        temperature=max(0.3, temperature)
                    )
                elif training_mode == 'rl':
                    stats = self.train_episode_rl()
                else:  # hybrid
                    if gen % 3 == 0:
                        stats = self.train_episode_rl()
                    else:
                        stats = self.train_generation(
                            max_tickets=100,
                            temperature=temperature
                        )

                # Print progress
                if gen % 10 == 0 or stats['coverage'] >= self.target_coverage:
                    elapsed = time.time() - start_time
                    print(f"Gen {gen:4d} | Coverage: {stats['coverage']:6.2f}% | "
                          f"Tickets: {stats['num_tickets']:3d} | "
                          f"Best: {self.best_coverage:6.2f}% ({len(self.best_tickets)} tickets) | "
                          f"Temp: {temperature:.2f} | Time: {elapsed:.1f}s")

                # Start new generation in visualization
                if self.viz and self.viz.running:
                    self.viz.new_generation()

                # Early stopping if we've found optimal solution
                if self.best_coverage >= self.target_coverage:
                    consecutive_optimal = sum(
                        1 for s in list(self.stats_history)[-10:]
                        if s['coverage'] >= self.target_coverage
                    )
                    if consecutive_optimal >= 5:
                        print(f"\n✅ Consistently achieving {self.target_coverage}% coverage!")
                        break

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")

        total_time = time.time() - start_time

        # Final summary
        print("\n" + "="*60)
        print("📊 TRAINING COMPLETE")
        print("="*60)
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Generations: {self.generation}")
        print(f"   Best coverage: {self.best_coverage:.2f}%")
        print(f"   Best ticket count: {len(self.best_tickets)}")
        print(f"   Theoretical minimum: ~{self.calculator.theoretical_minimum_tickets()}")

        if self.best_tickets:
            print(f"\n🎫 Best ticket set ({len(self.best_tickets)} tickets):")
            for i, ticket in enumerate(self.best_tickets[:20]):
                numbers = [str(n+1) for n in ticket]
                print(f"   #{i+1:3d}: {' '.join(f'{n:>2}' for n in numbers)}")
            if len(self.best_tickets) > 20:
                print(f"   ... and {len(self.best_tickets) - 20} more tickets")

        print("="*60 + "\n")

        # Cleanup visualization
        if self.viz:
            self.viz.cleanup()

        return {
            'best_coverage': self.best_coverage,
            'best_tickets': self.best_tickets,
            'total_generations': self.generation,
            'total_time': total_time
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neural Network Abbreviated Wheel / Covered Set Learner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Abbreviated Wheel Notation: "M of D from P" means:
  - P = Pool size (total numbers to choose from)
  - D = Draw size (numbers per ticket)
  - M = Match required (minimum numbers to match for coverage)

Examples:
    python main.py                              # Default: 3 of 5 from 36
    python main.py --pool 49 --draw 6 --match 4 # 4 of 6 from 49
    python main.py --pool 45 --draw 6 --match 3 # 3 of 6 from 45
    python main.py --headless --gens 500        # Train without visualization
    python main.py --mode rl                    # Pure reinforcement learning
        """
    )

    parser.add_argument('--pool', type=int, default=36,
                       help='Lottery pool size (default: 36)')
    parser.add_argument('--draw', type=int, default=5,
                       help='Numbers per draw (default: 5)')
    parser.add_argument('--match', type=int, default=3,
                       help='Minimum matches required (default: 3)')
    parser.add_argument('--target', type=float, default=100.0,
                       help='Target coverage percentage (default: 100)')
    parser.add_argument('--gens', type=int, default=500,
                       help='Number of generations (default: 500)')
    parser.add_argument('--mode', choices=['greedy', 'rl', 'hybrid'], default='hybrid',
                       help='Training mode (default: hybrid)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if CUDA_AVAILABLE:
            torch.cuda.manual_seed(args.seed)
        print(f"🎲 Random seed set to {args.seed}")

    # Validate arguments
    if args.match >= args.draw:
        print(f"❌ Error: match ({args.match}) must be less than draw ({args.draw})")
        sys.exit(1)

    if args.draw > args.pool:
        print(f"❌ Error: draw ({args.draw}) cannot exceed pool ({args.pool})")
        sys.exit(1)

    # Create and run learner
    learner = LotteryLearner(
        pool_size=args.pool,
        draw_size=args.draw,
        match_required=args.match,
        headless=args.headless,
        target_coverage=args.target
    )

    results = learner.run_training(
        num_generations=args.gens,
        training_mode=args.mode
    )

    return results


if __name__ == "__main__":
    main()
