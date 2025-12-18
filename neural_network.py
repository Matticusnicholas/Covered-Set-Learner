"""
Neural Network Architecture for Covered Set Learning
=====================================================
A policy-based neural network that learns to generate optimal lottery ticket combinations.

Uses a transformer-inspired architecture to understand relationships between numbers
and generate tickets that maximize coverage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for number positions."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class NumberEmbedding(nn.Module):
    """Embedding layer for lottery numbers."""

    def __init__(self, pool_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(pool_size, embed_dim)
        self.pool_size = pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class TicketGenerator(nn.Module):
    """
    Neural network that generates lottery tickets.

    Uses attention mechanisms to learn which number combinations
    provide good coverage.
    """

    def __init__(self, pool_size: int = 36, draw_size: int = 5,
                 embed_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        self.pool_size = pool_size
        self.draw_size = draw_size
        self.embed_dim = embed_dim

        # Number embedding
        self.number_embed = NumberEmbedding(pool_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, pool_size)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Coverage state encoder
        self.coverage_encoder = nn.Sequential(
            nn.Linear(pool_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Output heads
        self.number_selector = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, pool_size)
        )

        # Value head for reinforcement learning
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, coverage_state: torch.Tensor,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate probability distribution over numbers for ticket selection.

        Args:
            coverage_state: Binary tensor [batch, pool_size] indicating which
                           numbers are "hot" (frequently in uncovered draws)
            temperature: Softmax temperature for exploration

        Returns:
            logits: Raw scores for each number
            value: Estimated value of current state
        """
        batch_size = coverage_state.size(0)

        # Create number indices
        number_indices = torch.arange(self.pool_size, device=coverage_state.device)
        number_indices = number_indices.unsqueeze(0).expand(batch_size, -1)

        # Embed numbers
        number_embeds = self.number_embed(number_indices)
        number_embeds = self.pos_encoding(number_embeds)

        # Process through transformer
        transformed = self.transformer(number_embeds)

        # Global representation (mean pooling)
        global_repr = transformed.mean(dim=1)

        # Encode coverage state
        coverage_repr = self.coverage_encoder(coverage_state.float())

        # Combine representations
        combined = torch.cat([global_repr, coverage_repr], dim=-1)

        # Generate logits for number selection
        logits = self.number_selector(combined)

        # Value estimation
        value = self.value_head(combined)

        return logits / temperature, value


class CoveredSetPolicy(nn.Module):
    """
    Policy network for generating complete ticket sets.

    Uses the TicketGenerator to sequentially generate tickets
    until coverage goal is reached.
    """

    def __init__(self, pool_size: int = 36, draw_size: int = 5,
                 match_required: int = 3, device: str = None, **kwargs):
        super().__init__()

        self.pool_size = pool_size
        self.draw_size = draw_size
        self.match_required = match_required

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Main generator network
        self.generator = TicketGenerator(pool_size, draw_size, **kwargs)

        # Move to device
        self.to(self.device)

        print(f"🧠 Neural Network initialized on {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def generate_ticket(self, coverage_state: torch.Tensor,
                       temperature: float = 1.0,
                       greedy: bool = False) -> Tuple[Tuple[int, ...], torch.Tensor]:
        """
        Generate a single ticket based on current coverage state.

        Args:
            coverage_state: Current coverage information
            temperature: Exploration temperature
            greedy: If True, always pick highest probability numbers

        Returns:
            ticket: Tuple of selected numbers
            log_prob: Log probability of the selection
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.generator(coverage_state, temperature)

        # Select numbers one by one
        selected = []
        total_log_prob = 0

        mask = torch.zeros(self.pool_size, device=self.device)

        for _ in range(self.draw_size):
            # Mask already selected numbers
            masked_logits = logits.clone()
            masked_logits[0, mask.bool()] = float('-inf')

            # Sample or greedy select
            probs = F.softmax(masked_logits, dim=-1)

            if greedy:
                idx = probs.argmax(dim=-1).item()
            else:
                idx = torch.multinomial(probs[0], 1).item()

            selected.append(idx)
            mask[idx] = 1
            total_log_prob += torch.log(probs[0, idx] + 1e-10)

        return tuple(sorted(selected)), total_log_prob

    def generate_ticket_batch(self, coverage_states: torch.Tensor,
                             temperature: float = 1.0) -> List[Tuple[int, ...]]:
        """Generate multiple tickets in parallel."""
        batch_size = coverage_states.size(0)
        tickets = []

        for i in range(batch_size):
            ticket, _ = self.generate_ticket(
                coverage_states[i:i+1],
                temperature=temperature
            )
            tickets.append(ticket)

        return tickets

    def compute_coverage_state(self, tickets: List[Tuple[int, ...]],
                              uncovered_draws: List[Tuple[int, ...]]) -> torch.Tensor:
        """
        Compute coverage state from current tickets and uncovered draws.

        Returns a tensor indicating which numbers appear most frequently
        in uncovered draws (numbers we should prioritize).

        OPTIMIZED: Uses vectorized operations instead of Python loops.
        """
        state = torch.zeros(1, self.pool_size, device=self.device)

        if not uncovered_draws:
            # Return random state to start generating immediately
            return torch.rand(1, self.pool_size, device=self.device)

        # VECTORIZED: Convert to tensor and use scatter_add
        num_draws = len(uncovered_draws)
        draw_size = len(uncovered_draws[0])

        # Flatten draws to tensor
        draws_tensor = torch.tensor(uncovered_draws, device=self.device, dtype=torch.int64).flatten()

        # Use scatter_add for fast counting
        ones = torch.ones(num_draws * draw_size, device=self.device)
        state[0].scatter_add_(0, draws_tensor, ones)

        # Normalize
        state = state / (num_draws + 1e-10)

        return state


class EvolutionaryOptimizer:
    """
    Evolutionary strategy for optimizing the neural network.

    Uses population-based training with mutation and selection.
    """

    def __init__(self, policy: CoveredSetPolicy, population_size: int = 20,
                 elite_size: int = 4, mutation_rate: float = 0.1):
        self.policy = policy
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        # Create population of parameter perturbations
        self.population = []
        self._init_population()

    def _init_population(self):
        """Initialize population with random perturbations."""
        self.population = []
        for _ in range(self.population_size):
            perturbation = {}
            for name, param in self.policy.named_parameters():
                perturbation[name] = torch.randn_like(param) * self.mutation_rate
            self.population.append(perturbation)

    def apply_perturbation(self, perturbation: dict, scale: float = 1.0):
        """Apply a perturbation to the policy."""
        with torch.no_grad():
            for name, param in self.policy.named_parameters():
                param.add_(perturbation[name] * scale)

    def evolve(self, fitness_scores: List[float]):
        """Evolve population based on fitness scores."""
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]

        # Keep elites
        elite_perturbations = [self.population[i] for i in sorted_indices[:self.elite_size]]

        # Generate new population
        new_population = elite_perturbations.copy()

        while len(new_population) < self.population_size:
            # Select parent from elites
            parent_idx = np.random.randint(self.elite_size)
            parent = elite_perturbations[parent_idx]

            # Mutate
            child = {}
            for name, pert in parent.items():
                mutation = torch.randn_like(pert) * self.mutation_rate
                child[name] = pert + mutation

            new_population.append(child)

        self.population = new_population

        return fitness_scores[sorted_indices[0]]  # Return best fitness


class ReinforcementTrainer:
    """
    Reinforcement learning trainer using policy gradients.
    """

    def __init__(self, policy: CoveredSetPolicy, learning_rate: float = 1e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.policy.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train_episode(self, calculator, max_tickets: int = 1000,
                     target_coverage: float = 100.0) -> dict:
        """
        Train on a single episode of ticket generation.

        Returns training statistics.
        """
        self.policy.train()

        tickets = []
        log_probs = []
        rewards = []
        coverages = []

        # Get theoretical minimum for reward scaling
        theoretical_min = calculator.theoretical_minimum_tickets()

        # Initial state
        uncovered = calculator.get_uncovered_draws([], max_return=2000)
        coverage_state = self.policy.compute_coverage_state([], uncovered)

        for step in range(max_tickets):
            # Generate ticket
            ticket, log_prob = self.policy.generate_ticket(
                coverage_state,
                temperature=1.0,
                greedy=False
            )

            tickets.append(ticket)
            log_probs.append(log_prob)

            # Calculate new coverage
            result = calculator.calculate_coverage(tickets)
            coverage = result['coverage']
            coverages.append(coverage)

            # REWARD SYSTEM - Optimized for FEWER tickets
            prev_coverage = coverages[-2] if len(coverages) > 1 else 0
            improvement = coverage - prev_coverage

            # Base reward: coverage improvement (scaled up)
            reward = improvement * 2.0

            # PENALTY for each ticket (encourages efficiency)
            reward -= 0.05

            # BIG BONUS for high-value tickets (>1% coverage gain)
            if improvement > 1.0:
                reward += improvement * 0.5

            # Check if done - 100% coverage reached!
            if coverage >= target_coverage:
                # HUGE BONUS scaled by efficiency
                # Fewer tickets = bigger bonus
                # If you match theoretical minimum, get max bonus
                efficiency_ratio = theoretical_min / len(tickets)
                completion_bonus = 50.0 * efficiency_ratio  # Up to 50 points if optimal
                reward += completion_bonus
                rewards.append(reward)
                print(f"  🎯 100% coverage in {len(tickets)} tickets! (theoretical min: {theoretical_min})")
                break

            rewards.append(reward)

            # Update state (less frequently for speed)
            if step % 10 == 0:
                uncovered = calculator.get_uncovered_draws(tickets, max_return=2000)
                coverage_state = self.policy.compute_coverage_state(tickets, uncovered)

        # Compute returns and update policy
        returns = self.compute_returns(rewards)
        log_probs_tensor = torch.stack(log_probs)

        # Policy gradient loss
        loss = -(log_probs_tensor * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Statistics
        self.episode_rewards.append(sum(rewards))
        self.episode_lengths.append(len(tickets))

        return {
            'loss': loss.item(),
            'total_reward': sum(rewards),
            'final_coverage': coverages[-1],
            'num_tickets': len(tickets),
            'tickets': tickets
        }


if __name__ == "__main__":
    # Test the neural network
    policy = CoveredSetPolicy(pool_size=36, draw_size=5, match_required=3)

    # Test ticket generation
    coverage_state = torch.zeros(1, 36, device=policy.device)
    ticket, log_prob = policy.generate_ticket(coverage_state)

    print(f"\n🎫 Generated ticket: {ticket}")
    print(f"   Log probability: {log_prob.item():.4f}")

    # Test batch generation
    tickets = policy.generate_ticket_batch(
        torch.zeros(5, 36, device=policy.device),
        temperature=0.8
    )
    print(f"\n📦 Batch generated {len(tickets)} tickets:")
    for i, t in enumerate(tickets):
        print(f"   Ticket {i+1}: {t}")
