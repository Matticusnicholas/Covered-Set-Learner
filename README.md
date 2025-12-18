# 🎰 Neural Network Abbreviated Wheel / Covered Set Learner

A deep learning AI that learns to find optimal **abbreviated wheels** (covered sets) for lottery combinations using neural networks and CUDA acceleration. Features a beautiful, streaming-friendly visualization perfect for YouTube and TikTok content.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)

## 🎯 What is an Abbreviated Wheel / Covered Set?

An **abbreviated wheel** (also called a **covered set** or **lottery wheel**) guarantees that at least one ticket matches a minimum number of drawn numbers, no matter what combination is drawn.

### Example: "3 of 5 from 36"

This notation means:
- **Pool (36)**: Choose from 36 total numbers
- **Pick (5)**: Each ticket has 5 numbers
- **Match (3)**: Guarantee at least 3 matching numbers

So if you play this wheel and the lottery draws ANY 5 numbers from 1-36, at least one of your tickets will have 3+ matching numbers!

### Common Abbreviated Wheel Configurations

| Wheel | Pool | Pick | Match | Typical Tickets |
|-------|------|------|-------|-----------------|
| 3/5/36 | 36 | 5 | 3 | ~15-25 |
| 4/6/49 | 49 | 6 | 4 | ~50-100 |
| 3/6/45 | 45 | 6 | 3 | ~20-40 |
| 5/6/49 | 49 | 6 | 5 | ~200+ |

The AI learns to find the **minimum number of tickets** needed to guarantee coverage!

## ✨ Features

- 🧠 **Neural Network Learning**: Transformer-based architecture that learns optimal ticket selection
- 🚀 **CUDA Acceleration**: Leverages NVIDIA GPU for fast coverage calculations
- 📊 **Beautiful Visualization**: Streaming-ready interface with real-time animations
- 🔄 **Hybrid Training**: Combines greedy search with reinforcement learning
- 📈 **Live Statistics**: Watch the AI improve in real-time
- 🎨 **Cyberpunk Theme**: Neon-styled UI perfect for content creation

## 🖥️ Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended: RTX 3060 or better)
- PyTorch with CUDA
- Pygame for visualization

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Covered-Set-Learner.git
cd Covered-Set-Learner

# Install dependencies
pip install -r requirements.txt

# For CUDA support, install PyTorch with CUDA:
# Visit https://pytorch.org/get-started/locally/ for your specific setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Usage

### Basic Usage (Visual Mode)

```bash
python main.py
```

This launches the full visual experience with:
- Real-time neural network training
- Animated number grid
- Live coverage progress
- Streaming-ready interface (1280x720)

### Headless Mode (No GUI)

```bash
python main.py --headless
```

### Custom Abbreviated Wheel Configuration

```bash
# 3 of 5 from 36 (default)
python main.py --pool 36 --draw 5 --match 3

# 4 of 6 from 49 (like many national lotteries)
python main.py --pool 49 --draw 6 --match 4

# 3 of 6 from 45
python main.py --pool 45 --draw 6 --match 3

# More training generations
python main.py --gens 1000

# Different training modes
python main.py --mode rl        # Pure reinforcement learning
python main.py --mode greedy    # Greedy exploration
python main.py --mode hybrid    # Combined (default)
```

### All Options

```
python main.py --help

Options:
  --pool INT      Lottery pool size (default: 36)
  --draw INT      Numbers per draw (default: 5)
  --match INT     Minimum matches required (default: 3)
  --target FLOAT  Target coverage percentage (default: 100)
  --gens INT      Number of generations (default: 500)
  --mode          Training mode: greedy, rl, hybrid (default: hybrid)
  --headless      Run without visualization
  --seed INT      Random seed for reproducibility
```

## 🎥 Streaming Setup

The visualization is designed for streaming:

1. **Resolution**: 1280x720 by default (modify in `visualization.py`)
2. **FPS**: Capped at 60 FPS for smooth streaming
3. **Color Theme**: Cyberpunk/neon colors optimized for video compression

### OBS Capture Tips

- Use Window Capture for the pygame window
- Add a slight blur to reduce encoding artifacts
- Consider adding a webcam overlay in the corner

## 🧠 How It Works

### Architecture

1. **Covered Set Calculator**: GPU-accelerated coverage computation
2. **Neural Network**: Transformer-based ticket generator
3. **Training Loop**: Hybrid reinforcement learning + greedy exploration
4. **Visualization**: Real-time pygame interface

### Training Process

```
┌──────────────────────────────────────────────────────┐
│                 TRAINING LOOP                        │
├──────────────────────────────────────────────────────┤
│ 1. Neural network observes uncovered combinations   │
│ 2. Generates probability distribution over numbers  │
│ 3. Samples a new ticket based on probabilities      │
│ 4. Calculates coverage improvement (reward)         │
│ 5. Updates network weights via policy gradients     │
│ 6. Repeats until target coverage achieved           │
└──────────────────────────────────────────────────────┘
```

### Key Components

- **TicketGenerator**: Transformer encoder that learns number relationships
- **CoveredSetPolicy**: Wraps the generator with state management
- **ReinforcementTrainer**: Policy gradient training with returns
- **CoveredSetCalculator**: Fast GPU-based coverage computation

## 📊 Performance

On an NVIDIA RTX 3060:

| Configuration | Pool | Draw | Match | ~Min Tickets | Compute Time |
|--------------|------|------|-------|--------------|--------------|
| Default      | 36   | 5    | 3     | ~15-20       | <1 min       |
| Larger       | 45   | 6    | 3     | ~30-50       | ~2 min       |
| Hard         | 49   | 6    | 4     | ~100+        | ~5 min       |

## 🎨 Visualization Components

- **Number Grid**: Shows all pool numbers with heat map coloring
- **Progress Bar**: Animated coverage progress with glow effects
- **Stats Panel**: Real-time training statistics and mini-graph
- **Ticket Display**: Scrollable list of generated tickets
- **Neural Network Viz**: Animated network activity display
- **Particle Effects**: Celebration particles on improvements

## 📁 Project Structure

```
Covered-Set-Learner/
├── main.py              # Entry point and training orchestration
├── covered_set.py       # Coverage calculation (CUDA-accelerated)
├── neural_network.py    # Neural network architecture
├── visualization.py     # Streaming-friendly pygame interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔬 Mathematical Background

The covered set problem is related to **covering designs** in combinatorics:

- A covering design C(v, k, t) covers all t-subsets of v elements using k-subsets
- Finding optimal covers is NP-hard
- Our neural network approximates good solutions through learning

**Theoretical lower bound** (Schönheim bound):
```
L(n, k, m) ≥ ⌈(n/k) × ⌈((n-1)/(k-1)) × ... ⌈((n-m+1)/(k-m+1))⌉...⌉⌉
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more neural network architectures
- [ ] Implement evolutionary strategies
- [ ] Add support for different lottery types
- [ ] Improve visualization with more effects
- [ ] Add training checkpoints/resume

## 📄 License

MIT License - Feel free to use for content creation!

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Pygame community for the visualization library
- Covered set research by various mathematicians

---

**Made with 🧠 and 🎰 for content creators and math enthusiasts!**
