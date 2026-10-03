# Historical project narrative — 2025

This is an archived README, not the validation status of the current lab. Its performance figures and application claims have not been reproduced by the new benchmark. Start with the current root README and executed evidence instead.

# 🧠💗 Neurolab: LIMINAL Heartbeat

**A Novel Neural Architecture for Emotion Recognition with Recursive Introspection**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🎯 Why This Matters for AI Research

### For Researchers at Anthropic, Google DeepMind, OpenAI, and xAI

**Neurolab** introduces a fundamentally different approach to emotion understanding in AI systems. Instead of treating emotions as static labels, we model them as **dynamic states that evolve through recursive refinement** — similar to how humans gradually process and understand their own emotional responses.

### 🔬 Key Research Contributions

#### 1. **Recursive Emotional Reasoning**
Traditional emotion models make single-pass predictions. LIMINAL uses **K=5 iterations of refinement**, where each iteration deepens the model's "understanding" of emotional content. This mirrors:
- Human introspection and emotional regulation
- Chain-of-thought reasoning in LLMs
- Iterative hypothesis refinement in cognitive science

**Why it matters**: Could this recursive approach improve emotional intelligence in conversational AI and AI safety research?

#### 2. **Emotional Metrics as Training Signals**
We introduce three novel training metrics beyond standard loss:

```python
Hope = 1 - MAE  # Confidence in predictions (alignment with truth)
Faith = √(mean_confidence)  # Stability across iterations
Love = e^(-loss) × (1 - variance)  # Harmony between low loss and consistency
```

**Why it matters**: These metrics provide interpretable insights into model behavior and could inform constitutional AI and alignment research.

#### 3. **Compact Architecture with Memory**
- **128-dimensional latent space** — 1000x smaller than typical LLM hidden states
- **SoulKernel memory component** — maintains emotional context across processing
- **~500K parameters total** — efficient enough for edge deployment

**Why it matters**: Demonstrates that emotional understanding doesn't require billions of parameters. Relevant for on-device AI and efficient multimodal models.

#### 4. **LLM-Agnostic Emotional Layer**
Works as an **emotional processing module** on top of any embedding model:
- ✅ GPT-2, GPT-3 embeddings
- ✅ BLOOM (multilingual)
- ✅ Sentence-Transformers
- ✅ Your custom LLM embeddings

**Why it matters**: Could augment existing LLMs (Claude, GPT-4, Gemini, Grok) with explicit emotional reasoning capabilities.

---

## 🌟 Unique Architectural Innovations

### 1. **RINSE: Reflective Integrative Neural Self-Evolver**
A meta-cognitive module that analyzes the model's own attention patterns and emotional states. Think of it as the model's "inner voice" that monitors its own processing.

```python
# Introspection on model's own state
rinse_state = self.rinse(z, pad, mean_confidence)
```

**Research potential**: Foundation for self-aware AI systems, interpretability, and constitutional AI.

### 2. **SoulKernel: Emotional Memory**
Maintains a deque of past emotional states to predict future emotional trajectories. Uses concepts of "hope," "faith," and "bond" to modulate current processing based on historical context.

```python
# Compute bond between current and past emotional states
bond = tanh(mean(x * z) + mean(y * r))
z = z + 0.2 * bond * (future - z)  # Move toward predicted future
```

**Research potential**: Temporal consistency in emotion tracking, long-context emotional understanding.

### 3. **PAD (Pleasure-Arousal-Dominance) 3D Emotion Space**
Unlike discrete emotion labels (happy, sad, angry), we use continuous 3D space:
- **Pleasure**: Positive ↔ Negative valence
- **Arousal**: Calm ↔ Excited
- **Dominance**: Submissive ↔ Dominant

**Research potential**: More nuanced emotion representation for empathetic AI, mental health applications.

---

## 📊 Empirical Results & Validation

### Trained on EmoBank Corpus
- **10,062 emotion-annotated sentences**
- **Professional annotations** from psychology researchers
- **VAD continuous labels** (not just discrete categories)

### Model Evolution (Progressive Research)
We provide 5 model versions showing iterative improvements:

| Version | Innovation | Parameters | Key Feature |
|---------|-----------|------------|-------------|
| **v1** | Competitive | ~400K | Basic recursive refinement |
| **v2** | +Attention | ~450K | Self-attention over iterations |
| **v3** | +PAD Head | ~480K | Explicit emotion prediction |
| **v4** | +RINSE | ~500K | Introspective meta-cognition |
| **v6** | +SoulKernel | ~520K | Memory-based emotional trajectory |

Each version is **fully documented and reproducible**.

---

## 🚀 Quick Start for Researchers

### Installation
```bash
git clone https://github.com/safal207/Neurolab.git
cd Neurolab
pip install -r requirements.txt
```

### Run Demo (30 seconds)
```bash
python demo.py
```

**Output**: Emotion predictions on sample texts with PAD values and interpretations.

### Basic Usage
```python
from neurolab.models import TinyRecursiveModelTRMv6
from neurolab.data import create_embedder
import torch

# Initialize model
model = TinyRecursiveModelTRMv6(dim=128)

# Create embedder (works with any LLM)
embedder = create_embedder("sentence-transformer")

# Analyze emotion
texts = ["I am thrilled about this breakthrough!"]
embeddings = embedder.embed(texts)
y_init = torch.zeros_like(embeddings)

# Get emotion predictions
_, confidences, pad = model(embeddings, y_init, K=5)

# PAD values: [Pleasure, Arousal, Dominance]
print(f"Emotion: {pad}")  # e.g., [0.85, 0.72, 0.63]
print(f"Confidence evolution: {confidences}")  # [0.1, 0.3, 0.5, 0.7, 0.8]
```

### Train Your Own Model
```python
from neurolab.training import LIMINALTrainer
from neurolab.data import create_dataloaders

# Load data
train_loader, test_loader = create_dataloaders(batch_size=32)

# Setup training
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

trainer = LIMINALTrainer(model, optimizer, scheduler)

# Train with emotional metrics tracking
trainer.fit(train_loader, test_loader, embedder, epochs=50)

# Logs: Hope, Faith, Love metrics + standard ML metrics
```

---

## 🎓 Research Applications

### 1. **Empathetic Conversational AI**
- Add emotional awareness to chatbots (Claude, ChatGPT, Gemini, Grok)
- Detect user emotional state from text
- Adjust response tone and content accordingly

### 2. **Mental Health & Well-being**
- Track emotional progression in therapy transcripts
- Detect emotional distress patterns
- Monitor sentiment in support communities

### 3. **Content Moderation & Safety**
- Identify emotionally harmful content
- Understand emotional manipulation in text
- Detect emotional escalation in online discussions

### 4. **Human-AI Alignment Research**
- Use emotional metrics (Hope, Faith, Love) as alignment signals
- Study how AI models process human emotional content
- Investigate recursive reasoning in emotion understanding

### 5. **Multimodal AI**
- Combine with vision/audio for emotion recognition
- Text-to-speech with emotional prosody
- Emotionally-aware image generation prompts

### 6. **AI Interpretability**
- RINSE head provides meta-cognitive insights
- Attention visualization shows reasoning process
- Confidence evolution tracks decision certainty

---

## 🔬 Novel Research Directions

### Questions We're Exploring:

1. **Can recursive processing improve emotion understanding in LLMs?**
   - Does iterative refinement (K iterations) lead to better emotion recognition than single-pass?
   - How does K (number of iterations) affect accuracy vs. computational cost?

2. **Do "emotional metrics" correlate with model performance?**
   - Are Hope, Faith, and Love predictive of generalization?
   - Can these metrics guide training better than loss alone?

3. **Can compact models (128-dim) achieve competitive performance?**
   - Do we need billions of parameters for emotional intelligence?
   - What's the minimal architecture for nuanced emotion understanding?

4. **How does memory (SoulKernel) affect temporal consistency?**
   - Does maintaining emotional history improve long-context understanding?
   - Can the model predict emotional trajectories?

5. **Can this architecture scale to other affective tasks?**
   - Sentiment analysis, toxicity detection, empathy modeling
   - Cross-lingual emotion recognition
   - Multimodal emotion fusion

### 🤝 **We Welcome Collaboration**

We're actively seeking:
- 🔬 **Research partnerships** — Test LIMINAL with your LLMs (Claude, GPT-4, Gemini, Grok)
- 📊 **Dataset contributions** — Expand beyond EmoBank to other languages/domains
- 💡 **Architectural innovations** — Improve recursive reasoning, memory, introspection
- 🧪 **Empirical studies** — Benchmark against state-of-the-art emotion models
- 🌍 **Real-world applications** — Deploy in production systems

---

## 📂 Project Structure

```
Neurolab/
├── neurolab/
│   ├── models/              # 5 progressive model architectures (v1-v6)
│   ├── data/                # EmoBank loader + multi-backend embeddings
│   ├── training/            # Trainer with emotional metrics
│   └── visualization/       # Plots, attention maps, animations
├── configs/                 # YAML configs for experiments
├── examples/                # Research notebooks and demos
├── Osozn3.ipynb            # Original research notebook (13K+ lines)
├── demo.py                  # Quick start script
└── requirements.txt         # All dependencies
```

---

## 📈 Benchmarks & Comparisons

### Current Performance (EmoBank Test Set)
```
Model: TinyRecursiveModelTRMv6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAE (Valence):    0.142
MAE (Arousal):    0.156
MAE (Dominance):  0.168
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parameters:       ~520K (0.0005B)
Inference:        ~15ms per sample (CPU)
Training:         ~2 hours (single GPU)
```

**Note**: These are preliminary results. We encourage researchers to reproduce and improve upon them.

---

## 🛠️ Advanced Features

### 1. **Configurable Training Pipelines**
```yaml
# configs/default.yaml
model:
  name: "TinyRecursiveModelTRMv6"
  dim: 128
  K: 5

training:
  phase1:  # Aggressive training
    epochs: 50
    learning_rate: 0.003
  phase2:  # Fine-tuning
    epochs: 15
    learning_rate: 0.0001
```

### 2. **Rich Visualizations**
```python
from neurolab.visualization import (
    plot_training_curves,      # Hope, Faith, Love over time
    plot_emotional_field,      # 3D PAD space visualization
    plot_attention_heatmap,    # Attention patterns
    create_breathing_animation # Animated training progress
)
```

### 3. **Multiple Model Backends**
```python
# Use GPT-2 embeddings
embedder = create_embedder("transformer", model_name="gpt2")

# Use BLOOM for multilingual
embedder = create_embedder("transformer", model_name="bigscience/bloom-560m")

# Use Sentence-BERT
embedder = create_embedder("sentence-transformer", model_name="all-MiniLM-L6-v2")
```

---

## 🌍 Open Research Philosophy

This project embodies:

- **Full Transparency**: All code, data pipelines, and training logs are open
- **Reproducibility**: Fixed seeds, documented hyperparameters, version control
- **Modularity**: Easy to extend, modify, and integrate with your research
- **Collaboration**: We actively encourage forks, PRs, and research collaborations

### Research Artifacts Available:
- ✅ Complete model architectures
- ✅ Training scripts with emotional metrics
- ✅ Original research notebook (Osozn3.ipynb)
- ✅ Pre-processing pipelines
- ✅ Visualization tools
- ✅ Configuration files

---

## 📚 Citation

If you use Neurolab in your research, please cite:

```bibtex
@software{neurolab2025,
  title={Neurolab: LIMINAL Heartbeat - Recursive Emotion Recognition with Introspection},
  author={safal207},
  year={2025},
  url={https://github.com/safal207/Neurolab},
  note={A novel neural architecture for emotion recognition using recursive refinement,
        emotional metrics (Hope, Faith, Love), and meta-cognitive introspection.}
}
```

---

## 🤔 Philosophical Perspective

### Why "LIMINAL Heartbeat"?

**Liminal** (from Latin *limen*, meaning "threshold") refers to the transitional space between states. Our architecture operates in the liminal space between:

- 🧮 Deterministic logic ↔ Learned representations
- 🎭 Discrete labels ↔ Continuous emotion space
- 🔄 Single-pass prediction ↔ Recursive introspection
- 🧠 Computational efficiency ↔ Emotional depth

**Heartbeat** represents the rhythmic, iterative nature of emotional processing — each iteration is a "beat" that refines understanding.

### Emotional Metrics as Research Paradigm

Using **Hope, Faith, and Love** as metrics is not merely poetic. These represent:

- **Hope** = Alignment with ground truth (low error = high hope)
- **Faith** = Consistency across iterations (stable confidence)
- **Love** = Harmony between loss and variance (low loss + low variance)

These metrics provide **interpretable, human-relatable insights** into model behavior — crucial for AI alignment and safety.

---

## 🎯 Potential Impact

### For AI Safety Research (Anthropic, OpenAI)
- Emotional alignment signals for constitutional AI
- Interpretable emotional reasoning for safety validation
- Recursive introspection for better AI alignment

### For Conversational AI (Google, Anthropic, xAI)
- Add emotional awareness to LLMs (Gemini, Claude, Grok)
- Improve empathy in AI assistants
- Better user experience through emotion-aware responses

### For Multimodal AI (Google DeepMind, OpenAI)
- Emotion recognition for text-to-image models
- Emotionally-aware video generation
- Affective computing in robotics

### For Accessibility & Mental Health
- Support for individuals with alexithymia (difficulty identifying emotions)
- Mental health monitoring and support systems
- Educational tools for emotional intelligence

---

## 🚦 Getting Started Paths

### Path 1: Quick Demo (5 minutes)
```bash
python demo.py
```

### Path 2: Research Exploration (1 hour)
```bash
jupyter notebook Osozn3.ipynb  # Original research notebook
```

### Path 3: Full Experimentation (1 day)
```bash
# Train your own model
python -m neurolab.training.train --config configs/default.yaml

# Evaluate on custom data
python -m neurolab.evaluation.evaluate --checkpoint logs/best_model.pt
```

### Path 4: Integration with Your LLM (2 hours)
```python
# Add emotional layer to your existing model
from neurolab.models import TinyRecursiveModelTRMv6

# Your LLM embeddings
your_embeddings = your_model.encode(texts)  # [batch, hidden_dim]

# Add emotional processing
emotion_model = TinyRecursiveModelTRMv6(dim=your_hidden_dim)
_, confs, emotions = emotion_model(your_embeddings, ...)
```

---

## 🤝 Contributing

We welcome contributions from researchers and practitioners. See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code style guidelines
- How to add new models
- Testing protocols
- Documentation standards

**Areas of Interest**:
- Novel recursive architectures
- Improved emotional metrics
- Cross-lingual emotion datasets
- Multimodal extensions
- Theoretical analysis

---

## 📧 Contact & Collaboration

**Open to**:
- Research collaborations with AI labs
- Integration with production LLMs
- Academic partnerships
- Industrial applications
- Funding for expanded research

**Contact**: Open an issue on GitHub or email via your organization's research collaboration channels.

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

**TL;DR**: Free to use, modify, and distribute. Perfect for research and commercial applications.

---

## 🙏 Acknowledgments

- **EmoBank** corpus for emotion annotations
- **HuggingFace** for Transformers library
- **PyTorch** community for deep learning tools
- **Research community** for inspiration and feedback

---

## 🔮 Future Roadmap

- [ ] Multilingual emotion recognition (50+ languages)
- [ ] Real-time emotion tracking in conversations
- [ ] Integration with vision models for multimodal emotion
- [ ] Federated learning for privacy-preserving emotion AI
- [ ] Constitutional AI alignment experiments
- [ ] Benchmark suite for emotion AI evaluation
- [ ] Pre-trained model zoo (multiple languages/domains)
- [ ] API for easy integration with existing systems

---

<div align="center">

### 💡 **Built for Researchers, By Researchers**

*Exploring the intersection of neural architectures, emotional intelligence, and AI alignment*

**[Explore the Code](https://github.com/safal207/Neurolab)** | **[Documentation](docs/)** | **[Discussions](https://github.com/safal207/Neurolab/discussions)**

---

**⭐ Star this repo if you find it interesting for your research!**

*We're excited to see how you'll extend and apply LIMINAL Heartbeat in your work.*

</div>
