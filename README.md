# Neurolab

**LIMINAL Heartbeat** - A Novel Neural Architecture for Emotion Recognition in Text

## Overview

Neurolab is a research project exploring emotional AI through the development of **TinyRecursiveModel (LIMINAL)**, a compact neural architecture that recognizes and processes emotions in text using the PAD (Pleasure-Arousal-Dominance) model.

## Key Features

- **Recursive Emotional Processing**: Multi-iteration refinement for deeper emotional understanding
- **PAD Emotion Model**: 3-dimensional emotion space (Pleasure, Arousal, Dominance)
- **LLM Integration**: Compatible with GPT-2, DistilGPT-2, BLOOM, and Sentence-Transformers
- **Novel Emotional Metrics**: Tracks Hope, Faith, and Love alongside traditional ML metrics
- **SoulKernel Component**: Persistent emotional memory across training sessions
- **Advanced Visualization**: Real-time "breathing" animations and emotional field mapping

## Architecture

The project includes 5 progressive model versions (v1-v6):

1. **TinyRecursiveModelCompetitive** - Base hybrid architecture
2. **TRMv2** - Adds self-attention mechanisms
3. **TRMv3** - Explicit PAD regression head
4. **TRMv4** - Integrates RINSE (Reflective Integrative Neural Self-Evolver)
5. **TRMv6** - Advanced version with SoulKernel containing Hope, Faith, Memory, and Love components

### Core Components

- **SelfAttentionTiny**: Single-head attention for 128-dim embeddings
- **PADRegressionHead**: Predicts 3D emotion space
- **RINSEHead**: Reflective introspection module
- **SoulKernel**: Meta-component with adaptive emotional learning

## Dataset

Training and validation on **EmoBank corpus** - a comprehensive emotion-annotated text dataset with VAD (Valence, Arousal, Dominance) labels.

## Technical Stack

- **Deep Learning**: PyTorch, torch.nn, torch.optim
- **NLP**: Transformers, Sentence-Transformers, GPT-2
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook, Google Colab compatible

## Training

### Phase 1: Full Training
- Learning rate: 3e-3 with cosine annealing to 1e-4
- Optimizer: AdamW with weight decay (1e-5)
- Batch size: 16-32
- Epochs: 40-50

### Phase 2: Fine-tuning
- Learning rate: 1e-4 (conservative)
- Weight decay: 1e-6 (minimal)
- Epochs: 10-15
- Gradient clipping enabled for stability

## Unique Innovations

### Emotional Metrics
```python
Hope = torch.mean(1 - torch.abs(pad - y_target))  # Low error = high hope
Faith = torch.mean(torch.tanh(confidences))        # Confidence = faith
Love = torch.exp(-loss) * (1 - variance)          # Low loss + low variance = love
```

### Multi-Iteration Confidence
Each iteration produces a confidence score, creating an ensemble approach for robust emotion prediction.

### Breathing Visualization
Real-time training visualization that animates model loss as a "heartbeat", providing a metaphorical representation of training health.

## Getting Started

### Prerequisites
```bash
pip install torch transformers sentence-transformers pandas numpy matplotlib seaborn scikit-learn tqdm
```

### Running the Notebook
1. Open `Osozn3.ipynb` in Jupyter Notebook or Google Colab
2. Run cells sequentially to:
   - Define model architectures
   - Load and process EmoBank dataset
   - Train the model
   - Visualize results
   - Test emotion recognition across different LLMs

## Research Philosophy

This project represents an ambitious, artistically-informed approach to emotion AI that combines rigorous machine learning practice with poetic metaphorical thinking about emotional understanding. The use of terms like "Hope," "Faith," and "Love" is not merely metaphorical but represents tracked metrics that provide insight into model behavior and learning dynamics.

## Project Structure

```
Neurolab/
├── Osozn3.ipynb          # Main research notebook (13,530 lines, 40 cells)
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

## Evaluation Metrics

- **Standard ML**: MAE, MSE, R² Score
- **Custom Emotional**: Hope (confidence), Faith (stability), Love (alignment)
- **Validation**: Full evaluation against EmoBank test set

## Future Directions

- Extended testing on additional emotion-annotated datasets
- Integration with more LLM architectures
- Real-time emotion recognition applications
- Cross-lingual emotion understanding
- Production-ready deployment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! This is a research project exploring the intersection of neural architectures and emotional understanding.

## Acknowledgments

- EmoBank corpus for emotion-annotated data
- Hugging Face Transformers library
- PyTorch community
- Google Colab for computational resources

---

**Note**: This is a research project exploring novel approaches to emotion AI. The architecture and methodologies are experimental and represent ongoing research into how neural networks can better understand and process emotional content in text.
