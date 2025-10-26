"""
LIMINAL Heartbeat - Demo Script

Quick demonstration of emotion recognition using LIMINAL models.
"""

import torch
from neurolab.models import TinyRecursiveModelTRMv6
from neurolab.data import create_embedder, ProjectionLayer
from neurolab.visualization import plot_emotional_field, plot_pad_distribution

# Example texts with different emotional content
EXAMPLE_TEXTS = [
    "I am so happy and excited about this wonderful day!",
    "This is terrible and makes me very angry.",
    "I feel calm and peaceful, everything is okay.",
    "I'm worried and anxious about what might happen.",
    "This is absolutely amazing, I feel fantastic!",
    "I'm so sad and disappointed about this outcome.",
    "I feel neutral about this, neither good nor bad.",
    "I'm energized and ready to conquer the world!",
]


def main():
    print("=" * 60)
    print("LIMINAL Heartbeat - Emotion Recognition Demo")
    print("=" * 60)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create embedder
    print("\nLoading embedding model...")
    embedder = create_embedder(
        embedder_type="sentence-transformer",
        model_name="all-MiniLM-L6-v2",
        device=device,
    )

    # Create projection layer
    embedding_dim = embedder.embedding_dim
    projection = ProjectionLayer(input_dim=embedding_dim, output_dim=128).to(device)
    print(f"Embedding dimension: {embedding_dim} → 128")

    # Create model
    print("\nInitializing LIMINAL model (v6)...")
    model = TinyRecursiveModelTRMv6(dim=128, affect_w=0.3).to(device)
    model.eval()

    # Generate embeddings
    print("\nGenerating embeddings for example texts...")
    with torch.no_grad():
        embeddings = embedder.embed(EXAMPLE_TEXTS)
        embeddings = projection(embeddings)

    # Run emotion recognition
    print("\nRunning emotion recognition...")
    y_init = torch.zeros_like(embeddings)
    affect_vec = torch.zeros(len(EXAMPLE_TEXTS), 3).to(device)  # Neutral affect

    with torch.no_grad():
        y_output, confidences, pad_predictions = model(
            embeddings, y_init, affect_vec, K=5
        )

    # Display results
    print("\n" + "=" * 60)
    print("Emotion Recognition Results (PAD Model)")
    print("=" * 60)
    print("\nPAD Dimensions:")
    print("  P (Pleasure): Positive (1) vs Negative (-1)")
    print("  A (Arousal):  Excited (1) vs Calm (-1)")
    print("  D (Dominance): Dominant (1) vs Submissive (-1)")
    print("\n" + "-" * 60)

    pad_np = pad_predictions.cpu().numpy()

    for i, text in enumerate(EXAMPLE_TEXTS):
        pleasure, arousal, dominance = pad_np[i]
        conf = confidences[i] if i < len(confidences) else 0.0

        print(f"\n{i+1}. \"{text}\"")
        print(f"   P: {pleasure:+.3f} | A: {arousal:+.3f} | D: {dominance:+.3f}")
        print(f"   Confidence: {conf:.3f}")

        # Interpret emotion
        emotion_label = interpret_pad(pleasure, arousal, dominance)
        print(f"   → {emotion_label}")

    # Compute statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Average Pleasure:  {pad_np[:, 0].mean():+.3f} (±{pad_np[:, 0].std():.3f})")
    print(f"Average Arousal:   {pad_np[:, 1].mean():+.3f} (±{pad_np[:, 1].std():.3f})")
    print(f"Average Dominance: {pad_np[:, 2].mean():+.3f} (±{pad_np[:, 2].std():.3f})")

    # Visualize
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    # Plot PAD distribution
    plot_pad_distribution(pad_predictions, save_path="demo_pad_distribution.png")
    print("✓ Saved: demo_pad_distribution.png")

    # Plot emotional field (need targets for comparison, using predictions as proxy)
    plot_emotional_field(
        pad_predictions, pad_predictions, save_path="demo_emotional_field.png", show=False
    )
    print("✓ Saved: demo_emotional_field.png")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def interpret_pad(pleasure, arousal, dominance, threshold=0.3):
    """
    Interpret PAD values into emotion labels.

    Args:
        pleasure (float): Pleasure dimension (-1 to 1)
        arousal (float): Arousal dimension (-1 to 1)
        dominance (float): Dominance dimension (-1 to 1)
        threshold (float): Threshold for classification

    Returns:
        str: Emotion label
    """
    # High pleasure emotions
    if pleasure > threshold:
        if arousal > threshold:
            if dominance > threshold:
                return "Excited / Exuberant"
            else:
                return "Happy / Joyful"
        else:
            if dominance > threshold:
                return "Relaxed / Content"
            else:
                return "Peaceful / Serene"

    # Low pleasure emotions
    elif pleasure < -threshold:
        if arousal > threshold:
            if dominance > threshold:
                return "Angry / Furious"
            else:
                return "Anxious / Fearful"
        else:
            if dominance > threshold:
                return "Annoyed / Irritated"
            else:
                return "Sad / Depressed"

    # Neutral pleasure
    else:
        if arousal > threshold:
            return "Alert / Surprised"
        elif arousal < -threshold:
            return "Calm / Bored"
        else:
            return "Neutral"


if __name__ == "__main__":
    main()
