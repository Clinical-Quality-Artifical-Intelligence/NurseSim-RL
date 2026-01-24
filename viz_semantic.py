"""
Visualize "What the Agent Sees" - Semantic Embedding Projection

This script generates a 2D t-SNE visualization of how the RL agent
perceives clinical observations through NurseEmbed semantic vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from nursesim_rl import TriageEnv, NurseEmbedWrapper


def collect_observations(n_observations: int = 200, seed: int = 42):
    """Collect observations and their true categories."""
    
    print(f"[1] Collecting {n_observations} observations...")
    
    # Create semantic environment
    base_env = TriageEnv(max_steps=100, max_patients=50, seed=seed)
    env = NurseEmbedWrapper(base_env, use_vitals=True)
    
    observations = []
    categories = []
    complaints = []
    
    obs, info = env.reset(seed=seed)
    
    for i in range(n_observations):
        # Store observation and metadata
        observations.append(obs.copy())
        
        # Get true category from wrapped env
        if base_env.current_patient:
            categories.append(base_env.current_patient.true_category)
            complaints.append(base_env.current_patient.chief_complaint[:50])
        else:
            categories.append(3)  # Default
            complaints.append("No patient")
        
        # Take a random action to move to next patient
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    print(f"    Collected {len(observations)} observations")
    
    return np.array(observations), np.array(categories), complaints


def visualize_embeddings(observations, categories, complaints, output_path="viz"):
    """Create t-SNE visualization of embeddings."""
    
    print("[2] Computing t-SNE projection...")
    
    # Extract just the embedding part (first 384 dims)
    embeddings = observations[:, :384]
    
    # First reduce with PCA for speed (384 -> 50)
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    print(f"    PCA explained variance: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Then t-SNE to 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    print("[3] Creating visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color map for triage categories
    colors = {
        1: '#FF0000',  # Immediate - Red
        2: '#FF8C00',  # Very Urgent - Orange
        3: '#FFD700',  # Urgent - Yellow
        4: '#32CD32',  # Standard - Green
        5: '#1E90FF',  # Non-urgent - Blue
    }
    
    category_names = {
        1: 'P1: Immediate',
        2: 'P2: Very Urgent',
        3: 'P3: Urgent',
        4: 'P4: Standard',
        5: 'P5: Non-urgent'
    }
    
    # Plot each category
    for cat in sorted(set(categories)):
        mask = categories == cat
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors.get(cat, '#888888'),
            label=category_names.get(cat, f'Category {cat}'),
            alpha=0.7,
            s=100,
            edgecolors='white',
            linewidths=0.5
        )
    
    # Styling
    ax.set_title(
        'What the Agent Sees: Semantic Patient Clusters\n'
        '(t-SNE projection of NurseEmbed vectors)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Semantic Dimension 1', fontsize=11)
    ax.set_ylabel('Semantic Dimension 2', fontsize=11)
    
    # Legend
    ax.legend(loc='upper right', title='Triage Category', fontsize=10)
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(
        0.02, 0.02,
        'Patients with similar clinical presentations\ncluster together in semantic space.',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'semantic_clusters.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[4] Saved to: {output_file}")
    
    # Also show statistics
    print("\n[5] Cluster Statistics:")
    for cat in sorted(set(categories)):
        mask = categories == cat
        center = embeddings_2d[mask].mean(axis=0)
        print(f"    {category_names[cat]}: {mask.sum()} patients, center at ({center[0]:.1f}, {center[1]:.1f})")
    
    return output_file


def main():
    print("=" * 60)
    print("SEMANTIC EMBEDDING VISUALIZATION")
    print("'What the Agent Sees'")
    print("=" * 60 + "\n")
    
    # Collect data
    observations, categories, complaints = collect_observations(n_observations=150)
    
    # Visualize
    output_file = visualize_embeddings(observations, categories, complaints)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print(f"Open: {os.path.abspath(output_file)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
