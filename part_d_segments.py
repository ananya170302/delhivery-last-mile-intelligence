import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
import numpy as np
from sklearn.metrics import pairwise_distances


# ==========================================
# TASK 1 — HUB PROFILE
# ==========================================

# Load historical modelling data
df = pd.read_csv("data/modelling_table.csv")

# Group shipments by hub and calculate hub-level metrics
hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("hub_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
)

# Sort hubs in ascending order of hub_id
hub_profile = hub_profile.sort_values(by="hub_id").reset_index(drop=True)

# Save hub profile
hub_profile.to_csv("outputs/hub_profile.csv", index=False)

# Display results
print("\nTASK 1 — HUB PROFILE")
print("-" * 50)

print(f"Total hubs: {len(hub_profile)}")
print(f"Shipments per hub: {hub_profile['shipments'].unique()}")

print("\nHub Profile:")
print(hub_profile.to_string(index=False))







# ==========================================
# TASK 2 — K-MEANS HUB SEGMENTATION
# ==========================================

# Select hub-level metrics for clustering
clustering_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

# Prepare the clustering dataset
X = hub_profile[clustering_features]

# Scale the features to bring them to a comparable range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering with exactly 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Assign cluster labels to each hub
hub_profile["cluster"] = kmeans.fit_predict(X_scaled)

# Display cluster assignments
print("\nTASK 2 — K-MEANS HUB SEGMENTATION")
print("-" * 50)

print("\nCluster sizes:")
print(
    hub_profile["cluster"]
    .value_counts()
    .sort_index()
)

print("\nHub-to-cluster assignments:")
print(
    hub_profile[["hub_id", "cluster"]]
    .to_string(index=False)
)







# ==========================================
# TASK 3 — CLUSTER SEPARATION EVIDENCE
# ==========================================

# Calculate pairwise Euclidean distances between all hubs
distance_matrix = pairwise_distances(
    X_scaled,
    metric="euclidean"
)

# Evaluate different values of K
separation_results = []

for k in [2, 3, 4, 5]:

    # Fit K-Means for the current K
    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans_test.fit_predict(X_scaled)

    # Store distances for within-cluster and across-cluster pairs
    within_distances = []
    across_distances = []

    # Evaluate all unordered hub pairs
    for i, j in combinations(range(len(hub_profile)), 2):

        distance = distance_matrix[i, j]

        if cluster_labels[i] == cluster_labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    # Calculate mean distances
    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    # Calculate separation ratio
    separation_ratio = mean_across / mean_within

    # Calculate cluster sizes in descending order
    cluster_sizes = sorted(
        pd.Series(cluster_labels).value_counts().tolist(),
        reverse=True
    )

    cluster_sizes_str = "-".join(
        map(str, cluster_sizes)
    )

    # Store results
    separation_results.append({
        "K": k,
        "across_over_within": separation_ratio,
        "cluster_sizes": cluster_sizes_str
    })


# Create separation results DataFrame
separation_df = pd.DataFrame(separation_results)

# Save separation results
separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False
)

# Display results
print("\nTASK 3 — CLUSTER SEPARATION EVIDENCE")
print("-" * 50)

print(separation_df.to_string(index=False))









# ==========================================
# TASK 4 — SEGMENT NAMES AND TIER CHECK
# ==========================================

# Calculate average delay rate for each cluster
cluster_delay_rates = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

# Map cluster numbers to operational segment names
segment_names = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate"
}

# Assign operational segment names to each hub
hub_profile["segment"] = hub_profile["cluster"].map(segment_names)

# Load hub reference data
hubs = pd.read_csv("data/hubs.csv")

# Merge hub city and tier information
hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)

# Save hub-to-segment mapping
hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False
)

# Create segment-level profile using hub-level averages
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

# Arrange segments in operational order
segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = (
    segment_profile.sort_values("segment")
    .reset_index(drop=True)
)

# Save segment-level profile
segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False
)

# Define the expected segment for each hub tier
tier_segment_mapping = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

# Identify hubs whose actual segment differs from their tier expectation
hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(tier_segment_mapping)
)

tier_mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
][["hub_id", "segment", "hub_city", "hub_tier"]]

# Save tier-versus-segment mismatches
tier_mismatch.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False
)

# Display results
print("\nTASK 4 — SEGMENT NAMES AND TIER CHECK")
print("-" * 50)

print("\nCluster delay rates:")
print(cluster_delay_rates)

print("\nHub Segments:")
print(hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]].to_string(index=False))

print("\nSegment Profile:")
print(segment_profile.to_string(index=False))

print("\nTier vs Segment Mismatches:")
print(tier_mismatch.to_string(index=False))