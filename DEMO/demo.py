import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import json
import time
import os

# ===== Paths =====
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "real_project", "saved_parameters.json")
model_path = os.path.join(base_dir, "..", "real_project", "saved_model.pth")
data_path = os.path.join(base_dir, "..", "real_project", "new_merged.csv")

# ===== Load hyperparameters =====
with open(json_path, "r") as f:
    hyperparams = json.load(f)

input_dim = hyperparams.get("input_dim", 50)
hidden_dim = hyperparams.get("hidden_dim", 64)
num_layers = hyperparams.get("num_layers", 2)
dropout = hyperparams.get("dropout", 0.3)
use_attention = hyperparams.get("use_attention", True)
bidirectional = hyperparams.get("bidirectional", True)
sequence_length = hyperparams.get("sequence_length", 20)

# ===== AdvancedLSTM definition =====
class AdvancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3,
                 use_attention=True, bidirectional=True):
        super().__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            features = attended_out.mean(dim=1)
        else:
            features = lstm_out[:, -1, :]
        logits = self.classifier(features)
        return logits.squeeze(-1)

# ===== Load model =====
model = AdvancedLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    use_attention=use_attention,
    bidirectional=bidirectional
)
checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# ===== Streamlit UI =====
st.title("🔍 Intrusion Detection Streaming Demo")
st.write(f"**Model Configuration:** {sequence_length} timestep sequences, {input_dim} features")

# ===== Load dataset =====
if not os.path.exists(data_path):
    st.error(f"❌ Dataset not found at {data_path}")
    st.stop()

df = pd.read_csv(data_path)
st.write("### Dataset Preview")
st.dataframe(df.head())

if df.shape[1] != input_dim + 1:
    st.error(f"❌ Expected {input_dim + 1} columns (features + label), but CSV has {df.shape[1]} columns")
    st.stop()

features_df = df.iloc[:, :input_dim]
labels_df = df.iloc[:, input_dim]

st.write("### Class Distribution")
class_counts = labels_df.value_counts()
st.write(f"Benign: {class_counts.get(0, 0):,} ({100*class_counts.get(0, 0)/len(labels_df):.1f}%)")
st.write(f"Attack: {class_counts.get(1, 0):,} ({100*class_counts.get(1, 0)/len(labels_df):.1f}%)")

# ===== Normalize features =====
features_normalized = (features_df - features_df.mean()) / (features_df.std() + 1e-8)

# ===== Streaming Controls =====
st.write("### Streaming Controls")
col1, col2, col3 = st.columns(3)
with col1:
    streaming_speed = st.selectbox("Streaming Speed", options=[0.1, 0.3, 0.5, 1.0, 2.0], index=2,
                                   format_func=lambda x: f"{x}s per sample")
with col2:
    max_samples = st.number_input("Max Samples to Process", min_value=50, max_value=len(features_df),
                                  value=min(1000, len(features_df)))
with col3:
    skip_consecutive_after = st.number_input("Skip After N Consecutive Displayed", min_value=5, max_value=100,
                                             value=20)

# ===== Streaming Implementation =====
if st.button("🚀 Start Streaming Detection"):
    st.write("---")
    # Containers
    metrics_container = st.container()
    results_container = st.container()
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        accuracy_metric = col1.empty()
        precision_metric = col2.empty()
        recall_metric = col3.empty()
        f1_metric = col4.empty()

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_placeholder = results_container.empty()

    # Sliding window and results
    window_buffer = []
    displayed_results = []

    # Metrics
    correct_predictions = 0
    total_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    last_displayed_label = None
    consecutive_displayed = 0
    samples_displayed = 0

    for i in range(len(features_df)):
        row = features_normalized.iloc[i]
        true_label = int(labels_df.iloc[i])

        # Add to sliding window
        window_buffer.append(row.values)
        if len(window_buffer) > sequence_length:
            window_buffer.pop(0)

        if len(window_buffer) < sequence_length:
            continue

        # Model inference
        x = torch.tensor(window_buffer, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred_logit = model(x)
            y_pred_prob = torch.sigmoid(y_pred_logit).item()
            threshold = 0.3
            pred_label = 1 if y_pred_prob > threshold else 0

        # Honest metrics
        is_correct = pred_label == true_label
        correct_predictions += int(is_correct)
        total_predictions += 1
        true_positives += int(pred_label == 1 and true_label == 1)
        false_positives += int(pred_label == 1 and true_label == 0)
        false_negatives += int(pred_label == 0 and true_label == 1)

        # Decide whether to display
        display_this = True
        if last_displayed_label == pred_label:
            consecutive_displayed += 1
            if consecutive_displayed > skip_consecutive_after:
                display_this = False
        else:
            consecutive_displayed = 1
            last_displayed_label = pred_label

        if display_this:
            samples_displayed += 1
            displayed_results.append({
                "Sample": samples_displayed,
                "True": "🟢 Benign" if true_label == 0 else "🔴 Attack",
                "Predicted": "🟢 Benign" if pred_label == 0 else "🔴 Attack",
                "Confidence": f"{y_pred_prob:.3f}",
                "Status": "✅" if is_correct else "❌"
            })

        # Update metrics UI
        current_accuracy = (correct_predictions / total_predictions) * 100
        precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) else 0
        recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

        accuracy_metric.metric("Accuracy", f"{current_accuracy:.1f}%", f"{correct_predictions}/{total_predictions}")
        precision_metric.metric("Precision", f"{precision:.1f}%")
        recall_metric.metric("Recall", f"{recall:.1f}%")
        f1_metric.metric("F1-Score", f"{f1:.1f}")

        # Display all streamed sequences
        if displayed_results:
            display_df = pd.DataFrame(displayed_results)
            results_placeholder.dataframe(display_df, use_container_width=True)

        progress_bar.progress(min(samples_displayed / max_samples, 1.0))
        status_text.text(f"🔍 Processed {samples_displayed}/{max_samples} displayed samples | Row {i+1}/{len(features_df)}")

        time.sleep(streaming_speed)

        if samples_displayed >= max_samples:
            break

    # ===== Final Results =====
    st.write("---")
    st.success("🎉 Streaming Detection Complete!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples (sequences)", total_predictions)
    with col2:
        st.metric("Final Accuracy", f"{current_accuracy:.1f}%")
    with col3:
        attack_total = true_positives + false_negatives
        st.metric("Attack Detection Rate", f"{(true_positives/attack_total*100):.1f}%" if attack_total else "No attacks")

    st.write("### Detection Breakdown")
    breakdown_df = pd.DataFrame({
        'Metric': ['True Positives', 'False Positives', 'False Negatives', 'True Negatives'],
        'Count': [
            true_positives,
            false_positives,
            false_negatives,
            total_predictions - true_positives - false_positives - false_negatives
        ]
    })
    st.dataframe(breakdown_df, use_container_width=True)

    # ===== Download results =====
    csv = pd.DataFrame(displayed_results).to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results",
        data=csv,
        file_name=f"detection_results_{int(time.time())}.csv",
        mime="text/csv"
    )
