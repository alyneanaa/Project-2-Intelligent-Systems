"""
OPTIMIZED Intrusion Detection with CNN-LSTM
Training for FULL 100 EPOCHS - Synchronized Train/Val Curves

Configuration:
- Full 100 epochs training with synchronized train/val metrics
- Very aggressive dropout and regularization
- Smaller model to reduce memorization
- Lower learning rate for stable convergence
- Focus: Train and Val curves should move together
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Conv1D, LSTM, 
                         BatchNormalization, MaxPooling1D, Bidirectional)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
import seaborn as sns
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# ============================================
# TRAINING CONFIGURATION
# ============================================
FULL_EPOCHS = 100  # Train for full 100 epochs
USE_EARLY_STOPPING = False  # Set to True to enable early stopping
EARLY_STOP_PATIENCE = 20  # Only used if USE_EARLY_STOPPING = True
BATCH_SIZE = 32  # Smaller batch size for more frequent updates
LEARNING_RATE = 0.00025  # Much lower learning rate for stability
VALIDATION_SPLIT = 0.2

print("="*80)
print("INTRUSION DETECTION - SYNCHRONIZED TRAIN/VAL CURVES")
print("="*80)
print(f"\n⚙️  Training Configuration:")
print(f"   Epochs: {FULL_EPOCHS}")
print(f"   Early Stopping: {'Enabled' if USE_EARLY_STOPPING else 'Disabled'}")
print(f"   Batch Size: {BATCH_SIZE} (smaller = more stable)")
print(f"   Learning Rate: {LEARNING_RATE} (very low for sync)")
print(f"   Validation Split: {VALIDATION_SPLIT}")
print(f"   Focus: Synchronized Train/Val Curves")

# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================
print("\n" + "="*80)
print("📊 STEP 1: Load Dataset")
print("="*80 + "\n")

try:
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    print(f"✅ Training set: {train_df.shape}")
    
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')
    print(f"✅ Testing set: {test_df.shape}")
    
except FileNotFoundError:
    print("❌ Files not found! Download from:")
    print("https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
    exit(1)

# Check for label
if 'label' not in train_df.columns:
    possible = ['Label', 'attack_cat', 'Attack']
    for col in possible:
        if col in train_df.columns:
            train_df['label'] = train_df[col]
            test_df['label'] = test_df[col]
            print(f"Using '{col}' as label")
            break

print(f"\n🎯 Label Distribution:")
print("Training:")
print(train_df['label'].value_counts())
print("\nTesting:")
print(test_df['label'].value_counts())

# Convert to numeric if needed
if train_df['label'].dtype == 'object':
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])
    test_df['label'] = le.transform(test_df['label'])

# =============================================================================
# STEP 2: PREPROCESSING
# =============================================================================
print("\n" + "="*80)
print("🔧 STEP 2: Preprocessing")
print("="*80 + "\n")

def preprocess(df):
    df = df.copy()
    
    # Remove ID and attack_cat
    remove_cols = ['id', 'Id', 'ID', 'attack_cat']
    remove_cols = [c for c in remove_cols if c in df.columns]
    if remove_cols:
        df = df.drop(columns=remove_cols)
    
    # Handle missing
    for col in df.columns:
        if col == 'label':
            continue
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [c for c in cat_cols if c != 'label']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

train_processed = preprocess(train_df)
test_processed = preprocess(test_df)

print(f"✅ Train: {train_processed.shape}")
print(f"✅ Test: {test_processed.shape}")

# Separate X and y
X_train = train_processed.drop('label', axis=1).values
y_train = train_processed['label'].values
X_test = test_processed.drop('label', axis=1).values
y_test = test_processed['label'].values

print(f"\nX_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Normalized!")

# =============================================================================
# STEP 3: CREATE SEQUENCES
# =============================================================================
print("\n" + "="*80)
print("📦 STEP 3: Create Sequences")
print("="*80 + "\n")

SEQUENCE_LENGTH = 10

def create_sequences(X, y, seq_len):
    n = len(X) - seq_len + 1
    X_seq = np.zeros((n, seq_len, X.shape[1]))
    y_seq = np.zeros(n)
    
    for i in range(n):
        X_seq[i] = X[i:i + seq_len]
        y_seq[i] = np.max(y[i:i + seq_len])
    
    return X_seq, y_seq

print(f"Creating sequences (length={SEQUENCE_LENGTH})...")
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQUENCE_LENGTH)

print(f"✅ Train: {X_train_seq.shape}")
print(f"✅ Test: {X_test_seq.shape}")

# =============================================================================
# STEP 4: BUILD MODEL - VERY SMALL & SIMPLE
# =============================================================================
print("\n" + "="*80)
print("🏗️  STEP 4: Build Model (Small & Simple for Sync)")
print("="*80 + "\n")

n_features = X_train_seq.shape[2]

model = Sequential([
    # Conv Block 1 - Very conservative
    Conv1D(32, 3, activation='relu', padding='same',
           kernel_regularizer=l2(0.02),  # Very strong regularization
           input_shape=(SEQUENCE_LENGTH, n_features)),
    BatchNormalization(),
    Dropout(0.6),  # Very high dropout
    
    # Conv Block 2
    Conv1D(16, 3, activation='relu', padding='same',
           kernel_regularizer=l2(0.02)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.6),
    
    # LSTM Block 1 - Small and simple
    Bidirectional(LSTM(16, return_sequences=True)),  # Very small
    Dropout(0.6),
    
    # LSTM Block 2
    Bidirectional(LSTM(8)),  # Minimal capacity
    Dropout(0.6),
    
    # Dense Block 1 - Minimal
    Dense(16, activation='relu', kernel_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.6),
    
    # Output
    Dense(1, activation='sigmoid')
])

print("✅ Model built!")
model.summary()

# =============================================================================
# STEP 5: COMPILE
# =============================================================================
print("\n" + "="*80)
print("⚙️  STEP 5: Compile")
print("="*80 + "\n")

# Class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_seq),
    y=y_train_seq
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

print(f"Class weights: {class_weight_dict}")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
print("✅ Compiled!")

# =============================================================================
# STEP 6: TRAIN FOR FULL 100 EPOCHS
# =============================================================================
print("\n" + "="*80)
print(f"🚀 STEP 6: Train for {FULL_EPOCHS} Epochs")
print("="*80 + "\n")

# Setup callbacks
callbacks = []

# Always save best model
callbacks.append(
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
)

# Very conservative learning rate reduction
callbacks.append(
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.8,  # Smaller reduction factor
        patience=8,  # More patience
        min_lr=1e-8,
        verbose=1
    )
)

# Optional early stopping
if USE_EARLY_STOPPING:
    print(f"⚠️  Early stopping ENABLED (patience={EARLY_STOP_PATIENCE})")
    callbacks.append(
        EarlyStopping(
            monitor='val_auc',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
    )
else:
    print("✅ Early stopping DISABLED - Will train full 100 epochs")

print(f"\n🏃 Starting training for {FULL_EPOCHS} epochs...")
print(f"   Batch size: {BATCH_SIZE} (small for stability)")
print(f"   Learning rate: {LEARNING_RATE} (very low)")
print(f"   Validation split: {VALIDATION_SPLIT}")
print(f"   Training samples: {len(X_train_seq):,}")
print(f"   Steps per epoch: {len(X_train_seq) // BATCH_SIZE}")
print(f"   Focus: Synchronized Train/Val curves")
print("\n" + "-"*80 + "\n")

history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=VALIDATION_SPLIT,
    epochs=FULL_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Count actual epochs trained
epochs_trained = len(history.history['loss'])
print("\n" + "-"*80)
print(f"✅ Training complete!")
print(f"   Epochs completed: {epochs_trained}/{FULL_EPOCHS}")

if epochs_trained < FULL_EPOCHS:
    print(f"   Early stopped at epoch {epochs_trained}")
else:
    print(f"   Trained full {FULL_EPOCHS} epochs!")

# Save final model (after all epochs)
print("\n💾 Saving final model...")
model.save('final_model_100epochs.keras')
print("✅ Saved: final_model_100epochs.keras")
print("✅ Saved: best_model.keras (best during training)")

# =============================================================================
# STEP 7: EVALUATE
# =============================================================================
print("\n" + "="*80)
print("📊 STEP 7: Evaluate")
print("="*80 + "\n")

# Calculate sync metrics - correlation and differences
train_acc = np.array(history.history['accuracy'])
val_acc = np.array(history.history['val_accuracy'])
train_loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

# Correlation (should be close to 1 for synchronized curves)
acc_correlation = np.corrcoef(train_acc, val_acc)[0, 1]
loss_correlation = np.corrcoef(train_loss, val_loss)[0, 1]

# Average difference
avg_acc_diff = np.mean(np.abs(train_acc - val_acc))
avg_loss_diff = np.mean(np.abs(train_loss - val_loss))

print(f"📈 Curve Synchronization Analysis:")
print(f"   Accuracy Correlation (target: ~1.0): {acc_correlation:.4f}")
print(f"   Loss Correlation (target: ~1.0): {loss_correlation:.4f}")
print(f"   Average Accuracy Difference: {avg_acc_diff:.4f}")
print(f"   Average Loss Difference: {avg_loss_diff:.4f}")
print(f"   (Higher correlation = better synchronized curves)")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2.5, marker='o', markersize=3)
axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2.5, marker='s', markersize=3)
axes[0, 0].set_title(f'Model Accuracy (Correlation: {acc_correlation:.4f})', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2.5, marker='o', markersize=3)
axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2.5, marker='s', markersize=3)
axes[0, 1].set_title(f'Model Loss (Correlation: {loss_correlation:.4f})', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# AUC
axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2.5, marker='o', markersize=3)
axes[1, 0].plot(history.history['val_auc'], label='Val', linewidth=2.5, marker='s', markersize=3)
axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Synchronization info
axes[1, 1].text(0.5, 0.5, 
               f'Training Summary\n\n'
               f'Total Epochs: {epochs_trained}\n'
               f'Batch Size: {BATCH_SIZE}\n'
               f'Learning Rate: {LEARNING_RATE}\n'
               f'Early Stop: {"Yes" if USE_EARLY_STOPPING else "No"}\n\n'
               f'Final Train Acc: {history.history["accuracy"][-1]:.4f}\n'
               f'Final Val Acc: {history.history["val_accuracy"][-1]:.4f}\n'
               f'Acc Difference: {abs(history.history["accuracy"][-1] - history.history["val_accuracy"][-1]):.4f}\n\n'
               f'Acc Correlation: {acc_correlation:.4f}\n'
               f'Loss Correlation: {loss_correlation:.4f}',
               transform=axes[1, 1].transAxes,
               fontsize=11,
               verticalalignment='center',
               horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('training_history_100epochs.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: training_history_100epochs.png")

# Predictions on test set
print("\n🔮 Testing on test set...")
y_pred_prob = model.predict(X_test_seq, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test_seq, y_pred)
precision = precision_score(y_test_seq, y_pred)
recall = recall_score(y_test_seq, y_pred)
f1 = f1_score(y_test_seq, y_pred)
auc_score = roc_auc_score(y_test_seq, y_pred_prob)

print("\n" + "="*80)
print("🎯 FINAL TEST RESULTS")
print("="*80)
print(f"\n✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"✅ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"✅ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"✅ ROC-AUC:   {auc_score:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test_seq, y_pred, target_names=['Normal', 'Attack'], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test_seq, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Normal', 'Attack'],
           yticklabels=['Normal', 'Attack'])
plt.title(f'Confusion Matrix (After {epochs_trained} Epochs)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_100epochs.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: confusion_matrix_100epochs.png")

print(f"\n📊 Confusion Matrix Breakdown:")
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives:  {cm[1,1]:,}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_seq, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (After {epochs_trained} Epochs)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_100epochs.png', dpi=200)
plt.show()
print("✅ Saved: roc_curve_100epochs.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Training Summary:")
print(f"  ✓ Epochs Trained: {epochs_trained}/{FULL_EPOCHS}")
print(f"  ✓ Early Stopping: {'Used' if epochs_trained < FULL_EPOCHS else 'Not used'}")
print(f"  ✓ Best Val AUC: {max(history.history['val_auc']):.4f}")
print(f"  ✓ Final Val AUC: {history.history['val_auc'][-1]:.4f}")

print(f"\n🔗 Curve Synchronization (Higher = Better):")
print(f"  ✓ Accuracy Correlation: {acc_correlation:.4f} (target: ~1.0)")
print(f"  ✓ Loss Correlation: {loss_correlation:.4f} (target: ~1.0)")
print(f"  ✓ Avg Accuracy Gap: {avg_acc_diff:.4f}")
print(f"  ✓ Avg Loss Gap: {avg_loss_diff:.4f}")

print(f"\n📈 Test Performance:")
print(f"  ✓ Accuracy:  {accuracy*100:.2f}%")
print(f"  ✓ Precision: {precision*100:.2f}%")
print(f"  ✓ Recall:    {recall*100:.2f}%")
print(f"  ✓ F1-Score:  {f1*100:.2f}%")
print(f"  ✓ ROC-AUC:   {auc_score:.4f}")

print(f"\n📁 Generated Files:")
print("  ✓ training_history_100epochs.png")
print("  ✓ confusion_matrix_100epochs.png")
print("  ✓ roc_curve_100epochs.png")
print("  ✓ final_model_100epochs.keras (model after all epochs)")
print("  ✓ best_model.keras (best model during training)")

print("\n✅ All done!")
print("="*80)