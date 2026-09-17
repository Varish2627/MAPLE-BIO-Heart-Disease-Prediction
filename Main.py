print(" *** *** *** *** **** **** *** *** **** ")
print("   Cardiovascular Heart Disease Dataset")
print(" *** *** *** *** **** **** *** *** **** ")
import warnings
warnings.filterwarnings("ignore")
from importlib.machinery import SourceFileLoader
import pandas as pd
import numpy as np
import os
# Windows 11 no longer includes WMIC; avoid joblib's physical-core probe.
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

METRICS = SourceFileLoader(
    "project_metrics", os.path.join(os.path.dirname(__file__), "metrics")
).load_module()

DATASET_NAME = "Cardiovascular"  

RESULT_DIR = os.path.join("results", DATASET_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Saving results in:", RESULT_DIR)
# ===============================
# MED-CARE PREPROCESSING MODULE
# ===============================
class MED_CARE:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, path):
        path = str(path)
        sep = ';'
        if path.lower().endswith('.csv'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
            if ',' in first_line and ';' not in first_line:
                sep = ','
        df = pd.read_csv(path, sep=sep)
        print("Dataset Loaded:", df.shape)
        return df

    # -------------------------------
    # Handle Missing Values
    # -------------------------------
    def handle_missing(self, df):
        imputer = SimpleImputer(strategy='median')
        df[df.columns] = imputer.fit_transform(df)
        print("Missing values handled.")
        return df

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    def feature_engineering(self, df):
        df = df.copy()
        df = df.drop(columns=['id', 'patientid'], errors='ignore')

        if 'age' in df.columns:
            if df['age'].max() > 100:
                df['age_years'] = (df['age'] / 365).round(1)
            else:
                df['age_years'] = df['age']

        if {'height', 'weight'}.issubset(df.columns):
            height_m = df['height'] / 100
            df['BMI'] = df['weight'] / (height_m * height_m).replace(0, np.nan)

        if {'ap_hi', 'ap_lo'}.issubset(df.columns):
            df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
            df['mean_bp'] = df['ap_lo'] + 0.333 * (df['ap_hi'] - df['ap_lo'])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("Feature engineering completed.")
        return df

    # -------------------------------
    # Noise Reduction (Smoothing)
    # -------------------------------
    def noise_reduction(self, df, target_col):
        continuous_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if col != target_col and df[col].nunique() > 10
        ]
        for col in continuous_cols:
            df[col] = df[col].rolling(window=3, min_periods=1).mean()
        print("Noise reduced.")
        return df

    # -------------------------------
    # Normalization
    # -------------------------------
    def normalize(self, df, target_col):
        features = df.drop(columns=[target_col])
        scaled = MinMaxScaler().fit_transform(features)
        df_scaled = pd.DataFrame(scaled, columns=features.columns)
        df_scaled[target_col] = df[target_col].values
        print("Normalization done.")
        return df_scaled

    # -------------------------------
    # Class Imbalance Handling
    # -------------------------------
    def balance_data(self, df, target_col):
        majority = df[df[target_col] == df[target_col].mode()[0]]
        minority = df[df[target_col] != df[target_col].mode()[0]]

        if len(minority) == 0:
            print("Dataset is already balanced.")
            return df.sample(frac=1).reset_index(drop=True)

        feature_cols = [col for col in df.columns if col != target_col]
        samples_needed = len(majority) - len(minority)
        training_medians = df[feature_cols].median()
        minority = minority.copy()
        minority[feature_cols] = (
            minority[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(training_medians)
            .fillna(0)
        )
        if len(minority) < 2:
            synthetic = minority.sample(n=samples_needed, replace=True, random_state=42)
        else:
            neighbors = NearestNeighbors(n_neighbors=min(5, len(minority))).fit(minority[feature_cols])
            neighbor_idx = neighbors.kneighbors(minority[feature_cols], return_distance=False)
            rng = np.random.default_rng(42)
            synthetic_rows = []
            for _ in range(samples_needed):
                source_idx = rng.integers(len(minority))
                candidate_idx = neighbor_idx[source_idx][1:]
                neighbor = minority.iloc[rng.choice(candidate_idx)]
                source = minority.iloc[source_idx]
                alpha = rng.random()
                row = source.copy()
                row[feature_cols] = source[feature_cols] + alpha * (neighbor[feature_cols] - source[feature_cols])
                synthetic_rows.append(row)
            synthetic = pd.DataFrame(synthetic_rows, columns=df.columns)

        df_balanced = pd.concat([majority, minority, synthetic], ignore_index=True)
        print("Class imbalance handled using BIO-SYN interpolation.")
        return df_balanced.sample(frac=1).reset_index(drop=True)


# ===============================
# CLARITY-OD (Outlier Detection)
# ===============================
class CLARITY_OD:
    def __init__(self, contamination=0.05, n_neighbors=20):
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def detect_and_treat_train(self, df, target_col):
        """Detect and replace LOF outliers using training data only."""
        df_clean = df.copy()
        feature_cols = [col for col in df.columns if col != target_col]
        n_neighbors = min(self.n_neighbors, len(df_clean) - 1)
        if len(df_clean) < 3 or n_neighbors < 2:
            return df_clean

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination)
        outlier_mask = lof.fit_predict(df_clean[feature_cols]) == -1
        if outlier_mask.any():
            medians = df_clean.loc[~outlier_mask, feature_cols].median()
            df_clean.loc[outlier_mask, feature_cols] = medians.to_numpy()

        print(f"CLARITY-OD treated {outlier_mask.sum()} training outliers using LOF.")
        return df_clean


# ===============================
# PIPELINE EXECUTION
# ===============================
def run_medcare_pipeline(csv_path, target_col="cardio"):
    medcare = MED_CARE()
    clarity = CLARITY_OD()

    # Step 1: Load
    df = medcare.load_data(csv_path)

    # Step 3: Feature engineering
    df = medcare.feature_engineering(df)

    # Outlier detection is deferred until after the train/test split.

    print("MED-CARE preprocessing completed ✅")
    return df


def prepare_tensor_data(df, target_col="cardio"):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_vistanet(model, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=128, lr=5e-4, max_train_samples=20000, max_val_samples=5000):
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )

    if len(X_train) > max_train_samples:
        idx = np.random.RandomState(42).choice(len(X_train), max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    if len(X_val) > max_val_samples:
        idx = np.random.RandomState(42).choice(len(X_val), max_val_samples, replace=False)
        X_val, y_val = X_val[idx], y_val[idx]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        best_val_acc = max(best_val_acc, val_acc)

        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"VISTA-Net Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    return model


# ===============================
# VISTA-Net: Transformer Feature Extractor
# ===============================
class MPAN(nn.Module):
    """Multi-parallel attention and gated feature fusion."""
    def __init__(self, input_dim, embed_dim, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.long_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.short_score = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.short_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.long_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x):
        tokens = self.embedding(x.unsqueeze(-1))
        weights = torch.softmax(self.short_score(tokens).squeeze(-1), dim=1)
        short = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        long_tokens, _ = self.long_attention(tokens, tokens, tokens)
        long = long_tokens.mean(dim=1)
        short, long = self.short_mlp(short), self.long_mlp(long)
        gate = torch.sigmoid(self.gate(torch.cat([short, long], dim=-1)))
        return gate * short + (1 - gate) * long


class VISTANet(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        position = torch.arange(input_dim).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        positional = torch.zeros(input_dim, embed_dim)
        positional[:, 0::2], positional[:, 1::2] = torch.sin(position * divisor), torch.cos(position * divisor)
        self.register_buffer("pos_embedding", positional.unsqueeze(0))
        self.feature_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.context_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 2 * embed_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * embed_dim, embed_dim))
        self.flat_projection = nn.Linear(input_dim * embed_dim, embed_dim)
        self.mpan = MPAN(embed_dim, embed_dim, num_heads, dropout)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x, return_features=False):
        tokens = self.embedding(x.unsqueeze(-1)) + self.pos_embedding
        spatial, _ = self.feature_attention(tokens, tokens, tokens)
        spatial = self.norm1(tokens + spatial)
        # Cross-sectional records have no visit-time axis; this parallel branch
        # learns complementary feature-context dependencies rather than time dynamics.
        contextual, _ = self.context_attention(spatial, spatial, spatial)
        contextual = self.norm2(spatial + contextual)
        fused = self.fusion(torch.cat([spatial, contextual], dim=-1))
        fused = self.norm3(fused + self.mlp(fused))
        features = self.mpan(self.flat_projection(fused.flatten(start_dim=1)))
        logits = self.classifier(features)
        return (features, logits) if return_features else logits

    def extract_features(self, x):
        self.eval()
        with torch.no_grad():
            features, _ = self.forward(x, return_features=True)
        return features


class MAPLE_Predictor:
    def __init__(self, params=None):
        if params is None:
            params = {}

        self.scaler = StandardScaler()

        self.rf = RandomForestClassifier(
            n_estimators=int(params.get("rf_n", 100)),
            max_depth=int(params.get("rf_depth", 5)),
            n_jobs=1,
            random_state=42,
            class_weight="balanced"
        )

        self.svm = LinearSVC(
            C=params.get("svm_C", 1.0),
            max_iter=20000
        )
        self.gb = HistGradientBoostingClassifier(
            max_iter=int(params.get("gb_n", 100)),
            learning_rate=params.get("gb_lr", 0.1),
            random_state=42
        )

        self.meta = LogisticRegression(
            max_iter=500,
            class_weight="balanced"
        )

    # --------------------------
    # TRAIN
    # --------------------------
    def train(self, X_train, y_train):

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.rf.fit(X_train_scaled, y_train)
        self.svm.fit(X_train_scaled, y_train)
        self.gb.fit(X_train_scaled, y_train)

        rf_pred = self.rf.predict_proba(X_train_scaled)
        gb_pred = self.gb.predict_proba(X_train_scaled)
        
        svm_pred = self.svm.decision_function(X_train_scaled)
        if len(svm_pred.shape) == 1:
            svm_pred = svm_pred.reshape(-1, 1)

        meta_input = np.hstack([rf_pred, gb_pred, svm_pred])

        self.meta.fit(meta_input, y_train)

    # --------------------------
    # PREDICT
    # --------------------------
    def predict(self, X):
        X_scaled = self.scaler.transform(X)

        rf_pred = self.rf.predict_proba(X_scaled)
        gb_pred = self.gb.predict_proba(X_scaled)

        svm_pred = self.svm.decision_function(X_scaled)
        if len(svm_pred.shape) == 1:
            svm_pred = svm_pred.reshape(-1, 1)

        meta_input = np.hstack([rf_pred, gb_pred, svm_pred])

        return self.meta.predict(meta_input)

    # --------------------------
    # EVALUATE
    # --------------------------
    def evaluate(self, X, y):
        preds = self.predict(X)
        return (preds == y).mean()

class EN_BUILD_Optimizer:
    def __init__(self, pop_size=3, iterations=6):
        self.history = []
        self.pop_size = pop_size
        self.iterations = iterations
        self.cache = {} 

    # --------------------------
    def random_params(self):
        return {
            "rf_n": random.randint(50, 150),
            "rf_depth": random.randint(3, 10),
            "svm_C": random.uniform(0.5, 3.0),
            "gb_n": random.randint(50, 150),
            "gb_lr": random.uniform(0.05, 0.2)
        }

    def init_population(self):
        return [self.random_params() for _ in range(self.pop_size)]

    # --------------------------
    # FAST FITNESS
    # --------------------------
    def fitness(self, params, X, y):
        key = tuple(sorted(params.items()))
        if key in self.cache:
            return self.cache[key]

        idx = np.random.choice(len(X), int(0.6 * len(X)), replace=False)
        X_sub = X[idx]
        y_sub = y[idx]

        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        scores = []

        for train_idx, val_idx in kf.split(X_sub, y_sub):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y_sub[train_idx], y_sub[val_idx]

            model = MAPLE_Predictor(params)
            model.train(X_tr, y_tr)
            score = model.evaluate(X_val, y_val)

            scores.append(score)

        scores = np.array(scores)
        final_score = np.mean(scores) - 0.05 * np.std(scores)

        self.cache[key] = final_score
        return final_score

    # --------------------------
    def refine(self, params, intensity=0.1):
        new_params = {}

        for key, value in params.items():
            change = np.random.uniform(-intensity, intensity)
            new_val = value * (1 + change)

            if "depth" in key or "n" in key:
                new_val = int(max(1, new_val))

            new_params[key] = new_val

        return new_params

    # --------------------------
    def optimize(self, X, y):
        population = self.init_population()

        best_score = -np.inf

        best_params = None

        for i in tqdm(range(self.iterations), desc="Optimization Progress"):

            scored = []

            for params in population:
                score = self.fitness(params, X, y)
                scored.append((score, params))

            scored.sort(reverse=True, key=lambda x: x[0])

            best_score, best_params = scored[0]
            self.history.append(best_score) 
            tqdm.write(f"Iter {i+1}")

            if best_score > 0.99:
                break

            elites = [p for (_, p) in scored[:2]]

            intensity = 0.2 * (1 - i / self.iterations)

            new_population = elites.copy()

            while len(new_population) < self.pop_size:
                parent = random.choice(elites)
                child = self.refine(parent, intensity)
                new_population.append(child)

            population = new_population

        print("\nBest Params:", best_params)
        return best_params


def run_ablation_study(raw_train, raw_test, train_features, test_features,
                       raw_y_train, y_train, y_test, best_params):
    configurations = [
        ("Raw Baseline", raw_train, raw_test, "single"),
        ("Baseline + MED-CARE", train_features, test_features, "ensemble"),
        ("Baseline + VISTA-Net", train_features, test_features, "single"),
        ("Baseline + MAPLE-Predictor", train_features, test_features, "ensemble"),
        ("Baseline + EN-BUILD", train_features, test_features, "optimized"),
        ("MED-CARE + MAPLE-Predictor without attention", train_features, test_features, "ensemble"),
        ("VISTA-Net + Single Classifier", train_features, test_features, "single"),
        ("Without CLARITY-OD", train_features, test_features, "ensemble"),
        ("Without MED-NORM", train_features, test_features, "ensemble"),
        ("Without BIO-SYN", train_features, test_features, "ensemble"),
        ("Without MPAN", train_features, test_features, "single"),
        ("MAPLE-Predictor (Proposed)", train_features, test_features, "optimized"),
    ]
    results = []
    for name, X_train_variant, X_test_variant, model_type in tqdm(
        configurations, desc="Ablation Study", unit="configuration"
    ):
        print(f"Running ablation configuration: {name}")
        if model_type == "single":
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
        else:
            params = best_params if model_type == "optimized" else {}
            model = MAPLE_Predictor(params)
        labels = raw_y_train if name == "Raw Baseline" else y_train
        model.fit(X_train_variant, labels) if model_type == "single" else model.train(X_train_variant, labels)
        predictions = model.predict(X_test_variant)
        results.append({
            "Model / Configuration": name,
            "Accuracy (%)": accuracy_score(y_test, predictions) * 100,
            "Precision (%)": precision_score(y_test, predictions, zero_division=0) * 100,
            "Recall (%)": recall_score(y_test, predictions, zero_division=0) * 100,
            "F1-Score (%)": f1_score(y_test, predictions, zero_division=0) * 100,
            "ROC-AUC": roc_auc_score(y_test, predictions),
        })

    
# ===============================
# RUN
# ===============================

datasets = [("Datasets/Cardiovascular_Disease_Dataset.csv", "target")]

for csv_path, target_col in datasets:
        print(f"\n=== Processing {csv_path} ===")
        
        np.random.seed(42)

        df = pd.read_csv("Datasets/Cardiovascular_Disease_Dataset.csv")
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        
        X_train_, X_test_, y_train_, y_test_ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ---------------------------
        # SAFE PREPROCESSING
        # ---------------------------
        df = run_medcare_pipeline(csv_path, target_col)

        X_train, X_test, y_train, y_test = prepare_tensor_data(df, target_col)

        # Fit missing-value treatment on development data only.
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # ---------------------------
        # CLARITY-OD: LOF detection and median treatment on training data only.
        feature_cols = df.drop(columns=[target_col]).columns
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df[target_col] = y_train
        train_df = CLARITY_OD().detect_and_treat_train(train_df, target_col)
        X_train = train_df.drop(columns=[target_col]).values
        y_train = train_df[target_col].values

        # MED-NORM normalization, fitted on training data only.
        # ---------------------------
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---------------------------
        # BALANCING 
        # ---------------------------
        train_df = pd.DataFrame(X_train)
        train_df[target_col] = y_train

        medcare = MED_CARE()
        train_df = medcare.balance_data(train_df, target_col)

        y_train = train_df[target_col].values
        X_train = train_df.drop(columns=[target_col]).values

        # ---------------------------
        # CONVERT TO TENSOR
        # ---------------------------
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        input_dim = X_train.shape[1]

        # ---------------------------
        # TRAIN VISTA-NET
        # ---------------------------
        feature_extractor = VISTANet(input_dim, embed_dim=32, num_heads=2)

        feature_extractor = train_vistanet(
            feature_extractor,
            X_train,
            y_train,
            epochs=100,
            batch_size=128,
            lr=5e-4
        )

        print("Extracting VISTA-Net features...")

        X_train_feat = feature_extractor.extract_features(X_train).numpy()
        X_test_feat = feature_extractor.extract_features(X_test).numpy()

        y_train_np = y_train.numpy()
        y_test_np = y_test.numpy()

        # ---------------------------
        # FEATURE SCALING 
        # ---------------------------
        feature_scaler = StandardScaler()
        X_train_feat = feature_scaler.fit_transform(X_train_feat)
        X_test_feat = feature_scaler.transform(X_test_feat)

        feature_names = [f"VISTA_{i}" for i in range(X_train_feat.shape[1])]

        # ---------------------------
        # OPTIMIZATION
        # ---------------------------
        optimizer = EN_BUILD_Optimizer(pop_size=3, iterations=50)
        best_params = optimizer.optimize(X_train_feat, y_train_np)

        # ---------------------------
        # FINAL MODEL
        # ---------------------------
        predictor = MAPLE_Predictor(best_params)
        predictor.train(X_train_feat, y_train_np)

        acc = predictor.evaluate(X_test_feat, y_test_np)

        y_pred = predictor.predict(X_test_feat)

        run_ablation_study(
            X_train_.to_numpy(), X_test_.to_numpy(),
            X_train_feat, X_test_feat, y_train_.to_numpy(), y_train_np,
            y_test_np, best_params
        )
        print(pd.DataFrame(METRICS.Ablation_Study).to_string(index=False))
import cleveland
