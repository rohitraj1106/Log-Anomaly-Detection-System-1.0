"""
Feature Engineering Module — Transform Logs into ML-Ready Features.

Implements comprehensive feature engineering:
- TF-IDF text embeddings
- Log template extraction (via drain-like approach)
- N-gram features
- Time-window aggregation features
- Frequency-based anomaly indicators
- Statistical features

Design: All feature extractors are composable and produce
NumPy arrays or sparse matrices that can be stacked.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.config_loader import ConfigLoader
from utils.helpers import timer
from utils.logger import get_logger

logger = get_logger(__name__)


class LogTemplateExtractor:
    """
    Extract log templates by replacing variable parts with placeholders.

    Implements a simplified Drain-like algorithm for log template mining.
    """

    # Patterns for variable parts in log messages
    VARIABLE_PATTERNS = [
        (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<IP>"),
        (
            re.compile(
                r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
            ),
            "<UUID>",
        ),
        (re.compile(r"\b\d{10,13}\b"), "<TIMESTAMP>"),
        (re.compile(r"\b\d+\.\d+\b"), "<FLOAT>"),
        (re.compile(r"\b\d+\b"), "<NUM>"),
        (re.compile(r"\b[0-9a-fA-F]{6,}\b"), "<HEX>"),
        (re.compile(r"(?<==)\S+"), "<VALUE>"),
        (re.compile(r"(/\S+)+"), "<PATH>"),
    ]

    def __init__(self, max_templates: int = 500) -> None:
        self.max_templates = max_templates
        self._template_cache: dict[str, int] = {}
        self._template_counter: Counter = Counter()

    def extract_template(self, message: str) -> str:
        """Replace variable parts of a log message with placeholders."""
        template = message
        for pattern, placeholder in self.VARIABLE_PATTERNS:
            template = pattern.sub(placeholder, template)
        return template

    def fit_transform(self, messages: pd.Series) -> pd.Series:
        """Extract templates for a series of messages and build vocabulary."""
        templates = messages.astype(str).apply(self.extract_template)
        self._template_counter = Counter(templates)
        # Keep only top N templates
        top_templates = {t for t, _ in self._template_counter.most_common(self.max_templates)}
        self._template_cache = {t: i for i, t in enumerate(sorted(top_templates))}
        return templates

    def encode(self, templates: pd.Series) -> np.ndarray:
        """Encode templates to numeric IDs."""
        return templates.map(lambda t: self._template_cache.get(t, -1)).values


class FeatureEngineer:
    """
    Production feature engineering pipeline for log anomaly detection.

    Generates multiple feature types and combines them into a single
    feature matrix ready for model training/inference.
    """

    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        try:
            self._config = ConfigLoader.load(config_path)
            tfidf_cfg = self._config.get_section("features.tfidf")
            self._max_features = tfidf_cfg.get("max_features", 5000)
            self._ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 3]))
            self._min_df = tfidf_cfg.get("min_df", 2)
            self._max_df = tfidf_cfg.get("max_df", 0.95)
            self._sublinear_tf = tfidf_cfg.get("sublinear_tf", True)
            tw_cfg = self._config.get_section("features.time_windows")
            self._window_sizes = tw_cfg.get("window_sizes_minutes", [1, 5, 15, 60])
            tmpl_cfg = self._config.get_section("features.template")
            self._max_templates = tmpl_cfg.get("max_templates", 500)
        except (FileNotFoundError, Exception):
            self._max_features = 5000
            self._ngram_range = (1, 3)
            self._min_df = 2
            self._max_df = 0.95
            self._sublinear_tf = True
            self._window_sizes = [1, 5, 15, 60]
            self._max_templates = 500

        # Feature extractors
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._template_extractor = LogTemplateExtractor(max_templates=self._max_templates)
        self._is_fitted = False
        self._feature_names: list[str] = []

        logger.info("FeatureEngineer initialized")

    @timer
    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Fit all feature extractors and transform the input DataFrame.

        Args:
            df: Preprocessed log DataFrame with columns:
                timestamp, level, service, message, etc.

        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        logger.info(f"Feature engineering on {len(df)} records...")

        features_list: list[np.ndarray | sparse.spmatrix] = []
        feature_names: list[str] = []

        # 1. TF-IDF embeddings
        tfidf_features, tfidf_names = self._compute_tfidf(df, fit=True)
        features_list.append(tfidf_features)
        feature_names.extend(tfidf_names)

        # 2. Template features
        template_features, template_names = self._compute_template_features(df, fit=True)
        features_list.append(template_features)
        feature_names.extend(template_names)

        # 3. Statistical/numeric features
        stat_features, stat_names = self._compute_statistical_features(df)
        features_list.append(stat_features)
        feature_names.extend(stat_names)

        # 4. Time-window aggregation features
        time_features, time_names = self._compute_time_features(df)
        features_list.append(time_features)
        feature_names.extend(time_names)

        # 5. Frequency-based anomaly indicators
        freq_features, freq_names = self._compute_frequency_features(df)
        features_list.append(freq_features)
        feature_names.extend(freq_names)

        # Combine all features
        dense_features = []
        for f in features_list:
            if sparse.issparse(f):
                dense_features.append(f.toarray())
            else:
                dense_features.append(np.array(f) if not isinstance(f, np.ndarray) else f)

        # Ensure all have same number of rows
        feature_matrix = np.hstack(dense_features)
        self._feature_names = feature_names
        self._is_fitted = True

        logger.info(
            f"Feature matrix: {feature_matrix.shape[0]} samples × "
            f"{feature_matrix.shape[1]} features"
        )
        return feature_matrix, feature_names

    @timer
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted feature extractors."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")

        features_list = []

        # TF-IDF
        tfidf_features, _ = self._compute_tfidf(df, fit=False)
        features_list.append(tfidf_features)

        # Templates
        template_features, _ = self._compute_template_features(df, fit=False)
        features_list.append(template_features)

        # Statistical
        stat_features, _ = self._compute_statistical_features(df)
        features_list.append(stat_features)

        # Time features
        time_features, _ = self._compute_time_features(df)
        features_list.append(time_features)

        # Frequency features
        freq_features, _ = self._compute_frequency_features(df)
        features_list.append(freq_features)

        dense_features = []
        for f in features_list:
            if sparse.issparse(f):
                dense_features.append(f.toarray())
            else:
                dense_features.append(np.array(f) if not isinstance(f, np.ndarray) else f)

        return np.hstack(dense_features)

    def _compute_tfidf(
        self, df: pd.DataFrame, fit: bool = True
    ) -> tuple[np.ndarray | sparse.spmatrix, list[str]]:
        """Compute TF-IDF features from log messages."""
        messages = df["message"].astype(str).fillna("")

        if fit:
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=self._max_features,
                ngram_range=self._ngram_range,
                min_df=self._min_df if len(messages) > self._min_df else 1,
                max_df=self._max_df,
                sublinear_tf=self._sublinear_tf,
                stop_words="english",
            )
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(messages)
            feature_names = [
                f"tfidf_{name}" for name in self._tfidf_vectorizer.get_feature_names_out()
            ]
        else:
            if self._tfidf_vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer not fitted")
            tfidf_matrix = self._tfidf_vectorizer.transform(messages)
            feature_names = [
                f"tfidf_{name}" for name in self._tfidf_vectorizer.get_feature_names_out()
            ]

        logger.info(f"TF-IDF features: {tfidf_matrix.shape[1]} dimensions")
        return tfidf_matrix, feature_names

    def _compute_template_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> tuple[np.ndarray, list[str]]:
        """Compute log template-based features."""
        messages = df["message"].astype(str).fillna("")

        if fit:
            templates = self._template_extractor.fit_transform(messages)
        else:
            templates = messages.apply(self._template_extractor.extract_template)

        # Template ID encoding
        template_ids = self._template_extractor.encode(templates)

        # Template rarity score (inverse frequency)
        template_counts = Counter(templates)
        total = len(templates)
        rarity_scores = templates.map(lambda t: 1.0 - (template_counts.get(t, 1) / total)).values

        features = np.column_stack(
            [
                template_ids.reshape(-1, 1),
                rarity_scores.reshape(-1, 1),
            ]
        )

        feature_names = ["template_id", "template_rarity"]
        return features, feature_names

    def _compute_statistical_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Compute statistical/numeric features from log metadata."""
        features = []
        feature_names = []

        # Message length
        if "message_length" in df.columns:
            features.append(df["message_length"].fillna(0).values)
            feature_names.append("message_length")
        else:
            features.append(df["message"].astype(str).str.len().values)
            feature_names.append("message_length")

        # Log level numeric
        if "level_numeric" in df.columns:
            features.append(df["level_numeric"].fillna(1).values)
        else:
            level_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "WARN": 2, "ERROR": 3, "CRITICAL": 4}
            features.append(df["level"].astype(str).str.upper().map(level_map).fillna(1).values)
        feature_names.append("level_numeric")

        # Hour of day
        if "hour_of_day" in df.columns:
            features.append(df["hour_of_day"].fillna(12).values)
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            features.append(ts.dt.hour.fillna(12).values)
        else:
            features.append(np.full(len(df), 12))
        feature_names.append("hour_of_day")

        # Day of week
        if "day_of_week" in df.columns:
            features.append(df["day_of_week"].fillna(0).values)
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            features.append(ts.dt.dayofweek.fillna(0).values)
        else:
            features.append(np.full(len(df), 0))
        feature_names.append("day_of_week")

        # Has IP
        if "has_ip" in df.columns:
            features.append(df["has_ip"].fillna(0).values)
        else:
            features.append(np.zeros(len(df)))
        feature_names.append("has_ip")

        # Word count in message
        word_counts = df["message"].astype(str).str.split().str.len().fillna(0).values
        features.append(word_counts)
        feature_names.append("word_count")

        # Special character ratio
        special_chars = (
            df["message"]
            .astype(str)
            .apply(
                lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / max(len(x), 1)
            )
            .values
        )
        features.append(special_chars)
        feature_names.append("special_char_ratio")

        # Uppercase ratio
        upper_ratio = (
            df["message"]
            .astype(str)
            .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
            .values
        )
        features.append(upper_ratio)
        feature_names.append("uppercase_ratio")

        return np.column_stack(features), feature_names

    def _compute_time_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Compute time-window aggregation features."""
        features = []
        feature_names = []

        if "timestamp" not in df.columns:
            return np.zeros((len(df), 1)), ["time_placeholder"]

        ts = pd.to_datetime(df["timestamp"], errors="coerce")

        for window_min in self._window_sizes:
            window = f"{window_min}min"

            # Count of logs per time window
            df_temp = df.copy()
            df_temp["_ts"] = ts
            df_temp["_window"] = df_temp["_ts"].dt.floor(window)
            window_counts = df_temp.groupby("_window").size()
            df_temp["_window_count"] = df_temp["_window"].map(window_counts).fillna(0)
            features.append(df_temp["_window_count"].values)
            feature_names.append(f"log_count_{window_min}min")

        # Inter-arrival time (seconds between consecutive logs)
        if len(ts.dropna()) > 1:
            sorted_ts = ts.sort_values()
            inter_arrival = sorted_ts.diff().dt.total_seconds().fillna(0).values
            features.append(inter_arrival)
        else:
            features.append(np.zeros(len(df)))
        feature_names.append("inter_arrival_seconds")

        return np.column_stack(features), feature_names

    def _compute_frequency_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Compute frequency-based anomaly indicator features."""
        features = []
        feature_names = []

        # Service frequency (how common is this service in the batch)
        if "service" in df.columns:
            service_counts = df["service"].value_counts()
            total = len(df)
            service_freq = df["service"].map(service_counts / total).fillna(0).values
            features.append(service_freq)
            feature_names.append("service_frequency")

            # Service rarity (inverse frequency)
            service_rarity = 1.0 - service_freq
            features.append(service_rarity)
            feature_names.append("service_rarity")

        # Log level rarity within service
        if "level" in df.columns and "service" in df.columns:
            level_service = df.groupby(["service", "level"]).size().reset_index(name="count")
            level_service_total = df.groupby("service").size().reset_index(name="total")
            level_service = level_service.merge(level_service_total, on="service")
            level_service["ratio"] = level_service["count"] / level_service["total"]

            level_rarity_map = {}
            for _, row in level_service.iterrows():
                level_rarity_map[(row["service"], row["level"])] = 1.0 - row["ratio"]

            level_rarity = df.apply(
                lambda r: level_rarity_map.get((r["service"], r["level"]), 0.5),
                axis=1,
            ).values
            features.append(level_rarity)
            feature_names.append("level_rarity_in_service")

        # IP frequency
        if "source_ip" in df.columns:
            ip_counts = df["source_ip"].value_counts()
            ip_freq = df["source_ip"].map(ip_counts / len(df)).fillna(0).values
            features.append(ip_freq)
            feature_names.append("ip_frequency")

        if not features:
            return np.zeros((len(df), 1)), ["freq_placeholder"]

        return np.column_stack(features), feature_names

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names from the last fit_transform."""
        return self._feature_names

    @property
    def tfidf_vectorizer(self) -> TfidfVectorizer | None:
        """Return the fitted TF-IDF vectorizer."""
        return self._tfidf_vectorizer
