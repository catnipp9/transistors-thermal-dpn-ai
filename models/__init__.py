"""
DPN Classification Models Package
"""

from .data_loader import (
    ThermogramDataset,
    create_data_loaders,
    load_data_for_sklearn,
    get_transforms
)

from .preprocessing import (
    normalize_temperature,
    extract_statistical_features,
    extract_region_features,
    calculate_asymmetry_features,
    extract_all_features,
    apply_pca,
    select_best_features
)

from .model import (
    ThermogramCNN,
    LightweightCNN,
    TemperatureMatrixCNN,
    ThermogramResNet,
    create_random_forest,
    create_svm,
    create_gradient_boosting,
    create_mlp,
    create_logistic_regression,
    get_model
)

from .trainer import (
    CNNTrainer,
    SklearnTrainer,
    EarlyStopping,
    cross_validate_sklearn,
    print_metrics
)

__all__ = [
    # Data loading
    'ThermogramDataset',
    'create_data_loaders',
    'load_data_for_sklearn',
    'get_transforms',

    # Preprocessing
    'normalize_temperature',
    'extract_statistical_features',
    'extract_region_features',
    'calculate_asymmetry_features',
    'extract_all_features',
    'apply_pca',
    'select_best_features',

    # Models
    'ThermogramCNN',
    'LightweightCNN',
    'TemperatureMatrixCNN',
    'ThermogramResNet',
    'create_random_forest',
    'create_svm',
    'create_gradient_boosting',
    'create_mlp',
    'create_logistic_regression',
    'get_model',

    # Training
    'CNNTrainer',
    'SklearnTrainer',
    'EarlyStopping',
    'cross_validate_sklearn',
    'print_metrics'
]
