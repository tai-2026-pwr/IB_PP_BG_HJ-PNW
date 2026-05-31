from src.model_helpers.one_class import OneClassModels
from src.model_helpers.sklearn import ClassicalModel
from src.paths import (
    TEST_BALANCED_PATH,
    TEST_IMBALANCED_PATH,
    TRAIN_BALANCED_PATH,
    TRAIN_IMBALANCED_PATH,
)
from src.utils.representations import Representation
from src.utils.runner import build_output_dir

classical_ml_configs = [
    (ClassicalModel.SVM, "balanced", TRAIN_BALANCED_PATH, TEST_BALANCED_PATH),
    (ClassicalModel.SVM, "imbalanced", TRAIN_IMBALANCED_PATH, TEST_IMBALANCED_PATH),
    (ClassicalModel.RANDOM_FOREST, "balanced", TRAIN_BALANCED_PATH, TEST_BALANCED_PATH),
    (ClassicalModel.RANDOM_FOREST, "imbalanced", TRAIN_IMBALANCED_PATH, TEST_IMBALANCED_PATH),
    (ClassicalModel.GRADIENT_BOOSTING, "balanced", TRAIN_BALANCED_PATH, TEST_BALANCED_PATH),
    (ClassicalModel.GRADIENT_BOOSTING, "imbalanced", TRAIN_IMBALANCED_PATH, TEST_IMBALANCED_PATH),
]


one_class_ml_configs = [
    (OneClassModels.SVM, "balanced", TRAIN_BALANCED_PATH, TEST_BALANCED_PATH),
    (OneClassModels.IFOREST, "balanced", TRAIN_BALANCED_PATH, TEST_BALANCED_PATH),
]

output_dir_aac_classical = build_output_dir("results/classical_ml/aac")
output_dir_dpc_classical = build_output_dir("results/classical_ml/dpc")
output_dir_physicochemical_classical = build_output_dir("results/classical_ml/physicochemical")

representations_configs_classic_ml = [
    (Representation.AAC, output_dir_aac_classical),
    (Representation.DPC, output_dir_dpc_classical),
    (Representation.PHYSICOCHEMICAL, output_dir_physicochemical_classical),
]

output_dir_aac_oneclass = build_output_dir("results/one_class/aac")
output_dir_physicochemical_oneclass = build_output_dir("results/one_class/physicochemical")

representations_configs_one_class = [
    (Representation.AAC, output_dir_aac_oneclass),
    (Representation.PHYSICOCHEMICAL, output_dir_physicochemical_oneclass),
]
