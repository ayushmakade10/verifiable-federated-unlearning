"""
evidence/ — Evidence Bundle Components
========================================

Provides model/file hashing, the participation log access layer,
and evidence bundle assembly utilities.
"""

from evidence.hashing import hash_file, hash_model
from evidence.participation_log import ParticipationLog
from evidence.bundle import (
    build_manifest,
    create_unlearning_request,
    save_frozen_config,
    save_frozen_config_from_dict,
)

__all__ = [
    "hash_model",
    "hash_file",
    "ParticipationLog",
    "build_manifest",
    "create_unlearning_request",
    "save_frozen_config",
    "save_frozen_config_from_dict",
]
