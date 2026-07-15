from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.propagation.topology import concatenate
from witwin.channel_native.propagation.topology import export


def test_concatenate_helpers_are_same_object_compatibility_exports():
    for name in (
        "_empty_path_block",
        "_block_sequence_width",
        "concatenate_path_blocks",
        "_sort_order",
        "_interaction_type_sequence",
        "canonical_sequence_key",
        "_canonical_selection_order",
        "_pad_topology_sequences",
    ):
        assert getattr(legacy, name) is getattr(concatenate, name)


def test_export_helper_is_same_object_compatibility_export():
    assert legacy._ensure_topology_fields is export._ensure_topology_fields
