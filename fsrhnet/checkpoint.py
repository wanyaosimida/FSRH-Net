"""Checkpoint-key compatibility for the canonical FSRH-Net module names.

Only exact module-path prefixes are translated. Tensor values are neither
copied nor discarded, and ambiguous keys are rejected. Training-framework
containers and prefixes such as ``net.`` or ``module.`` must be handled by
the caller before loading a model-only state dictionary.
"""

from collections import OrderedDict
from collections.abc import Mapping


def _model_prefix_renames():
    renames = [("backbone.last_layer", "backbone.compatibility_head")]
    for level in (4, 3, 2):
        old = f"decoder.block{level}"
        new = f"decoder.refine{level}"
        for old_suffix, new_suffix in (
            ("norm1", "context_norm"),
            ("norm2", "scan_norm"),
            ("frequency_descriptor", "spatial_descriptor"),
            ("scale_router", "scale_predictor"),
            ("state_mixer.fc1", "scan_refinement.input_projection"),
            ("state_mixer.fc2", "scan_refinement.output_projection"),
            ("state_mixer.act", "scan_refinement.activation"),
            ("state_mixer.attn.state_router", "scan_refinement.routed_scan.route_predictor"),
            ("state_mixer.attn.branch_gain", "scan_refinement.routed_scan.modulation_gain"),
            ("state_mixer.attn.local_context_gain", "scan_refinement.routed_scan.refinement_gain"),
            ("state_mixer.attn", "scan_refinement.routed_scan"),
            ("state_mixer", "scan_refinement"),
        ):
            renames.append((f"{old}.{old_suffix}", f"{new}.{new_suffix}"))
        renames.append((old, new))

    for level in (3, 2, 1):
        old = f"decoder.skip{level}"
        new = f"decoder.skip_projection{level}"
        renames.extend((
            (f"{old}.current_proj", f"{new}.input_projection"),
            (f"{old}.fuse", f"{new}.refinement"),
            (old, new),
            (f"decoder.hssf{level}", f"decoder.fusion{level}"),
        ))
    renames.append(("decoder.reduce_to_l1", "decoder.shallow_projection"))

    old, new = "decoder.seg_head", "decoder.segmentation_head"
    for old_suffix, new_suffix in (
        ("MSC.b0", "multiscale_context.branch1"),
        ("MSC.b1", "multiscale_context.branch2"),
        ("MSC.b2", "multiscale_context.branch3"),
        ("MSC.project", "multiscale_context.output_projection"),
        ("MSC", "multiscale_context"),
        ("avgpool", "channel_pool"),
        ("fc", "channel_gate"),
        ("spat_atten", "spatial_gate"),
        ("kv", "compatibility_projection"),
        ("head", "classifier"),
    ):
        renames.append((f"{old}.{old_suffix}", f"{new}.{new_suffix}"))
    renames.append((old, new))
    return tuple(sorted(renames, key=lambda pair: len(pair[0]), reverse=True))


_MODEL_PREFIX_RENAMES = _model_prefix_renames()
_BACKBONE_PREFIX_RENAMES = (("last_layer", "compatibility_head"),)


def _rename_path(path, renames):
    if not isinstance(path, str):
        raise TypeError("Checkpoint keys and metadata paths must be strings.")
    for old, new in renames:
        if path == old or path.startswith(old + "."):
            return new + path[len(old):]
    return path


def _remap_mapping(source, renames, kind):
    mapped = OrderedDict()
    originals = {}
    for name, value in source.items():
        target = _rename_path(name, renames)
        if target in mapped:
            raise RuntimeError(
                f"Checkpoint {kind} collision: {originals[target]!r} and "
                f"{name!r} both map to {target!r}."
            )
        mapped[target] = value
        originals[target] = name
    return mapped


def _remap_state_dict(state_dict, renames):
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a model-only state dictionary, not a checkpoint container.")
    result = _remap_mapping(state_dict, renames, "key")
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("state_dict._metadata must be a mapping.")
        remapped_metadata = _remap_mapping(metadata, renames, "metadata")
        # PyTorch may add assign-related flags to each module's metadata.
        # Keep those changes separate from the caller's checkpoint object.
        result._metadata = OrderedDict(
            (name, dict(value) if isinstance(value, Mapping) else value)
            for name, value in remapped_metadata.items()
        )
    return result


def remap_model_state_dict(state_dict):
    """Translate supported earlier model keys to the current module names.

    Current keys pass through unchanged. Unknown keys are retained so that
    ``load_state_dict(..., strict=True)`` can report them. If both an earlier
    and a current key name refer to the same destination, conversion fails
    rather than choosing one value. Module-version metadata is also renamed;
    unused metadata entries are preserved for PyTorch to handle.
    """
    return _remap_state_dict(state_dict, _MODEL_PREFIX_RENAMES)


def remap_backbone_state_dict(state_dict):
    """Translate encoder-local checkpoint keys without stripping prefixes."""
    return _remap_state_dict(state_dict, _BACKBONE_PREFIX_RENAMES)
