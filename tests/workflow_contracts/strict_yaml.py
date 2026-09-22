"""A YAML loader for workflows that refuses what PyYAML would resolve silently.

PyYAML keeps the last of two equal mapping keys and reports nothing. Every
workflow contract reads its documents through this loader instead, so a
duplicate is a failure the contract names rather than a value it never saw.

Run via ``make test-workflow-contracts``.
"""

import typing as typ

import yaml


class StrictWorkflowLoader(yaml.SafeLoader):
    """A safe loader that refuses a mapping declaring one key twice.

    PyYAML keeps the last of two equal keys and says nothing, so a job
    declaring `runs-on` twice reads as whichever label came second while
    the first is what a reviewer sees. GitHub does not agree to that
    reading, and neither may a contract: a duplicate is refused here, before
    any reader can answer a question about a value that was discarded.

    Examples
    --------
    >>> parse_workflow_text("runs-on: a\\nruns-on: b\\n")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    yaml.constructor.ConstructorError: ...duplicate key 'runs-on'...
    """


def _construct_unique_mapping(
    loader: StrictWorkflowLoader, node: yaml.MappingNode, *, deep: bool = False
) -> dict[typ.Any, typ.Any]:
    """Construct one mapping, refusing a key it has already declared."""
    loader.flatten_mapping(node)
    seen: set[typ.Any] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


StrictWorkflowLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def parse_workflow_text(text: str) -> object:
    """Parse workflow text, refusing a duplicate mapping key.

    Examples
    --------
    >>> parse_workflow_text("on: push\\n")
    {True: 'push'}
    """
    return yaml.load(text, Loader=StrictWorkflowLoader)
