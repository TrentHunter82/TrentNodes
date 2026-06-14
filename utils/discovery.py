"""
Node discovery utilities for TrentNodes.

Automatically discovers and registers ComfyUI nodes from the nodes/ package.
"""
import importlib
import inspect
import pkgutil
import re


def pretty_name(name: str) -> str:
    """
    Convert CamelCase class name to spaced display name.

    Args:
        name: CamelCase class name (e.g., "HelloImageNode")

    Returns:
        Spaced name (e.g., "Hello Image Node")
    """
    return re.sub(r"(?<!^)([A-Z])", r" \1", name).strip()


def is_comfyui_node(obj) -> bool:
    """
    Check if an object looks like a ComfyUI node class.

    Args:
        obj: Object to check

    Returns:
        True if object has required ComfyUI node attributes
    """
    required_attrs = ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION")
    return inspect.isclass(obj) and all(
        hasattr(obj, attr) for attr in required_attrs
    )


def register_node(
    node_class,
    class_mappings: dict,
    display_mappings: dict,
    default_category: str = "Trent/Auto"
):
    """
    Register a single node class into the mappings.

    Args:
        node_class: The node class to register
        class_mappings: NODE_CLASS_MAPPINGS dict to update
        display_mappings: NODE_DISPLAY_NAME_MAPPINGS dict to update
        default_category: Default category if node has none
    """
    name = node_class.__name__
    class_mappings[name] = node_class

    # Ensure a category so it groups properly
    if not hasattr(node_class, "CATEGORY"):
        setattr(node_class, "CATEGORY", default_category)

    # Set display name if not already provided
    display_name = getattr(node_class, "DISPLAY_NAME", pretty_name(name))
    display_mappings.setdefault(name, display_name)


def discover_nodes(
    package_name: str,
    class_mappings: dict,
    display_mappings: dict
):
    """
    Discover and register all nodes from the nodes subpackage.

    Args:
        package_name: The root package name (e.g., __name__ from __init__.py)
        class_mappings: NODE_CLASS_MAPPINGS dict to populate
        display_mappings: NODE_DISPLAY_NAME_MAPPINGS dict to populate
    """
    # Import the "nodes" subpackage
    try:
        nodes_pkg = importlib.import_module(f"{package_name}.nodes")
    except Exception as e:
        print(f"[TrentNodes] Could not import 'nodes' package: {e}")
        return

    prefix = f"{package_name}.nodes."

    for _finder, modname, _ispkg in pkgutil.walk_packages(
        nodes_pkg.__path__, prefix=prefix
    ):
        try:
            module = importlib.import_module(modname)
        except Exception as e:
            print(f"[TrentNodes] Import error in {modname}: {e}")
            continue

        # Preferred: module provides explicit mappings
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                display_mappings.update(
                    getattr(module, "NODE_DISPLAY_NAME_MAPPINGS")
                )
            continue

        # Fallback: auto-register classes that look like ComfyUI nodes
        for attr, obj in vars(module).items():
            if is_comfyui_node(obj):
                register_node(obj, class_mappings, display_mappings)
