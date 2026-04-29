"""
Wildcard type marker for ComfyUI node inputs and outputs.

ComfyUI's type matcher compares strings literally, so a bare "*" is
treated as a type named "*" and won't connect to anything. AnyType is a
str subclass whose __ne__ always returns False, so equality checks
against any other type string return True.
"""


class AnyType(str):
    """Wildcard type that matches any input/output type."""

    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")
