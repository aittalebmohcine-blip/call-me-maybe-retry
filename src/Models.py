from pydantic import BaseModel, Field
from typing import Dict, Literal


ParamType = Literal["number", "integer", "string", "boolean"]


class ParameterDefinition(BaseModel):
    type: ParamType


class FunctionDefinition(BaseModel):
    """Represents the definition of a function that can be called.

    This model encapsulates metadata about a function including its signature,
    description, and parameter specifications.

    Attributes:
        name: The name of the function.
        description: A human-readable description of what the function does.
        parameters: A dictionary mapping parameter names to their type
        specifications.
            Each parameter entry contains type information
            (e.g., 'type': 'string').
        returns: A dictionary describing the return value of the function.
    """
    name: str = Field(..., description="The name of the function")
    description: str = Field(...,
                             description="The description of the function")
    parameters: Dict[str, ParameterDefinition
                     ] = Field(..., description="Parameters the function take")
    returns: ParameterDefinition = Field(
        ...,
        description="What the function returns"
    )
