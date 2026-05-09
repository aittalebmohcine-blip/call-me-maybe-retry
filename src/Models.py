from pydantic import BaseModel, Field
from typing import Dict, Any


class FunctionDefinition(BaseModel):
    """Represents the definition of a function that can be called.
    
    This model encapsulates metadata about a function including its signature,
    description, and parameter specifications.
    
    Attributes:
        name: The name of the function.
        description: A human-readable description of what the function does.
        parameters: A dictionary mapping parameter names to their type specifications.
            Each parameter entry contains type information (e.g., 'type': 'string').
        returns: A dictionary describing the return value of the function.
    """
    name: str = Field(..., description="The name of the function")
    description: str = Field(...,
                             description="The description of the function")
    parameters: Dict[str, Dict[str, str]
                     ] = Field(..., description="Parameters the function take")
    returns: Dict[str, str] = Field(...,
                                    description="What the function returns")


class FunctionCallOutput(BaseModel):
    """Represents the output of parsing a function call from a prompt.
    
    This model captures the extracted function call information including the
    original prompt, the identified function name, and the parsed parameters.
    
    Attributes:
        prompt: The original user prompt.
        name: The name of the function identified to fulfill the prompt.
        parameters: Extracted function parameters with their values.
    """
    prompt: str = Field(..., description="The user prompt")
    name: str = Field(
        ...,
        description="The function name coresponding to the prompt"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Extratcted parameters from the prompt"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary representation.
        
        Returns:
            A dictionary with keys 'prompt', 'name', and 'parameters'.
        """
        return {
            "prompt": self.prompt,
            "name": self.name,
            "parameters": self.parameters
        }
