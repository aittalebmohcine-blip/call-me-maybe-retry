from pydantic import BaseModel, Field
from typing import Dict, Any


class FunctionDefinition(BaseModel):
    name: str = Field(..., description="The name of the function")
    description: str = Field(...,
                             description="The description of the function")
    parameters: Dict[str, Dict[str, str]
                     ] = Field(..., description="Parameters the function take")
    returns: Dict[str, str] = Field(...,
                                    description="What the function returns")


class FunctionCallOutput(BaseModel):
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
        return {
            "prompt": self.prompt,
            "name": self.name,
            "parameters": self.parameters
        }
