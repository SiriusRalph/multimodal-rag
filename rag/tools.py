"""Étape 5 : le retriever multimodal exposé comme outil d'agent."""
from dataclasses import asdict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .retrieval import Retriever, format_results

# Le LLM ne choisit ni k ni la modalité : llama3.2 demandait k=1 ou des valeurs
# invalides et n'interrogeait qu'une modalité, ce qui rendait l'hybride impossible.
TEXT_RESULTS = 5
IMAGE_RESULTS = 2

TOOL_DESCRIPTION = """\
Search the paper "Attention Is All You Need" (the Transformer paper). Every call returns
the best text passages and the best figures of the paper, each with its page number.
Use it for any question about the paper's content, numbers, architecture or figures.
Call it again with a reworded query if the results do not answer the question.
Figures are returned with their caption and file path: you read the caption, not the pixels.
"""


class RetrieverInput(BaseModel):
    query: str | None = Field(
        default=None,
        description="Search terms, always in English because the paper is in English. "
        "Example: 'number of parallel attention layers heads'. Leave empty when image_path is set.",
    )
    image_path: str | None = Field(
        default=None,
        description="Path of an image given by the user, to find similar figures of the paper. "
        "Leave empty when query is set.",
    )


def build_retriever_tool(retriever: Retriever) -> StructuredTool:
    def retriever_tool(query: str | None = None, image_path: str | None = None) -> tuple[str, list[dict]]:
        # Le LLM remplit parfois les deux champs ; l'image fournie par l'utilisateur prime alors.
        request = {"image_path": image_path} if image_path else {"query": query or None}
        try:
            results = retriever.search(**request, k=IMAGE_RESULTS, content_type="image")
            results += retriever.search(**request, k=TEXT_RESULTS, content_type="text")
        except (ValueError, FileNotFoundError) as error:
            return f"Recherche impossible : {error}", []
        return format_results(results), [asdict(r) for r in results]

    return StructuredTool.from_function(
        func=retriever_tool,
        name="retriever_tool",
        description=TOOL_DESCRIPTION,
        args_schema=RetrieverInput,
        response_format="content_and_artifact",
    )
