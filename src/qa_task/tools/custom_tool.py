from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
import json


class KnowledgeGraphReaderToolInput(BaseModel):
    """Input schema for KnowledgeGraphReaderTool."""
    path: str = Field(
        default="knowledge_graph.json",
        description="Path to the knowledge graph JSON file.",
    )


class KnowledgeGraphReaderTool(BaseTool):
    name: str = "Knowledge Graph File Reader"
    description: str = (
        "Read knowledge graph triples from knowledge_graph.json. "
        "Use this tool to access triples for answering."
    )
    args_schema: Type[BaseModel] = KnowledgeGraphReaderToolInput

    def _run(self, path: str = "knowledge_graph.json") -> str:
        kg_path = Path(path)
        if not kg_path.is_absolute():
            kg_path = Path.cwd() / kg_path
        if not kg_path.exists():
            raise FileNotFoundError(
                f"knowledge_graph.json not found at {kg_path}. "
                "Run knowledge_graph_task first."
            )
        raw = kg_path.read_text().strip()
        if not raw:
            raise ValueError(
                f"knowledge_graph.json is empty at {kg_path}. "
                "Run knowledge_graph_task first and ensure it writes valid JSON."
            )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"knowledge_graph.json contains invalid JSON at {kg_path}: {exc}"
            ) from exc
        return json.dumps(data)


class ParagraphRetrievalToolInput(BaseModel):
    """Input schema for ParagraphRetrievalTool."""
    query: str = Field(
        description="The search query or keywords to find relevant paragraphs."
    )


class ParagraphRetrievalTool(BaseTool):
    name: str = "Paragraph Retrieval Tool"
    description: str = (
        "Search and retrieve relevant paragraphs from the document collection "
        "based on a query. Returns matching paragraphs with their indices and titles."
    )
    args_schema: Type[BaseModel] = ParagraphRetrievalToolInput
    paragraphs: List[Dict[str, Any]] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        """
        Simple keyword-based retrieval from paragraphs.
        Returns paragraphs that contain any of the query keywords.
        """
        if not self.paragraphs:
            return json.dumps({"error": "No paragraphs available", "results": []})
        
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        results = []
        for para in self.paragraphs:
            para_text = para.get("paragraph_text", "").lower()
            para_title = para.get("title", "").lower()
            
            # Simple scoring: count keyword matches
            score = sum(1 for kw in query_keywords if kw in para_text or kw in para_title)
            
            if score > 0:
                results.append({
                    "idx": para.get("idx"),
                    "title": para.get("title"),
                    "paragraph_text": para.get("paragraph_text"),
                    "score": score,
                    "is_supporting": para.get("is_supporting", False)
                })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return json.dumps({
            "query": query,
            "num_results": len(results),
            "results": results[:10]  # Return top 10 matches
        }, ensure_ascii=False, indent=2)
