from __future__ import annotations

import operator
import os
import re
import json

from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from json_repair import repair_json

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

load_dotenv()

# =========================================================
# SCHEMAS
# =========================================================

class Task(BaseModel):
    id: int

    title: str

    goal: str

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
    )

    target_words: int

    tags: List[str] = Field(
        default_factory=list
    )

    requires_research: bool = False

    requires_citations: bool = False

    requires_code: bool = False


class Plan(BaseModel):

    blog_title: str

    audience: str

    tone: str

    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
    ] = "explainer"

    constraints: List[str] = Field(
        default_factory=list
    )

    tasks: List[Task]


class EvidenceItem(BaseModel):

    title: str

    url: str

    published_at: Optional[str] = None

    snippet: Optional[str] = None

    source: Optional[str] = None


class RouterDecision(BaseModel):

    needs_research: bool

    mode: Literal[
        "closed_book",
        "hybrid",
        "open_book",
    ]

    reason: str

    queries: List[str] = Field(
        default_factory=list
    )

    max_results_per_query: int = 5


class EvidencePack(BaseModel):

    evidence: List[EvidenceItem] = Field(
        default_factory=list
    )


class ImageSpec(BaseModel):

    placeholder: str

    filename: str

    alt: str

    caption: str

    prompt: str

    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024",
    ] = "1024x1024"

    quality: Literal[
        "low",
        "medium",
        "high",
    ] = "medium"


class GlobalImagePlan(BaseModel):

    md_with_placeholders: str

    images: List[ImageSpec] = Field(
        default_factory=list
    )


# =========================================================
# STATE
# =========================================================

class State(TypedDict):

    topic: str

    mode: str

    needs_research: bool

    queries: List[str]

    evidence: List[EvidenceItem]

    plan: Optional[Plan]

    as_of: str

    recency_days: int

    sections: Annotated[
        List[tuple[int, str]],
        operator.add
    ]

    merged_md: str

    md_with_placeholders: str

    image_specs: List[dict]

    final: str


# =========================================================
# MODEL
# =========================================================

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",

    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    ),

    temperature=0.2,

    max_new_tokens=4096,

    streaming=True,
)

model = ChatHuggingFace(
    llm=llm
)


# =========================================================
# STRUCTURED OUTPUT HELPER
# =========================================================

def structured_llm_output(
    messages,
    schema,
):

    response = model.invoke(
        messages
    )

    text = response.content.strip()

    # remove markdown
    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    try:

        # normal parse
        data = json.loads(text)

    except Exception:

        try:

            # repair broken json
            repaired = repair_json(
                text
            )

            data = json.loads(
                repaired
            )

        except Exception as e:

            print(
                "\nFAILED JSON:\n"
            )

            print(text)

            raise e

    return schema(**data)


# =========================================================
# ROUTER
# =========================================================

ROUTER_SYSTEM = """
You are a routing module.

Return ONLY pure JSON.

DO NOT explain anything.
DO NOT use markdown.

Schema:
{
    "needs_research": true,
    "mode": "closed_book",
    "reason": "text",
    "queries": [],
    "max_results_per_query": 5
}
"""


def router_node(state: State):

    messages = [

        SystemMessage(
            content=ROUTER_SYSTEM
        ),

        HumanMessage(
            content=f"""
Topic:
{state['topic']}

As-of date:
{state['as_of']}
"""
        ),
    ]

    decision = structured_llm_output(
        messages,
        RouterDecision
    )

    if decision.mode == "open_book":

        recency_days = 7

    elif decision.mode == "hybrid":

        recency_days = 45

    else:

        recency_days = 3650

    return {
        "needs_research":
            decision.needs_research,

        "mode":
            decision.mode,

        "queries":
            decision.queries,

        "recency_days":
            recency_days,
    }


def route_next(state: State):

    if state["needs_research"]:
        return "research"

    return "orchestrator"


# =========================================================
# TAVILY SEARCH
# =========================================================

def _tavily_search(
    query: str,
    max_results: int = 5
):

    if not os.getenv(
        "TAVILY_API_KEY"
    ):
        return []

    try:

        from langchain_community.tools.tavily_search import (
            TavilySearchResults,
        )

        tool = TavilySearchResults(
            max_results=max_results
        )

        results = tool.invoke(
            {"query": query}
        )

        out = []

        for r in results or []:

            out.append(
                {
                    "title":
                        r.get("title", ""),

                    "url":
                        r.get("url", ""),

                    "snippet":
                        r.get("content", ""),

                    "published_at":
                        r.get(
                            "published_date"
                        ),

                    "source":
                        r.get("source"),
                }
            )

        return out

    except Exception as e:

        print(e)

        return []


def _iso_to_date(
    s: Optional[str]
):

    if not s:
        return None

    try:

        return date.fromisoformat(
            s[:10]
        )

    except Exception:
        return None


# =========================================================
# RESEARCH
# =========================================================

RESEARCH_SYSTEM = """
You are a research synthesizer.

Return ONLY pure JSON.

DO NOT explain anything.
DO NOT use markdown.

Schema:
{
  "evidence": [
    {
      "title": "text",
      "url": "https://...",
      "published_at": "2025-01-01",
      "snippet": "text",
      "source": "text"
    }
  ]
}
"""


def research_node(
    state: State
):

    queries = state.get(
        "queries",
        []
    )

    raw = []

    for q in queries:

        raw.extend(
            _tavily_search(
                q,
                max_results=6
            )
        )

    if not raw:

        return {
            "evidence": []
        }

    messages = [

        SystemMessage(
            content=RESEARCH_SYSTEM
        ),

        HumanMessage(
            content=f"""
As-of date:
{state['as_of']}

Raw Results:
{raw}
"""
        ),
    ]

    pack = structured_llm_output(
        messages,
        EvidencePack
    )

    dedup = {}

    for e in pack.evidence:

        if e.url:

            dedup[e.url] = e

    evidence = list(
        dedup.values()
    )

    if state["mode"] == "open_book":

        as_of = date.fromisoformat(
            state["as_of"]
        )

        cutoff = as_of - timedelta(
            days=state["recency_days"]
        )

        evidence = [
            e
            for e in evidence
            if (
                d := _iso_to_date(
                    e.published_at
                )
            )
            and d >= cutoff
        ]

    return {
        "evidence": evidence
    }


# =========================================================
# ORCHESTRATOR
# =========================================================

ORCH_SYSTEM = """
You are a senior technical writer.

Create a blog outline.

Return ONLY pure JSON.

DO NOT explain anything.
DO NOT use markdown.

Schema:
{
  "blog_title": "text",
  "audience": "text",
  "tone": "text",
  "blog_kind": "explainer",
  "constraints": [],
  "tasks": []
}
"""


def orchestrator_node(
    state: State
):

    messages = [

        SystemMessage(
            content=ORCH_SYSTEM
        ),

        HumanMessage(
            content=f"""
Topic:
{state['topic']}

Mode:
{state['mode']}

Evidence:
{[e.model_dump() for e in state.get('evidence', [])][:15]}
"""
        ),
    ]

    plan = structured_llm_output(
        messages,
        Plan
    )

    return {
        "plan": plan
    }


# =========================================================
# FANOUT
# =========================================================

def fanout(
    state: State
):

    return [

        Send(
            "worker",

            {
                "task":
                    task.model_dump(),

                "plan":
                    state["plan"].model_dump(),

                "topic":
                    state["topic"],

                "mode":
                    state["mode"],

                "evidence": [
                    e.model_dump()
                    for e in state.get(
                        "evidence",
                        []
                    )
                ],
            },
        )

        for task in state["plan"].tasks
    ]


# =========================================================
# WORKER
# =========================================================

WORKER_SYSTEM = """
You are a senior technical writer.

Write ONE markdown section.

Return only markdown.
"""


def worker_node(
    payload: dict
):

    task = Task(
        **payload["task"]
    )

    plan = Plan(
        **payload["plan"]
    )

    evidence = [

        EvidenceItem(**e)

        for e in payload.get(
            "evidence",
            []
        )
    ]

    bullets_text = "\n".join(
        f"- {b}"
        for b in task.bullets
    )

    evidence_text = "\n".join(
        f"- {e.title} | {e.url}"
        for e in evidence[:10]
    )

    response = model.invoke(
        [

            SystemMessage(
                content=WORKER_SYSTEM
            ),

            HumanMessage(
                content=f"""
Blog Title:
{plan.blog_title}

Section Title:
{task.title}

Goal:
{task.goal}

Bullets:
{bullets_text}

Evidence:
{evidence_text}
"""
            ),
        ]
    )

    section_md = (
        response.content.strip()
    )

    return {
        "sections": [
            (
                task.id,
                section_md
            )
        ]
    }


# =========================================================
# MERGE CONTENT
# =========================================================

def merge_content(
    state: State
):

    plan = state["plan"]

    ordered_sections = [

        md

        for _, md in sorted(
            state["sections"],
            key=lambda x: x[0]
        )
    ]

    body = "\n\n".join(
        ordered_sections
    )

    merged_md = f"""
# {plan.blog_title}

{body}
"""

    return {
        "merged_md": merged_md
    }


# =========================================================
# DECIDE IMAGES
# =========================================================

DECIDE_IMAGES_SYSTEM = """
Decide if images are needed.

Return ONLY pure JSON.

DO NOT explain anything.
DO NOT use markdown.

Schema:
{
  "md_with_placeholders": "markdown",
  "images": []
}
"""


def decide_images(
    state: State
):

    messages = [

        SystemMessage(
            content=DECIDE_IMAGES_SYSTEM
        ),

        HumanMessage(
            content=state["merged_md"]
        ),
    ]

    image_plan = structured_llm_output(
        messages,
        GlobalImagePlan
    )

    return {
        "md_with_placeholders":
            image_plan.md_with_placeholders,

        "image_specs": [
            img.model_dump()
            for img in image_plan.images
        ],
    }


# =========================================================
# GEMINI IMAGE
# =========================================================

def _gemini_generate_image_bytes(
    prompt: str
):

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.getenv(
            "GOOGLE_API_KEY"
        )
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",

        contents=prompt,

        config=types.GenerateContentConfig(
            response_modalities=[
                "IMAGE"
            ]
        ),
    )

    parts = (
        resp.candidates[0]
        .content
        .parts
    )

    for part in parts:

        inline = getattr(
            part,
            "inline_data",
            None
        )

        if (
            inline
            and getattr(
                inline,
                "data",
                None
            )
        ):
            return inline.data

    raise RuntimeError(
        "No image generated"
    )


def _safe_slug(
    title: str
):

    s = title.lower().strip()

    s = re.sub(
        r"[^a-z0-9 _-]",
        "",
        s
    )

    s = re.sub(
        r"\s+",
        "_",
        s
    )

    return s


# =========================================================
# GENERATE IMAGES
# =========================================================

def generate_and_place_images(
    state: State
):

    plan = state["plan"]

    md = (
        state.get(
            "md_with_placeholders"
        )
        or state["merged_md"]
    )

    image_specs = state.get(
        "image_specs",
        []
    )

    if not image_specs:

        filename = (
            f"{_safe_slug(plan.blog_title)}.md"
        )

        Path(filename).write_text(
            md,
            encoding="utf-8"
        )

        return {
            "final": md
        }

    images_dir = Path(
        "images"
    )

    images_dir.mkdir(
        exist_ok=True
    )

    for spec in image_specs:

        placeholder = spec[
            "placeholder"
        ]

        filename = spec[
            "filename"
        ]

        out_path = (
            images_dir / filename
        )

        try:

            img_bytes = (
                _gemini_generate_image_bytes(
                    spec["prompt"]
                )
            )

            out_path.write_bytes(
                img_bytes
            )

            img_md = f"""
![{spec['alt']}](images/{filename})

*{spec['caption']}*
"""

            md = md.replace(
                placeholder,
                img_md
            )

        except Exception as e:

            md = md.replace(
                placeholder,
                f"\nImage generation failed: {e}\n"
            )

    filename = (
        f"{_safe_slug(plan.blog_title)}.md"
    )

    Path(filename).write_text(
        md,
        encoding="utf-8"
    )

    return {
        "final": md
    }


# =========================================================
# REDUCER SUBGRAPH
# =========================================================

reducer_graph = StateGraph(
    State
)

reducer_graph.add_node(
    "merge_content",
    merge_content
)

reducer_graph.add_node(
    "decide_images",
    decide_images
)

reducer_graph.add_node(
    "generate_and_place_images",
    generate_and_place_images
)

reducer_graph.add_edge(
    START,
    "merge_content"
)

reducer_graph.add_edge(
    "merge_content",
    "decide_images"
)

reducer_graph.add_edge(
    "decide_images",
    "generate_and_place_images"
)

reducer_graph.add_edge(
    "generate_and_place_images",
    END
)

reducer_subgraph = (
    reducer_graph.compile()
)


# =========================================================
# MAIN GRAPH
# =========================================================

g = StateGraph(
    State
)

g.add_node(
    "router",
    router_node
)

g.add_node(
    "research",
    research_node
)

g.add_node(
    "orchestrator",
    orchestrator_node
)

g.add_node(
    "worker",
    worker_node
)

g.add_node(
    "reducer",
    reducer_subgraph
)

g.add_edge(
    START,
    "router"
)

g.add_conditional_edges(
    "router",

    route_next,

    {
        "research":
            "research",

        "orchestrator":
            "orchestrator",
    },
)

g.add_edge(
    "research",
    "orchestrator"
)

g.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"]
)

g.add_edge(
    "worker",
    "reducer"
)

g.add_edge(
    "reducer",
    END
)

app = g.compile()

print(
    "Graph Compiled Successfully"
)