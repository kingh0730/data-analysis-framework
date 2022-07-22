from itertools import chain
import json
from typing import Any
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from utils import django_setup


from base import models
from base.views import get_all_history_of_specific_data_frame_file

st.write(
    """
# Data frame file history
"""
)


# Helper functions


def nodes_from_data_frame_file_json(dff_json: Any) -> tuple[list[Node], list[Edge]]:
    if dff_json is None:
        return [], []

    this_id = f"DataFrameFile ({dff_json['pk']})"

    # This edge
    upper_job_json = dff_json["upper_job"]
    upper_job_id = (
        f"Job ({upper_job_json['pk']})" if upper_job_json is not None else None
    )
    this_edges = (
        [
            Edge(
                target=this_id,
                source=upper_job_id,
                color="red",
            )
        ]
        if upper_job_json is not None
        else []
    )

    # This node
    this_node = Node(
        id=this_id,
        # label=f"DataFrameFile({dff_json['pk']})",
        label=dff_json["pk"],
        symbolType="cross" if upper_job_json is None else None,
        color="deepSkyBlue" if upper_job_json is None else None,
    )

    # Nodes and edges
    upper_nodes, upper_edges = nodes_from_job_json(upper_job_json)
    combined_nodes = [this_node] + upper_nodes
    combined_edges = this_edges + upper_edges

    return combined_nodes, combined_edges


def nodes_from_job_json(job_json: Any) -> tuple[list[Node], list[Edge]]:
    if job_json is None:
        return [], []

    inputs_dict_json = job_json["inputs_dict"]

    # This node
    this_id = f"Job ({job_json['pk']})"
    this_node = Node(
        id=this_id,
        label=json.dumps(
            {
                "id": job_json["pk"],
                "job": job_json["model"],
            }
            | {
                k: v
                for k, v in job_json["fields"].items()
                if k[:5] != "input" and k[:6] != "output"
            },
            indent=2,
            sort_keys=False,
        ),
        color="green" if inputs_dict_json else "deepSkyBlue",
        symbolType=None if inputs_dict_json else "cross",
    )

    # This edges
    this_edges = []
    for item_or_items_json in inputs_dict_json.values():
        if item_or_items_json is None:
            continue
        if isinstance(item_or_items_json, dict):
            item_id = f"DataFrameFile ({item_or_items_json['pk']})"
            this_edges.append(
                Edge(
                    target=this_id,
                    source=item_id,
                )
            )
        if isinstance(item_or_items_json, list):
            for item_json in item_or_items_json:
                item_id = f"DataFrameFile ({item_json['pk']})"
                this_edges.append(
                    Edge(
                        target=this_id,
                        source=item_id,
                    )
                )

    # Upper nodes and edges
    def nodes_from_data_frame_file_or_iterable_json(
        item_or_items_json: Any,
    ) -> tuple[list[Node], list[Edge]]:
        if item_or_items_json is None:
            return [], []
        if isinstance(item_or_items_json, dict):
            return nodes_from_data_frame_file_json(item_or_items_json)
        if isinstance(item_or_items_json, list):
            all_nodes_and_edges = [
                nodes_from_data_frame_file_json(item_json)
                for item_json in item_or_items_json
            ]
            return list(
                chain.from_iterable([nodes for nodes, edges in all_nodes_and_edges])
            ), list(
                chain.from_iterable([edges for nodes, edges in all_nodes_and_edges])
            )
        raise ValueError("This JSON is not correct.")

    upper_nodes_and_edges = [
        nodes_from_data_frame_file_or_iterable_json(item_or_items_json)
        for item_or_items_json in inputs_dict_json.values()
    ]

    # Combination
    combined_nodes = [this_node] + list(
        chain.from_iterable([nodes for nodes, edges in upper_nodes_and_edges])
    )
    combined_edges = this_edges + list(
        chain.from_iterable([edges for nodes, edges in upper_nodes_and_edges])
    )
    return combined_nodes, combined_edges


# Inputs


with st.sidebar:
    dff = st.selectbox("Data frame file", models.DataFrameFile.objects.all())


st.write("---")


specific_dff = dff.get_specific_data_frame_file()
json_res = get_all_history_of_specific_data_frame_file(specific_dff)


nodes, edges = nodes_from_data_frame_file_json(json_res)
nodes[0].color = "gold"
nodes[0].symbolType = "star"
if edges:
    edges[0].strokeWidth = 4
else:
    st.error("Only one node.")
# st.write(nodes)
# st.write(edges)


# nodes.append(Node(id="Spider Man", label="Peter Parker"))
# nodes.append(Node(id="Captain Marvel", label="Nobody"))
# edges.append(Edge(source="Captain Marvel", label="friend_of", target="Spider Man"))


config = Config(
    node={"labelProperty": "label"},
    link={"labelProperty": "label", "renderLabel": True},
    d3={"alphaTarget": 0},
)

agraph(nodes=nodes, edges=edges, config=config)


st.write("---")


st.write(json_res)
