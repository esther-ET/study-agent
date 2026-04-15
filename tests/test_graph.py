from src.agent.graph import build_graph, app

def test_graph_exists():
    assert app is not None
    assert hasattr(app, 'invoke')

def test_graph_nodes():
    graph = build_graph()
    node_names = list(graph.nodes.keys())
    assert "intent_parser" in node_names
    assert "search_executor" in node_names
    assert "formatter" in node_names