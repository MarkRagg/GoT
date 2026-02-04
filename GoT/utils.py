def parse_response(res) -> str:
    return res["messages"][-1].content